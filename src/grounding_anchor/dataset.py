"""
src/grounding_anchor/dataset.py

Fully independent dataset class for the anchor-based grounding head (v2) --
does NOT subclass src/grounding/dataset.py's MomentRetrievalDataset, per
explicit choice. It DOES reproduce the same local-crop + normalization logic
(same context_pad_sec, same "normalize against the local crop, not the whole
video" fix documented in v1's dataset.py) so the coordinate system matches
src/pipeline.py exactly. If that crop logic ever changes in
src/grounding/dataset.py, mirror the change here too -- there's no shared
code to keep it in sync automatically.

Anchor labels depend on the batch's padded max length, so they're built in
collate_fn below, not per-item in __getitem__.
"""
import random

MIN_PAD_SEC = 2.0
MAX_PAD_SEC = 30.0
import json
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.clip_extractor import ClipEmbedder

import glob
import re

_CHUNK_RE = re.compile(r"^(.*)_([0-9.]+)_([0-9.]+)\.npz$")


def _load_video_features(feature_dir: str, vid: str):
    """
    Loads a video's CLIP features. Handles two on-disk layouts:
      1. a single '{vid}.npz' with the whole video's embeddings/window_bounds
      2. multiple '{vid}_{start}_{end}.npz' chunks (150s segments) -- confirmed
         on this dataset that window_bounds INSIDE each chunk are
         chunk-relative (0..~150), not absolute video time. Each chunk's start
         (parsed from its filename) is added back in before chunks are sorted
         and concatenated into one (T, D) / (T, 2) pair covering the whole
         video -- matching the single-file layout's contract exactly, so
         nothing downstream (local-crop logic, normalization) needs to change.
    """
    single_path = os.path.join(feature_dir, f"{vid}.npz")
    if os.path.exists(single_path):
        data = np.load(single_path)
        return data["embeddings"], data["window_bounds"]

    chunk_paths = sorted(
        glob.glob(os.path.join(feature_dir, f"{vid}_*.npz")),
        key=lambda p: float(_CHUNK_RE.match(os.path.basename(p)).group(2)),
    )
    if not chunk_paths:
        raise FileNotFoundError(f"No feature file(s) found for vid={vid} in {feature_dir}")

    embeddings_parts, bounds_parts = [], []
    for p in chunk_paths:
        m = _CHUNK_RE.match(os.path.basename(p))
        chunk_start = float(m.group(2))
        d = np.load(p)
        embeddings_parts.append(d["embeddings"])
        bounds_parts.append(d["window_bounds"] + chunk_start)  # relative -> absolute

    return np.concatenate(embeddings_parts, axis=0), np.concatenate(bounds_parts, axis=0)

CONTEXT_PAD_SEC = 5.0             # keep in sync with MomentRetrievalPipeline.context_pad_sec
SCALES: List[int] = [1, 2, 3, 5, 8]  # must match model.py's SCALES (same ordering)
POS_IOU_THRESH = 0.5


class AnchorGroundingDataset(Dataset):
    def __init__(self, jsonl_path: str, feature_dir: str, embedder: ClipEmbedder = None,
                 context_pad_sec: float = CONTEXT_PAD_SEC):
        self.items = [json.loads(l) for l in open(jsonl_path)]
        self.feature_dir = feature_dir
        self.context_pad_sec = context_pad_sec
        self.embedder = embedder or ClipEmbedder()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        clip_feats_full, window_bounds_full = _load_video_features(self.feature_dir, item["vid"])

        query_feat = torch.from_numpy(self.embedder.embed_text(item["query"])).float()
        gt_start, gt_end = item["relevant_windows"][0]

        video_start = window_bounds_full[0, 0]
        video_end = window_bounds_full[-1, 1]
        front_pad = random.uniform(MIN_PAD_SEC, MAX_PAD_SEC)
        back_pad = random.uniform(MIN_PAD_SEC, MAX_PAD_SEC)
        ctx_start = max(video_start, gt_start - front_pad)
        ctx_end = min(video_end, gt_end + back_pad)

        mask = (window_bounds_full[:, 0] >= ctx_start) & (window_bounds_full[:, 1] <= ctx_end)
        if not mask.any():
            mask = np.ones(len(window_bounds_full), dtype=bool)

        clip_feats = torch.from_numpy(clip_feats_full[mask]).float()       # (T, D)
        local_bounds = torch.from_numpy(window_bounds_full[mask]).float()  # (T, 2) absolute sec

        return {
            "clip_feats": clip_feats,
            "query_feat": query_feat,
            "local_bounds": local_bounds,
            "gt_start": float(gt_start),
            "gt_end": float(gt_end),
        }


def _iou_1d(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def build_anchor_targets(local_bounds: torch.Tensor, T: int, gt_start: float, gt_end: float,
                          scales: List[int] = None, pos_iou_thresh: float = POS_IOU_THRESH):
    """
    local_bounds: (T_real, 2) absolute-second window bounds for ONE sample.
    T: the batch's padded length (Tmax) -- same value passed to
       AnchorGroundingHead.forward's clip_feats, so anchor layout/normalization
       matches exactly.

    Returns, for anchors laid out as [scale0: t=0..T-1, scale1: t=0..T-1, ...]
    (matching AnchorGroundingHead.build_anchors's layout):
        valid: (T*len(scales),) bool -- False for anchors starting at t >= T_real
        label: (T*len(scales),) float in {0,1} -- 1 if IoU(anchor, GT) >= thresh;
               if no anchor clears the threshold, the single best-IoU valid
               anchor is forced positive (SSD-style fallback) so every example
               contributes at least one positive.
    """
    scales = scales or SCALES
    T_real = local_bounds.shape[0]
    num_anchors = T * len(scales)
    valid = torch.zeros(num_anchors, dtype=torch.bool)
    ious = torch.zeros(num_anchors, dtype=torch.float32)

    idx = 0
    for scale in scales:
        for t in range(T):
            if t < T_real:
                end_idx = min(t + scale, T_real)  # exclusive, clipped to real length
                a_start_sec = local_bounds[t, 0].item()
                a_end_sec = local_bounds[end_idx - 1, 1].item()
                ious[idx] = _iou_1d(a_start_sec, a_end_sec, gt_start, gt_end)
                valid[idx] = True
            idx += 1

    label = torch.zeros(num_anchors, dtype=torch.float32)
    if valid.any():
        label[valid & (ious >= pos_iou_thresh)] = 1.0
        if label.sum() == 0:
            masked_iou = torch.where(valid, ious, torch.full_like(ious, -1.0))
            label[torch.argmax(masked_iou)] = 1.0
    return valid, label


def collate_fn(batch):
    B = len(batch)
    Tmax = max(item["clip_feats"].shape[0] for item in batch)
    D = batch[0]["clip_feats"].shape[1]

    clip_feats = torch.zeros(B, Tmax, D)
    feat_mask = torch.zeros(B, Tmax, dtype=torch.bool)
    query_feats = torch.stack([item["query_feat"] for item in batch])

    num_anchors = Tmax * len(SCALES)
    anchor_valid = torch.zeros(B, num_anchors, dtype=torch.bool)
    anchor_label = torch.zeros(B, num_anchors, dtype=torch.float32)

    gt_center = torch.zeros(B)
    gt_width = torch.zeros(B)
    local_start = torch.zeros(B)
    local_end = torch.zeros(B)

    for i, item in enumerate(batch):
        T_real = item["clip_feats"].shape[0]
        clip_feats[i, :T_real] = item["clip_feats"]
        feat_mask[i, :T_real] = True

        valid, label = build_anchor_targets(item["local_bounds"], Tmax, item["gt_start"], item["gt_end"])
        anchor_valid[i] = valid
        anchor_label[i] = label

        ls = item["local_bounds"][0, 0].item()
        le = item["local_bounds"][-1, 1].item()
        local_start[i], local_end[i] = ls, le
        duration = le - ls
        gt_center[i] = ((item["gt_start"] + item["gt_end"]) / 2.0 - ls) / duration
        gt_width[i] = (item["gt_end"] - item["gt_start"]) / duration

    return {
        "clip_feats": clip_feats,
        "feat_mask": feat_mask,
        "query_feat": query_feats,
        "anchor_valid": anchor_valid,
        "anchor_label": anchor_label,
        "gt_center": gt_center,
        "gt_width": gt_width,
        "local_start": local_start,
        "local_end": local_end,
    }