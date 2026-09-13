"""
src/grounding_span/dataset.py

Independent dataset for the span-based grounding head (v3). Same local-crop +
normalization contract as v1/v2 (context_pad_sec, local-time coordinates
matching src/pipeline.py) and the same chunked-feature-file loader as v2
(feature cache format didn't change, only the model/task did) -- but targets
here are WINDOW INDICES (start_idx, end_idx into the local crop), since the
span predictor outputs per-window logits, not normalized center/width.
"""
import random

MIN_PAD_SEC = 2.0
MAX_PAD_SEC = 30.0
import glob
import json
import os
import re
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.clip_extractor import ClipEmbedder

CONTEXT_PAD_SEC = 5.0
_CHUNK_RE = re.compile(r"^(.*)_([0-9.]+)_([0-9.]+)\.npz$")


def _load_video_features(feature_dir: str, vid: str):
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
        bounds_parts.append(d["window_bounds"] + chunk_start)

    return np.concatenate(embeddings_parts, axis=0), np.concatenate(bounds_parts, axis=0)


class SpanGroundingDataset(Dataset):
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

        clip_feats = torch.from_numpy(clip_feats_full[mask]).float()
        local_bounds = window_bounds_full[mask]  # (T, 2) absolute sec, numpy
        T = local_bounds.shape[0]

        # GT window-index span: every window overlapping [gt_start, gt_end).
        overlap = (local_bounds[:, 1] > gt_start) & (local_bounds[:, 0] < gt_end)
        if overlap.any():
            idxs = np.nonzero(overlap)[0]
            start_idx, end_idx = int(idxs[0]), int(idxs[-1])
            highlight = torch.from_numpy(overlap.astype(np.float32))
        else:
            # GT fell in a gap between windows (shouldn't normally happen given
            # the crop construction) -- fall back to the single nearest window.
            centers = (local_bounds[:, 0] + local_bounds[:, 1]) / 2.0
            nearest = int(np.argmin(np.abs(centers - (gt_start + gt_end) / 2.0)))
            start_idx, end_idx = nearest, nearest
            highlight = torch.zeros(T, dtype=torch.float32)
            highlight[nearest] = 1.0

        return {
            "clip_feats": clip_feats,
            "query_feat": query_feat,
            "local_bounds": torch.from_numpy(local_bounds).float(),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "highlight": highlight,
        }


def collate_fn(batch):
    B = len(batch)
    Tmax = max(item["clip_feats"].shape[0] for item in batch)
    D = batch[0]["clip_feats"].shape[1]

    clip_feats = torch.zeros(B, Tmax, D)
    feat_mask = torch.zeros(B, Tmax, dtype=torch.bool)
    query_feats = torch.stack([item["query_feat"] for item in batch])
    start_idx = torch.zeros(B, dtype=torch.long)
    end_idx = torch.zeros(B, dtype=torch.long)
    highlight = torch.zeros(B, Tmax, dtype=torch.float32)
    local_start = torch.zeros(B)
    local_end = torch.zeros(B)

    for i, item in enumerate(batch):
        T_real = item["clip_feats"].shape[0]
        clip_feats[i, :T_real] = item["clip_feats"]
        feat_mask[i, :T_real] = True
        start_idx[i] = item["start_idx"]
        end_idx[i] = item["end_idx"]
        highlight[i, :T_real] = item["highlight"]
        local_start[i] = item["local_bounds"][0, 0].item()
        local_end[i] = item["local_bounds"][-1, 1].item()

    return {
        "clip_feats": clip_feats,
        "feat_mask": feat_mask,
        "query_feat": query_feats,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "highlight": highlight,
        "local_start": local_start,
        "local_end": local_end,
    }