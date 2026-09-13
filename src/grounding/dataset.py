"""
Dataset wrapper for QVHighlights / Charades-STA once features are cached
as .npz (see src/features/clip_extractor.py). Expects an annotations jsonl
with fields: {vid, query, relevant_windows: [[start_sec, end_sec], ...]}.

QVHighlights ships this format natively (see moment_detr/lighthouse data dir).
For Charades-STA, convert its .txt query files to this jsonl format first —
a converter stub is included below.
"""
import random

MIN_PAD_SEC = 2.0
MAX_PAD_SEC = 30.0
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.clip_extractor import ClipEmbedder

# Keep this in sync with MomentRetrievalPipeline's context_pad_sec in
# src/pipeline.py -- see the note below on why this must match.
CONTEXT_PAD_SEC = 5.0


class MomentRetrievalDataset(Dataset):
    def __init__(self, jsonl_path: str, feature_dir: str, embedder: ClipEmbedder = None,
                 context_pad_sec: float = CONTEXT_PAD_SEC):
        self.items = [json.loads(l) for l in open(jsonl_path)]
        self.feature_dir = feature_dir
        self.context_pad_sec = context_pad_sec
        # text embedding done lazily; reuse one embedder instance across the dataset
        self.embedder = embedder or ClipEmbedder()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        feat_path = os.path.join(self.feature_dir, f"{item['vid']}.npz")
        data = np.load(feat_path)
        clip_feats_full = data["embeddings"]        # (T, D)
        window_bounds_full = data["window_bounds"]  # (T, 2)

        query_feat = torch.from_numpy(self.embedder.embed_text(item["query"])).float()

        gt_start, gt_end = item["relevant_windows"][0]

        # IMPORTANT: at inference (src/pipeline.py), the grounding model never sees
        # the whole video -- it only sees a local crop of windows around a candidate
        # from stage 1, padded by context_pad_sec on each side, and it decodes
        # predictions relative to THAT crop's bounds (see MomentDETRLite.to_timestamps
        # in src/pipeline.py). If we normalized against the whole video here instead,
        # the model would learn "0.5 = middle of the whole video" but at inference
        # "0.5" gets decoded as "middle of this 15-second local snippet" -- a silent
        # coordinate-system mismatch that eval.py alone would never catch, since
        # eval.py just compares normalized-space predictions to normalized-space GT.
        #
        # Fix: mimic the same local crop here at training time.
        video_start = window_bounds_full[0, 0]
        video_end = window_bounds_full[-1, 1]
        front_pad = random.uniform(MIN_PAD_SEC, MAX_PAD_SEC)
        back_pad = random.uniform(MIN_PAD_SEC, MAX_PAD_SEC)
        ctx_start = max(video_start, gt_start - front_pad)
        ctx_end = min(video_end, gt_end + back_pad)

        mask = (window_bounds_full[:, 0] >= ctx_start) & (window_bounds_full[:, 1] <= ctx_end)
        if not mask.any():
            # Very short clips can end up with no window fully inside the padded
            # crop -- fall back to the whole video rather than crashing.
            mask = np.ones(len(window_bounds_full), dtype=bool)

        clip_feats = torch.from_numpy(clip_feats_full[mask]).float()
        local_bounds = window_bounds_full[mask]
        local_start, local_end = local_bounds[0, 0], local_bounds[-1, 1]

        # ground-truth span normalized to [0,1] over the LOCAL crop, matching how
        # MomentDETRLite.to_timestamps decodes predictions at inference time.
        duration = local_end - local_start
        center = ((gt_start + gt_end) / 2 - local_start) / duration
        width = (gt_end - gt_start) / duration

        return {
            "clip_feats": clip_feats,
            "query_feat": query_feat,
            "gt_center": torch.tensor(center, dtype=torch.float32),
            "gt_width": torch.tensor(width, dtype=torch.float32),
            "video_start": local_start,
            "video_end": local_end,
        }


def charades_txt_to_jsonl(txt_path: str, out_jsonl: str):
    """
    Charades-STA lines look like:
        "AO8RW 0.0 5.9##a person turns on a light."
    Converts to the jsonl format MomentRetrievalDataset expects.
    """
    with open(txt_path) as f, open(out_jsonl, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta, query = line.split("##")
            vid, start, end = meta.split(" ")
            out.write(
                json.dumps(
                    {
                        "vid": vid,
                        "query": query.strip(),
                        "relevant_windows": [[float(start), float(end)]],
                    }
                )
                + "\n"
            )