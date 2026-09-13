"""
eval_e2e_anchor.py

Honest end-to-end eval for the anchor-based grounding head (v2). Runs the
full two-stage pipeline (CLIP retrieval -> grounding) on real video files and
scores the TOP prediction against GT via IoU -- GT is never used to pick the
stage-1 window, matching the no-leakage contract eval_e2e.py established for
v1. Standalone: does not import eval_e2e.py or src/pipeline.py.

Usage:
    python eval_e2e_anchor.py --test_jsonl <path> --video_dir <dir> \
        --ckpt grounding_anchor_v1.pt
"""
import argparse
import json
import os

import numpy as np
import torch

from src.features.clip_extractor import ClipEmbedder
from src.retrieval.coarse_retrieval import cosine_topk, merge_adjacent_candidates
from src.grounding_anchor.model import AnchorGroundingHead


def iou_1d(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


class AnchorPipeline:
    """Mirrors MomentRetrievalPipeline.run() (src/pipeline.py) with
    AnchorGroundingHead swapped in for MomentDETRLite. Kept standalone here --
    not touching src/pipeline.py until the anchor model is validated and you
    decide whether to merge it in."""

    def __init__(self, grounding_ckpt=None, window_sec=7.5, top_k_windows=5,
                 context_pad_sec=5.0, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = ClipEmbedder(device=self.device)
        self.window_sec = window_sec
        self.top_k_windows = top_k_windows
        self.context_pad_sec = context_pad_sec

        self.grounder = AnchorGroundingHead().to(self.device)
        if grounding_ckpt:
            self.grounder.load_state_dict(torch.load(grounding_ckpt, map_location=self.device))
        self.grounder.eval()

    def run(self, video_path, query):
        window_embs, windows = self.embedder.embed_video(video_path, self.window_sec)
        query_emb = self.embedder.embed_text(query)
        bounds = np.array([[w.start_sec, w.end_sec] for w in windows])

        effective_k = min(self.top_k_windows, max(1, len(windows) // 2))
        candidates = cosine_topk(window_embs, query_emb, bounds, top_k=effective_k)
        merged = merge_adjacent_candidates(candidates)

        predictions = []
        for cand in merged:
            ctx_start = max(0.0, cand.start_sec - self.context_pad_sec)
            ctx_end = cand.end_sec + self.context_pad_sec
            mask = (bounds[:, 0] >= ctx_start) & (bounds[:, 1] <= ctx_end)
            local_feats = torch.from_numpy(window_embs[mask]).float().unsqueeze(0).to(self.device)
            if local_feats.shape[1] == 0:
                continue
            q_feat = torch.from_numpy(query_emb).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                centers, widths, scores = self.grounder(local_feats, q_feat)
            probs = torch.sigmoid(scores)[0]
            best_i = int(probs.argmax())

            local_bounds = bounds[mask]
            start_sec, end_sec = AnchorGroundingHead.to_timestamps(
                centers[0, best_i].item(), widths[0, best_i].item(),
                local_bounds[0, 0], local_bounds[-1, 1],
            )
            predictions.append((start_sec, end_sec, float(probs[best_i])))

        predictions.sort(key=lambda p: -p[2])
        return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--window_sec", type=float, default=7.5)
    ap.add_argument("--top_k_windows", type=int, default=5)
    ap.add_argument("--context_pad_sec", type=float, default=5.0)
    ap.add_argument("--min_gt_duration", type=float, default=0.0)
    ap.add_argument("--max_gt_duration", type=float, default=float("inf"))
    args = ap.parse_args()

    pipeline = AnchorPipeline(grounding_ckpt=args.ckpt, window_sec=args.window_sec,
                               top_k_windows=args.top_k_windows,
                               context_pad_sec=args.context_pad_sec)

    items = [json.loads(l) for l in open(args.test_jsonl)]
    hits_05, hits_07, total = 0, 0, 0

    for item in items:
        video_path = os.path.join(args.video_dir, f"{item['vid']}.mp4")
        if not os.path.exists(video_path):
            continue
        gt_start, gt_end = item["relevant_windows"][0]
        gt_duration = gt_end - gt_start
        if not (args.min_gt_duration <= gt_duration <= args.max_gt_duration):
            continue
        preds = pipeline.run(video_path, item["query"])
        total += 1
        if not preds:
            continue
        top_start, top_end, _ = preds[0]
        iou = iou_1d(top_start, top_end, gt_start, gt_end)
        hits_05 += int(iou >= 0.5)
        hits_07 += int(iou >= 0.7)

    print(f"n={total}")
    if total:
        print(f"R@1 IoU=0.5: {hits_05/total:.4f}")
        print(f"R@1 IoU=0.7: {hits_07/total:.4f}")
    else:
        print("no examples evaluated")


if __name__ == "__main__":
    main()