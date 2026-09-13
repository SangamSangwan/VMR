"""
Quick benchmark script: runs honest end-to-end evaluation using precomputed .npz
features (100x faster than reading raw .mp4) and evaluates Stage 1 + Stage 2
score fusion across different alpha values.
"""
import argparse
import json
import os
import numpy as np
import torch

from src.features.clip_extractor import ClipEmbedder
from src.retrieval.coarse_retrieval import cosine_topk, merge_adjacent_candidates
from src.grounding.model import MomentDETRLite


def iou_1d(pred, gt):
    pred_s, pred_e = pred
    gt_s, gt_e = gt
    inter = max(0.0, min(pred_e, gt_e) - max(pred_s, gt_s))
    union = max(pred_e, gt_e) - min(pred_s, gt_s)
    if union <= 0:
        return 0.0
    return inter / union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="data/charades_sta/test_subset.jsonl")
    parser.add_argument("--feature_dir", default="data/charades_sta/features")
    parser.add_argument("--ckpt", default="grounding_head.pt")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--context_pad_sec", type=float, default=5.0)
    parser.add_argument("--top_k_windows", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ClipEmbedder(device=device)

    grounder = MomentDETRLite().to(device)
    grounder.load_state_dict(torch.load(args.ckpt, map_location=device))
    grounder.eval()

    items = [json.loads(l) for l in open(args.jsonl)]
    if args.limit:
        items = items[:args.limit]

    alphas = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    hits_05 = {a: 0 for a in alphas}
    hits_07 = {a: 0 for a in alphas}
    total = 0

    print(f"Running evaluation on {len(items)} examples with {args.ckpt}...")

    with torch.no_grad():
        for i, item in enumerate(items):
            feat_path = os.path.join(args.feature_dir, f"{item['vid']}.npz")
            if not os.path.exists(feat_path):
                continue

            data = np.load(feat_path)
            window_embs = data["embeddings"]
            bounds = data["window_bounds"]
            gt_start, gt_end = item["relevant_windows"][0]

            query_emb = embedder.embed_text(item["query"])
            effective_k = min(args.top_k_windows, max(1, len(bounds) // 2))
            candidates = cosine_topk(window_embs, query_emb, bounds, top_k=effective_k)
            merged = merge_adjacent_candidates(candidates)

            candidate_preds = []
            for cand in merged:
                ctx_start = max(0.0, cand.start_sec - args.context_pad_sec)
                ctx_end = cand.end_sec + args.context_pad_sec

                mask = (bounds[:, 0] >= ctx_start) & (bounds[:, 1] <= ctx_end)
                local_feats = torch.from_numpy(window_embs[mask]).float().unsqueeze(0).to(device)
                if local_feats.shape[1] == 0:
                    continue
                q_feat = torch.from_numpy(query_emb).float().unsqueeze(0).to(device)

                centers, widths, conf = grounder(local_feats, q_feat)
                conf_probs = torch.sigmoid(conf)[0]
                best_i = int(conf_probs.argmax())

                local_bounds = bounds[mask]
                start_sec, end_sec = MomentDETRLite.to_timestamps(
                    centers[0, best_i].item(),
                    widths[0, best_i].item(),
                    local_bounds[0, 0],
                    local_bounds[-1, 1],
                )
                conf_val = float(conf_probs[best_i])
                sim_val = max(0.0, cand.score)  # Stage 1 cosine similarity
                candidate_preds.append((start_sec, end_sec, conf_val, sim_val))

            if not candidate_preds:
                continue

            total += 1
            for a in alphas:
                # Score fusion: sim^a * conf^(1-a)
                ranked = sorted(
                    candidate_preds,
                    key=lambda p: (p[3] ** a) * (p[2] ** (1.0 - a)),
                    reverse=True,
                )
                top = ranked[0]
                iou = iou_1d((top[0], top[1]), (gt_start, gt_end))
                if iou >= 0.5:
                    hits_05[a] += 1
                if iou >= 0.7:
                    hits_07[a] += 1

    print(f"\nEvaluated {total} examples. Results across alpha (alpha=0 is pure Stage 2, alpha=1 is pure Stage 1):")
    print(f"{'Alpha':<8} | {'R@1 IoU=0.5':<12} | {'R@1 IoU=0.7':<12}")
    print("-" * 38)
    for a in alphas:
        r05 = hits_05[a] / total
        r07 = hits_07[a] / total
        label = " (Baseline)" if a == 0.0 else (" (Coarse only)" if a == 1.0 else "")
        print(f"{a:<8.2f} | {r05:<12.4f} | {r07:<12.4f}{label}")


if __name__ == "__main__":
    main()
