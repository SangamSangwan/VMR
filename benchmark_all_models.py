"""
Comprehensive benchmark: evaluates DETR, Anchor, and Span grounding heads
with and without Stage 1 + Stage 2 score fusion on precomputed features.
"""
import argparse
import json
import os
import numpy as np
import torch

from src.features.clip_extractor import ClipEmbedder
from src.retrieval.coarse_retrieval import cosine_topk, merge_adjacent_candidates
from src.grounding.model import MomentDETRLite
from src.grounding_anchor.model import AnchorGroundingHead
from src.grounding_anchor.model_v2 import EnhancedAnchorGroundingHead
from src.grounding_span.model import SpanGroundingHead


def iou_1d(pred, gt):
    pred_s, pred_e = pred
    gt_s, gt_e = gt
    inter = max(0.0, min(pred_e, gt_e) - max(pred_s, gt_s))
    union = max(pred_e, gt_e) - min(pred_s, gt_s)
    if union <= 0:
        return 0.0
    return inter / union


def evaluate_model(model_type, ckpt_path, items, feature_dir, device, embedder,
                   context_pad_sec=5.0, top_k_windows=5, alphas=[0.0, 0.4, 0.5, 1.0]):
    if model_type == "detr":
        model = MomentDETRLite().to(device)
    elif model_type == "anchor":
        model = AnchorGroundingHead().to(device)
    elif model_type == "anchor_v2":
        model = EnhancedAnchorGroundingHead().to(device)
    elif model_type == "span":
        model = SpanGroundingHead().to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    hits_05 = {a: 0 for a in alphas}
    hits_07 = {a: 0 for a in alphas}
    total = 0

    with torch.no_grad():
        for item in items:
            feat_path = os.path.join(feature_dir, f"{item['vid']}.npz")
            if not os.path.exists(feat_path):
                continue

            data = np.load(feat_path)
            window_embs = data["embeddings"]
            bounds = data["window_bounds"]
            gt_start, gt_end = item["relevant_windows"][0]

            query_emb = embedder.embed_text(item["query"])
            effective_k = min(top_k_windows, max(1, len(bounds) // 2))
            candidates = cosine_topk(window_embs, query_emb, bounds, top_k=effective_k)
            merged = merge_adjacent_candidates(candidates)

            candidate_preds = []
            for cand in merged:
                ctx_start = max(0.0, cand.start_sec - context_pad_sec)
                ctx_end = cand.end_sec + context_pad_sec

                mask = (bounds[:, 0] >= ctx_start) & (bounds[:, 1] <= ctx_end)
                local_feats = torch.from_numpy(window_embs[mask]).float().unsqueeze(0).to(device)
                if local_feats.shape[1] == 0:
                    continue
                q_feat = torch.from_numpy(query_emb).float().unsqueeze(0).to(device)
                local_bounds = bounds[mask]

                if model_type == "detr":
                    centers, widths, conf = model(local_feats, q_feat)
                    conf_probs = torch.sigmoid(conf)[0]
                    best_i = int(conf_probs.argmax())
                    s_sec, e_sec = MomentDETRLite.to_timestamps(
                        centers[0, best_i].item(), widths[0, best_i].item(),
                        local_bounds[0, 0], local_bounds[-1, 1],
                    )
                    conf_val = float(conf_probs[best_i])

                elif model_type in ("anchor", "anchor_v2"):
                    centers, widths, scores = model(local_feats, q_feat)
                    probs = torch.sigmoid(scores)[0]
                    best_i = int(probs.argmax())
                    decode_fn = EnhancedAnchorGroundingHead.to_timestamps if model_type == "anchor_v2" else AnchorGroundingHead.to_timestamps
                    s_sec, e_sec = decode_fn(
                        centers[0, best_i].item(), widths[0, best_i].item(),
                        local_bounds[0, 0], local_bounds[-1, 1],
                    )
                    conf_val = float(probs[best_i])

                elif model_type == "span":
                    start_logits, end_logits, _ = model(local_feats, q_feat)
                    s_sec, e_sec = SpanGroundingHead.decode(start_logits[0], end_logits[0], local_bounds)
                    start_lp = torch.log_softmax(start_logits[0], dim=-1)
                    end_lp = torch.log_softmax(end_logits[0], dim=-1)
                    s_idx = int(np.argmin(np.abs(local_bounds[:, 0] - s_sec)))
                    e_idx = int(np.argmin(np.abs(local_bounds[:, 1] - e_sec)))
                    conf_val = float(torch.exp(start_lp[s_idx] + end_lp[e_idx]))

                sim_val = max(1e-4, cand.score)
                candidate_preds.append((s_sec, e_sec, conf_val, sim_val))

            if not candidate_preds:
                continue

            total += 1
            for a in alphas:
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

    return total, hits_05, hits_07


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="data/charades_sta/test_subset.jsonl")
    parser.add_argument("--feature_dir", default="data/charades_sta/features")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ClipEmbedder(device=device)

    items = [json.loads(l) for l in open(args.jsonl)]
    if args.limit:
        items = items[:args.limit]

    models_to_test = [
        ("Enhanced Anchor v2 (BiGRU + Charades)", "anchor_v2", "grounding_anchor_v2_charades.pt"),
        ("DETR v2 (fixed conf)", "detr", "grounding_head_v2_confloss.pt"),
        ("Anchor v1 cropfix", "anchor", "grounding_anchor_v1_cropfix.pt"),
        ("Span v1 cropfix", "span", "grounding_span_v1_cropfix.pt"),
    ]


    alphas = [0.0, 0.4, 0.5, 1.0]

    print("=" * 65)
    print(f"BENCHMARK ON {len(items)} SAMPLES (Charades-STA Test Subset)")
    print("=" * 65)

    for name, mtype, ckpt in models_to_test:
        if not os.path.exists(ckpt):
            print(f"Skipping {name}: {ckpt} not found.")
            continue
        total, h05, h07 = evaluate_model(mtype, ckpt, items, args.feature_dir, device, embedder, alphas=alphas)
        print(f"\nModel: {name} ({ckpt}) [n={total}]")
        print(f"{'Metric':<18} | {'Raw (alpha=0.0)':<16} | {'Fused (alpha=0.5)':<18} | {'Delta':<10}")
        print("-" * 65)
        raw_05 = h05[0.0] / total
        fused_05 = h05[0.5] / total
        delta_05 = fused_05 - raw_05
        raw_07 = h07[0.0] / total
        fused_07 = h07[0.5] / total
        delta_07 = fused_07 - raw_07
        print(f"{'R@1 IoU=0.5':<18} | {raw_05:<16.4f} | {fused_05:<18.4f} | {delta_05:+.4f}")
        print(f"{'R@1 IoU=0.7':<18} | {raw_07:<16.4f} | {fused_07:<18.4f} | {delta_07:+.4f}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
