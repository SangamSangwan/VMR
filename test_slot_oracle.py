"""
Investigate MomentDETRLite query slot oracle and multi-query NMS:
1. What is the oracle R@1 if the best of the 5 queries is chosen?
2. What happens if we consider all 5 queries with 1D Soft-NMS / NMS?
"""
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


def nms_1d(proposals, iou_thresh=0.5):
    """proposals: list of (start, end, score) sorted by score desc"""
    keep = []
    while proposals:
        best = proposals.pop(0)
        keep.append(best)
        proposals = [p for p in proposals if iou_1d((p[0], p[1]), (best[0], best[1])) < iou_thresh]
    return keep


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ClipEmbedder(device=device)

    model = MomentDETRLite().to(device)
    model.load_state_dict(torch.load("grounding_head_v2_confloss.pt", map_location=device))
    model.eval()

    items = [json.loads(l) for l in open("data/charades_sta/test_subset.jsonl")][:200]
    feature_dir = "data/charades_sta/features"

    hits_argmax_05, hits_argmax_07 = 0, 0
    hits_fused_05, hits_fused_07 = 0, 0
    hits_oracle_slot_05, hits_oracle_slot_07 = 0, 0
    hits_all_queries_nms_05, hits_all_queries_nms_07 = 0, 0
    total = 0

    with torch.no_grad():
        for item in items:
            feat_path = os.path.join(feature_dir, f"{item['vid']}.npz")
            if not os.path.exists(feat_path):
                continue

            data = np.load(feat_path)
            window_embs = data["embeddings"]
            bounds = data["window_bounds"]
            gt = item["relevant_windows"][0]

            query_emb = embedder.embed_text(item["query"])
            candidates = cosine_topk(window_embs, query_emb, bounds, top_k=5)
            merged = merge_adjacent_candidates(candidates)

            argmax_preds = []
            fused_preds = []
            all_queries = []
            cand_oracle_hits_05 = False
            cand_oracle_hits_07 = False

            for cand in merged:
                ctx_start = max(0.0, cand.start_sec - 5.0)
                ctx_end = cand.end_sec + 5.0

                mask = (bounds[:, 0] >= ctx_start) & (bounds[:, 1] <= ctx_end)
                local_feats = torch.from_numpy(window_embs[mask]).float().unsqueeze(0).to(device)
                if local_feats.shape[1] == 0:
                    continue
                q_feat = torch.from_numpy(query_emb).float().unsqueeze(0).to(device)

                centers, widths, conf = model(local_feats, q_feat)
                conf_probs = torch.sigmoid(conf)[0]
                local_bounds = bounds[mask]

                # 1. Standard argmax slot
                best_i = int(conf_probs.argmax())
                s_best, e_best = MomentDETRLite.to_timestamps(
                    centers[0, best_i].item(), widths[0, best_i].item(),
                    local_bounds[0, 0], local_bounds[-1, 1],
                )
                argmax_preds.append((s_best, e_best, float(conf_probs[best_i])))

                sim = max(1e-4, cand.score)
                fused_score = (sim ** 0.5) * (float(conf_probs[best_i]) ** 0.5)
                fused_preds.append((s_best, e_best, fused_score))

                # Check all 5 query slots
                for q in range(centers.shape[1]):
                    s_q, e_q = MomentDETRLite.to_timestamps(
                        centers[0, q].item(), widths[0, q].item(),
                        local_bounds[0, 0], local_bounds[-1, 1],
                    )
                    iou_q = iou_1d((s_q, e_q), gt)
                    if iou_q >= 0.5:
                        cand_oracle_hits_05 = True
                    if iou_q >= 0.7:
                        cand_oracle_hits_07 = True

                    q_score = (sim ** 0.5) * (float(conf_probs[q]) ** 0.5)
                    all_queries.append((s_q, e_q, q_score))

            if not argmax_preds:
                continue

            total += 1

            # Argmax baseline
            argmax_preds.sort(key=lambda p: -p[2])
            if iou_1d((argmax_preds[0][0], argmax_preds[0][1]), gt) >= 0.5:
                hits_argmax_05 += 1
            if iou_1d((argmax_preds[0][0], argmax_preds[0][1]), gt) >= 0.7:
                hits_argmax_07 += 1

            # Fused
            fused_preds.sort(key=lambda p: -p[2])
            if iou_1d((fused_preds[0][0], fused_preds[0][1]), gt) >= 0.5:
                hits_fused_05 += 1
            if iou_1d((fused_preds[0][0], fused_preds[0][1]), gt) >= 0.7:
                hits_fused_07 += 1

            # All queries with NMS
            all_queries.sort(key=lambda p: -p[2])
            nms_ranked = nms_1d(all_queries, iou_thresh=0.6)
            if nms_ranked and iou_1d((nms_ranked[0][0], nms_ranked[0][1]), gt) >= 0.5:
                hits_all_queries_nms_05 += 1
            if nms_ranked and iou_1d((nms_ranked[0][0], nms_ranked[0][1]), gt) >= 0.7:
                hits_all_queries_nms_07 += 1

            # Oracle
            if cand_oracle_hits_05:
                hits_oracle_slot_05 += 1
            if cand_oracle_hits_07:
                hits_oracle_slot_07 += 1

    print(f"\n[n={total}] Slot Diagnosis & Multi-Query Results:")
    print(f"{'Method':<32} | {'R@1 IoU=0.5':<12} | {'R@1 IoU=0.7':<12}")
    print("-" * 62)
    print(f"{'1. Raw Baseline (alpha=0.0)':<32} | {hits_argmax_05/total:<12.4f} | {hits_argmax_07/total:<12.4f}")
    print(f"{'2. Fused (alpha=0.5, 1 slot/cand)':<32} | {hits_fused_05/total:<12.4f} | {hits_fused_07/total:<12.4f}")
    print(f"{'3. Fused + All 5 Queries + NMS':<32} | {hits_all_queries_nms_05/total:<12.4f} | {hits_all_queries_nms_07/total:<12.4f}")
    print(f"{'4. Oracle Slot Selection Ceiling':<32} | {hits_oracle_slot_05/total:<12.4f} | {hits_oracle_slot_07/total:<12.4f}")


if __name__ == "__main__":
    main()
