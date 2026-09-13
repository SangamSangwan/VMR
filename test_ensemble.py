"""
test_ensemble.py
Analyzes the error overlap and ensemble potential between MomentDETR v2
and EnhancedAnchorGroundingHead v2 on Charades-STA.
"""
import json
import os
import numpy as np
import torch

from src.features.clip_extractor import ClipEmbedder
from src.retrieval.coarse_retrieval import cosine_topk, merge_adjacent_candidates
from src.grounding.model import MomentDETRLite
from src.grounding_anchor.model_v2 import EnhancedAnchorGroundingHead
from benchmark_fusion import iou_1d


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ClipEmbedder(device=device)

    detr = MomentDETRLite().to(device)
    detr.load_state_dict(torch.load("grounding_head_v2_confloss.pt", map_location=device))
    detr.eval()

    anchor = EnhancedAnchorGroundingHead().to(device)
    anchor.load_state_dict(torch.load("grounding_anchor_v2_charades.pt", map_location=device))
    anchor.eval()

    items = [json.loads(l) for l in open("data/charades_sta/test_subset.jsonl")][:200]
    feature_dir = "data/charades_sta/features"

    detr_hits_05, anchor_hits_05, union_hits_05 = 0, 0, 0
    detr_hits_07, anchor_hits_07, union_hits_07 = 0, 0, 0
    blended_hits_05, blended_hits_07 = 0, 0
    total = 0

    with torch.no_grad():
        for item in items:
            p = os.path.join(feature_dir, f"{item['vid']}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            embs = d["embeddings"]
            bnds = d["window_bounds"]
            gt = item["relevant_windows"][0]

            q_emb = embedder.embed_text(item["query"])
            cands = merge_adjacent_candidates(cosine_topk(embs, q_emb, bnds, top_k=5))

            detr_preds, anchor_preds, blended_preds = [], [], []

            for c in cands:
                mask = (bnds[:, 0] >= max(0.0, c.start_sec - 5.0)) & (bnds[:, 1] <= c.end_sec + 5.0)
                lf = torch.from_numpy(embs[mask]).float().unsqueeze(0).to(device)
                if lf.shape[1] == 0:
                    continue
                qf = torch.from_numpy(q_emb).float().unsqueeze(0).to(device)
                lb = bnds[mask]
                sim = max(1e-4, c.score)

                # DETR
                ct, wt, cf = detr(lf, qf)
                bi = int(torch.sigmoid(cf)[0].argmax())
                s_d, e_d = MomentDETRLite.to_timestamps(ct[0, bi].item(), wt[0, bi].item(), lb[0, 0], lb[-1, 1])
                score_d = (sim ** 0.5) * (float(torch.sigmoid(cf)[0][bi]) ** 0.5)
                detr_preds.append((s_d, e_d, score_d))

                # Anchor
                ca, wa, sa = anchor(lf, qf)
                bia = int(torch.sigmoid(sa)[0].argmax())
                s_a, e_a = EnhancedAnchorGroundingHead.to_timestamps(ca[0, bia].item(), wa[0, bia].item(), lb[0, 0], lb[-1, 1])
                score_a = (sim ** 0.5) * (float(torch.sigmoid(sa)[0][bia]) ** 0.5)
                anchor_preds.append((s_a, e_a, score_a))

                # Blended: average boundary, multiplied confidence
                s_blend = 0.6 * s_d + 0.4 * s_a
                e_blend = 0.6 * e_d + 0.4 * e_a
                score_blend = 0.6 * score_d + 0.4 * score_a
                blended_preds.append((s_blend, e_blend, score_blend))

            if not detr_preds or not anchor_preds:
                continue

            total += 1
            detr_preds.sort(key=lambda x: -x[2])
            anchor_preds.sort(key=lambda x: -x[2])
            blended_preds.sort(key=lambda x: -x[2])

            d_hit_05 = iou_1d((detr_preds[0][0], detr_preds[0][1]), gt) >= 0.5
            a_hit_05 = iou_1d((anchor_preds[0][0], anchor_preds[0][1]), gt) >= 0.5
            b_hit_05 = iou_1d((blended_preds[0][0], blended_preds[0][1]), gt) >= 0.5

            d_hit_07 = iou_1d((detr_preds[0][0], detr_preds[0][1]), gt) >= 0.7
            a_hit_07 = iou_1d((anchor_preds[0][0], anchor_preds[0][1]), gt) >= 0.7
            b_hit_07 = iou_1d((blended_preds[0][0], blended_preds[0][1]), gt) >= 0.7

            if d_hit_05: detr_hits_05 += 1
            if a_hit_05: anchor_hits_05 += 1
            if d_hit_05 or a_hit_05: union_hits_05 += 1
            if b_hit_05: blended_hits_05 += 1

            if d_hit_07: detr_hits_07 += 1
            if a_hit_07: anchor_hits_07 += 1
            if d_hit_07 or a_hit_07: union_hits_07 += 1
            if b_hit_07: blended_hits_07 += 1

    print("\n" + "=" * 65)
    print(f"ENSEMBLE ANALYSIS [n={total}]")
    print("=" * 65)
    print(f"{'Model / Ensemble Strategy':<35} | {'R@1 IoU=0.5':<12} | {'R@1 IoU=0.7':<12}")
    print("-" * 65)
    print(f"{'1. MomentDETR v2 (Standalone)':<35} | {detr_hits_05/total:<12.4f} | {detr_hits_07/total:<12.4f}")
    print(f"{'2. Enhanced Anchor v2 (Standalone)':<35} | {anchor_hits_05/total:<12.4f} | {anchor_hits_07/total:<12.4f}")
    print(f"{'3. Simple Boundary/Score Blend':<35} | {blended_hits_05/total:<12.4f} | {blended_hits_07/total:<12.4f}")
    print(f"{'4. Theoretical Oracle Union':<35} | {union_hits_05/total:<12.4f} | {union_hits_07/total:<12.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
