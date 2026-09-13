"""
Honest end-to-end evaluation: runs the REAL two-stage pipeline (retrieval +
grounding, exactly as pipeline.py does it) on each test example, and computes
R@1 at IoU=0.5/0.7 against ground truth.

Unlike src/grounding/eval.py, this does NOT use the oracle GT-centered local
crop from dataset.py -- it gives the model a real video and query, same as
production use, and stage 1's retrieval mistakes are fully "in play".

Usage:
    python eval_e2e.py --jsonl data/charades_sta/test_subset.jsonl \
        --video_dir data/charades_sta/videos_subset --ckpt grounding_head.pt \
        --limit 100
"""
import argparse
import json
from src.pipeline import MomentRetrievalPipeline


def iou_1d(pred, gt):
    pred_s, pred_e = pred
    gt_s, gt_e = gt
    inter = max(0.0, min(pred_e, gt_e) - max(pred_s, gt_s))
    union = max(pred_e, gt_e) - min(pred_s, gt_s)
    if union <= 0:
        return 0.0
    return inter / union


def main(args):
    items = [json.loads(l) for l in open(args.jsonl)]
    if args.limit:
        items = items[: args.limit]

    pipeline = MomentRetrievalPipeline(grounding_ckpt=args.ckpt)

    hits_05, hits_07, total, skipped = 0, 0, 0, 0
    for i, item in enumerate(items):
        if args.feature_dir:
            input_path = f"{args.feature_dir}/{item['vid']}.npz"
        else:
            input_path = f"{args.video_dir}/{item['vid']}.mp4"

        gt_start, gt_end = item["relevant_windows"][0]
        try:
            preds = pipeline.run(
                input_path,
                item["query"],
                use_grounding=True,
                fusion_alpha=args.fusion_alpha,
            )
        except FileNotFoundError:
            skipped += 1
            continue
        if not preds:
            skipped += 1
            continue

        top = preds[0]
        iou = iou_1d((top.start_sec, top.end_sec), (gt_start, gt_end))
        total += 1
        if iou >= 0.5:
            hits_05 += 1
        if iou >= 0.7:
            hits_07 += 1

        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(items)} processed  "
                  f"(running R@1 IoU=0.5: {hits_05/total:.4f}, IoU=0.7: {hits_07/total:.4f})")

    print(f"\n=== Honest end-to-end results ({total} examples, {skipped} skipped, alpha={args.fusion_alpha}) ===")
    print(f"R@1 IoU=0.5: {hits_05/total:.4f}")
    print(f"R@1 IoU=0.7: {hits_07/total:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--video_dir", default=None, help="Path to raw mp4 directory")
    ap.add_argument("--feature_dir", default=None, help="Path to cached .npz features directory (much faster)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fusion_alpha", type=float, default=0.5,
                    help="Weight for Stage 1 retrieval similarity (0.0=pure grounding, 1.0=pure retrieval)")
    args = ap.parse_args()
    if not args.video_dir and not args.feature_dir:
        ap.error("Either --video_dir or --feature_dir must be provided.")
    main(args)