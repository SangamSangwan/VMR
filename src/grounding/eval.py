"""
Computes R@1 at IoU=0.5/0.7 against ground truth, so you can compare your
numbers to lighthouse's published results for the same dataset/model config.

Usage:
    python -m src.grounding.eval --jsonl data/qvhighlights/val.jsonl \
        --feature_dir data/qvhighlights/features --ckpt grounding_head.pt
"""
import argparse
import torch
from torch.utils.data import DataLoader

from src.grounding.dataset import MomentRetrievalDataset
from src.grounding.model import MomentDETRLite
from src.grounding.train import collate


def iou(pred_start, pred_end, gt_start, gt_end):
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return inter / union if union > 0 else 0.0


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MomentRetrievalDataset(args.jsonl, args.feature_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    model = MomentDETRLite().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    hits_05, hits_07, total = 0, 0, 0
    with torch.no_grad():
        for clip_feats, query_feats, gt_center, gt_width in dl:
            clip_feats, query_feats = clip_feats.to(device), query_feats.to(device)
            centers, widths, conf = model(clip_feats, query_feats)
            best = conf.argmax(dim=1)[0]

            pred_c, pred_w = centers[0, best].item(), widths[0, best].item()
            gt_c, gt_w = gt_center.item(), gt_width.item()

            pred_s, pred_e = pred_c - pred_w / 2, pred_c + pred_w / 2
            gt_s, gt_e = gt_c - gt_w / 2, gt_c + gt_w / 2

            score = iou(pred_s, pred_e, gt_s, gt_e)
            hits_05 += score >= 0.5
            hits_07 += score >= 0.7
            total += 1

    print(f"R@1 IoU=0.5: {hits_05/total:.4f}")
    print(f"R@1 IoU=0.7: {hits_07/total:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    evaluate(args)
