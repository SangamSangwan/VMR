"""
Trains MomentDETRLite on cached CLIP features. (Diagnostic version: prints
per-component loss so we can see which term is/isn't improving.)
"""
import argparse
import torch
from torch.utils.data import DataLoader

from src.grounding.model import MomentDETRLite
from src.grounding.dataset import MomentRetrievalDataset


def collate(batch):
    max_t = max(b["clip_feats"].shape[0] for b in batch)
    feat_dim = batch[0]["clip_feats"].shape[1]
    padded = torch.zeros(len(batch), max_t, feat_dim)
    for i, b in enumerate(batch):
        t = b["clip_feats"].shape[0]
        padded[i, :t] = b["clip_feats"]
    query_feats = torch.stack([b["query_feat"] for b in batch])
    gt_center = torch.stack([b["gt_center"] for b in batch])
    gt_width = torch.stack([b["gt_width"] for b in batch])
    return padded, query_feats, gt_center, gt_width


def span_giou(pred_center, pred_width, gt_center, gt_width):
    pred_s, pred_e = pred_center - pred_width / 2, pred_center + pred_width / 2
    gt_s, gt_e = gt_center - gt_width / 2, gt_center + gt_width / 2
    inter = (torch.minimum(pred_e, gt_e) - torch.maximum(pred_s, gt_s)).clamp(min=0)
    pred_len = (pred_e - pred_s).clamp(min=1e-6)
    gt_len = (gt_e - gt_s).clamp(min=1e-6)
    union = (pred_len + gt_len - inter).clamp(min=1e-6)
    iou = inter / union
    enclose = (torch.maximum(pred_e, gt_e) - torch.minimum(pred_s, gt_s)).clamp(min=1e-6)
    return iou - (enclose - union) / enclose


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MomentRetrievalDataset(args.jsonl, args.feature_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model = MomentDETRLite().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        sum_l1, sum_giou, sum_conf, n = 0.0, 0.0, 0.0, 0
        for clip_feats, query_feats, gt_center, gt_width in dl:
            clip_feats, query_feats = clip_feats.to(device), query_feats.to(device)
            gt_center, gt_width = gt_center.to(device), gt_width.to(device)

            centers, widths, conf = model(clip_feats, query_feats)

            l1 = (centers - gt_center.unsqueeze(1)).abs() + (widths - gt_width.unsqueeze(1)).abs()
            best_q = l1.argmin(dim=1)
            b_idx = torch.arange(centers.size(0), device=device)
            pred_center = centers[b_idx, best_q]
            pred_width = widths[b_idx, best_q]

            l1_loss = torch.nn.functional.l1_loss(pred_center, gt_center) + \
                      torch.nn.functional.l1_loss(pred_width, gt_width)
            giou = span_giou(pred_center, pred_width, gt_center, gt_width)
            giou_loss = (1 - giou).mean()

            conf_target = torch.zeros_like(conf)
            conf_target[b_idx, best_q] = 1.0
            conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(conf, conf_target)

            loss = l1_loss + giou_loss + conf_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_l1 += l1_loss.item()
            sum_giou += giou_loss.item()
            sum_conf += conf_loss.item()
            n += 1

        print(f"epoch {epoch+1}/{args.epochs}  l1={sum_l1/n:.4f}  giou={sum_giou/n:.4f}  conf={sum_conf/n:.4f}  total={(sum_l1+sum_giou+sum_conf)/n:.4f}")

    torch.save(model.state_dict(), args.out_ckpt)
    print(f"saved checkpoint to {args.out_ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out_ckpt", default="grounding_head.pt")
    args = ap.parse_args()
    train(args)