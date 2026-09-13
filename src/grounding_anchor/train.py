"""
src/grounding_anchor/train.py

Training script for the anchor-based grounding head (v2). Independent of
src/grounding/train.py -- no shared classes, no shared checkpoint names.

Loss: BCEWithLogits over anchor quality scores (masked to anchors flagged
valid in anchor_valid) + SmoothL1 regression on (center, width) for anchors
flagged positive in anchor_label. No slot-matching / Hungarian assignment
needed -- each anchor's label comes from its own fixed IoU with GT, so this
sidesteps the "confidence head can't tell slots apart" failure mode v1 hit.

Checkpoints follow the same pattern v1's train.py used:
    <ckpt_name>.pt                          (best-val copy at project root)
    checkpoints/<ckpt_name>_latest.pt
    checkpoints/<ckpt_name>_best.pt
    checkpoints/<ckpt_name>_epochN.pt
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.grounding_anchor.dataset import AnchorGroundingDataset, collate_fn
from src.grounding_anchor.model import AnchorGroundingHead


def compute_loss(centers, widths, scores, anchor_valid, anchor_label, gt_center, gt_width):
    pos = anchor_valid & (anchor_label == 1.0)

    if anchor_valid.any():
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            scores[anchor_valid], anchor_label[anchor_valid], reduction="mean"
        )
    else:
        bce = scores.sum() * 0.0

    if pos.any():
        B, A = centers.shape
        gt_c = gt_center.unsqueeze(1).expand(B, A)[pos]
        gt_w = gt_width.unsqueeze(1).expand(B, A)[pos]
        reg = (torch.nn.functional.smooth_l1_loss(centers[pos], gt_c)
               + torch.nn.functional.smooth_l1_loss(widths[pos], gt_w))
    else:
        reg = scores.sum() * 0.0

    total = bce + reg
    return total, bce.detach(), reg.detach()


def save_ckpt(model, ckpt_dir, base_name, tag):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{base_name}_{tag}.pt"))


def run_epoch(model, loader, device, opt=None):
    train_mode = opt is not None
    model.train() if train_mode else model.eval()
    loss_sum, bce_sum, reg_sum, n = 0.0, 0.0, 0.0, 0

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            clip_feats = batch["clip_feats"].to(device)
            feat_mask = batch["feat_mask"].to(device)
            query_feat = batch["query_feat"].to(device)
            anchor_valid = batch["anchor_valid"].to(device)
            anchor_label = batch["anchor_label"].to(device)
            gt_center = batch["gt_center"].to(device)
            gt_width = batch["gt_width"].to(device)

            centers, widths, scores = model(clip_feats, query_feat, feat_mask)
            loss, bce, reg = compute_loss(centers, widths, scores, anchor_valid,
                                           anchor_label, gt_center, gt_width)

            if train_mode:
                opt.zero_grad()
                loss.backward()
                opt.step()

            loss_sum += loss.item()
            bce_sum += bce.item()
            reg_sum += reg.item()
            n += 1

    n = max(1, n)
    return loss_sum / n, bce_sum / n, reg_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--feature_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ckpt_name", default="grounding_anchor_v1")
    p.add_argument("--ckpt_dir", default="checkpoints")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = AnchorGroundingDataset(args.train_jsonl, args.feature_dir)
    val_ds = AnchorGroundingDataset(args.val_jsonl, args.feature_dir, embedder=train_ds.embedder)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = AnchorGroundingHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(args.epochs):
        train_loss, train_bce, train_reg = run_epoch(model, train_loader, device, opt)
        val_loss, val_bce, val_reg = run_epoch(model, val_loader, device, opt=None)

        print(f"epoch {epoch+1}/{args.epochs}  "
              f"train_loss={train_loss:.4f} (bce={train_bce:.4f} reg={train_reg:.4f})  "
              f"val_loss={val_loss:.4f} (bce={val_bce:.4f} reg={val_reg:.4f})")

        save_ckpt(model, args.ckpt_dir, args.ckpt_name, "latest")
        save_ckpt(model, args.ckpt_dir, args.ckpt_name, f"epoch{epoch+1}")
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(model, args.ckpt_dir, args.ckpt_name, "best")
            torch.save(model.state_dict(), f"{args.ckpt_name}.pt")

    print(f"done. best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()