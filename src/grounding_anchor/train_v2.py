"""
src/grounding_anchor/train_v2.py

Training script for the Enhanced Anchor Grounding Head (v2.1).
Key improvements over v1:
1. Trains EnhancedAnchorGroundingHead with temporal BiGRU context layer and boundary-aware pooling.
2. Supports training directly on Charades-STA (12,408 examples) or QVHighlights.
3. Enhanced loss: Focal-weighted BCE on anchor quality scores + Smooth L1 regression.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.grounding_anchor.dataset import AnchorGroundingDataset, collate_fn
from src.grounding_anchor.model_v2 import EnhancedAnchorGroundingHead


def compute_loss(centers, widths, scores, anchor_valid, anchor_label, gt_center, gt_width,
                 pos_weight=2.0, gamma=1.5):
    pos = anchor_valid & (anchor_label == 1.0)

    if anchor_valid.any():
        valid_scores = scores[anchor_valid]
        valid_labels = anchor_label[anchor_valid]
        p = torch.sigmoid(valid_scores)
        # Focal weight to handle anchor class imbalance
        focal_weight = torch.where(
            valid_labels == 1.0,
            pos_weight * (1.0 - p) ** gamma,
            p ** gamma,
        )
        bce = (torch.nn.functional.binary_cross_entropy_with_logits(
            valid_scores, valid_labels, reduction="none"
        ) * focal_weight).mean()
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

            loss_sum += loss.item()
            bce_sum += bce.item()
            reg_sum += reg.item()
            n += 1

    n = max(1, n)
    return loss_sum / n, bce_sum / n, reg_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", default="data/charades_sta/train_subset.jsonl")
    p.add_argument("--val_jsonl", default="data/charades_sta/test_subset.jsonl")
    p.add_argument("--feature_dir", default="data/charades_sta/features")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ckpt_name", default="grounding_anchor_v2_charades")
    p.add_argument("--ckpt_dir", default="checkpoints")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training EnhancedAnchorGroundingHead on device: {device}")
    print(f"Train: {args.train_jsonl} | Val: {args.val_jsonl}")

    train_ds = AnchorGroundingDataset(args.train_jsonl, args.feature_dir)
    val_ds = AnchorGroundingDataset(args.val_jsonl, args.feature_dir, embedder=train_ds.embedder)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = EnhancedAnchorGroundingHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")
    for epoch in range(args.epochs):
        train_loss, train_bce, train_reg = run_epoch(model, train_loader, device, opt)
        val_loss, val_bce, val_reg = run_epoch(model, val_loader, device, opt=None)
        scheduler.step()

        print(f"epoch {epoch+1}/{args.epochs}  "
              f"train_loss={train_loss:.4f} (bce={train_bce:.4f} reg={train_reg:.4f})  "
              f"val_loss={val_loss:.4f} (bce={val_bce:.4f} reg={val_reg:.4f})  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        save_ckpt(model, args.ckpt_dir, args.ckpt_name, "latest")
        if (epoch + 1) % 5 == 0:
            save_ckpt(model, args.ckpt_dir, args.ckpt_name, f"epoch{epoch+1}")
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(model, args.ckpt_dir, args.ckpt_name, "best")
            torch.save(model.state_dict(), f"{args.ckpt_name}.pt")
            print(f"  -> new best val_loss={best_val:.4f} saved to {args.ckpt_name}.pt")

    print(f"done. best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
