"""
src/grounding_span/train.py

Training script for the span-based grounding head (v3). Independent of
v1's and v2's train.py.

Loss = CrossEntropy(start_logits, start_idx) + CrossEntropy(end_logits, end_idx)
     + BCEWithLogits(highlight_logits, highlight_label) [masked to real windows]
Standard span-QA loss + VSLNet-style auxiliary highlighting loss.

Checkpoints follow the same pattern as v1/v2:
    <ckpt_name>.pt                          (best-val copy at project root)
    checkpoints/<ckpt_name>_latest.pt
    checkpoints/<ckpt_name>_best.pt
    checkpoints/<ckpt_name>_epochN.pt
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.grounding_span.dataset import SpanGroundingDataset, collate_fn
from src.grounding_span.model import SpanGroundingHead


def compute_loss(start_logits, end_logits, highlight_logits, start_idx, end_idx,
                  highlight, feat_mask, highlight_weight: float = 1.0):
    ce_start = torch.nn.functional.cross_entropy(start_logits, start_idx)
    ce_end = torch.nn.functional.cross_entropy(end_logits, end_idx)
    hl_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        highlight_logits[feat_mask], highlight[feat_mask], reduction="mean"
    )
    total = ce_start + ce_end + highlight_weight * hl_loss
    return total, ce_start.detach(), ce_end.detach(), hl_loss.detach()


def save_ckpt(model, ckpt_dir, base_name, tag):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{base_name}_{tag}.pt"))


def run_epoch(model, loader, device, opt=None):
    train_mode = opt is not None
    model.train() if train_mode else model.eval()
    sums = {"loss": 0.0, "ce_s": 0.0, "ce_e": 0.0, "hl": 0.0}
    n = 0

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            clip_feats = batch["clip_feats"].to(device)
            feat_mask = batch["feat_mask"].to(device)
            query_feat = batch["query_feat"].to(device)
            start_idx = batch["start_idx"].to(device)
            end_idx = batch["end_idx"].to(device)
            highlight = batch["highlight"].to(device)

            start_logits, end_logits, highlight_logits = model(clip_feats, query_feat, feat_mask)
            loss, ce_s, ce_e, hl = compute_loss(start_logits, end_logits, highlight_logits,
                                                 start_idx, end_idx, highlight, feat_mask)

            if train_mode:
                opt.zero_grad()
                loss.backward()
                opt.step()

            sums["loss"] += loss.item()
            sums["ce_s"] += ce_s.item()
            sums["ce_e"] += ce_e.item()
            sums["hl"] += hl.item()
            n += 1

    n = max(1, n)
    return {k: v / n for k, v in sums.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--feature_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ckpt_name", default="grounding_span_v1")
    p.add_argument("--ckpt_dir", default="checkpoints")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SpanGroundingDataset(args.train_jsonl, args.feature_dir)
    val_ds = SpanGroundingDataset(args.val_jsonl, args.feature_dir, embedder=train_ds.embedder)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = SpanGroundingHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(args.epochs):
        tr = run_epoch(model, train_loader, device, opt)
        va = run_epoch(model, val_loader, device, opt=None)

        print(f"epoch {epoch+1}/{args.epochs}  "
              f"train_loss={tr['loss']:.4f} (ce_s={tr['ce_s']:.4f} ce_e={tr['ce_e']:.4f} hl={tr['hl']:.4f})  "
              f"val_loss={va['loss']:.4f} (ce_s={va['ce_s']:.4f} ce_e={va['ce_e']:.4f} hl={va['hl']:.4f})")

        save_ckpt(model, args.ckpt_dir, args.ckpt_name, "latest")
        save_ckpt(model, args.ckpt_dir, args.ckpt_name, f"epoch{epoch+1}")
        if va["loss"] < best_val:
            best_val = va["loss"]
            save_ckpt(model, args.ckpt_dir, args.ckpt_name, "best")
            torch.save(model.state_dict(), f"{args.ckpt_name}.pt")

    print(f"done. best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()