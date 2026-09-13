"""
Trains MomentDETRLite on cached CLIP features. (Diagnostic version: prints
per-component loss so we can see which term is/isn't improving.)
"""
import argparse
import os
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

    # --- checkpointing setup ---
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    out_ckpt_base = os.path.splitext(os.path.basename(args.out_ckpt))[0]
    best_loss = float("inf")
    best_path = os.path.join(ckpt_dir, f"{out_ckpt_base}_best.pt")
    latest_path = os.path.join(ckpt_dir, f"{out_ckpt_base}_latest.pt")

    start_epoch = 0

    # --- resume logic ---
    if args.resume:
        if os.path.exists(latest_path):
            ckpt = torch.load(latest_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                # new-format checkpoint: full resume with optimizer + epoch
                print(f"resuming from {latest_path}")
                model.load_state_dict(ckpt["model_state_dict"])
                opt.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"]
                best_loss = ckpt.get("best_loss", float("inf"))
                print(f"resumed at epoch {start_epoch}, best_loss so far = {best_loss:.4f}")
            else:
                # old-format checkpoint (plain state_dict, from before resume support):
                # load weights only, optimizer state and epoch count are lost
                print(f"found old-format checkpoint at {latest_path} (no optimizer/epoch info)")
                print("loading model weights only; epoch count restarts at 0")
                model.load_state_dict(ckpt)
                start_epoch = 0
        else:
            print(f"--resume passed but no checkpoint found at {latest_path}, starting fresh")

    for epoch in range(start_epoch, args.epochs):
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

            # conf: (B, num_queries) raw logits. Treat this as a single classification
            # problem per example -- "which of the num_queries slots is the match?" --
            # rather than num_queries independent yes/no decisions. With only one
            # positive among several negatives every time, independent BCE tends to
            # just push all logits down instead of learning a sharp preference.
            conf_loss = torch.nn.functional.cross_entropy(conf, best_q)

            loss = l1_loss + giou_loss + conf_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_l1 += l1_loss.item()
            sum_giou += giou_loss.item()
            sum_conf += conf_loss.item()
            n += 1

        epoch_total = (sum_l1 + sum_giou + sum_conf) / n
        print(f"epoch {epoch+1}/{args.epochs}  l1={sum_l1/n:.4f}  giou={sum_giou/n:.4f}  conf={sum_conf/n:.4f}  total={epoch_total:.4f}")

        # always overwrite "latest" (full resume state: model + optimizer + epoch)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epoch + 1,
            "best_loss": min(best_loss, epoch_total),
        }, latest_path)

        # save best-so-far by train loss (plain state_dict, for easy loading in eval/inference)
        if epoch_total < best_loss:
            best_loss = epoch_total
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best (total={best_loss:.4f}), saved to {best_path}")

        # periodic numbered checkpoint every N epochs (plain state_dict)
        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
            numbered_path = os.path.join(ckpt_dir, f"{out_ckpt_base}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), numbered_path)
            print(f"  -> periodic checkpoint saved to {numbered_path}")

    torch.save(model.state_dict(), args.out_ckpt)
    print(f"saved final checkpoint to {args.out_ckpt}")
    print(f"best checkpoint (by train loss) saved to {best_path} (loss={best_loss:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out_ckpt", default="grounding_head.pt")
    ap.add_argument("--ckpt_dir", default="checkpoints",
                     help="directory for periodic/best/latest checkpoints")
    ap.add_argument("--ckpt_every", type=int, default=10,
                     help="save a numbered checkpoint every N epochs (0 to disable)")
    ap.add_argument("--resume", action="store_true",
                     help="resume training from checkpoints/<out_ckpt>_latest.pt if it exists")
    args = ap.parse_args()
    train(args)