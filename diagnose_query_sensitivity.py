"""
diagnose_query_sensitivity.py

Checks whether SpanGroundingHead is actually using the query, or just
learned a positional prior over window index that ignores query_feat
entirely. Runs each validation batch twice: once with real queries, once
with queries shuffled across the batch (same videos, wrong queries paired
in). If predicted start/end windows barely change between the two runs,
the model isn't conditioning on the query -- it's exploiting a positional
bias in where GT windows tend to fall in this dataset.

Usage:
    python diagnose_query_sensitivity.py --val_jsonl <path> --feature_dir <dir> --ckpt grounding_span_v1.pt
"""
import argparse

import torch
from torch.utils.data import DataLoader

from src.grounding_span.dataset import SpanGroundingDataset, collate_fn
from src.grounding_span.model import SpanGroundingHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_ds = SpanGroundingDataset(args.val_jsonl, args.feature_dir)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = SpanGroundingHead().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    total, same_start, same_end, same_both = 0, 0, 0, 0
    start_prob_diff_sum, end_prob_diff_sum = 0.0, 0.0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.max_batches:
                break
            clip_feats = batch["clip_feats"].to(device)
            feat_mask = batch["feat_mask"].to(device)
            query_feat = batch["query_feat"].to(device)

            B = query_feat.shape[0]
            if B < 2:
                continue  # need at least 2 examples to shuffle meaningfully

            # roll by 1 so every example gets a DIFFERENT example's query,
            # never its own -- a clean "wrong query" test
            shuffled_query_feat = torch.roll(query_feat, shifts=1, dims=0)

            start_logits, end_logits, _ = model(clip_feats, query_feat, feat_mask)
            start_logits_shuf, end_logits_shuf, _ = model(clip_feats, shuffled_query_feat, feat_mask)

            start_pred = start_logits.argmax(dim=-1)
            end_pred = end_logits.argmax(dim=-1)
            start_pred_shuf = start_logits_shuf.argmax(dim=-1)
            end_pred_shuf = end_logits_shuf.argmax(dim=-1)

            start_probs = torch.softmax(start_logits, dim=-1)
            start_probs_shuf = torch.softmax(start_logits_shuf, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)
            end_probs_shuf = torch.softmax(end_logits_shuf, dim=-1)

            # L1 distance between the two probability distributions per example
            start_prob_diff_sum += (start_probs - start_probs_shuf).abs().sum(dim=-1).sum().item()
            end_prob_diff_sum += (end_probs - end_probs_shuf).abs().sum(dim=-1).sum().item()

            same_start += (start_pred == start_pred_shuf).sum().item()
            same_end += (end_pred == end_pred_shuf).sum().item()
            same_both += ((start_pred == start_pred_shuf) & (end_pred == end_pred_shuf)).sum().item()
            total += B

    print(f"n={total}")
    print(f"same argmax start index (real vs wrong query): {same_start/total:.4f}")
    print(f"same argmax end index   (real vs wrong query): {same_end/total:.4f}")
    print(f"same BOTH start+end     (real vs wrong query): {same_both/total:.4f}")
    print(f"avg L1 diff in start prob dist: {start_prob_diff_sum/total:.4f}")
    print(f"avg L1 diff in end prob dist:   {end_prob_diff_sum/total:.4f}")
    print()
    print("Interpretation: if 'same start/end' is high (>0.5-0.6) and L1 diffs")
    print("are small, the model is largely ignoring the query and predicting")
    print("based on window position/structure alone -- a positional-prior bug,")
    print("not a fundamental flaw in the span-based approach.")


if __name__ == "__main__":
    main()