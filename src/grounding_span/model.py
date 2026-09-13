"""
src/grounding_span/model.py

Span-based grounding head (v3) - VSLNet-style boundary regression, replacing
both v1's DETR-style query slots and v2's dense anchor scoring.

Why this design: v1's slot-matching head had a documented ceiling (needs
large-scale pretraining to learn slot assignment). v2's dense-anchor design
removed slot-matching but introduced a NEW ceiling -- anchors only exist at
fixed (window-count) durations, so ~46% of QVHighlights GT moments (measured
on the real test set: 28.5% shorter than one window, 17% longer than the
widest anchor) were mathematically unreachable regardless of training
quality. See EXPERIMENT_LOG.md for both findings.

VSLNet-style span prediction removes the duration ceiling by construction:
instead of scoring a fixed candidate set, it predicts a probability
distribution over START window index and END window index independently, via
softmax over the whole (variable-length) local crop. Any (start, end) pair is
reachable -- from a single window up to the entire crop -- with no scale list
to run out of range on. This is an established approach (VSLNet, Zhang et al.
2020) proven to work on frozen features without large-scale pretraining.

ADAPTATION NOTE: the original VSLNet uses per-token (GloVe) query embeddings
for fine-grained Context-Query Attention (QANet-style). This pipeline only
ever produces a single pooled CLIP query embedding (ClipEmbedder.embed_text
returns one (D,) vector -- the same contract the retrieval stage and v1/v2
depend on), so there's no token-level query sequence available without
changing ClipEmbedder, which is out of this module's scope. Instead, this
model:
  - runs real self-attention over the T video windows (unlike v2's fully
    independent per-window features before pooling), so boundary decisions
    get cross-window temporal context
  - fuses the single pooled query vector via FiLM-style modulation
    (scale+shift), a lighter substitute for full cross-attention
  - keeps VSLNet's Query-Guided Highlighting (QGH) as an auxiliary per-window
    foreground/background head, trained jointly
If this still underperforms, extracting per-token query features (a change to
ClipEmbedder + retrieval's query encoding) would enable true QANet-style
Context-Query Attention -- noted here, not built, since it touches shared
code outside this module.
"""
from typing import Tuple

import torch
import torch.nn as nn
import math

def sinusoidal_positional_encoding(T: int, hidden_dim: int, device) -> torch.Tensor:
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, hidden_dim, 2, device=device).float() * (-math.log(10000.0) / hidden_dim))
    pe = torch.zeros(T, hidden_dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (T, H)

class SpanGroundingHead(nn.Module):
    def __init__(self, clip_dim: int = 512, hidden_dim: int = 256,
                 num_attn_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.feat_proj = nn.Linear(clip_dim, hidden_dim)
        self.query_proj = nn.Linear(clip_dim, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)

        # Query-Guided Highlighting: per-window foreground/background logit
        self.highlight_head = nn.Linear(hidden_dim, 1)

        # Conditional span predictor: end prediction is conditioned on the
        # (soft) start distribution, like VSLNet's conditional span
        # predictor, instead of two fully independent heads.
        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_gru = nn.GRU(hidden_dim + 1, hidden_dim, batch_first=True, bidirectional=True)
        self.end_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, clip_feats: torch.Tensor, query_feat: torch.Tensor,
                feat_mask: torch.Tensor = None):
        """
        clip_feats: (B, T, D) local-crop CLIP window features, zero-padded on
            the right if T varies across the batch.
        query_feat: (B, D) pooled query embedding.
        feat_mask: (B, T) bool, True for real (non-padded) windows. If None,
            assumes no padding (e.g. inference with batch size 1).

        Returns:
            start_logits: (B, T) -- padded positions set to a large negative value
            end_logits:   (B, T) -- same
            highlight_logits: (B, T) -- QGH auxiliary foreground/background logit
        """
        B, T, D = clip_feats.shape
        device = clip_feats.device
        if feat_mask is None:
            feat_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        feat_h = self.feat_proj(clip_feats)  # (B, T, H)
        query_h = self.query_proj(query_feat).unsqueeze(1).expand(-1, T, -1)  # (B, T, H)
        h = self.fuse(torch.cat([feat_h, query_h], dim=-1))  # (B, T, H)
        pe = sinusoidal_positional_encoding(T, self.hidden_dim, device)  # (T, H)
        h = h + pe.unsqueeze(0)  # broadcast over batch
        # transformer expects True = position to IGNORE
        h = self.encoder(h, src_key_padding_mask=~feat_mask)  # (B, T, H)

        highlight_logits = self.highlight_head(h).squeeze(-1)  # (B, T)

        start_logits_raw = self.start_head(h).squeeze(-1)  # (B, T)
        neg_inf = torch.finfo(start_logits_raw.dtype).min
        start_logits_masked = start_logits_raw.masked_fill(~feat_mask, neg_inf)
        start_probs = torch.softmax(start_logits_masked, dim=-1)

        end_in = torch.cat([h, start_probs.unsqueeze(-1)], dim=-1)  # (B, T, H+1)
        end_h, _ = self.end_gru(end_in)  # (B, T, 2H)
        end_logits_raw = self.end_head(end_h).squeeze(-1)  # (B, T)

        start_logits = start_logits_raw.masked_fill(~feat_mask, neg_inf)
        end_logits = end_logits_raw.masked_fill(~feat_mask, neg_inf)

        return start_logits, end_logits, highlight_logits

    @staticmethod
    def decode(start_logits: torch.Tensor, end_logits: torch.Tensor,
               local_bounds) -> Tuple[float, float]:
        """
        Single-example decode (no batch dim): picks the (start_idx, end_idx)
        pair with start_idx <= end_idx maximizing
        start_log_prob[start_idx] + end_log_prob[end_idx] -- standard span-QA
        decode. O(T) via a running max of start log-prob, not O(T^2).

        local_bounds: (T, 2) array of absolute-second window bounds for this
            example's local crop.
        Returns (start_sec, end_sec).
        """
        start_lp = torch.log_softmax(start_logits, dim=-1)
        end_lp = torch.log_softmax(end_logits, dim=-1)
        T = start_lp.shape[0]

        best_score = float("-inf")
        best_s, best_e = 0, 0
        running_best_start_lp = float("-inf")
        running_best_start_idx = 0
        for t in range(T):
            if start_lp[t].item() > running_best_start_lp:
                running_best_start_lp = start_lp[t].item()
                running_best_start_idx = t
            score = running_best_start_lp + end_lp[t].item()
            if score > best_score:
                best_score = score
                best_s, best_e = running_best_start_idx, t

        start_sec = float(local_bounds[best_s][0])
        end_sec = float(local_bounds[best_e][1])
        return start_sec, end_sec