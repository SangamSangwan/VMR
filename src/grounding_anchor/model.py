"""
src/grounding_anchor/model.py

Dense anchor-based grounding head (v2) - 2D-TAN style anchor scoring instead of
DETR-style query slots. See HANDOFF_anchor_model.md for why.

Interface mirrors MomentDETRLite (src/grounding/model.py) so it COULD be
swapped into src/pipeline.py later without changing pipeline.py's argmax+decode
logic -- but this file does not import from or modify src/grounding/*.

    forward(clip_feats, query_feat, feat_mask=None) -> (centers, widths, scores)
    each shaped (B, num_anchors), all in local-crop-normalized [0,1] space.

Anchors are deterministic (start_window, scale) pairs generated from the
sequence length -- no slot-matching ambiguity, each anchor gets its own
independent quality score.
"""
from typing import List, Tuple

import torch
import torch.nn as nn

# Anchor scales, in units of local CLIP windows. Must match SCALES in
# src/grounding_anchor/dataset.py -- the two files build the same anchor grid
# independently, and this ordering (scale outer loop, t inner loop) has to
# match dataset.py's build_anchor_targets exactly or labels line up with the
# wrong anchors.
SCALES: List[int] = [1, 2, 3, 5, 8]


class AnchorGroundingHead(nn.Module):
    def __init__(self, clip_dim: int = 512, hidden_dim: int = 256, scales: List[int] = None):
        super().__init__()
        self.scales = scales or SCALES
        self.hidden_dim = hidden_dim

        self.feat_proj = nn.Linear(clip_dim, hidden_dim)
        self.query_proj = nn.Linear(clip_dim, hidden_dim)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # per-anchor quality score (logit) -- "does this anchor overlap GT well"
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # per-anchor (delta_center, delta_log_width) refinement, SSD-style
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

    def build_anchors(self, T: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (anchor_t, anchor_scale), each (num_anchors,) int64. Anchor i
        covers local window indices [anchor_t[i], anchor_t[i] + anchor_scale[i]),
        before any per-sample clipping. Deterministic given T.
        """
        ts, scales = [], []
        for s in self.scales:
            for t in range(T):
                ts.append(t)
                scales.append(s)
        return (torch.tensor(ts, dtype=torch.long, device=device),
                torch.tensor(scales, dtype=torch.long, device=device))

    def _pool_anchors(self, window_repr: torch.Tensor, anchor_start: torch.Tensor,
                       anchor_end: torch.Tensor) -> torch.Tensor:
        """
        window_repr: (B, T, H). anchor_start/anchor_end: (B, A) exclusive-end
        window index ranges, ALREADY per-sample clipped to that sample's real
        (unpadded) length. Mean-pools via a cumsum trick -- no python loop.
        Returns (B, A, H).
        """
        B, T, H = window_repr.shape
        zeros = window_repr.new_zeros(B, 1, H)
        cs = torch.cat([zeros, window_repr.cumsum(dim=1)], dim=1)  # (B, T+1, H)

        A = anchor_start.shape[1]
        start_idx = anchor_start.long().unsqueeze(-1).expand(B, A, H)
        end_idx = anchor_end.long().unsqueeze(-1).expand(B, A, H)
        sum_start = torch.gather(cs, 1, start_idx)
        sum_end = torch.gather(cs, 1, end_idx)
        span_len = (anchor_end - anchor_start).clamp(min=1).unsqueeze(-1).float()
        return (sum_end - sum_start) / span_len

    def forward(self, clip_feats: torch.Tensor, query_feat: torch.Tensor,
                feat_mask: torch.Tensor = None):
        """
        clip_feats: (B, T, D). If batched with variable-length crops, right-pad
            with zeros to T and pass feat_mask (B, T) bool marking real windows
            -- see src/grounding_anchor/dataset.py's collate_fn. At inference
            with batch size 1 (no padding), feat_mask can be omitted.
        query_feat: (B, D)

        IMPORTANT: anchor spans are clipped per-sample to that sample's real
        (unpadded) length, taken from feat_mask -- NOT to the batch-wide padded
        T. Without this, an anchor near the end of a short real sequence would
        silently pool in zero-padded windows from longer sequences in the same
        batch, which would not match how dataset.py's build_anchor_targets
        computes IoU/labels (it clips at the real length too). Getting this
        wrong would be exactly the kind of silent train/inference coordinate
        mismatch v1 already got bitten by once (see EXPERIMENT_LOG.md) --
        keep pooling and target-building clipped the same way if you touch this.

        Returns centers, widths, scores -- each (B, num_anchors).
        """
        B, T, D = clip_feats.shape
        device = clip_feats.device
        if feat_mask is None:
            feat_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        T_real = feat_mask.long().sum(dim=1)  # (B,)

        feat_h = self.feat_proj(clip_feats)
        query_h = self.query_proj(query_feat).unsqueeze(1).expand(-1, T, -1)
        window_repr = self.fuse(torch.cat([feat_h, query_h], dim=-1))  # (B, T, H)

        anchor_t, anchor_scale = self.build_anchors(T, device)  # (A,)
        A = anchor_t.shape[0]
        anchor_t_b = anchor_t.unsqueeze(0).expand(B, A)
        raw_end = anchor_t_b + anchor_scale.unsqueeze(0)
        T_real_b = T_real.unsqueeze(1).expand(B, A)

        anchor_end = torch.minimum(raw_end, T_real_b).clamp(min=1)
        anchor_start = torch.minimum(anchor_t_b, (T_real_b - 1).clamp(min=0))

        anchor_repr = self._pool_anchors(window_repr, anchor_start, anchor_end)
        scores = self.score_head(anchor_repr).squeeze(-1)     # (B, A)
        deltas = self.reg_head(anchor_repr)                    # (B, A, 2)

        # normalized against global T (batch's padded length) so this space is
        # consistent across every sample in the batch -- matches how
        # dataset.py's build_anchor_targets normalizes (it's given the same T).
        a_center = (anchor_start.float() + anchor_end.float()) / 2.0 / T
        a_width = (anchor_end.float() - anchor_start.float()) / float(T)

        delta_center = deltas[..., 0]
        delta_log_width = deltas[..., 1].clamp(-4.0, 4.0)  # stability
        centers = a_center + delta_center * a_width
        widths = a_width * torch.exp(delta_log_width)

        return centers, widths, scores

    @staticmethod
    def to_timestamps(center: float, width: float, local_start: float, local_end: float) -> Tuple[float, float]:
        """Same decode contract as MomentDETRLite.to_timestamps in
        src/grounding/model.py -- duplicated here (not imported) to keep this
        module fully independent of v1, per request."""
        duration = local_end - local_start
        center_sec = local_start + center * duration
        half_width_sec = (width * duration) / 2.0
        return center_sec - half_width_sec, center_sec + half_width_sec