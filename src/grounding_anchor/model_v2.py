"""
src/grounding_anchor/model_v2.py

Enhanced Anchor Grounding Head (v2.1):
1. Temporal Context Layer (BiGRU): Allows adjacent windows to exchange temporal dynamics
   before anchor pooling, so the model recognizes action onsets and completions.
2. Boundary-Aware Pooling: Concatenates the start window feature, span mean feature,
   and end window feature [h_start, h_mean, h_end] into the anchor representation.
3. Smooth SSD-style center/width regression for sub-window and continuous boundaries.
"""
from typing import List, Tuple
import torch
import torch.nn as nn

SCALES: List[int] = [1, 2, 3, 5, 8]


class EnhancedAnchorGroundingHead(nn.Module):
    def __init__(self, clip_dim: int = 512, hidden_dim: int = 256, scales: List[int] = None):
        super().__init__()
        self.scales = scales or SCALES
        self.hidden_dim = hidden_dim

        self.feat_proj = nn.Linear(clip_dim, hidden_dim)
        self.query_proj = nn.Linear(clip_dim, hidden_dim)

        # Multi-modal fusion
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Temporal Context Layer: Bi-directional temporal context
        self.temporal_context = nn.GRU(
            hidden_dim, hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Boundary-aware representation: [h_start, h_mean, h_end] -> 3 * hidden_dim
        anchor_in_dim = hidden_dim * 3

        # Anchor Quality Score Head (predicts continuous IoU quality)
        self.score_head = nn.Sequential(
            nn.Linear(anchor_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        # Refinement Head: delta center and delta log-width
        self.reg_head = nn.Sequential(
            nn.Linear(anchor_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def build_anchors(self, T: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Boundary-aware pooling:
        Extracts start feature h_start, end feature h_end, and mean feature h_mean.
        Returns (B, A, 3 * H).
        """
        B, T, H = window_repr.shape
        zeros = window_repr.new_zeros(B, 1, H)
        cs = torch.cat([zeros, window_repr.cumsum(dim=1)], dim=1)  # (B, T+1, H)

        A = anchor_start.shape[1]
        start_idx = anchor_start.long().unsqueeze(-1).expand(B, A, H)
        end_idx = anchor_end.long().unsqueeze(-1).expand(B, A, H)

        # Mean representation over the span
        sum_start = torch.gather(cs, 1, start_idx)
        sum_end = torch.gather(cs, 1, end_idx)
        span_len = (anchor_end - anchor_start).clamp(min=1).unsqueeze(-1).float()
        h_mean = (sum_end - sum_start) / span_len  # (B, A, H)

        # Exact boundary features at onset and offset
        h_start = torch.gather(window_repr, 1, start_idx)
        # end_idx is exclusive, so last window is clamp(end - 1, min=0)
        last_window_idx = (anchor_end - 1).clamp(min=0).long().unsqueeze(-1).expand(B, A, H)
        h_end = torch.gather(window_repr, 1, last_window_idx)

        return torch.cat([h_start, h_mean, h_end], dim=-1)  # (B, A, 3H)

    def forward(self, clip_feats: torch.Tensor, query_feat: torch.Tensor,
                feat_mask: torch.Tensor = None):
        B, T, D = clip_feats.shape
        device = clip_feats.device
        if feat_mask is None:
            feat_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        T_real = feat_mask.long().sum(dim=1)

        feat_h = self.feat_proj(clip_feats)
        query_h = self.query_proj(query_feat).unsqueeze(1).expand(-1, T, -1)
        fused = self.fuse(torch.cat([feat_h, query_h], dim=-1))  # (B, T, H)

        # Pass through temporal context
        window_repr, _ = self.temporal_context(fused)  # (B, T, H)

        anchor_t, anchor_scale = self.build_anchors(T, device)
        A = anchor_t.shape[0]
        anchor_t_b = anchor_t.unsqueeze(0).expand(B, A)
        raw_end = anchor_t_b + anchor_scale.unsqueeze(0)
        T_real_b = T_real.unsqueeze(1).expand(B, A)

        anchor_end = torch.minimum(raw_end, T_real_b).clamp(min=1)
        anchor_start = torch.minimum(anchor_t_b, (T_real_b - 1).clamp(min=0))

        # Boundary-aware pooling
        anchor_repr = self._pool_anchors(window_repr, anchor_start, anchor_end)

        scores = self.score_head(anchor_repr).squeeze(-1)     # (B, A)
        deltas = self.reg_head(anchor_repr)                    # (B, A, 2)

        a_center = (anchor_start.float() + anchor_end.float()) / 2.0 / T
        a_width = (anchor_end.float() - anchor_start.float()) / float(T)

        delta_center = deltas[..., 0]
        delta_log_width = deltas[..., 1].clamp(-4.0, 4.0)
        centers = a_center + delta_center * a_width
        widths = a_width * torch.exp(delta_log_width)

        return centers, widths, scores

    @staticmethod
    def to_timestamps(center: float, width: float, local_start: float, local_end: float) -> Tuple[float, float]:
        duration = local_end - local_start
        center_sec = local_start + center * duration
        half_width_sec = (width * duration) / 2.0
        return center_sec - half_width_sec, center_sec + half_width_sec
