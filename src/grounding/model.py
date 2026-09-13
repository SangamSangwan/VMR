"""
Stage 2: fine-grained grounding head, Moment-DETR-lite.

Given a candidate window (expanded with some context on each side) as a
sequence of per-clip CLIP features, plus the query embedding, predict a
small set of (center, width, confidence) moment proposals in normalized
[0,1] coordinates relative to the candidate window. This is a deliberately
small version of jayleicn/moment_detr's architecture: a transformer encoder
over [query_token, clip_tokens...] followed by a few learned query slots
(like DETR object queries) that regress moment spans.

Not a full reproduction — good enough as a v1 grounding head; swap in the
real Moment-DETR (see reference repo) once this pipeline is validated.
"""
import torch
import torch.nn as nn


class MomentDETRLite(nn.Module):
    def __init__(
        self,
        feat_dim: int = 512,      # CLIP embedding dim (ViT-B/32 -> 512)
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        num_queries: int = 5,     # how many candidate spans to predict per window
        max_clips: int = 64,      # max clip tokens per candidate window
    ):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.query_proj = nn.Linear(feat_dim, d_model)

        self.pos_embed = nn.Parameter(torch.randn(1, max_clips + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.moment_queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # regress (center, width) in [0,1] + confidence logit
        self.span_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)
        )
        self.conf_head = nn.Linear(d_model, 1)

    def forward(self, clip_feats: torch.Tensor, query_feat: torch.Tensor):
        """
        clip_feats: (B, T, feat_dim) per-clip CLIP embeddings within the candidate window
        query_feat: (B, feat_dim) CLIP text embedding
        returns: centers (B, num_queries), widths (B, num_queries), confidences (B, num_queries)
        """
        B, T, _ = clip_feats.shape
        tokens = self.input_proj(clip_feats)                       # (B, T, d_model)
        q_tok = self.query_proj(query_feat).unsqueeze(1)           # (B, 1, d_model)
        seq = torch.cat([q_tok, tokens], dim=1)                    # (B, T+1, d_model)
        seq = seq + self.pos_embed[:, : seq.size(1), :]

        memory = self.encoder(seq)                                 # (B, T+1, d_model)

        queries = self.moment_queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d_model)
        decoded = self.decoder(queries, memory)                    # (B, Q, d_model)

        span = self.span_head(decoded).sigmoid()                   # (B, Q, 2) center,width in [0,1]
        conf = self.conf_head(decoded).squeeze(-1)                 # (B, Q)

        centers, widths = span[..., 0], span[..., 1]
        return centers, widths, conf

    @staticmethod
    def to_timestamps(center, width, window_start_sec, window_end_sec):
        """Convert normalized (center,width) back to absolute [start_sec, end_sec]."""
        dur = window_end_sec - window_start_sec
        c_sec = window_start_sec + center * dur
        w_sec = width * dur
        return c_sec - w_sec / 2, c_sec + w_sec / 2
