"""
End-to-end: video + text query -> ranked timestamp predictions.
This is the "meet at the candidate windows interface" point between the
two halves of the project (retrieval/embedding owner vs grounding/serving owner).
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch

from src.features.clip_extractor import ClipEmbedder
from src.retrieval.coarse_retrieval import cosine_topk, merge_adjacent_candidates, Candidate
from src.grounding.model import MomentDETRLite


@dataclass
class MomentPrediction:
    start_sec: float
    end_sec: float
    confidence: float


class MomentRetrievalPipeline:
    def __init__(
        self,
        grounding_ckpt: Optional[str] = None,
        window_sec: float = 7.5,
        top_k_windows: int = 5,
        context_pad_sec: float = 5.0,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = ClipEmbedder(device=self.device)
        self.window_sec = window_sec
        self.top_k_windows = top_k_windows
        self.context_pad_sec = context_pad_sec

        self.grounder = MomentDETRLite().to(self.device)
        if grounding_ckpt:
            self.grounder.load_state_dict(torch.load(grounding_ckpt, map_location=self.device))
        self.grounder.eval()

    def run(self, video_path: str, query: str, use_grounding: bool = True) -> List[MomentPrediction]:
        # --- Stage 1: coarse retrieval ---
        window_embs, windows = self.embedder.embed_video(video_path, self.window_sec)
        query_emb = self.embedder.embed_text(query)
        bounds = np.array([[w.start_sec, w.end_sec] for w in windows])

        effective_k = min(self.top_k_windows, max(1, len(windows) // 2))
        candidates: List[Candidate] = cosine_topk(
            window_embs, query_emb, bounds, top_k=effective_k
        )

        if not use_grounding:
            return [MomentPrediction(c.start_sec, c.end_sec, c.score) for c in candidates]

        # --- Stage 2: fine-grained grounding on top candidates only ---
        merged = merge_adjacent_candidates(candidates)
        predictions = []
        for cand in merged:
            ctx_start = max(0.0, cand.start_sec - self.context_pad_sec)
            ctx_end = cand.end_sec + self.context_pad_sec

            mask = (bounds[:, 0] >= ctx_start) & (bounds[:, 1] <= ctx_end)
            local_feats = torch.from_numpy(window_embs[mask]).float().unsqueeze(0).to(self.device)
            if local_feats.shape[1] == 0:
                continue
            q_feat = torch.from_numpy(query_emb).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                centers, widths, conf = self.grounder(local_feats, q_feat)
            conf_probs = torch.sigmoid(conf)[0]
            best_i = int(conf_probs.argmax())

            local_bounds = bounds[mask]
            start_sec, end_sec = MomentDETRLite.to_timestamps(
                centers[0, best_i].item(),
                widths[0, best_i].item(),
                local_bounds[0, 0],
                local_bounds[-1, 1],
            )
            predictions.append(
                MomentPrediction(start_sec, end_sec, float(conf_probs[best_i]))
            )

        predictions.sort(key=lambda p: -p.confidence)
        return predictions