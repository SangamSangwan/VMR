"""
Stage 1: coarse retrieval.
Given per-window CLIP embeddings + a text query embedding, rank windows
by cosine similarity and return the top-k candidates. This alone is v0.
"""
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Candidate:
    start_sec: float
    end_sec: float
    score: float


def cosine_topk(
    window_embeddings: np.ndarray,  # (N, D), L2-normalized
    query_embedding: np.ndarray,    # (D,), L2-normalized
    window_bounds: np.ndarray,      # (N, 2) [start_sec, end_sec]
    top_k: int = 5,
) -> List[Candidate]:
    sims = window_embeddings @ query_embedding  # (N,) cosine sim since both are normalized
    order = np.argsort(-sims)[:top_k]
    return [
        Candidate(
            start_sec=float(window_bounds[i, 0]),
            end_sec=float(window_bounds[i, 1]),
            score=float(sims[i]),
        )
        for i in order
    ]


def merge_adjacent_candidates(candidates: List[Candidate], gap_tol: float = 0.5) -> List[Candidate]:
    """Optional: merge top-k windows that are contiguous in time into one span
    before handing to the grounding stage, so the head sees more context."""
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda c: c.start_sec)
    merged = [cands[0]]
    for c in cands[1:]:
        last = merged[-1]
        if c.start_sec - last.end_sec <= gap_tol:
            merged[-1] = Candidate(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, c.end_sec),
                score=max(last.score, c.score),
            )
        else:
            merged.append(c)
    return merged
