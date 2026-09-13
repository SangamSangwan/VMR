"""
CLIP feature extraction (frozen backbone) for:
  - video windows -> one pooled embedding per window
  - text query -> one embedding

Usage (single video, in-process):
    from src.features.clip_extractor import ClipEmbedder
    embedder = ClipEmbedder()
    window_embs, windows = embedder.embed_video("my_video.mp4", window_sec=7.5)
    query_emb = embedder.embed_text("find where the dog jumps into the pool")

Usage (batch, precompute + cache to disk):
    python -m src.features.clip_extractor --video_dir data/charades_sta/videos \
        --out_dir data/charades_sta/features --window_sec 5
"""
import argparse
import os
import glob
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

from src.features.video_utils import chunk_video, Window


class ClipEmbedder:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False  # frozen backbone, per project brief
        self._text_cache = {}

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        if text in self._text_cache:
            return self._text_cache[text]
        tokens = self.tokenizer([text]).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        arr = feat.cpu().numpy()[0]
        self._text_cache[text] = arr
        return arr


    @torch.no_grad()
    def embed_frames(self, frames: np.ndarray) -> np.ndarray:
        """frames: (N, H, W, 3) uint8 RGB -> pooled (D,) embedding averaged over N frames."""
        imgs = torch.stack(
            [self.preprocess(Image.fromarray(f)) for f in frames]
        ).to(self.device)
        feats = self.model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        pooled = feats.mean(dim=0)
        pooled = pooled / pooled.norm()
        return pooled.cpu().numpy()

    def embed_video(self, video_path: str, window_sec: float = 7.5, frames_per_window: int = 3):
        """Returns (window_embeddings [num_windows, D], windows: List[Window])."""
        windows = chunk_video(video_path, window_sec, frames_per_window)
        embs = np.stack([self.embed_frames(w.frames) for w in windows])
        return embs, windows


def precompute_features(video_dir: str, out_dir: str, window_sec: float, frames_per_window: int):
    os.makedirs(out_dir, exist_ok=True)
    embedder = ClipEmbedder()
    video_paths = sorted(
        glob.glob(os.path.join(video_dir, "*.mp4")) + glob.glob(os.path.join(video_dir, "*.avi"))
    )
    for vp in tqdm(video_paths, desc="extracting CLIP features"):
        vid_id = os.path.splitext(os.path.basename(vp))[0]
        out_path = os.path.join(out_dir, f"{vid_id}.npz")
        if os.path.exists(out_path):
            continue  # cache hit, don't recompute
        embs, windows = embedder.embed_video(vp, window_sec, frames_per_window)
        bounds = np.array([[w.start_sec, w.end_sec] for w in windows])
        np.savez(out_path, embeddings=embs, window_bounds=bounds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window_sec", type=float, default=7.5)
    ap.add_argument("--frames_per_window", type=int, default=3)
    args = ap.parse_args()
    precompute_features(args.video_dir, args.out_dir, args.window_sec, args.frames_per_window)
