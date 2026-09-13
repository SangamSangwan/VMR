"""
Frame sampling utilities for chunking a video into fixed-length windows
and pulling representative frames from each window.
"""
from dataclasses import dataclass
from typing import List
import cv2
import numpy as np


@dataclass
class Window:
    start_sec: float
    end_sec: float
    frames: np.ndarray  # (num_frames, H, W, 3) uint8, RGB


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps if fps > 0 else 0.0


def chunk_video(
    video_path: str,
    window_sec: float = 7.5,
    frames_per_window: int = 3,
) -> List[Window]:
    """
    Splits the video into non-overlapping windows of `window_sec` seconds each,
    and samples `frames_per_window` evenly spaced frames from each window.
    These frames get CLIP-embedded and averaged to represent the window.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    windows: List[Window] = []
    t = 0.0
    while t < duration:
        end = min(t + window_sec, duration)
        sample_times = np.linspace(t, end, frames_per_window, endpoint=False)
        frames = []
        for st in sample_times:
            frame_idx = int(st * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if frames:
            windows.append(Window(start_sec=t, end_sec=end, frames=np.stack(frames)))
        t += window_sec

    cap.release()
    return windows
