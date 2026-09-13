"""
Copies a small subset of Charades-STA videos (matching your train/test jsonl)
into a separate folder, so you can run clip_extractor.py on a manageable
sample instead of all ~9,848 videos.

Run from inside your vmr project root, with the venv active:
    python make_charades_subset.py
"""
import json
import os
import shutil
from pathlib import Path

ANNOTATION_FILES = [
    ("data/charades_sta/train.jsonl", 2000),
    ("data/charades_sta/test.jsonl", 400),
]
FULL_VIDEO_DIR = Path("data/charades_sta/videos")
SUBSET_DIR = Path("data/charades_sta/videos_subset")

SUBSET_DIR.mkdir(parents=True, exist_ok=True)

vids = []
seen = set()
for ann_path, cap in ANNOTATION_FILES:
    if not os.path.exists(ann_path):
        print(f"WARNING: {ann_path} not found, skipping")
        continue
    file_vids = []
    for line in open(ann_path, "r"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row["vid"] not in seen:
            seen.add(row["vid"])
            file_vids.append(row["vid"])
    sampled = file_vids[:cap]
    print(f"{ann_path}: {len(file_vids)} unique videos total, sampling {len(sampled)}")
    vids.extend(sampled)

print(f"Copying {len(vids)} videos into {SUBSET_DIR} ...")

missing = []
for i, vid in enumerate(vids):
    src = FULL_VIDEO_DIR / f"{vid}.mp4"
    dst = SUBSET_DIR / f"{vid}.mp4"
    if dst.exists():
        continue
    if not src.exists():
        missing.append(vid)
        continue
    shutil.copy(src, dst)
    if (i + 1) % 50 == 0:
        print(f"  ...{i + 1}/{len(vids)} copied")

print(f"\nDone. {len(vids) - len(missing)} copied, {len(missing)} missing from videos/.")
if missing:
    print("Missing video IDs (first 20):")
    for vid in missing[:20]:
        print(f"  {vid}")