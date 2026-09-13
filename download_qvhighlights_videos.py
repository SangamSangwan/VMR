"""
Downloads only the QVHighlights raw videos referenced in your annotation
jsonl files, from the Hugging Face mirror (ayushsdev/qvhighlights-videos).

Run from inside your vmr project root, with the venv active:
    python download_qvhighlights_videos.py

This version first fetches the REAL file listing from the HF repo, so it
doesn't have to guess folder/sharding structure.
"""
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

ANNOTATION_FILES = [
    ("data/qvhighlights/highlight_train_release.jsonl", None),
    ("data/qvhighlights/highlight_val_release.jsonl", None),
    ("data/qvhighlights/highlight_test_with_gt.jsonl", None),
]

OUT_DIR = Path("data/qvhighlights/videos")
REPO_ID = "ayushsdev/qvhighlights-videos"

OUT_DIR.mkdir(parents=True, exist_ok=True)

vids = set()
for ann_path, cap in ANNOTATION_FILES:
    if not os.path.exists(ann_path):
        print(f"WARNING: {ann_path} not found, skipping")
        continue
    file_vids = []
    seen = set()
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["vid"] not in seen:
                seen.add(row["vid"])
                file_vids.append(row["vid"])
    sampled = file_vids if cap is None else file_vids[:cap]
    print(f"{ann_path}: {len(file_vids)} unique videos total, sampling {len(sampled)}")
    vids.update(sampled)

print(f"Found {len(vids)} unique videos to download (subset, not the full dataset).")

print("Fetching real file listing from Hugging Face repo (this can take a bit)...")
all_repo_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
name_to_path = {}
for p in all_repo_files:
    if p.endswith(".mp4"):
        name_to_path[os.path.basename(p)] = p
print(f"Repo has {len(name_to_path)} mp4 files total.")

failed = []
for i, vid in enumerate(sorted(vids)):
    filename = f"{vid}.mp4"
    out_path = OUT_DIR / filename
    if out_path.exists():
        continue
    repo_path = name_to_path.get(filename)
    if repo_path is None:
        failed.append(vid)
        continue
    try:
        local_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=repo_path,
        cache_dir="E:/vmr/.hf_cache",
        )
        shutil.move(local_path, out_path)
    except Exception as e:
        print(f"  failed {vid}: {e}")
        failed.append(vid)
    if (i + 1) % 25 == 0:
        print(f"  ...{i + 1}/{len(vids)} processed")

print(f"\nDone. {len(vids) - len(failed)} downloaded, {len(failed)} failed.")
if failed:
    print("Failed video IDs (first 20):")
    for vid in failed[:20]:
        print(f"  {vid}")
    with open("data/qvhighlights/missing_videos.txt", "w") as f:
        f.write("\n".join(failed))
    print("Full list written to data/qvhighlights/missing_videos.txt")