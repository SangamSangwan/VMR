import json
import os

have = set(f.replace(".npz", "") for f in os.listdir("data/charades_sta/features"))
print(f"{len(have)} videos have features")

for split in ["train", "test"]:
    src = f"data/charades_sta/{split}.jsonl"
    dst = f"data/charades_sta/{split}_subset.jsonl"
    kept, total = 0, 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            total += 1
            row = json.loads(line)
            if row["vid"] in have:
                out.write(line)
                kept += 1
    print(f"{split}: kept {kept}/{total} -> {dst}")