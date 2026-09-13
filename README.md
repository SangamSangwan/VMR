# Video Moment Retrieval (retrieve-then-ground)

## 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Sanity-check the reference pipeline (milestone 1)

Before writing anything else, get `lighthouse` running on QVHighlights with a
pretrained checkpoint. This is *not* part of this repo — clone it separately:

```bash
git clone https://github.com/line/lighthouse.git
cd lighthouse
# follow its README to download QVHighlights features + a pretrained checkpoint,
# then run its Gradio demo. Confirms your env/data pipeline works before you
# write a line of your own code.
```

## 2. Get the datasets

```bash
bash data/download_qvhighlights.sh   # annotations + link to pre-extracted features
bash data/download_charades.sh       # annotations + raw videos, you extract features yourself
```

For Charades-STA, convert the raw `.txt` query files to this repo's jsonl format:

```python
from src.grounding.dataset import charades_txt_to_jsonl
charades_txt_to_jsonl("data/charades_sta/charades_sta_train.txt", "data/charades_sta/train.jsonl")
charades_txt_to_jsonl("data/charades_sta/charades_sta_test.txt", "data/charades_sta/test.jsonl")
```

## 3. Build the CLIP feature cache (milestone 2)

Only needed for datasets that don't ship pre-extracted features (Charades-STA):

```bash
python -m src.features.clip_extractor \
    --video_dir data/charades_sta/videos \
    --out_dir data/charades_sta/features \
    --window_sec 5
```

This caches one `.npz` per video (`embeddings`, `window_bounds`) so you never
recompute CLIP features on repeated runs.

## 4. Coarse retrieval only — v0 demo (milestone 3)

```python
from src.pipeline import MomentRetrievalPipeline

pipeline = MomentRetrievalPipeline()  # no grounding_ckpt -> coarse windows only
preds = pipeline.run("my_video.mp4", "the dog jumps into the pool", use_grounding=False)
for p in preds:
    print(f"{p.start_sec:.1f}s - {p.end_sec:.1f}s  score={p.confidence:.3f}")
```

## 5. Train the grounding head (milestone 4)

```bash
python -m src.grounding.train \
    --jsonl data/qvhighlights/train.jsonl \
    --feature_dir data/qvhighlights/features \
    --epochs 20 --batch_size 8 \
    --out_ckpt grounding_head.pt
```

Evaluate against lighthouse's published R@1 numbers:

```bash
python -m src.grounding.eval \
    --jsonl data/qvhighlights/val.jsonl \
    --feature_dir data/qvhighlights/features \
    --ckpt grounding_head.pt
```

## 6. Full retrieve-then-ground pipeline

```python
pipeline = MomentRetrievalPipeline(grounding_ckpt="grounding_head.pt")
preds = pipeline.run("my_video.mp4", "the dog jumps into the pool")
```

## 7. Serve it (milestone 5)

```bash
export GROUNDING_CKPT=grounding_head.pt   # omit to fall back to coarse-only
uvicorn serving.app:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/search \
  -F "file=@my_video.mp4" \
  -F "query=the dog jumps into the pool"
```

Or the Gradio demo instead of curl/React:

```bash
python -m ui.gradio_app
```

Validate the retrieve-then-ground scaling claim on a real long video (>10 min)
before calling this done — that's the whole point of the two-stage design.

## Project layout

```
vmr/
  data/                          download scripts + converters
  src/
    features/
      video_utils.py             frame sampling / chunking
      clip_extractor.py          CLIP embeddings (frozen backbone), disk cache
    retrieval/
      coarse_retrieval.py        cosine-similarity top-k over windows
    grounding/
      model.py                   MomentDETRLite (Moment-DETR-lite)
      dataset.py                 QVHighlights/Charades-STA loader
      train.py                   training loop
      eval.py                    R@1 @ IoU=0.5/0.7
    pipeline.py                  glues retrieval + grounding end to end
  serving/
    app.py                       FastAPI: POST /search
    schemas.py
  ui/
    gradio_app.py                quick demo UI
```

## Team split

- **Person A (retrieval/embeddings/data):** `src/features/`, `src/retrieval/`,
  dataset download + conversion scripts, benchmarking against lighthouse.
- **Person B (grounding/serving):** `src/grounding/`, `serving/`, `ui/`.
- **Shared interface:** `src/pipeline.py` — the list of `Candidate` windows
  handed from coarse retrieval to the grounding head. Agree on this shape
  early (window embeddings array + `[start_sec, end_sec]` bounds) and you can
  build in parallel.

## Reference repos (don't reinvent this)

- https://github.com/line/lighthouse — benchmark harness, 7 models, 6 datasets
- https://github.com/showlab/UniVTG — unified architecture reference
- https://github.com/jayleicn/moment_detr — original Moment-DETR, the model
  `src/grounding/model.py` is a lightweight stand-in for. Swap it in directly
  once this pipeline is validated, if you want the real published numbers.
