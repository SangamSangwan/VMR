"""
FastAPI backend.

Run:
    uvicorn serving.app:app --reload --port 8000

Endpoints:
    POST /search  (multipart form: file=<video>, query=<text>) -> SearchResponse
    GET  /health
"""
import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from serving.schemas import SearchResponse, MomentOut
from src.pipeline import MomentRetrievalPipeline

app = FastAPI(title="Video Moment Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before deploying publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

# load once at startup; pass a trained checkpoint path once you have one
GROUNDING_CKPT = os.environ.get("GROUNDING_CKPT")  # None -> coarse-only fallback
pipeline = MomentRetrievalPipeline(grounding_ckpt=GROUNDING_CKPT)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(file: UploadFile = File(...), query: str = Form(...)):
    if not query.strip():
        raise HTTPException(400, "query must not be empty")

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        use_grounding = GROUNDING_CKPT is not None
        preds = pipeline.run(tmp_path, query, use_grounding=use_grounding)
    finally:
        os.remove(tmp_path)

    return SearchResponse(
        query=query,
        video_id=str(uuid.uuid4()),
        moments=[
            MomentOut(start_sec=p.start_sec, end_sec=p.end_sec, confidence=p.confidence)
            for p in preds
        ],
    )
