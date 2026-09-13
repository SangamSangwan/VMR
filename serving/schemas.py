from pydantic import BaseModel
from typing import List


class MomentOut(BaseModel):
    start_sec: float
    end_sec: float
    confidence: float


class SearchResponse(BaseModel):
    query: str
    video_id: str
    moments: List[MomentOut]
