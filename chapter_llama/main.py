# chapter_llama/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os

# Import your HTTP client/provider
from app.Providers.chapter_llama_provider import ChapterLlamaProvider

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Prefer an explicit upstream env var to avoid pointing at our own server.
# If CHAPTER_LLAMA_UPSTREAM is not set, falls back to CHAPTER_LLAMA_URL or
# the provider's default (http://127.0.0.1:8000).
UPSTREAM_URL = os.getenv("CHAPTER_LLAMA_UPSTREAM") or os.getenv("CHAPTER_LLAMA_URL")

# -----------------------------------------------------------------------------
# App & Provider
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Chapter Llama Wrapper",
    version="1.0.0",
    description="A FastAPI wrapper around an upstream Chapter-Llama service.",
)

# Pass base_url explicitly if provided, so we don't accidentally call ourselves
# when running uvicorn on the same port.
if UPSTREAM_URL:
    cl = ChapterLlamaProvider(base_url=UPSTREAM_URL)
else:
    cl = ChapterLlamaProvider()

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class DetectReq(BaseModel):
    audio_segments: List[Dict]
    ocr_segments: List[Dict]
    win_sec: int = 60
    overlap_sec: int = 10
    language: str = "zh-hant"

class TitleBlock(BaseModel):
    start: float
    end: float
    content: str

class TitleReq(BaseModel):
    blocks: List[TitleBlock]
    language: str = "zh-hant"
    max_chars_per_block: int = 2000

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "chapter-llama-wrapper",
        "docs": "/docs",
        "upstream": UPSTREAM_URL or "provider default (e.g. http://127.0.0.1:8000)",
    }

@app.get("/health")
def health():
    return {"ok": cl.health()}

@app.post("/v1/chapter/boundaries")
def detect(req: DetectReq):
    out = cl.detect_boundaries(
        audio_segments=req.audio_segments,
        ocr_segments=req.ocr_segments,
        win_sec=req.win_sec,
        overlap_sec=req.overlap_sec,
        language=req.language,
    )
    if out is None:
        raise HTTPException(
            status_code=502,
            detail="Upstream Chapter-Llama unavailable or returned no data",
        )
    return {"chapters": out}

@app.post("/v1/chapter/titles")
def title(req: TitleReq):
    blocks = [{"start": b.start, "end": b.end, "content": b.content} for b in req.blocks]
    out = cl.title_chapters(
        blocks,
        language=req.language,
        max_chars_per_block=req.max_chars_per_block,
    )
    if out is None:
        raise HTTPException(
            status_code=502,
            detail="Upstream Chapter-Llama unavailable or returned no data",
        )
    return {"chapters": out}

# -----------------------------------------------------------------------------
# Optional: run with `python chapter_llama/main.py` (not required for uvicorn -m)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chapter_llama.main:app", host="0.0.0.0", port=8001, reload=True)
