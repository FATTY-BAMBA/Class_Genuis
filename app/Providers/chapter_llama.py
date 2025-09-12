# app/Providers/chapter_llama.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List

class Segment(BaseModel):
    start: float
    end: float
    text: str

class ChapterLlamaPayload(BaseModel):
    audio_segments: List[Segment]
    ocr_segments: List[Segment]
    win_sec: int = Field(default=240, ge=30, le=600)
    overlap_sec: int = Field(default=45, ge=0, le=120)
    language: str = "zh-hant"