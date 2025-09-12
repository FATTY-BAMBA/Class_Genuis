# chapter_llama/tools/extract/ocr_asr_aligner.py
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

# env knob: sample every Nth ASR line to reduce OCR work
_OCR_SAMPLE_EVERY_N = max(1, int(os.getenv("OCR_SAMPLE_EVERY_N", "25")))

# "HH:MM:SS: text"
_ASR_LINE = re.compile(r"^\s*(\d{2}):(\d{2}):(\d{2}):\s*(.+?)\s*$")

def parse_existing_asr(asr_file_path: Path) -> List[Dict[str, Any]]:
    """
    Parses ASR file lines ("HH:MM:SS: text") into:
      [{"start_sec": int, "text": str}, ...]
    """
    if not asr_file_path.exists():
        raise FileNotFoundError(f"ASR file not found: {asr_file_path}")
    items: List[Dict[str, Any]] = []
    for line in asr_file_path.read_text(encoding="utf-8").splitlines():
        m = _ASR_LINE.match(line.strip())
        if not m:
            continue
        hh, mm, ss, text = m.groups()
        start = int(hh) * 3600 + int(mm) * 60 + int(ss)
        text = text.strip()
        if text:
            items.append({"start_sec": start, "text": text})
    return items

def get_ocr_context_for_asr(video_path: Path, asr_file_path: Path, ocr_processor) -> str:
    """
    Aligns OCR with ASR timestamps by scanning a small window around each sampled ASR line.
    Expects ocr_processor.get_text_for_many_segments(video_path, segments=[(start, end), ...]) -> List[dict]
    Returns a single markdown string (Chinese title kept for your UI).
    """
    asr_lines = parse_existing_asr(asr_file_path)
    if not asr_lines:
        return ""

    sampled = asr_lines[::_OCR_SAMPLE_EVERY_N]
    # Small window after each ASR timestamp; change 15s if you want a wider context
    segments: List[Tuple[int, int]] = [(max(0, x["start_sec"]), x["start_sec"] + 15) for x in sampled]

    results = ocr_processor.get_text_for_many_segments(video_path=video_path, segments=segments)

    lines = ["# 從投影片與螢幕捕捉到的相關文字："]
    for item in results:
        s = int(item.get("start", 0))
        texts = item.get("texts", []) or []
        if not texts:
            continue
        lines.append(f"*   於 {sec_to_hms(s)} 左右捕捉到:")
        for t in texts:
            # Normalize whitespace a bit for nicer bullets
            lines.append(f"    - 「{str(t).strip()}」")

    return "\n".join(lines) + ("\n" if lines else "")

def sec_to_hms(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
