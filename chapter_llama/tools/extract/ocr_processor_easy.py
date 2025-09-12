# chapter_llama/tools/extract/ocr_processor.py
import os
import time
from pathlib import Path
from typing import List, Tuple, Iterable, Dict

import cv2
import numpy as np
import easyocr

# ---------------- Env knobs (all optional) ----------------
# OPTIMIZED: Faster defaults for speed
_OCR_FPS           = float(os.getenv("OCR_FPS", "2.0"))            # Increased from 0.5 to 2.0 FPS
_OCR_DOWNSCALE_W   = int(os.getenv("OCR_DOWNSCALE_W", "640"))      # Reduced from 960 to 640px
_OCR_CONF          = float(os.getenv("OCR_CONF", "0.65"))          # Lower confidence threshold
_OCR_BATCH         = int(os.getenv("OCR_BATCH", "32"))             # Larger batch size for GPU
_OCR_PARAGRAPH     = os.getenv("OCR_PARAGRAPH", "false").lower() == "true"
_OCR_LANGS         = os.getenv("OCR_LANGS", "ch_tra,en").split(",")
_OCR_GPU           = os.getenv("OCR_GPU", "true").lower() == "true"
_OCR_MAX_FRAMES    = int(os.getenv("OCR_MAX_FRAMES", "50"))        # Reduced safety cap
_OCR_LOG_EVERY_N   = int(os.getenv("OCR_LOG_EVERY_N", "50"))       # Less frequent logging


def _sec_to_hms(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _downscale(frame: np.ndarray, target_w: int) -> np.ndarray:
    if target_w <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame
    scale = target_w / float(w)
    new_w, new_h = target_w, max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _unique_filtered_texts(texts: Iterable[str]) -> List[str]:
    """
    Dedup + quick denylist + length filter, language-agnostic.
    """
    denylist = {
        'user', '使用者', 'user:', '使用者:', 'username',
        'ta', '助教', 'ta:', '助教:',
        'file', '檔案', 'edit', '編輯', 'view', '檢視',
        'settings', '設定', 'options', '選項',
        'loading...', '載入中...', 'please wait', '請稍候',
        'welcome', '歡迎', 'thank you', '謝謝', 'logout', '登出',
        'next', '下一步', 'previous', '上一步', 'submit', '提交'
    }
    seen = set()
    out = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        tl = t.lower()
        if len(t) < 2 or tl in denylist:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


class OCRProcessor:
    """
    OPTIMIZED OCR processor with faster frame extraction and processing
    """

    def __init__(self):
        t0 = time.time()
        print("🔄 初始化 EasyOCR Reader ...")
        # GPU=True will silently fallback if CUDA not available; we also allow OCR_GPU=false to force CPU
        self.reader = easyocr.Reader(_OCR_LANGS, gpu=_OCR_GPU)
        self._has_batched = hasattr(self.reader, "readtext_batched")
        print(f"✅ OCR Reader ready (gpu={_OCR_GPU}, batched={self._has_batched}) in {time.time()-t0:.1f}s")

    # ----------- Public (compat) single-window API -----------
    def get_text_from_segment(self, video_path: Path, start_sec: int, end_sec: int) -> str:
        """
        Keeps your original signature & output format.
        """
        items = self.get_text_for_many_segments(
            video_path=video_path,
            segments=[(start_sec, end_sec)],
            fps=_OCR_FPS,
            downscale_w=_OCR_DOWNSCALE_W,
        )
        # items is a list of dicts [{"start":, "end":, "texts":[...]}]
        if not items:
            return ""
        seg = items[0]
        texts = seg.get("texts", [])
        if not texts:
            return ""
        ts = _sec_to_hms(int(start_sec))
        formatted = [f"*   於 {ts} 左右捕捉到:"]
        formatted += [f"    - 「{t}」" for t in texts]
        return "\n".join(formatted)

    # ----------- New: multi-window fast path -----------
    def get_text_for_many_segments(
        self,
        video_path: Path,
        segments: List[Tuple[int, int]],
        fps: float = _OCR_FPS,
        downscale_w: int = _OCR_DOWNSCALE_W,
    ) -> List[Dict]:
        """
        Extract OCR texts for many (start,end) windows efficiently.

        Returns:
          [
            {"start": s, "end": e, "texts": ["...", "..."]},
            ...
          ]
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        results = []
        try:
            for idx, (s, e) in enumerate(segments, 1):
                frames = self._extract_frames_optimized(cap, s, e, fps=fps, downscale_w=downscale_w)
                texts = self._run_ocr_optimized(frames)
                texts = _unique_filtered_texts(texts)
                results.append({"start": s, "end": e, "texts": texts})

                if idx % _OCR_LOG_EVERY_N == 0:
                    print(f"🖼️ OCR progress: {idx}/{len(segments)} windows processed")
        finally:
            cap.release()

        return results

    # ----------- OPTIMIZED Frame Extraction -----------
    def _extract_frames_optimized(
        self,
        cap: cv2.VideoCapture,
        start_sec: int,
        end_sec: int,
        fps: float,
        downscale_w: int,
    ) -> List[np.ndarray]:
        """
        OPTIMIZED: Extract consecutive frames without repeated seeking - much faster!
        """
        start_ms = max(0, int(start_sec * 1000))
        end_ms = max(start_ms, int(end_sec * 1000))
        
        # Seek to start position ONCE
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        
        # Get video FPS to calculate frame interval
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # fallback for invalid FPS
        
        frame_interval = max(1, int(video_fps / fps))
        
        frames = []
        count = 0
        current_ms = start_ms
        
        while current_ms <= end_ms and count < _OCR_MAX_FRAMES:
            ok, bgr = cap.read()
            if not ok:
                break
            
            # Only process every Nth frame based on desired FPS
            if count % frame_interval == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = _downscale(rgb, downscale_w)
                frames.append(rgb)
            
            count += 1
            current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        return frames

    # ----------- OPTIMIZED OCR Processing -----------
    def _run_ocr_optimized(self, frames: List[np.ndarray]) -> List[str]:
        """
        OPTIMIZED: Larger batches and more efficient processing
        """
        out: List[str] = []
        if not frames:
            return out

        # batched path (when available)
        if self._has_batched and len(frames) > 1:
            # Process ALL frames in maximum-sized batches
            batch_size = min(_OCR_BATCH, len(frames))
            for i in range(0, len(frames), batch_size):
                chunk = frames[i:i + batch_size]
                results = self.reader.readtext_batched(
                    chunk,
                    detail=1,
                    paragraph=_OCR_PARAGRAPH,
                    batch_size=batch_size,
                )
                for res_per_image in results:
                    for item in res_per_image:
                        if isinstance(item, (list, tuple)) and len(item) >= 3:
                            _, text, conf = item[:3]
                            if conf is None or conf >= _OCR_CONF:
                                out.append(text)
        else:
            # Fallback for single frames or no batched support
            for img in frames:
                res = self.reader.readtext(img, detail=1, paragraph=_OCR_PARAGRAPH)
                for item in res:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        _, text, conf = item[:3]
                        if conf is None or conf >= _OCR_CONF:
                            out.append(text)

        return out

    # ----------- Keep original for backward compatibility -----------
    def _extract_frames(
        self,
        cap: cv2.VideoCapture,
        start_sec: int,
        end_sec: int,
        fps: float,
        downscale_w: int,
    ) -> List[np.ndarray]:
        """Original method kept for compatibility"""
        return self._extract_frames_optimized(cap, start_sec, end_sec, fps, downscale_w)

    def _run_ocr(self, frames: List[np.ndarray]) -> List[str]:
        """Original method kept for compatibility"""
        return self._run_ocr_optimized(frames)