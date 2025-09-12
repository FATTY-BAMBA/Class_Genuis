# app/video_processing.py
# optimized_video_processing.py

import os
import cv2
import subprocess
from io import BytesIO
import logging
import re
import numpy as np
from collections import Counter
from pydub import AudioSegment
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel
)
from faster_whisper import WhisperModel
import hashlib
import pickle
import json
import tempfile
import time
import contextlib
from pathlib import Path


# ==================== Optional GPU OCR ====================
OCRReader = None  # Force EasyOCR to be disabled
OCR_GPU = True    # PaddleOCR will still use GPU if available

# ==================== Logging Setup ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==================== Device & AMP Setup ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Whisper config (GPU if available) ====================
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v2")
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if WHISPER_DEVICE == "cuda" else "int8"
WHISPER_DOWNLOAD_ROOT = os.getenv("WHISPER_CACHE", "/workspace/models")

# helper
def _log_runtime_config():
    logging.info(
        f"CFG → ASR_WINDOW_SEC={ASR_WINDOW_SEC} ASR_VAD={ASR_VAD} "
        f"FILTER_OCR_EDGES={FILTER_EDGES} TRIM_BY_METADATA={TRIM_BY_METADATA}"
    )


# ASR behavior (env-tunable)
ASR_CHUNK_SEC = int(os.getenv("ASR_CHUNK_SEC", "600"))                  # 10 min chunks for robustness
ASR_CHUNK_OVERLAP = int(os.getenv("ASR_CHUNK_OVERLAP", "5"))            # seconds of overlap
ASR_BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE", "2"))                    # 1 = fastest, 2 = balanced
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "zh")
ASR_VAD = os.getenv("ASR_VAD", "false").lower() == "true"               # default OFF (we control granularity post-merge)
ASR_VAD_MIN_SPEECH = float(os.getenv("ASR_VAD_MIN_SPEECH", "1.0"))
ASR_VAD_MIN_SILENCE = float(os.getenv("ASR_VAD_MIN_SILENCE", "1.0"))

# Post-merge behavior (env-tunable)
ASR_ATOMIC_TARGET_SEC = float(os.getenv("ASR_ATOMIC_TARGET_SEC", "45"))
ASR_ATOMIC_MAX_SEC = float(os.getenv("ASR_ATOMIC_MAX_SEC", "90"))
ASR_ATOMIC_MAX_GAP = float(os.getenv("ASR_ATOMIC_MAX_GAP", "1.2"))
ASR_WINDOW_SEC = int(os.getenv("ASR_WINDOW_SEC", "600"))               # exact window size in seconds

# ==================== OCR Filtering Configuration ====================
FILTER_EDGES = os.getenv("FILTER_OCR_EDGES", "false").lower() == "true"
EDGE_HEAD_SEC = int(os.getenv("OCR_FILTER_HEAD_SEC", "600"))
EDGE_TAIL_SEC = int(os.getenv("OCR_FILTER_TAIL_SEC", "300"))
OCR_FILTER_BLACKLIST = ['新竹小幫手', '老師', '助教']

# ==================== Trimming by Metadata (config) ====================
TRIM_BY_METADATA = os.getenv("TRIM_BY_METADATA", "true").lower() == "true"
TRIM_MARGIN_SEC = int(os.getenv("TRIM_MARGIN_SEC", "2"))

# Accept multiple languages/casing/variants
SHARE_START_PATTERNS = [r"sharing\s*started", r"start\s*sharing", r"開始共享", r"開始分享"]
SHARE_STOP_PATTERNS  = [r"sharing\s*stopp?ed", r"stop\s*sharing", r"停止共享", r"停止分享"]

def _match_any(title: str, patterns) -> bool:
    t = (title or "").strip().lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)

# ==================== FFmpeg ====================
FFMPEG_BINARY = os.getenv("FFMPEG_PATH", "ffmpeg")

def check_ffmpeg():
    try:
        subprocess.run([FFMPEG_BINARY, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logging.info(f"✅ FFmpeg found: {FFMPEG_BINARY}")
        return FFMPEG_BINARY
    except Exception:
        logging.error("❌ FFmpeg not found or not executable!")
        raise FileNotFoundError("FFmpeg binary not found or not working.")
FFMPEG_BINARY = check_ffmpeg()

def _extract_frames_ffmpeg(video_path, every_sec=30, start_ts=0, end_ts=None):
    """Fallback: extract frames with ffmpeg when OpenCV can't decode."""
    import tempfile, os, cv2, subprocess, glob

    out = []
    with tempfile.TemporaryDirectory() as td:
        ss = ["-ss", str(start_ts)] if start_ts else []
        dur = []
        if end_ts is not None and end_ts > (start_ts or 0):
            dur = ["-t", str(float(end_ts) - float(start_ts))]

        dst = os.path.join(td, "frame_%06d.jpg")
        cmd = [FFMPEG_BINARY, *ss, "-i", video_path, *dur,
               "-vf", f"fps=1/{every_sec}", "-qscale:v", "2", dst]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        files = sorted(glob.glob(os.path.join(td, "frame_*.jpg")))
        for i, fp in enumerate(files):
            img = cv2.imread(fp)
            if img is None:
                continue  # guard against decode hiccups
            ts = (i) * every_sec + (start_ts or 0)
            out.append({"timestamp": float(ts), "frame": img, "frame_index": i+1})
    return out

# ==================== Helper ====================

def seed_topics_from_signals(ocr_texts, asr_windows=None, top_k=12):
    kw = []
    for t in (ocr_texts or []):
        kw.extend(extract_keywords(t or "", top_k=3))
    # (Optional) you can pass ASR windows later if you have them here:
    asr_text = ""
    if asr_windows:
        asr_text = " ".join([w.get("text","") for w in asr_windows])
    asr_tokens = re.findall(r"\b\w{5,}\b", asr_text.lower())
    from collections import Counter as _Ctr
    asr_common = [w for w,_ in _Ctr(asr_tokens).most_common(top_k)]
    topics, seen = [], set()
    for t in kw + asr_common:
        if t and t not in seen:
            seen.add(t)
            topics.append(t)
    topics += ["chart", "graph", "diagram", "code editor", "terminal", "equation", "table"]
    return topics[:max(8, top_k)]


# ==================== Vision Models Toggles ====================
ENABLE_BLIP = os.getenv("ENABLE_BLIP", "true").lower() == "true"
ENABLE_CLIP = os.getenv("ENABLE_CLIP", "true").lower() == "true"
BLIP_MODEL_NAME = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-large")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

_BLIP = {"proc": None, "model": None}
_CLIP = {"proc": None, "model": None}

def get_blip():
    if not ENABLE_BLIP:
        return None, None
    if _BLIP["proc"] is None or _BLIP["model"] is None:
        logging.info(f"🖼️ Loading BLIP model: {BLIP_MODEL_NAME} on {DEVICE}…")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        proc = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(DEVICE)
        model.eval()
        _BLIP["proc"], _BLIP["model"] = proc, model
        logging.info("✅ BLIP ready")
    return _BLIP["proc"], _BLIP["model"]

def get_clip():
    if not ENABLE_CLIP:
        return None, None
    if _CLIP["proc"] is None or _CLIP["model"] is None:
        logging.info(f"🔎 Loading CLIP model: {CLIP_MODEL_NAME} on {DEVICE}…")
        from transformers import CLIPProcessor, CLIPModel
        proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        model.eval()
        _CLIP["proc"], _CLIP["model"] = proc, model
        logging.info("✅ CLIP ready")
    return _CLIP["proc"], _CLIP["model"]


# ==================== Caching ====================
class VideoProcessingCache:
    def __init__(self, cache_dir="./video_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logging.info(f"📁 Cache directory: {cache_dir}")
    
    def get_video_hash(self, video_path):
        hasher = hashlib.md5()
        hasher.update(video_path.encode())
        hasher.update(str(os.path.getsize(video_path)).encode())
        hasher.update(str(os.path.getmtime(video_path)).encode())
        with open(video_path, 'rb') as f:
            hasher.update(f.read(1024*1024))
        return hasher.hexdigest()
    
    def cache_result(self, video_path, stage, result, metadata=None):
        video_hash = self.get_video_hash(video_path)
        cache_file = f"{self.cache_dir}/{video_hash}_{stage}.pkl"
        cache_data = {'result': result, 'metadata': metadata or {}, 'timestamp': time.time(), 'video_path': video_path}
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logging.info(f"💾 Cached {stage} results for {os.path.basename(video_path)}")
        except Exception as e:
            logging.warning(f"Failed to cache {stage}: {e}")
    
    def get_cached_result(self, video_path, stage, max_age_hours=24):
        video_hash = self.get_video_hash(video_path)
        cache_file = f"{self.cache_dir}/{video_hash}_{stage}.pkl"
        if not os.path.exists(cache_file):
            return None
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            age_hours = (time.time() - cache_data['timestamp']) / 3600
            if age_hours > max_age_hours:
                logging.info(f"🕒 Cache expired for {stage} ({age_hours:.1f}h old)")
                return None
            logging.info(f"🎯 Using cached {stage} results")
            return cache_data['result']
        except Exception as e:
            logging.warning(f"Failed to load cached {stage}: {e}")
            return None
    
    def clear_cache(self, video_path=None):
        if video_path:
            video_hash = self.get_video_hash(video_path)
            pattern = f"{video_hash}_*.pkl"
        else:
            pattern = "*.pkl"
        import glob
        files = glob.glob(os.path.join(self.cache_dir, pattern))
        for f in files:
            os.remove(f)
        logging.info(f"🗑️ Cleared {len(files)} cache files")

# ==================== Frame Preprocessor ====================
class FramePreprocessor:
    def __init__(self, similarity_threshold=0.95, blur_threshold=100):
        self.similarity_threshold = similarity_threshold
        self.blur_threshold = blur_threshold
        
    def compute_frame_hash(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 16))
        mean = np.mean(resized)
        binary = resized > mean
        return ''.join(['1' if b else '0' for b in binary.flatten()])
    
    def hamming_distance(self, h1, h2):
        return sum(c1 != c2 for c1, c2 in zip(h1, h2))
    
    def is_blurry(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.blur_threshold
    
    def detect_slide_change(self, frame1, frame2, threshold=0.3):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        small1 = cv2.resize(gray1, (64, 64))
        small2 = cv2.resize(gray2, (64, 64))
        diff = cv2.absdiff(small1, small2)
        similarity = 1 - (np.sum(diff) / (64 * 64 * 255))
        return similarity < (1 - threshold)
    
    def extract_optimized_frames(self, video_path, interval_sec=300, min_interval_sec=30, start_ts=0, end_ts=None):
        logging.info(f"🎬 Extracting optimized frames from {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0
        logging.info(f"🎥 VideoCapture stats: fps={fps:.3f} frames={total_frames} duration≈{duration:.2f}s")

        # If decoding fails in container, fall back to ffmpeg sampling
        if not fps or fps <= 1 or total_frames <= 0:
            logging.warning("⚠️ OpenCV cannot decode this video, falling back to ffmpeg-based frame extraction.")
            return _extract_frames_ffmpeg(
                video_path,
                every_sec=max(30, min_interval_sec),
                start_ts=start_ts,
                end_ts=end_ts
            )

        selected_frames = []
        prev_hash = None
        prev_frame = None
        last_selected_time = -min_interval_sec
        check_interval = int(fps * min_interval_sec) if fps else 1
        
        frames_checked = 0
        frames_selected = 0
        
        for frame_idx in range(0, total_frames, check_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames_checked += 1
            current_time = frame_idx / fps if fps else 0

            if current_time < start_ts or (end_ts is not None and current_time > end_ts):
                continue
            if current_time - last_selected_time < min_interval_sec:
                continue
            if self.is_blurry(frame):
                continue
            
            frame_hash = self.compute_frame_hash(frame)
            if prev_hash and self.hamming_distance(frame_hash, prev_hash) < 5:
                continue
            
            if prev_frame is not None:
                if not self.detect_slide_change(prev_frame, frame):
                    if current_time - last_selected_time < interval_sec:
                        continue
            
            selected_frames.append({'timestamp': current_time, 'frame': frame.copy(), 'frame_index': frame_idx})
            prev_hash = frame_hash
            prev_frame = frame.copy()
            last_selected_time = current_time
            frames_selected += 1
        
        cap.release()
        kept_pct = (frames_selected/frames_checked*100) if frames_checked else 0
        logging.info(f"✨ Frame optimization: {frames_checked} checked → {frames_selected} selected ({kept_pct:.1f}% kept)")
        return selected_frames

# ==================== Shared PaddleOCR ====================
OCR_INSTANCE = None
def get_shared_ocr():
    global OCR_INSTANCE
    if OCR_INSTANCE is None:
        from paddleocr import PaddleOCR
        logging.info("🔍 Loading shared PaddleOCR (lang='ch', GPU=%s)...", OCR_GPU)
        OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=OCR_GPU)
    return OCR_INSTANCE

# ==================== Metadata Extraction ====================
def extract_chapter_events(video_path):
    try:
        cmd = [FFMPEG_BINARY, "-i", video_path, "-f", "ffmetadata", "-"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        metadata = result.stdout.decode()

        events = []
        chapter = {}
        for line in metadata.splitlines():
            if line.startswith("[CHAPTER]"):
                if chapter:
                    events.append(chapter)
                chapter = {}
            elif "=" in line:
                key, val = line.split("=", 1)
                chapter[key.strip()] = val.strip()
        if chapter:
            events.append(chapter)

        parsed = []
        for ch in events:
            parsed.append({
                "start_sec": int(ch.get("START", 0)) / 1000,
                "end_sec": int(ch.get("END", 0)) / 1000,
                "title": ch.get("title", "").strip()
            })
        return parsed
    except Exception as e:
        logging.warning(f"⚠️ Failed to extract metadata: {e}")
        return []

# ==================== Text Cleanup ====================
def remove_repeated_phrases(text, min_repeat=3):
    pattern = r"((\S+)(?:\s*\2){%d,})" % (min_repeat - 1)
    return re.sub(pattern, r"\2", text)

def truncate_repetitions(text, max_repeat=5):
    tokens = text.split()
    seen = {}
    result = []
    for token in tokens:
        seen[token] = seen.get(token, 0) + 1
        if seen[token] <= max_repeat:
            result.append(token)
    return " ".join(result)

def clean_transcript_text(text):
    text = remove_repeated_phrases(text, min_repeat=3)
    text = truncate_repetitions(text, max_repeat=5)
    return text.strip()

# ==================== Whisper Singleton ====================
_WHISPER_MODEL = None
def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        logging.info(f"🧠 Loading Faster-Whisper model '{WHISPER_MODEL_NAME}' on {WHISPER_DEVICE} (compute={WHISPER_COMPUTE_TYPE})...")
        _WHISPER_MODEL = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=WHISPER_DOWNLOAD_ROOT
        )
        logging.info("✅ Faster-Whisper model ready")
    return _WHISPER_MODEL

# ==================== Audio Helpers (chunked) ====================
def _extract_audio_bytes(video_path: str) -> BytesIO:
    """FFmpeg to mono 16k WAV in-memory (once per video)."""
    logging.info("🎵 Extracting full audio to WAV (mono 16k)...")
    cmd = [FFMPEG_BINARY, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return BytesIO(proc.stdout)

def _yield_chunks(audio_wav: BytesIO, chunk_sec=600, overlap_sec=5):
    """Yield (start_s, end_s, AudioSegment) with small overlaps for continuity."""
    audio = AudioSegment.from_file(audio_wav, format="wav")
    total_ms = len(audio)
    size = int(chunk_sec * 1000)
    step = int((chunk_sec - overlap_sec) * 1000)
    for ms in range(0, total_ms, step):
        end = min(ms + size, total_ms)
        yield ms/1000.0, end/1000.0, audio[ms:end]

# ==================== Smarter Post-Merge ====================
def _ends_with_punc(t: str) -> bool:
    return bool(re.search(r'[。．！!？?：:；;」”\.\,\)]\s*$', t.strip()))

def merge_segments_smart(segments, target_sec=45, max_sec=90, max_gap=1.2, prefer_punctuation=True):
    """Build mid-sized, sentence-friendly chunks before aggregating to fixed windows."""
    if not segments:
        return []
    out = []
    cur = {"start": None, "end": None, "text": ""}

    for s in segments:
        if cur["start"] is None:
            cur["start"] = s["start"]
            cur["end"] = s["end"]
            cur["text"] = s["text"]
            continue

        gap = s["start"] - cur["end"]
        new_len = s["end"] - cur["start"]

        if gap > max_gap or new_len > max_sec:
            out.append({
                "start": round(cur["start"], 2),
                "end": round(cur["end"], 2),
                "text": clean_transcript_text(cur["text"])
            })
            cur = {"start": s["start"], "end": s["end"], "text": s["text"]}
            continue

        cur["end"] = s["end"]
        cur["text"] += (" " if not cur["text"].endswith(" ") else "") + s["text"]

        if (cur["end"] - cur["start"]) >= target_sec:
            if not prefer_punctuation or _ends_with_punc(cur["text"]):
                out.append({
                    "start": round(cur["start"], 2),
                    "end": round(cur["end"], 2),
                    "text": clean_transcript_text(cur["text"])
                })
                cur = {"start": None, "end": None, "text": ""}

    if cur["text"]:
        out.append({
            "start": round(cur["start"], 2),
            "end": round(cur["end"], 2),
            "text": clean_transcript_text(cur["text"])
        })
    return out

def merge_atomic_to_windows(atomic, window_sec=300):
    """Aggregate atomic chunks into exact window sizes (e.g., 5 minutes)."""
    if not atomic:
        return []
    windows = []
    w = {"start": atomic[0]["start"], "end": atomic[0]["end"], "text": atomic[0]["text"]}
    for s in atomic[1:]:
        if (s["end"] - w["start"]) <= window_sec:
            w["end"] = s["end"]
            w["text"] += "\n" + s["text"]
        else:
            windows.append({k: (round(v, 2) if k in ("start", "end") else v) for k, v in w.items()})
            w = {"start": s["start"], "end": s["end"], "text": s["text"]}
    windows.append({k: (round(v, 2) if k in ("start", "end") else v) for k, v in w.items()})
    return windows

# Backward-compat simple merger (kept)
def merge_segments_to_fixed_blocks(segments, block_sec=600):
    grouped = []
    current = {"start": None, "end": None, "text": ""}
    for seg in segments:
        if current["start"] is None:
            current["start"] = seg["start"]
        current["end"] = seg["end"]
        current["text"] += " " + seg["text"]
        if current["end"] - current["start"] >= block_sec:
            grouped.append({
                "start": round(current["start"], 2),
                "end": round(current["end"], 2),
                "text": clean_transcript_text(current["text"])
            })
            current = {"start": None, "end": None, "text": ""}
    if current["text"]:
        grouped.append({
            "start": round(current["start"], 2),
            "end": round(current["end"], 2),
            "text": clean_transcript_text(current["text"])
        })
    return grouped

# ==================== Audio Processing (Chunked, VAD-OFF default) ====================

def process_audio_pipeline_chunked(video_path, cache, start_ts=0, end_ts=None):
    """
    Transcribe in fixed chunks (10 min default), VAD disabled,
    then aggregate into EXACT 5-minute windows for LLM usage.
    (Returns only 10-min windows for speed/memory.)
    """
    cached = cache.get_cached_result(video_path, "audio_transcription")
    if cached:
        return cached

    logging.info("🎵 Chunked audio transcription starting…")
    whisper_model = get_whisper_model()

    audio_bytes = _extract_audio_bytes(video_path)

    all_segments = []
    chunk_idx = 0
    for c_start, c_end, seg_audio in _yield_chunks(audio_bytes, ASR_CHUNK_SEC, ASR_CHUNK_OVERLAP):
        if c_end <= start_ts or (end_ts is not None and c_start >= end_ts):
            continue
        chunk_idx += 1
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / f"asr_chunk_{chunk_idx}.wav"
            seg_audio.export(str(tmp_path), format="wav")
            logging.info(f"🎙️ ASR chunk {chunk_idx}: {c_start:.1f}s–{c_end:.1f}s (beam={ASR_BEAM_SIZE}, vad={ASR_VAD})")
            segments, info = whisper_model.transcribe(
                str(tmp_path),
                language=ASR_LANGUAGE,
                task="transcribe",
                beam_size=ASR_BEAM_SIZE,
                vad_filter=ASR_VAD,
                vad_parameters=({
                    "min_speech_duration_s": ASR_VAD_MIN_SPEECH,
                    "min_silence_duration_s": ASR_VAD_MIN_SILENCE
                } if ASR_VAD else None),
                word_timestamps=False,
                initial_prompt=""
            )
            for s in segments:
                abs_start = s.start + c_start
                abs_end = s.end + c_start
                if abs_end > start_ts and (end_ts is None or abs_start < end_ts):
                    all_segments.append({"start": abs_start, "end": abs_end, "text": s.text})

    all_segments.sort(key=lambda x: x["start"])

    # Keep the smart merge for quality, but don’t return/store raw/atomic
    atomic = merge_segments_smart(
        all_segments,
        target_sec=ASR_ATOMIC_TARGET_SEC,
        max_sec=ASR_ATOMIC_MAX_SEC,
        max_gap=ASR_ATOMIC_MAX_GAP,
        prefer_punctuation=True
    )
    merged = merge_atomic_to_windows(atomic, window_sec=ASR_WINDOW_SEC)

    cache.cache_result(video_path, "audio_transcription", merged)
    return merged


# ==================== Video Processing (OCR/BLIP/CLIP) ====================
def extract_keywords(text, top_k=5):
    tokens = re.findall(r"\b\w{4,}\b", text.lower())
    freq = Counter(tokens)
    return [w for w,_ in freq.most_common(top_k)]

def generate_blip_captions(frames, processor, model, batch_size=8, max_new_tokens=20):
    if not frames:
        return []
    captions = []
    model_device = next(model.parameters()).device
    try:
        model.eval()
        with torch.no_grad():
            is_cuda = model_device.type == "cuda"
            amp_ctx = torch.cuda.amp.autocast if is_cuda else contextlib.nullcontext
            for i in range(0, len(frames), batch_size):
                pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames[i:i+batch_size]]
                inputs = processor(images=pil_imgs, return_tensors="pt", padding=True).to(model_device)
                with amp_ctx():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=3,
                        do_sample=False
                    )
                caps = processor.batch_decode(out, skip_special_tokens=True)
                captions.extend([c.strip() for c in caps])
    except Exception as e:
        logging.warning(f"BLIP captioning failed: {e}")
        # Fallback: produce a minimal caption per frame so downstream isn’t empty
        captions = ["" for _ in frames]
    # Ensure non-empty strings (tiny heuristic fallback)
    return [c if c else "a slide or scene with text/graphics" for c in captions]


def _l2_normalize(t: torch.Tensor, eps=1e-12):
    return t / (t.norm(dim=-1, keepdim=True) + eps)

def match_clip_topics_batch(frames, topics, processor, model, top_k=1):
    if not frames or not topics:
        return [[] for _ in frames]
    device = next(model.parameters()).device
    model.eval()
    # 1) Encode topics once
    with torch.no_grad():
        txt = processor(text=topics, return_tensors="pt", padding=True).to(device)
        text_feats = _l2_normalize(model.get_text_features(**txt))  # (T, D)

    results = []
    bs = 16
    with torch.no_grad():
        for i in range(0, len(frames), bs):
            pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames[i:i+bs]]
            img_inputs = processor(images=pil, return_tensors="pt", padding=True).to(device)
            img_feats = _l2_normalize(model.get_image_features(**img_inputs))  # (B, D)
            # cosine = img_feats @ text_feats.T
            sims = img_feats @ text_feats.T
            topv, topi = sims.topk(k=min(top_k, len(topics)), dim=1)
            for row in topi.tolist():
                results.append([topics[j] for j in row])
    return results

def process_video_pipeline(video_path, cache, preprocessor, 
                           blip_processor=None, blip_model=None,
                           clip_processor=None, clip_model=None,
                           predefined_topics=None,
                           start_ts=0, end_ts=None):
    cached_result = cache.get_cached_result(video_path, "video_analysis")

    if cached_result:
        if isinstance(cached_result, dict) and "filtered" in cached_result:
            return cached_result
        else:
            return {"raw": None, "filtered": cached_result}
    
    # Auto-load models if not provided and toggles are enabled
    if blip_processor is None and blip_model is None and ENABLE_BLIP:
        blip_processor, blip_model = get_blip()
    if clip_processor is None and clip_model is None and ENABLE_CLIP:
        clip_processor, clip_model = get_clip()

    logging.info("🎬 Starting video processing pipeline...")
    frame_data = preprocessor.extract_optimized_frames(
        video_path,
        interval_sec=ASR_WINDOW_SEC,
        start_ts=start_ts,
        end_ts=end_ts
    )

    if not frame_data:
        logging.error("No frames extracted from video")
        return None
    
    frames = [fd['frame'] for fd in frame_data]
    times = [fd['timestamp'] for fd in frame_data]
    logging.info(f"🖼️ Processing {len(frames)} optimized frames...")

    texts = []
    ocr = get_shared_ocr()
    if OCRReader and len(frames) > 1:
        try:
            reader = OCRReader(['en', 'ch'], gpu=OCR_GPU)
            raw_results = reader.readtext(frames, detail=0, batch_size=4)
            texts = ["\n".join(result) if result else "" for result in raw_results]
        except Exception as e:
            logging.warning(f"EasyOCR batch failed: {e}, falling back to PaddleOCR")
            texts = []
    if not texts:
        for i, frame in enumerate(frames):
            try:
                res = ocr.ocr(frame, cls=True)
                if res and res[0]:
                    text = "\n".join([line[1][0] for line in res[0] if line[1][0].strip()])
                    texts.append(text)
                else:
                    texts.append("")
            except Exception as e:
                logging.warning(f"OCR failed for frame {i}: {e}")
                texts.append("")
    keywords = [extract_keywords(t) for t in texts]

    # Ensure we have topics even if none were provided
    if predefined_topics is None:
        predefined_topics = seed_topics_from_signals(texts, asr_windows=None, top_k=12)


    captions = []
    if blip_processor and blip_model:
        try:
            logging.info("🖼️ Generating BLIP captions...")
            captions = generate_blip_captions(frames, blip_processor, blip_model)
        except Exception as e:
            logging.warning(f"BLIP processing failed: {e}")
            captions = [""] * len(frames)

    clip_hits = []
    if clip_processor and clip_model and predefined_topics:
        try:
            logging.info("🔍 Matching CLIP topics...")
            clip_hits = match_clip_topics_batch(frames, predefined_topics, clip_processor, clip_model)
        except Exception as e:
            logging.warning(f"CLIP processing failed: {e}")
            clip_hits = [[]] * len(frames)
    
    # ---- Vision metrics AFTER we have captions/clip_hits ----
    cap_rate = (sum(1 for c in captions if str(c).strip()) / max(1, len(captions))) if captions is not None else 0.0
    clip_rate = (sum(1 for hits in clip_hits if hits) / max(1, len(clip_hits))) if clip_hits is not None else 0.0

    vision_meta = {
        "blip_enabled": bool(blip_processor and blip_model),
        "clip_enabled": bool(clip_processor and clip_model),
        "caption_nonempty_rate": cap_rate,
        "clip_nonempty_rate": clip_rate
    }

    STRICT_VISION = os.getenv("STRICT_VISION", "false").lower() == "true"
    if STRICT_VISION and vision_meta["blip_enabled"] and cap_rate == 0:
        raise RuntimeError("STRICT_VISION: BLIP produced no captions.")
    if STRICT_VISION and vision_meta["clip_enabled"] and clip_rate == 0:
        raise RuntimeError("STRICT_VISION: CLIP produced no topic matches.")


    ocr_segs = []
    for i, timestamp in enumerate(times):
        ocr_segs.append({
            "start": round(timestamp, 2),
            "end": round(times[i+1] if i+1 < len(times) else timestamp + ASR_WINDOW_SEC, 2),
            "timestamp": timestamp,
            "text": texts[i] if i < len(texts) else "",
            "keywords": keywords[i] if i < len(keywords) else [],
            "blip_caption": captions[i] if i < len(captions) else "",
            "clip_topics": clip_hits[i] if i < len(clip_hits) else []
        })
    
    # Duration clamp
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    cap.release()
    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
    else:
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                 "default=noprint_wrappers=1:nokey=1", video_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
            )
            duration = float(probe.stdout.strip())
        except Exception:
            logging.warning("⚠️ Could not determine duration; defaulting to large value.")
            duration = 18000.0
    for seg in ocr_segs:
        seg["end"] = min(seg.get("end", 0), duration)

    ocr_segs_raw = list(ocr_segs)
    filtered_segs = apply_ocr_filters(ocr_segs, duration)

    cache.cache_result(video_path, "video_analysis", {"raw": ocr_segs_raw, "filtered": filtered_segs})
    return {
        "raw": ocr_segs_raw,
        "filtered": filtered_segs,
        "vision_meta": vision_meta   ### ADD: keep meta in output
    }

# ==================== Orchestrator (Sequential) ====================
def sequential_video_processing(video_path, cache_dir="./video_cache",
                                blip_processor=None, blip_model=None,
                                clip_processor=None, clip_model=None,
                                predefined_topics=None):
    """
    Main function that processes audio (chunked) and video sequentially.
    Audio: 10-min chunks, VAD OFF (default), then post-merge to EXACT 10-min windows.
    """
    _log_runtime_config()
    logging.info(f"🚀 Starting video processing (sequential): {os.path.basename(video_path)}")
    cache = VideoProcessingCache(cache_dir)
    preprocessor = FramePreprocessor()

    # Optional metadata-based trimming
    events = extract_chapter_events(video_path)
    start_ts, end_ts = 0, None

    if TRIM_BY_METADATA and events:
        for evt in events:
            title = evt.get("title", "")
            if _match_any(title, SHARE_START_PATTERNS):
                start_ts = evt.get("start_sec", start_ts)
            if _match_any(title, SHARE_STOP_PATTERNS):
                end_ts = evt.get("end_sec", end_ts)
        if start_ts == 0 and end_ts is None and events:
            start_ts = max(0, events[0].get("start_sec", 0))
            end_ts = max(start_ts, events[-1].get("end_sec", start_ts))
        if start_ts or (end_ts is not None):
            start_ts = max(0, start_ts - TRIM_MARGIN_SEC)
            if end_ts is not None:
                end_ts = max(start_ts, end_ts + TRIM_MARGIN_SEC)

    if start_ts == 0 and end_ts is None:
        logging.info("📺 No usable share metadata — processing entire video.")
    else:
        logging.info(f"✂️ Trimming to metadata window: start={start_ts:.2f}s, end={end_ts:.2f}s (with margins)")

    start_time = time.time()

    # --- 1) Audio (Chunked) ---
    logging.info("🎧 Starting chunked audio processing…")
    try:
        audio_data_merged = process_audio_pipeline_chunked(
            video_path, cache, start_ts=start_ts, end_ts=end_ts
        )
        if audio_data_merged is None:
            return None

        logging.info(f"✅ Audio done | windows({ASR_WINDOW_SEC//60}m)={len(audio_data_merged)}")
        
    except Exception as e:
        logging.error(f"❌ Audio processing failed: {e}")
        return None

    # --- 2) Video (OCR/BLIP/CLIP) ---
    logging.info("🎬 Starting video processing…")
    ocr_raw, ocr_filtered = [], []

    video_out = {"raw": [], "filtered": [], "vision_meta": {}}

    try:
        video_out = process_video_pipeline(
            video_path, cache, preprocessor,
            blip_processor, blip_model,
            clip_processor, clip_model,
            predefined_topics,
            start_ts=start_ts, end_ts=end_ts
        ) or {"raw": [], "filtered": [], "vision_meta": {}}
        if video_out:
            ocr_raw = video_out.get("raw") or []
            ocr_filtered = video_out.get("filtered") or []
            logging.info("✅ Video processing completed")
        else:
            logging.warning("⚠️ Video processing returned no result; continuing with audio-only.")
    except Exception as e:
        logging.error(f"⚠️ OCR pipeline failed; continuing with audio-only: {e}")
        # keep defaults: video_out stays as the safe dict; ocr_* already []

    processing_time = time.time() - start_time
    final_result = {
        "video_path": video_path,
        "processing_time": processing_time,
        # ✅ AUDIO → 5-min windows for LLM input
        "audio_transcription": audio_data_merged,
        # ✅ OCR → filtered segments for LLM input (may be empty)
        "vision_meta": video_out.get("vision_meta", {}), 
        "video_analysis": ocr_filtered,
        "video_analysis_raw": ocr_raw,
        "timestamp": time.time(),
        "method": "sequential_chunked_no_vad",
    }
    cache.cache_result(video_path, "final_result", final_result)
    logging.info(f"🎉 Sequential processing completed in {processing_time:.1f}s")
    return final_result

# ==================== Helper Functions (unchanged) ====================
def apply_ocr_filters(ocr_segments, total_video_duration):
    blacklist_regex = re.compile("|".join(map(re.escape, OCR_FILTER_BLACKLIST)))
    filtered = []
    for seg in ocr_segments:
        text = seg.get("text", "").strip()
        blip = seg.get("blip_caption", "").strip()
        clip = seg.get("clip_topics", [])
        if blacklist_regex.fullmatch(text):
            continue
        if FILTER_EDGES:
            if seg['timestamp'] < EDGE_HEAD_SEC or seg['timestamp'] > (total_video_duration - EDGE_TAIL_SEC):
                continue
        if len(text) < 30 and not blip and not clip:
            continue
        filtered.append(seg)
    return filtered

def group_slides_by_time(ocr_segments, max_gap=90, max_blocks=12):
    if not ocr_segments:
        return []
    grouped = []
    current_group = [ocr_segments[0]]
    for seg in ocr_segments[1:]:
        gap = seg['timestamp'] - current_group[-1]['timestamp']
        if gap > max_gap:
            grouped.append(current_group)
            current_group = []
        current_group.append(seg)
    if current_group:
        grouped.append(current_group)
    while len(grouped) > max_blocks:
        min_gap = float('inf')
        min_index = 0
        for i in range(len(grouped) - 1):
            gap = grouped[i+1][0]['timestamp'] - grouped[i][-1]['timestamp']
            if gap < min_gap:
                min_gap = gap
                min_index = i
        grouped[min_index].extend(grouped[min_index + 1])
        del grouped[min_index + 1]
    return grouped

def align_topic_timestamps(topics, ocr_blocks):
    aligned_topics = []
    for i, topic in enumerate(topics):
        if i >= len(ocr_blocks):
            break
        block = ocr_blocks[i]
        start = block[0]["timestamp"]
        end = block[-1]["timestamp"]
        aligned = topic.copy()
        aligned["start"] = round(start, 2)
        aligned["end"] = round(end, 2)
        aligned["order_hint"] = i + 1
        aligned_topics.append(aligned)
    return aligned_topics

#------------------ main---------------
if __name__ == "__main__":
    import os, sys, json
    from time import time as _now
    from datetime import datetime
    # helpful helper near the top of __main__
    win_label = f"{ASR_WINDOW_SEC//60}min"


    # 🎯 Hard-code your video here OR override via env VIDEO_PATH
    video_path = os.environ.get("VIDEO_PATH", r"D:\Codes\Class_Genius-best-safe\app\input_video.mp4")

    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)

    # Config
    start_ts = 0
    end_ts = None  # e.g., 1800 for first 30 minutes
    cache_dir = "./video_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # No BLIP/CLIP for faster test (wire later if needed)
    blip_processor = blip_model = clip_processor = clip_model = None
    predefined_topics = None

    t0 = _now()
    cache = VideoProcessingCache(cache_dir)

    # --- AUDIO ---
    audio_windows = process_audio_pipeline_chunked(
        video_path, cache, start_ts=start_ts, end_ts=end_ts
    )
    if audio_windows is None:
        print("❌ Audio processing failed.")
        sys.exit(2)

    # --- VIDEO (OCR) — best-effort, won't crash the run if OCR stack isn't available ---
    try:
        preprocessor = FramePreprocessor()
        video_out = process_video_pipeline(
            video_path, cache, preprocessor,
            blip_processor, blip_model,
            clip_processor, clip_model,
            predefined_topics,
            start_ts=start_ts, end_ts=end_ts
        ) or {"raw": [], "filtered": []}
    except Exception as e:
        logging.error(f"⚠️ OCR pipeline failed, continuing with audio-only results: {e}")
        video_out = {"raw": [], "filtered": []}

    elapsed = _now() - t0

    final_result = {
        "video_path": video_path,
        "processing_time": elapsed,
        # audio
        "audio_transcription": audio_windows,  # 10-min windows
        # video
        "video_analysis": video_out.get("filtered", []),
        "video_analysis_raw": video_out.get("raw", []),
        "method": "cli_chunked_no_vad"
    }

    # === Summary to console ===
    aw = final_result.get("audio_transcription") or []
    ov = final_result.get("video_analysis") or []
    print("\n=== SUMMARY ===")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Audio {ASR_WINDOW_SEC//60}-min windows: {len(aw)}")
    print(f"\nSample {ASR_WINDOW_SEC//60}-min window:")
    print(f"OCR segments (filtered): {len(ov)}")
    if aw:
        sample = aw[min(1, len(aw)-1)]
        text_preview = (sample.get("text", "")[:240] or "").replace("\n", " ")
        print(f"\nSample {ASR_WINDOW_SEC//60}-min window:")
        print(f"  {sample.get('start', 0.0):.2f}s – {sample.get('end', 0.0):.2f}s | {len(sample.get('text',''))} chars")
        print("  " + text_preview + ("..." if len(sample.get("text","")) > 240 else ""))

    # === Confirmed saves (timestamped, absolute paths) ===
    base_dir = os.path.dirname(__file__)
    runs_dir = os.path.join(base_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Combined JSON (everything)
    json_path = os.path.join(runs_dir, f"{base_name}_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    # Optional: a readable transcript preview (5-min windows)
    txt_path = os.path.join(runs_dir, f"{base_name}_{stamp}_{win_label}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for w in final_result["audio_transcription"]:
            f.write(f"[{w.get('start',0.0):.2f}–{w.get('end',0.0):.2f}] {w.get('text','')}\n\n")

    # --- Extra saves: keep audio + OCR in separate files ---

    # 1) Audio (5-min windows) as JSON + TXT
    audio_json_path = os.path.join(runs_dir, f"{base_name}_{stamp}_audio_{win_label}.json")
    with open(audio_json_path, "w", encoding="utf-8") as f:
        json.dump(final_result["audio_transcription"], f, ensure_ascii=False, indent=2)

    audio_txt_path = os.path.join(runs_dir, f"{base_name}_{stamp}_audio_{win_label}.txt")
    with open(audio_txt_path, "w", encoding="utf-8") as f:
        for w in final_result["audio_transcription"] or []:
            f.write(f"[{w.get('start',0.0):.2f}–{w.get('end',0.0):.2f}] {w.get('text','')}\n\n")

    # 2) OCR as JSON (filtered + raw)
    ocr_filtered_path = os.path.join(runs_dir, f"{base_name}_{stamp}_ocr_filtered.json")
    with open(ocr_filtered_path, "w", encoding="utf-8") as f:
        json.dump(final_result["video_analysis"], f, ensure_ascii=False, indent=2)

    ocr_raw_path = os.path.join(runs_dir, f"{base_name}_{stamp}_ocr_raw.json")
    with open(ocr_raw_path, "w", encoding="utf-8") as f:
        json.dump(final_result["video_analysis_raw"], f, ensure_ascii=False, indent=2)

    print(f"\n💾 Saved JSON → {os.path.abspath(json_path)}")
    print(f"📝 Saved {ASR_WINDOW_SEC//60}-min transcript preview → {os.path.abspath(txt_path)}")
    print(f"💾 Saved audio ({ASR_WINDOW_SEC//60}m) JSON → {os.path.abspath(audio_json_path)}")
    print(f"📝 Saved audio ({ASR_WINDOW_SEC//60}m) TXT  → {os.path.abspath(audio_txt_path)}")
    print(f"💾 Saved OCR (filtered)  → {os.path.abspath(ocr_filtered_path)}")
    print(f"💾 Saved OCR (raw)       → {os.path.abspath(ocr_raw_path)}")
