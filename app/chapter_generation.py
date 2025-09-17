# app/video_chaptering.py
"""
Module for generating video chapters from *raw* ASR and OCR inputs.
- No ASR/OCR preprocessing/merging; we use what you pass in.
- Enforces a ~128k token prompt budget (approx) for large-context models.
- Parses, lightly cleans, balances chapters, and converts titles to Traditional Chinese.
- Exposes RAW LLM output so you can inspect what the model produced before any parsing/balancing.

CLI examples:
    python video_chaptering.py --asr-file raw_asr.txt --duration 3600 --video-id test_01
    python video_chaptering.py --asr-file raw_asr.txt --ocr-file ocr_raw.txt --duration 1800 --video-id test_02
    # Show both RAW and FINAL in console:
    python video_chaptering.py --asr-file raw_asr.txt --duration 1800 --video-id debug_run --debug
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional Azure AI Inference imports (only if used)
try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential
except Exception:  # optional at runtime
    ChatCompletionsClient = None  # type: ignore
    SystemMessage = None  # type: ignore
    UserMessage = None  # type: ignore
    AzureKeyCredential = None  # type: ignore

# Optional OpenAI import (only if used)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Optional Simplified→Traditional conversion (OpenCC preferred)
try:
    from opencc import OpenCC
    _opencc = OpenCC('s2t')
except Exception:
    _opencc = None

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
@dataclass
class ChapterConfig:
    """Configuration for chapter generation service"""
    service_type: str = os.getenv("CHAPTER_SERVICE_TYPE", "openai")  # "openai" or "azure"
    openai_model: str = os.getenv("CHAPTER_OPENAI_MODEL", "gpt-4o")
    azure_model: str = os.getenv("CHAPTER_AZURE_MODEL", "Meta-Llama-3.1-8B-Instruct")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    azure_endpoint: Optional[str] = os.getenv("AZURE_AI_ENDPOINT")
    azure_key: Optional[str] = os.getenv("AZURE_AI_KEY")
    azure_api_version: str = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")

def validate_config(config: ChapterConfig) -> bool:
    """Validate that required configuration is present"""
    if config.service_type == "azure":
        if not config.azure_endpoint or not config.azure_key:
            logger.error("Azure AI credentials not configured. Set AZURE_AI_ENDPOINT and AZURE_AI_KEY.")
            return False
    elif config.service_type == "openai":
        if not config.openai_api_key:
            logger.error("OpenAI API key not configured. Set OPENAI_API_KEY.")
            return False
    else:
        logger.error(f"Unknown service type: {config.service_type}")
        return False
    return True

def get_content_hash(transcript: str, ocr_context: str, duration: float) -> str:
    """Generate hash for content to enable caching"""
    content = f"{transcript}{ocr_context}{duration}"
    return hashlib.md5(content.encode()).hexdigest()

# ─────────────────────────
# Utilities
# ─────────────────────────
CHAPTER_LINE_RE = re.compile(
    r"""
    ^\s*
    (?:[\-\*\u2022]\s*)?
    \[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?\s*
    (?:[\-–—:]\s*)?
    (?P<title>.+?)
    \s*$
    """,
    re.VERBOSE,
)

def sec_to_hms(sec: int) -> str:
    """Convert seconds to HH:MM:SS format"""
    if sec < 0:
        sec = 0
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _is_cjk(ch: str) -> bool:
    return '\u4e00' <= ch <= '\u9fff'

def clean_chapter_titles(chapters: Dict[str, str]) -> Dict[str, str]:
    """
    Clean up chapter titles by removing filler words and improving clarity.
    For Chinese titles, avoid English-centric capitalization; if over-trimmed (<4 chars),
    fall back to the original title.
    """
    cleaned: Dict[str, str] = {}
    filler_words = ['那', '所以', '這個', '那個', '就是', '呢', '啊', '喔', '然後', '接著']
    for ts, original_title in chapters.items():
        title = original_title
        for word in filler_words:
            title = title.replace(word, '')
        title = re.sub(r'[。，“”、！？\.!?,]+$', '', title.strip())
        title = re.sub(r'\s+', ' ', title)

        # If cleaning made it too short, revert to the original
        if 0 < len(title) < 4:
            title = original_title.strip()

        cleaned[ts] = title
    return cleaned

def count_tokens_llama(text: str) -> int:
    """Approximate token counting for mixed Chinese/English (≈1 token per CJK char; 1/4 per other chars)"""
    chinese_chars = sum(1 for char in text if _is_cjk(char))
    non_chinese_len = len(text) - chinese_chars
    return chinese_chars + max(1, non_chinese_len // 4)

def truncate_text_by_tokens(text: str, max_tokens: int = 120_000) -> str:
    """Truncate text to approximately max_tokens, preserving sentence boundaries where possible"""
    if max_tokens <= 0:
        return ""
    if count_tokens_llama(text) <= max_tokens:
        return text
    logger.warning(f"Truncating transcript from {count_tokens_llama(text):,} tokens to {max_tokens:,} tokens")
    sentences = re.split(r'(?<=[。！？.!?])', text)
    truncated = ""
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = count_tokens_llama(sentence)
        if current_tokens + sentence_tokens > max_tokens:
            break
        truncated += sentence
        current_tokens += sentence_tokens
    return truncated

# ─────────────────────────
# Chapter policy & parsing
# ─────────────────────────
def chapter_policy(duration_sec: int) -> Tuple[int, Tuple[int, int], int]:
    """Determine chapter generation parameters based on video duration"""
    if duration_sec < 30 * 60:
        return 90,  (6, 12), 30
    if duration_sec < 60 * 60:
        return 180, (8, 16), 40
    if duration_sec < 120 * 60:
        return 300, (10, 20), 50
    if duration_sec < 180 * 60:
        return 540, (12, 24), 60
    return 600, (14, 28), 80

def _normalize_ts(ts: str) -> str:
    """Normalize timestamp format to HH:MM:SS"""
    parts = ts.split(":")
    if len(parts) == 2:
        return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    if len(parts) == 3:
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    return ts

def parse_chapters_from_output(output_text: str) -> Dict[str, str]:
    """Parse chapter timestamps and titles from LLM output"""
    chapters: Dict[str, str] = {}
    
    # Direct parsing for "HH:MM:SS - Title" format
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        
        # Look for the pattern "HH:MM:SS - Title"
        if ' - ' in line:
            parts = line.split(' - ', 1)
            if len(parts) == 2:
                ts = parts[0].strip()
                title = parts[1].strip()
                # Validate timestamp format
                if re.match(r'\d{2}:\d{2}:\d{2}', ts):
                    chapters[ts] = title
    
    # If no chapters found, try the original regex approach
    if not chapters:
        for line in output_text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = CHAPTER_LINE_RE.match(line)
            if m:
                ts = _normalize_ts(m.group("ts").strip())
                title = m.group("title").strip()
                if title:
                    chapters.setdefault(ts, title)
    
    return chapters

def globally_balance_chapters(
    chapters: Dict[str, str],
    duration_sec: int,
    min_gap_sec: int,
    target_range: Tuple[int, int],
    max_caps: int,
) -> Dict[str, str]:
    """Balance chapters across the video duration with center-biased selection"""
    def ts_to_s(ts: str) -> int:
        p = ts.split(":")
        if len(p) == 2:
            return int(p[0]) * 60 + int(p[1])
        if len(p) == 3:
            return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
        return 0

    cands = [(ts_to_s(ts), ts, t.strip()) for ts, t in chapters.items() if 0 <= ts_to_s(ts) <= duration_sec]
    cands.sort(key=lambda x: x[0])
    if not cands:
        return {}

    # Deduplicate by minimum gap; keep the longer title as a proxy for richness.
    dedup = []
    for s, ts, title in cands:
        if dedup and (s - dedup[-1][0]) < min_gap_sec:
            if len(title) > len(dedup[-1][2]):
                dedup[-1] = (s, ts, title)
        else:
            dedup.append((s, ts, title))

    t_low, t_high = target_range
    if t_low <= len(dedup) <= t_high:
        return {ts: title for _, ts, title in dedup}

    # Too many chapters: choose the one nearest to each segment's center
    if len(dedup) > t_high:
        selected = []
        segment_length = max(1, duration_sec // t_high)
        for i in range(t_high):
            segment_start = i * segment_length
            segment_end = (i + 1) * segment_length if i < t_high - 1 else duration_sec + 1
            segment_center = (segment_start + segment_end) // 2
            segment_chapters = [c for c in dedup if segment_start <= c[0] < segment_end]
            if segment_chapters:
                chosen = min(segment_chapters, key=lambda c: abs(c[0] - segment_center))
                selected.append(chosen)
        selected.sort(key=lambda x: x[0])
        return {ts: title for _, ts, title in selected}

    # Not enough chapters: just cap to max_caps to be safe.
    return {ts: title for _, ts, title in dedup[:max_caps]}

# ─────────────────────────
# OCR handling (optional legacy "segments" mode)
# ─────────────────────────
def load_ocr_segments(file_obj, filename: str) -> List[Dict]:
    """
    Accepts:
      - JSON array:        [ { "start": 0, "end": 3, "text": "..." }, ... ]
      - Wrapped JSON:      { "segments": [ {...}, ... ] }
      - JSON Lines (JSONL): one JSON object per line
      - Plain text (.txt): whole file becomes a single segment at t=0
    Returns: List[Dict] with keys: start (int), end (int, optional), text (str)
    """
    try:
        data = json.load(file_obj)
        if isinstance(data, dict) and "segments" in data and isinstance(data["segments"], list):
            segments = data["segments"]
        elif isinstance(data, list):
            segments = data
        else:
            return []
        out = []
        for item in segments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start = int(item.get("start", 0))
            end = int(item.get("end", start))
            out.append({"start": start, "end": end, "text": text})
        return out
    except json.JSONDecodeError:
        pass

    # Try JSONL
    try:
        file_obj.seek(0)
        segments = []
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                segments = None
                break
            if not isinstance(obj, dict):
                continue
            text = str(obj.get("text", "")).strip()
            if not text:
                continue
            start = int(obj.get("start", 0))
            end = int(obj.get("end", start))
            segments.append({"start": start, "end": end, "text": text})
        if segments is not None:
            return segments
    except Exception:
        pass

    # Plain text fallback
    try:
        file_obj.seek(0)
        txt = file_obj.read().strip()
    except Exception:
        txt = ""
    return [{"start": 0, "end": 0, "text": txt}] if txt else []

def build_ocr_context_from_segments(ocr_segments: List[Dict]) -> str:
    """Legacy minimal OCR formatting: timestamped lines with a simple header."""
    if not ocr_segments:
        return ""
    lines = ["# 螢幕/投影片擷取文字（原始）："]
    for seg in ocr_segments:
        start = int(seg.get("start", 0))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"* {sec_to_hms(start)}：{text}")
    return "\n".join(lines)

# ─────────────────────────
# Simplified→Traditional conversion
# ─────────────────────────
_S2T_FALLBACK_MAP = {
    "体": "體", "台": "臺", "后": "後", "广": "廣", "画": "畫", "录": "錄", "观": "觀",
    "面": "麵", "发": "發", "门": "門", "问": "問", "类": "類", "网": "網", "图": "圖",
    "书": "書", "记": "記", "读": "讀", "党": "黨", "术": "術", "层": "層", "约": "約",
}

def to_traditional(text: str) -> str:
    """Convert a string to Traditional Chinese. Uses OpenCC if available; otherwise minimal mapping."""
    if not text:
        return text
    if _opencc is not None:
        try:
            return _opencc.convert(text)
        except Exception:
            pass
    return ''.join(_S2T_FALLBACK_MAP.get(ch, ch) for ch in text)

def ensure_traditional_chapters(chapters: Dict[str, str]) -> Dict[str, str]:
    """Convert all chapter titles to Traditional Chinese (idempotent if already Traditional)."""
    return {ts: to_traditional(title) for ts, title in chapters.items()}

# ─────────────────────────
# Client Initialization & LLM call
# ─────────────────────────
def initialize_client(service_type: str, **kwargs) -> Any:
    """Initialize the appropriate LLM client"""
    if service_type == "azure":
        if ChatCompletionsClient is None or AzureKeyCredential is None:
            raise RuntimeError("Azure dependencies are not available in this environment.")
        return ChatCompletionsClient(
            endpoint=kwargs["endpoint"],
            credential=AzureKeyCredential(kwargs["key"]),
            api_version=kwargs.get("api_version", "2024-05-01-preview"),
        )
    elif service_type == "openai":
        if OpenAI is None:
            raise RuntimeError("OpenAI client is not available in this environment.")
        return OpenAI(
            api_key=kwargs["api_key"],
            base_url=kwargs.get("base_url", "https://api.openai.com/v1/"),
        )
    else:
        raise ValueError(f"Unknown service type: {service_type}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
def call_llm(
    service_type: str,
    client: Any,
    system_message: str,
    user_message: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> Any:
    """Call LLM API with retry logic"""
    if service_type == "azure":
        return client.complete(
            messages=[
                SystemMessage(content=system_message),
                UserMessage(content=user_message),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model=model,
        )
    elif service_type == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response
    else:
        raise ValueError(f"Unknown service type: {service_type}")

# ─────────────────────────
# Prompt builder (ASR first, OCR second, OCR verbatim supported)
# ─────────────────────────
def build_prompt_body(
    transcript: str,
    duration_sec: int,
    ocr_context: str = "",
) -> str:
    """Build the prompt for chapter generation (ASR as the primary source)."""
    duration_hms = sec_to_hms(int(duration_sec))
    min_gap_sec, (t_low, t_high), max_caps = chapter_policy(int(duration_sec))
    intro = (
        "你是一位資深的教育內容編輯專家。你的任務是為以下影片逐字稿生成清晰、專業且簡潔的 YouTube 章節標題（**繁體中文**）。\n\n"
        f"影片總長度：{duration_hms}。請生成 **{t_low}–{t_high} 個章節**（最多 {max_caps} 個）。\n"
        f"每個章節間隔至少 **{min_gap_sec//60} 分鐘**（若主題延續請勿切分）。\n\n"
        "## 分析步驟：\n"
        "1.  **辨識主題：** 以 ASR 逐字稿為主要依據" + ("，並參考螢幕文字（OCR）" if ocr_context else "") + "。\n"
        "2.  **提取關鍵主題：** 找出影片中所教授的主要課程、模組或技能。\n"
        "3.  **創建章節標題：** 用清晰的標題總結每個段落，反映其核心教學價值。\n\n"
    )
    asr_block = (
        "## 輸出格式要求（必須嚴格遵守）：\n"
        "`HH:MM:SS - 標題`（不要編號、不要額外說明、不要裝飾符號）\n\n"
        "## 實際影片逐字稿內容（原始 ASR，作為主要依據）：\n"
        f"{transcript}\n\n"
    )
    ocr_block = ""
    if ocr_context.strip():
        # OCR is appended verbatim (no extra processing). Kept under a heading for clarity.
        ocr_block = (
            "## 螢幕文字輔助資訊（OCR，僅作輔助參考）：\n"
            f"{ocr_context}\n\n"
        )
    # Always put ASR first if present; OCR after.
    return intro + asr_block + ocr_block

# ─────────────────────────
# MAIN FUNCTIONS
# ─────────────────────────
def generate_chapters_debug(
    raw_asr_text: str,
    ocr_segments: List[Dict],
    duration: float,
    video_id: str,
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    *,
    ocr_context_override: Optional[str] = None,  # pass raw OCR here to inject verbatim
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """
    Returns a tuple: (raw_llm_text, parsed_raw_chapters, balanced_final_chapters)
    - raw_llm_text: exact text returned by the LLM BEFORE any parsing/cleaning/balancing.
    - parsed_raw_chapters: chapters parsed from raw_llm_text (no balancing yet), lightly cleaned & Traditionalized.
    - balanced_final_chapters: after policy-based dedup/down-selection.
    """
    if progress_callback:
        progress_callback("initializing", 0)

    if run_dir is None:
        run_dir = Path(f"/tmp/chapter_generation/{video_id}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting chapter generation for video {video_id} (duration: {duration}s)")

        # Load configuration
        config = ChapterConfig()
        if not validate_config(config):
            logger.warning("Configuration validation failed, using time-based fallback")
            fallback = create_time_based_fallback(int(duration))
            fallback = ensure_traditional_chapters(fallback)
            return ("", {}, fallback)

        if progress_callback:
            progress_callback("processing_inputs", 10)

        # Build OCR context:
        # - If override is provided, use it verbatim.
        # - Else, if ocr_segments is provided and you choose 'segments' mode upstream, format minimally.
        if ocr_context_override is not None:
            ocr_context = ocr_context_override
        else:
            ocr_context = build_ocr_context_from_segments(ocr_segments) if ocr_segments else ""

        min_gap_sec, target_range, max_caps = chapter_policy(int(duration))
        
        # Save raw inputs (for debugging/auditing)
        with open(run_dir / "raw_asr_text.txt", "w", encoding="utf-8") as f:
            f.write(raw_asr_text)
        if ocr_context_override is not None:
            with open(run_dir / "ocr_raw.txt", "w", encoding="utf-8") as f:
                f.write(ocr_context_override)
        else:
            with open(run_dir / "ocr_segments.json", "w", encoding="utf-8") as f:
                json.dump(ocr_segments, f, ensure_ascii=False, indent=2)

        if progress_callback:
            progress_callback("initializing_client", 20)

        # Initialize client
        service_type = config.service_type
        model = config.openai_model if service_type == "openai" else config.azure_model

        if service_type == "azure":
            client = initialize_client(
                service_type="azure",
                endpoint=config.azure_endpoint,
                key=config.azure_key,
                api_version=config.azure_api_version,
            )
        else:
            client = initialize_client(
                service_type="openai",
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
            )

        if progress_callback:
            progress_callback("building_prompt", 30)

        # Prompt with budgeting (~128k total)
        prompt_template = build_prompt_body("", int(duration), ocr_context)
        template_tokens = count_tokens_llama(prompt_template)
        CONTEXT_BUDGET = 128_000
        max_transcript_tokens = max(0, CONTEXT_BUDGET - template_tokens)
        transcript_for_prompt = truncate_text_by_tokens(raw_asr_text, max_transcript_tokens)
        full_prompt = build_prompt_body(transcript_for_prompt, int(duration), ocr_context)

        with open(run_dir / "full_prompt.txt", "w", encoding="utf-8") as f:
            f.write(full_prompt)

        if progress_callback:
            progress_callback("calling_llm", 50)

        # Call LLM — ASR-first priority
        enhanced_system_message = (
            "你是一個協助創建影片章節的助手。"
            "以『ASR 逐字稿』為主要依據產生章節；"
            "『OCR 文字』僅作為輔助參考，當與 ASR 衝突時，一律以 ASR 為準。"
            "請『只輸出』章節清單，每行格式：`HH:MM:SS - 標題`（繁體中文）。"
            "回應中請勿包含任何其他文字、評論或解釋。"
        )

        logger.info(f"Calling {service_type} API for chapter generation...")
        t0 = time.time()
        resp = call_llm(
            service_type=service_type,
            client=client,
            system_message=enhanced_system_message,
            user_message=full_prompt,
            model=model,
            max_tokens=2048,
            temperature=0.2,
            top_p=0.9,
        )
        dt = time.time() - t0
        logger.info(f"LLM API call completed in {dt:.2f}s")

        if progress_callback:
            progress_callback("parsing_response", 70)

        # RAW text from LLM (BEFORE any parsing/cleaning/balancing)
        if service_type == "azure":
            raw_llm_text = resp.choices[0].message.content
        else:
            raw_llm_text = resp.choices[0].message.content

        with open(run_dir / "llm_output_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw_llm_text)

        # Parse → "raw chapters" (no balancing yet)
        parsed_raw = parse_chapters_from_output(raw_llm_text)  # timestamps -> title
        # Light title clean & Traditionalize for readability (still pre-balance)
        parsed_raw_clean_trad = ensure_traditional_chapters(clean_chapter_titles(parsed_raw))

        with open(run_dir / "parsed_raw_chapters.json", "w", encoding="utf-8") as f:
            json.dump(parsed_raw_clean_trad, f, ensure_ascii=False, indent=2)

        if progress_callback:
            progress_callback("balancing_chapters", 80)

        # Balance according to policy
        chapters_final = globally_balance_chapters(
            parsed_raw_clean_trad, int(duration), min_gap_sec, target_range, max_caps
        )
        if not chapters_final:
            raise RuntimeError("No chapters left after balancing")

        with open(run_dir / "chapters_final.json", "w", encoding="utf-8") as f:
            json.dump(chapters_final, f, ensure_ascii=False, indent=2)

        if progress_callback:
            progress_callback("completed", 100)

        return (raw_llm_text, parsed_raw_clean_trad, chapters_final)

    except Exception as e:
        logger.error(f"Chapter generation failed: {e}", exc_info=True)
        # Fallback
        fallback = ensure_traditional_chapters(create_time_based_fallback(int(duration)))
        return ("", {}, fallback)

def create_time_based_fallback(duration_sec: int) -> Dict[str, str]:
    """Create fallback chapters based on time intervals"""
    fallback_chapters: Dict[str, str] = {}
    interval = 300  # 5 minutes
    for i in range(0, int(duration_sec), interval):
        fallback_chapters[sec_to_hms(i)] = "章節 " + str((i // interval) + 1)
    logger.info(f"Created {len(fallback_chapters)} time-based fallback chapters")
    return fallback_chapters

def generate_chapters(
    raw_asr_text: str,
    ocr_segments: List[Dict],
    duration: float,
    video_id: str,
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    *,
    ocr_context_override: Optional[str] = None,
) -> Dict[str, str]:
    """
    Backward-compatible wrapper returning only the FINAL balanced chapters.
    If you want raw model output as well, call `generate_chapters_debug` instead.
    """
    _raw_text, _parsed_raw, final_chapters = generate_chapters_debug(
        raw_asr_text=raw_asr_text,
        ocr_segments=ocr_segments,
        duration=duration,
        video_id=video_id,
        run_dir=run_dir,
        progress_callback=progress_callback,
        ocr_context_override=ocr_context_override,
    )
    return final_chapters

# ─────────────────────────
# CLI
# ─────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate video chapters from raw ASR and optional OCR.")
    parser.add_argument('--asr-file', type=argparse.FileType('r', encoding='utf-8'), required=True,
                        help='Path to file containing raw ASR text with timestamps.')
    parser.add_argument('--ocr-file', type=argparse.FileType('r', encoding='utf-8'),
                        help='Optional path to OCR file. In verbatim mode this is read as raw text.')
    parser.add_argument('--duration', type=float, required=True,
                        help='Duration of the video in seconds.')
    parser.add_argument('--video-id', type=str, required=True,
                        help='Unique identifier for the video (used for output directory).')
    parser.add_argument('--output-dir', type=str, default='./chapter_debug',
                        help='Directory to save debug outputs. Default: ./chapter_debug')
    parser.add_argument('--debug', action='store_true', help='Print RAW LLM output and parsed chapters too.')
    parser.add_argument('--ocr-mode', choices=['none', 'verbatim', 'segments'], default='verbatim',
                        help="How to include OCR: 'none' (omit), 'verbatim' (raw text), or 'segments' (legacy minimal formatting).")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    # Read ASR
    logger.info(f"Reading ASR text from {args.asr_file.name}...")
    raw_asr_text = args.asr_file.read()
    args.asr_file.close()

    # Read OCR according to the chosen mode
    ocr_segments: List[Dict] = []
    ocr_context_override: Optional[str] = None
    if args.ocr_file:
        if args.ocr_mode == 'none':
            logger.info("OCR mode: none (omit OCR from prompt).")
            try:
                args.ocr_file.close()
            except Exception:
                pass
        elif args.ocr_mode == 'verbatim':
            logger.info(f"OCR mode: verbatim. Reading {args.ocr_file.name} as raw text...")
            try:
                ocr_context_override = args.ocr_file.read()
            finally:
                try:
                    args.ocr_file.close()
                except Exception:
                    pass
            logger.info("OCR loaded verbatim.")
        else:
            logger.info(f"OCR mode: segments. Reading OCR segments from {args.ocr_file.name}...")
            try:
                ocr_segments = load_ocr_segments(args.ocr_file, args.ocr_file.name)
                args.ocr_file.close()
                logger.info(f"Loaded {len(ocr_segments)} OCR segments")
            except Exception as e:
                logger.warning(f"OCR file load failed, proceeding without OCR. Detail: {e}")
                try:
                    args.ocr_file.close()
                except Exception:
                    pass
                ocr_segments = []

    # Output directory
    run_dir = Path(args.output_dir) / args.video_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving debug outputs to: {run_dir}")

    # Simple progress callback
    def cli_progress_callback(stage: str, percent: int):
        logger.info(f"Progress: {percent}% - {stage}")

    # Generate
    logger.info("Starting chapter generation...")
    raw_text, parsed_raw, final_chapters = generate_chapters_debug(
        raw_asr_text=raw_asr_text,
        ocr_segments=ocr_segments,
        duration=args.duration,
        video_id=args.video_id,
        run_dir=run_dir,
        progress_callback=cli_progress_callback,
        ocr_context_override=ocr_context_override,  # raw OCR passes straight through
    )

    # Console output
    print("\n" + "="*50)
    print("✅ CHAPTER GENERATION COMPLETE")
    print("="*50)

    if args.debug:
        print("\n--- RAW LLM OUTPUT (as returned) ---")
        print(raw_text if raw_text else "(empty/raw fallback)")
        print("\n--- PARSED (pre-balance) ---")
        for ts, title in parsed_raw.items():
            print(f"{ts} - {title}")

    print("\n--- FINAL (balanced) ---")
    for ts, title in final_chapters.items():
        print(f"{ts} - {title}")

    # Save final chapters to a clean file
    output_file = run_dir / "final_chapters.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for timestamp, title in final_chapters.items():
            f.write(f"{timestamp} - {title}\n")
    logger.info(f"Final chapters saved to: {output_file}")

    # Also save a pre-balance view for convenience
    pre_file = run_dir / "parsed_raw_chapters.txt"
    with open(pre_file, 'w', encoding='utf-8') as f:
        for timestamp, title in parsed_raw.items():
            f.write(f"{timestamp} - {title}\n")
    logger.info(f"Parsed (pre-balance) chapters saved to: {pre_file}")

if __name__ == "__main__":
    main()
