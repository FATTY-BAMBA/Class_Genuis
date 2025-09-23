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

def parse_summary_from_output(output_text: str) -> Dict[str, str]:
    """Extract the structured summary from the LLM output"""
    summary = {}
    lines = output_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('課程主題：'):
            summary['topic'] = line.replace('課程主題：', '').strip()
        elif line.startswith('核心內容：'):
            summary['core_content'] = line.replace('核心內容：', '').strip()
        elif line.startswith('學習目標：'):
            summary['learning_objectives'] = line.replace('學習目標：', '').strip()
        elif line.startswith('適合對象：'):
            summary['target_audience'] = line.replace('適合對象：', '').strip()
        elif line.startswith('難度級別：'):
            summary['difficulty'] = line.replace('難度級別：', '').strip()
    
    # Apply Traditional Chinese conversion to summary fields
    if _opencc:
        for key in summary:
            summary[key] = to_traditional(summary[key])
    
    return summary

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
    duration_hms = sec_to_hms(int(duration_sec))
    min_gap_sec, (t_low, t_high), max_caps = chapter_policy(int(duration_sec))
    
    # Extract first and last timestamps for concrete examples
    timestamps = []
    for line in transcript.split('\n'):
        if ':' in line and len(line.split(':')) >= 4:
            ts_part = ':'.join(line.split(':')[:3])
            if re.match(r'\d{2}:\d{2}:\d{2}', ts_part):
                timestamps.append(ts_part)
    
    first_ts = timestamps[0] if timestamps else "00:00:00"
    last_ts = timestamps[-1] if timestamps else duration_hms
    
    prompt = f"""
# 教育章節設計專家 - 時間戳記精準對應版
你是資深線上課程設計專家，負責將教學影片轉化為專業教育章節結構。

# 🚨 最重要的規則 - 時間戳記必須精準對應
**逐字稿實際時間範圍：{first_ts} 到 {last_ts}**

## 絕對禁止的行為：
❌ 生成 00:00:00 章節（除非逐字稿真的從 00:00:00 開始）
❌ 規律時間間隔：每15分鐘、每30分鐘等固定模式
❌ 憑空想像時間點（必須對應逐字稿中的實際時間戳）
❌ 忽略逐字稿的時間範圍

## 必須遵守的規則：
✅ 第一個章節時間 >= {first_ts}（逐字稿開始時間）
✅ 最後一個章節時間 <= {last_ts}（逐字稿結束時間）  
✅ 每個章節時間必須接近逐字稿中實際討論該主題的時間戳（±60秒內）
✅ 基於內容自然轉折點，而非固定間隔

# 如何找到真實的章節轉折點：
## 語言信號詞（講師轉換話題）：
- 「接下來我們要講...」「現在進入...」「首先...第二...」
- 「我們來看一下...」「這個部分完成後，我們來看...」
- 「有了基礎概念，現在來實際操作...」
- 「問與答時間」「總結一下」「我們來練習...」

## 教學內容轉換：
- 新概念/技術的首次詳細解釋
- 理論講解 → 實際操作的轉換
- 不同工具/軟體的切換時間點
- 範例演示的開始與結束
- 練習題/互動環節的開始

## 視覺/操作轉換（參考OCR）：
- 畫面切換到新投影片/軟體界面
- 開始實際操作示範
- 檔案開啟/工具切換的時間點

# 錯誤示範 vs 正確做法：
## ❌ 錯誤（絕對避免）：
00:00:00 - 課程介紹
00:15:00 - 基礎概念  
00:30:00 - 進階應用
00:45:00 - 實作練習

## ✅ 正確（基於實際內容）：
{first_ts} - 課程開場與學習目標說明
[尋找逐字稿中第一個主題轉換的時間戳] - 第一個主要概念講解
[尋找逐字稿中理論轉實作的時間戳] - 實際操作演示開始
[尋找逐字稿中重要範例的時間戳] - 關鍵範例分析

# 影片資訊
- 總時長: {duration_hms}
- 逐字稿時間範圍: {first_ts} 到 {last_ts}
- 目標章節: {t_low}-{t_high} 個學習單元
- 最小間隔: {min_gap_sec//60} 分鐘

# 分析步驟：
1. **識別時間範圍**：確認逐字稿從 {first_ts} 開始，到 {last_ts} 結束
2. **通讀內容**：理解整體教學流程和知識架構
3. **標記轉折**：找出 {t_low}-{t_high} 個最重要的主題轉換點
4. **時間對應**：每個章節時間必須對應逐字稿中實際討論的時間
5. **標題精準**：用具體術語描述該時間點開始的教學內容

# 內容資料
## 主要逐字稿（包含真實時間戳）：
{transcript[:50000]}... [其餘內容已載入]

## 輔助視覺內容：
{ocr_context if ocr_context else "（無螢幕內容參考）"}

# 輸出格式
## 第一部分：章節列表
嚴格遵守：`HH:MM:SS - 具體章節標題`
- 時間戳必須是逐字稿中實際存在或非常接近（±60秒內）的時間
- 標題用繁體中文，具體描述該時間點開始的教學內容

## 第二部分：課程摘要（章節列表完成後，空一行輸出）
請提供結構化的課程摘要，格式如下：

課程主題：[主要教學領域，如：Python程式設計、Premiere Pro剪輯]
核心內容：[2-3個最重要的技術或概念]
學習目標：[學生完成後應具備的能力]
適合對象：[目標學員背景]
難度級別：[初級/中級/高級]

# 最終檢查
生成每個章節前，問自己：
1. 這個時間點在逐字稿中是否有對應的內容轉換？
2. 章節時間是否在 {first_ts} 到 {last_ts} 範圍內？
3. 標題是否準確反映從這個時間點開始的教學內容？

完成章節後，檢查摘要：
1. 課程主題是否準確反映核心教學內容？
2. 核心內容是否包含最重要的2-3個技術點？
3. 學習目標是否具體可衡量？
"""
    return prompt

# ─────────────────────────
# Hierarchical Multi-Pass Generation (NEW)
# ─────────────────────────

def should_use_hierarchical(duration: float, transcript_length: int) -> bool:
    """Determine if hierarchical multi-pass should be used"""
    # Use hierarchical for longer, content-rich educational videos
    return (duration >= 1800 and  # 30+ minutes
            transcript_length >= 5000 and  # Substantial content
            duration <= 14400)  # Under 4 hours (very long videos might need different handling)

def hierarchical_multipass_generation(
    raw_asr_text: str,
    duration: float,
    ocr_context: str,
    client: Any,
    config: ChapterConfig,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """
    Three-pass hierarchical generation for high-quality educational chapters
    Returns: (raw_llm_text, chapters, metadata)
    """
    
    # PASS 1: Course Structure Analysis (10% of budget)
    if progress_callback:
        progress_callback("analyzing_course_structure", 40)
    
    structure_prompt = f"""
作為資深教學設計專家，分析這個{sec_to_hms(int(duration))}教學影片的整體架構：

【核心學習目標】
1. 學生完成本課程後應掌握哪些關鍵能力？
2. 有哪些必須理解的核心理論或概念？
3. 有哪些需要熟練的實用技能？

【知識架構分析】
- 基礎鋪陳：哪些是前提知識或基礎概念？
- 核心教學：最重要的理論/方法/技術是什麼？
- 應用延伸：如何將所學應用於實際場景？
- 總結整合：如何將零散知識系統化？

【教學方法識別】
- 理論講解 vs. 實例演示 vs. 操作練習 的比例分佈
- 是否有問答互動、思考題、重點回顧？

影片內容摘要（前40,000字符）：
{truncate_text_by_tokens(raw_asr_text, 10000)}

輔助視覺內容：
{truncate_text_by_tokens(ocr_context, 2000) if ocr_context else "無"}
"""
    
    structure_response = call_llm(
        service_type=config.service_type,
        client=client,
        system_message="你是課程架構分析專家，擅長識別教學影片的整體學習目標和知識體系",
        user_message=structure_prompt,
        model=config.openai_model if config.service_type == "openai" else config.azure_model,
        max_tokens=1200,
        temperature=0.3
    )
    
    structure_text = (structure_response.choices[0].message.content 
                     if config.service_type == "openai" 
                     else structure_response.choices[0].message.content)
    
    # PASS 2: Learning Modules Identification (30% of budget)
    if progress_callback:
        progress_callback("identifying_learning_modules", 60)
    
    modules_prompt = f"""
基於課程結構分析：
{structure_text}

現在識別具體的學習模塊（4-8個），每個模塊應滿足：
1. 有明確的學習目標
2. 包含完整的教學閉環（講解→範例→練習）
3. 時長合理（10-45分鐘）
4. 有清晰的開始和結束標記

特別注意以下教學轉折信號：
- 主題轉換："接下來我們進入"、"現在開始講"、"第二部分"
- 深度變化："有了基礎我們來看"、"更深入的問題是"
- 應用轉向："理論講完了我們來實際操作"、"來看一個例子"

完整逐字稿（精簡至80,000字符）：
{truncate_text_by_tokens(raw_asr_text, 30000)}

請輸出格式：
模塊名稱 ~ 預估時間範圍 ~ 核心學習點 ~ 教學方法
範例：演算法基礎 ~ 00:00-00:25 ~ 時間複雜度分析 ~ 理論講解+範例演示
"""
    
    modules_response = call_llm(
        service_type=config.service_type,
        client=client,
        system_message="你是課程模塊設計師，擅長將教學內容分解為邏輯連貫的學習單元",
        user_message=modules_prompt,
        model=config.openai_model if config.service_type == "openai" else config.azure_model,
        max_tokens=1500,
        temperature=0.2
    )
    
    modules_text = (modules_response.choices[0].message.content 
                   if config.service_type == "openai" 
                   else modules_response.choices[0].message.content)
    
    # PASS 3: Detailed Chapter Generation (60% of budget)
    if progress_callback:
        progress_callback("generating_detailed_chapters", 80)
    
    chapters_prompt = f"""
【課程整體結構】
{structure_text}

【學習模塊規劃】  
{modules_text}

現在為每個模塊生成具體的章節時間點（總共15-30個章節），要求：

【章節設計原則】
1. 每個章節代表一個完整的學習子目標
2. 標記關鍵概念的首次詳細解釋
3. 標記重要範例或案例分析的開始
4. 標記練習題或互動環節
5. 標記重點回顧或總結處

【時間點選擇優先級】
高優先級：理論首次講解、核心公式推導、重要範例開始
中優先級：次要概念、補充說明、小練習
低優先級：重複強調、過渡語句、技術操作細節

【標題規範】
- 使用專業術語，反映具體學習內容
- 包含所屬模塊標籤（如：[基礎模塊]）
- 明確指出是講解、範例、練習還是總結

完整內容：
{raw_asr_text}

總時長：{sec_to_hms(int(duration))}

輸出格式：HH:MM:SS - [模塊標籤] 具體章節標題
範例：00:15:30 - [演算法基礎] 時間複雜度Big O表示法講解
"""
    
    final_response = call_llm(
        service_type=config.service_type,
        client=client,
        system_message="你是細心的章節設計師，擅長為學習模塊創建精確的時間標記",
        user_message=chapters_prompt,
        model=config.openai_model if config.service_type == "openai" else config.azure_model,
        max_tokens=2500,
        temperature=0.1
    )
    
    final_text = (final_response.choices[0].message.content 
                 if config.service_type == "openai" 
                 else final_response.choices[0].message.content)
    
    # Parse chapters
    chapters = parse_chapters_from_output(final_text)
    # Parse structured summary
    course_summary = parse_summary_from_output(final_text)
    
    # Extract educational metadata
    metadata = {
        'generation_method': 'hierarchical_multi_pass',
        'structure_analysis': structure_text,
        'modules_analysis': modules_text,
        'educational_quality_score': estimate_educational_quality(chapters, structure_text),
        'course_summary': course_summary 
    }
    
    return final_text, chapters, metadata

def estimate_educational_quality(chapters: Dict[str, str], structure: str) -> float:
    """Simple heuristic to estimate educational quality of chapters"""
    quality_indicators = [
        '講解', '原理', '範例', '練習', '實作', '應用', '總結', '重點',
        '概念', '方法', '技巧', '步驟', '案例', '分析'
    ]
    
    title_text = ' '.join(chapters.values())
    indicator_count = sum(1 for indicator in quality_indicators 
                         if indicator in title_text)
    
    total_titles = len(chapters)
    return min(1.0, indicator_count / max(1, total_titles * 0.7))

# ─────────────────────────
# Enhanced Main Function with Smart Routing
# ─────────────────────────

def generate_chapters_debug(
    raw_asr_text: str,
    ocr_segments: List[Dict],
    duration: float,
    video_id: str,
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    *,
    ocr_context_override: Optional[str] = None,
    # NEW: Add control parameter
    force_generation_method: Optional[str] = None,  # 'hierarchical' or 'single_pass'
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """
    Enhanced version with smart routing between hierarchical and single-pass generation
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

        # Build OCR context (existing logic)
        if ocr_context_override is not None:
            ocr_context = ocr_context_override
        else:
            ocr_context = build_ocr_context_from_segments(ocr_segments) if ocr_segments else ""

        min_gap_sec, target_range, max_caps = chapter_policy(int(duration))
        
        # Save raw inputs
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

        # Initialize client (existing logic)
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

        # 🎯 NEW: Smart Generation Method Selection
        use_hierarchical = False
        if force_generation_method == 'hierarchical':
            use_hierarchical = True
        elif force_generation_method == 'single_pass':
            use_hierarchical = False
        else:
            # Auto-detect based on content characteristics
            use_hierarchical = should_use_hierarchical(duration, len(raw_asr_text))
        
        logger.info(f"Using generation method: {'hierarchical_multi_pass' if use_hierarchical else 'single_pass'}")

        if use_hierarchical:
            if progress_callback:
                progress_callback("hierarchical_analysis", 30)
            
            # Use hierarchical multi-pass generation
            raw_llm_text, chapters, metadata = hierarchical_multipass_generation(
                raw_asr_text=raw_asr_text,
                duration=duration,
                ocr_context=ocr_context,
                client=client,
                config=config,
                progress_callback=progress_callback
            )
            
            # Save hierarchical metadata
            with open(run_dir / "hierarchical_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            with open(run_dir / "course_structure.txt", "w", encoding="utf-8") as f:
                f.write(metadata.get('structure_analysis', ''))
            with open(run_dir / "learning_modules.txt", "w", encoding="utf-8") as f:
                f.write(metadata.get('modules_analysis', ''))
                
        else:
            if progress_callback:
                progress_callback("single_pass_processing", 30)
            
            # Use original single-pass generation
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

            enhanced_system_message = (
                "你是專業的線上課程設計專家，擅長為各種學科創建高品質教育章節結構。"
                "自動識別課程領域並使用適當專業術語，專注於學習價值和教育連貫性。"
                "嚴格避免重複模式，創建反映真實教育進程的專業章節標題。"
                "僅輸出章節清單，每行格式: `HH:MM:SS - 標題`（繁體中文）。"
            )

            logger.info(f"Calling {service_type} API for single-pass chapter generation...")
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

            if service_type == "azure":
                raw_llm_text = resp.choices[0].message.content
            else:
                raw_llm_text = resp.choices[0].message.content

            # Parse chapters
            chapters = parse_chapters_from_output(raw_llm_text)
            # Parse structured summary
            course_summary = parse_summary_from_output(raw_llm_text)
            metadata = {'generation_method': 'single_pass',
                        'course_summary': course_summary}

        # COMMON POST-PROCESSING (existing logic)
        if progress_callback:
            progress_callback("parsing_response", 70)

        with open(run_dir / "llm_output_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw_llm_text)

        # Apply cleaning and Traditional Chinese conversion
        parsed_raw_clean_trad = ensure_traditional_chapters(clean_chapter_titles(chapters))

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

        # Save generation method info
        with open(run_dir / "generation_method.txt", "w", encoding="utf-8") as f:
            f.write(metadata.get('generation_method', 'unknown'))

        if progress_callback:
            progress_callback("completed", 100)

        return (raw_llm_text, parsed_raw_clean_trad, chapters_final)

    except Exception as e:
        logger.error(f"Chapter generation failed: {e}", exc_info=True)
        fallback = ensure_traditional_chapters(create_time_based_fallback(int(duration)))
        return ("", {}, fallback)
        
# ─────────────────────────
# MAIN FUNCTIONS
# ─────────────────────────

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
    # NEW: Add the same parameter here for consistency
    force_generation_method: Optional[str] = None,  # 'hierarchical' or 'single_pass'
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
        force_generation_method=force_generation_method  # Pass through the new parameter
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
