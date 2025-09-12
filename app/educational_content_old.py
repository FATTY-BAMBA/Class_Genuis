
# app/educational_content.py
"""
Module for generating educational content (MCQs and Lecture Notes) from pre-processed ASR and OCR segments.
Designed for enhancing student learning and review of educational materials.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

# Azure AI Inference imports
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# OpenAI import
from openai import OpenAI

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# ==================== PROGRESS ====================
STAGES = {
    "initializing": 5,
    "processing_inputs": 15,
    "initializing_client": 25,
    "generating_mcqs": 55,
    "generating_notes": 80,
    "processing_results": 92,
    "completed": 100,
}
def report(stage: str, progress_callback: Optional[Callable[[str, int], None]]):
    if progress_callback:
        progress_callback(stage, STAGES.get(stage, 0))

# ==================== CONFIGURATION ====================
@dataclass
class EducationalContentConfig:
    """Configuration for educational content generation service"""
    service_type: str = os.getenv("EDU_SERVICE_TYPE", "openai")
    openai_model: str = os.getenv("EDU_OPENAI_MODEL", "gpt-4o")
    azure_model: str = os.getenv("EDU_AZURE_MODEL", "Meta-Llama-3.1-8B-Instruct")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    azure_endpoint: Optional[str] = os.getenv("AZURE_AI_ENDPOINT")
    azure_key: Optional[str] = os.getenv("AZURE_AI_KEY")
    azure_api_version: str = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
    max_questions: int = int(os.getenv("MAX_QUESTIONS", "10"))
    max_notes_pages: int = int(os.getenv("MAX_NOTES_PAGES", "5"))
    enable_cache: bool = os.getenv("EDU_ENABLE_CACHE", "true").lower() == "true"
    force_traditional: bool = os.getenv("EDU_FORCE_TRADITIONAL", "true").lower() == "true"

def validate_config(config: EducationalContentConfig) -> bool:
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

def get_content_hash(transcript: str, ocr_context: str, content_type: str) -> str:
    """Generate hash for content to enable caching"""
    content = f"{transcript}{ocr_context}{content_type}"
    return hashlib.md5(content.encode()).hexdigest()

# ==================== DATA STRUCTURES ====================
@dataclass
class MCQ:
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    difficulty: str
    topic: str

@dataclass
class LectureNoteSection:
    title: str
    content: str
    key_points: List[str]
    examples: List[str]

@dataclass
class EducationalContentResult:
    mcqs: List[MCQ]
    lecture_notes: List[LectureNoteSection]
    summary: str

# ==================== UTILITIES ====================
def sec_to_hms(sec: int) -> str:
    """Convert seconds to HH:MM:SS format"""
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def count_tokens_llama(text: str) -> int:
    """Approximate token counting for Llama-like tokenization"""
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    non_chinese_len = len(text) - chinese_chars
    return chinese_chars + max(1, non_chinese_len // 4)

def truncate_text_by_tokens(text: str, max_tokens: int = 120_000) -> str:
    """Truncate to approx max_tokens, preserving whole sentences"""
    if max_tokens <= 0:
        return ""
    tokens = count_tokens_llama(text)
    if tokens <= max_tokens:
        return text
    logger.warning(f"Truncating content from {tokens:,} tokens to {max_tokens:,} tokens")
    sentences = re.split(r'(?<=[。！？.!?])', text)
    truncated, current = [], 0
    for sentence in sentences:
        t = count_tokens_llama(sentence)
        if current + t > max_tokens:
            break
        truncated.append(sentence)
        current += t
    return "".join(truncated)

def build_ocr_context_from_segments(ocr_segments: List[Dict]) -> str:
    """Convert OCR segments into a descriptive context string (sentence-level bullets)."""
    if not ocr_segments:
        return ""
    context_lines = ["# 從投影片與螢幕捕捉到的相關文字："]
    SENT_SPLIT = re.compile(r"[。；;！？!?]\s*|\n+")
    for seg in ocr_segments:
        start = int(seg.get('start', 0))
        text = (seg.get('text') or "").strip()
        if not text:
            continue
        timestamp = sec_to_hms(start)
        context_lines.append(f"*   於 {timestamp} 左右捕捉到:")
        for sent in filter(None, (s.strip() for s in SENT_SPLIT.split(text))):
            context_lines.append(f"    - 「{sent}」")
    return "\n".join(context_lines)

# ---------- Simplified -> Traditional conversion ----------
def _init_opencc():
    try:
        from opencc import OpenCC  # type: ignore
        return OpenCC('s2t')  # Simplified to Traditional
    except Exception:
        return None

_OPENCC = _init_opencc()

# Minimal fallback mapping for common characters if OpenCC is unavailable.
_S2T_FALLBACK = str.maketrans({
    "后": "後", "里": "裡", "台": "臺", "万": "萬", "与": "與", "书": "書", "体": "體",
    "价": "價", "优": "優", "儿": "兒", "动": "動", "华": "華", "发": "發", "后": "後",
    "复": "復", "国": "國", "广": "廣", "汉": "漢", "会": "會", "纪": "紀", "简": "簡",
    "经": "經", "历": "歷", "马": "馬", "门": "門", "面": "麵", "内": "內", "气": "氣",
    "权": "權", "确": "確", "实": "實", "术": "術", "体": "體", "万": "萬", "云": "雲",
    "众": "眾", "为": "為", "从": "從", "众": "眾", "冲": "衝", "读": "讀", "爱": "愛",
    "战": "戰", "钟": "鐘", "级": "級", "术": "術", "师": "師", "学": "學", "习": "習",
    "声": "聲", "观": "觀", "台": "臺", "这": "這", "这": "這", "里": "裡", "复": "複"
})

def to_traditional(text: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese.
    Uses OpenCC if available; otherwise falls back to a small character map.
    """
    if not text:
        return text
    if _OPENCC is not None:
        try:
            return _OPENCC.convert(text)
        except Exception:
            pass
    # Fallback: basic character-level conversion
    return text.translate(_S2T_FALLBACK)

# ==================== PROMPT BUILDERS ====================
def build_mcq_prompt(transcript: str, ocr_context: str, max_questions: int) -> str:
    """Build prompt for MCQ generation"""
    prompt = f"""
你是一位資深的教育專家，專門為學生創建高質量的多選題。請根據以下課程內容創建 {max_questions} 個多選題。

## 分析步驟：
1. **識別關鍵概念**：從逐字稿和螢幕文字中找出重要概念、定義、公式、原理和關鍵知識點
2. **創建多樣化題目**：包含記憶性、理解性、應用性和分析性題目
3. **設計有價值的選項**：正確答案明確，錯誤選項具有教育意義（常見誤解）
4. **提供詳細解釋**：解釋為什麼正確答案正確，錯誤答案為什麼錯誤

## 內容來源：
{ocr_context if ocr_context else "# 無螢幕文字內容"}

## 課程逐字稿：
{transcript}

## 輸出格式要求（必須嚴格遵守 JSON 格式）：
```json
{{
  "mcqs": [
    {{
      "question": "問題內容（繁體中文）",
      "options": ["選項A", "選項B", "選項C", "選項D"],
      "correct_answer": "正確選項字母（A、B、C或D）",
      "explanation": "詳細解釋為什麼這個答案正確",
      "difficulty": "easy/medium/hard",
      "topic": "相關主題或概念"
    }}
  ]
}}
```

請確保題目：
- 覆蓋課程的主要概念
- 難度分布均衡（簡單30%、中等50%、困難20%）
- 避免模糊或歧義的問題
- 每個選項都看起來合理但只有一個正確
"""
    return prompt

def build_lecture_notes_prompt(transcript: str, ocr_context: str, max_pages: int) -> str:
    """Build prompt for lecture notes generation"""
    prompt = f"""
你是一位優秀的教學設計專家，請將以下課程內容轉換為結構清晰、易於理解的講義筆記。

## 目標：
創建 {max_pages} 頁的綜合講義，幫助學生學習、複習和理解課程內容。

## 內容要求：
1. **結構化組織**：按邏輯順序組織內容，包含標題、子標題
2. **關鍵重點**：突出重要概念、公式、定義和原理
3. **實用例子**：包含相關範例和應用場景
4. **視覺化提示**：標記適合圖表、表格或圖形表示的內容
5. **學習提示**：添加學習技巧和常見錯誤提醒

## 內容來源：
{ocr_context if ocr_context else "# 無螢幕文字內容"}

## 課程逐字稿：
{transcript}

## 輸出格式要求（必須嚴格遵守 JSON 格式）：
```json
{{
  "sections": [
    {{
      "title": "章節標題",
      "content": "詳細內容說明...",
      "key_points": ["重點1", "重點2", "重點3"],
      "examples": ["範例1", "範例2"]
    }}
  ],
  "summary": "整個課程的簡要總結"
}}
```

請確保講義：
- 語言清晰簡潔（繁體中文）
- 技術術語準確
- 適合不同學習風格的學生
- 包含實際應用和例子
"""
    return prompt

# ==================== CLIENT INITIALIZATION ====================
def initialize_client(service_type: str, **kwargs) -> Any:
    """Initialize the appropriate LLM client"""
    if service_type == "azure":
        return ChatCompletionsClient(
            endpoint=kwargs["endpoint"],
            credential=AzureKeyCredential(kwargs["key"]),
            api_version=kwargs.get("api_version", "2024-05-01-preview")
        )
    elif service_type == "openai":
        return OpenAI(
            api_key=kwargs["api_key"],
            base_url=kwargs.get("base_url", "https://api.openai.com/v1/")
        )
    else:
        raise ValueError(f"Unknown service type: {service_type}")

# ==================== LLM API CALLS ====================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def call_llm(
    service_type: str,
    client: Any,
    system_message: str,
    user_message: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    top_p: float = 0.9
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
            model=model
        )
    elif service_type == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response
    else:
        raise ValueError(f"Unknown service type: {service_type}")

def extract_text_from_response(resp, service_type: str) -> str:
    """Handle Azure/OpenAI response shape differences safely."""
    try:
        if service_type == "azure":
            choice_list = getattr(resp, "choices", None)
            if not choice_list:
                return ""
            choice0 = choice_list[0]
            msg = getattr(choice0, "message", None)
            if msg and getattr(msg, "content", None):
                return msg.content
            if isinstance(choice0, dict):
                msg = choice0.get("message") or {}
                return msg.get("content", "") or ""
            return ""
        else:
            return resp.choices[0].message.content
    except Exception:
        logger.exception("Failed to extract content from LLM response")
        return ""

# ==================== SAFE JSON HELPERS ====================
def _safe_load_json(text: str) -> Optional[dict]:
    """Extract JSON from fenced block if present and gently repair common issues."""
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL) or re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = (m.group(1) if m else text).strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        blob2 = (blob
                 .replace("“", '"').replace("”", '"')
                 .replace("’", "'").replace("\u0000", ""))
        try:
            return json.loads(blob2)
        except Exception:
            logger.error("JSON parse failed. First 2k chars:\n%s", blob[:2000])
            return None

def _coerce_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def _norm_mcq(d: dict) -> dict:
    """Normalize MCQ schema fields/types."""
    d = dict(d)
    if "correct_option" in d and "correct_answer" not in d:
        d["correct_answer"] = d.pop("correct_option")
    d["options"] = [str(o) for o in _coerce_list(d.get("options"))][:4]
    diff = str(d.get("difficulty", "medium")).lower()
    d["difficulty"] = diff if diff in {"easy", "medium", "hard"} else "medium"
    return d

# ==================== RESPONSE PARSING ====================
def parse_mcq_response(response_text: str, force_traditional: bool = True) -> List[MCQ]:
    """Parse MCQ response from LLM and convert to Traditional if requested"""
    data = _safe_load_json(response_text)
    if not data:
        return []
    mcqs: List[MCQ] = []
    for mcq_data in data.get('mcqs', []):
        d = _norm_mcq(mcq_data)
        q = d.get('question', '')
        opts = d.get('options', [])
        exp = d.get('explanation', '')
        topic = d.get('topic', '')
        if force_traditional:
            q = to_traditional(q)
            opts = [to_traditional(o) for o in opts]
            exp = to_traditional(exp)
            topic = to_traditional(topic)
        mcqs.append(MCQ(
            question=q,
            options=opts,
            correct_answer=d.get('correct_answer', ''),
            explanation=exp,
            difficulty=d.get('difficulty', 'medium'),
            topic=topic
        ))
    return mcqs

def parse_lecture_notes_response(response_text: str, force_traditional: bool = True) -> Tuple[List[LectureNoteSection], str]:
    """Parse lecture notes response from LLM and convert to Traditional if requested"""
    data = _safe_load_json(response_text)
    if not data:
        return [], ''
    sections: List[LectureNoteSection] = []
    for section_data in data.get('sections', []):
        title = section_data.get('title', '')
        content = section_data.get('content', '')
        key_points = section_data.get('key_points', [])
        examples = section_data.get('examples', [])
        if force_traditional:
            title = to_traditional(title)
            content = to_traditional(content)
            key_points = [to_traditional(x) for x in key_points]
            examples = [to_traditional(x) for x in examples]
        sections.append(LectureNoteSection(
            title=title,
            content=content,
            key_points=key_points,
            examples=examples
        ))
    summary = data.get('summary', '')
    if force_traditional:
        summary = to_traditional(summary)
    return sections, summary

# ==================== ASR PREPROCESSING ====================
def preprocess_asr_text(raw_asr_text: str, min_chunk_duration: int = 60, max_gap: int = 10) -> str:
    """
    Preprocess raw ASR text by combining lines into meaningful chunks.

    Args:
        raw_asr_text: Raw ASR with timestamped lines
        min_chunk_duration: Minimum duration in seconds for each chunk
        max_gap: Start a new chunk if time gap between lines exceeds this value (seconds)

    Returns:
        Cleaned, combined transcript text
    """
    lines = raw_asr_text.strip().split('\n')
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_start_time: Optional[int] = None
    current_end_time: Optional[int] = None

    for line in lines:
        if not line.strip():
            continue

        # Parse timestamp and content
        if ':' in line and len(line.split(':', 1)) > 1:
            time_part, content = line.split(':', 1)
            time_part = time_part.strip()
            content = content.strip()
            if not content:
                continue

            # Convert timestamp to seconds
            try:
                if time_part.count(':') == 2:  # HH:MM:SS
                    h, m, s = map(int, time_part.split(':'))
                    timestamp_sec = h * 3600 + m * 60 + s
                elif time_part.count(':') == 1:  # MM:SS
                    m, s = map(int, time_part.split(':'))
                    timestamp_sec = m * 60 + s
                else:
                    continue
            except ValueError:
                continue

            # Start new chunk or continue
            if current_start_time is None:
                current_start_time = timestamp_sec
                current_end_time = timestamp_sec
                current_chunk = [content]
            elif timestamp_sec - current_end_time > max_gap:
                # Save existing chunk if long enough
                if current_chunk and current_end_time - current_start_time >= min_chunk_duration:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(f"[{sec_to_hms(current_start_time)}-{sec_to_hms(current_end_time)}] {chunk_text}")
                # Start new chunk
                current_start_time = timestamp_sec
                current_end_time = timestamp_sec
                current_chunk = [content]
            else:
                current_chunk.append(content)
                current_end_time = timestamp_sec
        else:
            # Lines without timestamps: attach to current chunk if any
            if current_chunk:
                current_chunk.append(line.strip())

    # Add the final chunk (keep if only chunk even if short)
    if current_chunk and current_end_time is not None and current_start_time is not None:
        dur = current_end_time - current_start_time
        if dur >= min_chunk_duration or not chunks:
            chunk_text = ' '.join(current_chunk)
            chunks.append(f"[{sec_to_hms(current_start_time)}-{sec_to_hms(current_end_time)}] {chunk_text}")

    # Fallback if no valid chunks
    if not chunks:
        cleaned_lines = []
        for line in lines:
            if ':' in line and len(line.split(':', 1)) > 1:
                _, content = line.split(':', 1)
                if content.strip():
                    cleaned_lines.append(content.strip())
        return ' '.join(cleaned_lines)

    return '\n\n'.join(chunks)

# ==================== MAIN FUNCTIONS ====================
def generate_educational_content(
    raw_asr_text: str,
    ocr_segments: List[Dict],
    video_id: str,
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> EducationalContentResult:
    """
    Main function to generate educational content from pre-processed segments.
    """
    report("initializing", progress_callback)

    if run_dir is None:
        run_dir = Path(f"/tmp/educational_content/{video_id}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting educational content generation for video {video_id}")

        config = EducationalContentConfig()
        if not validate_config(config):
            raise RuntimeError("Configuration validation failed")

        report("processing_inputs", progress_callback)

        # PREPROCESS ASR TEXT
        transcript = (raw_asr_text)
        ocr_context = build_ocr_context_from_segments(ocr_segments)
        logger.info(f"Preprocessed transcript: {len(transcript)} chars, OCR context: {len(ocr_context)} chars")

        # Save intermediate files for debugging
        with open(run_dir / "raw_asr_text.txt", "w", encoding="utf-8") as f:
            f.write(raw_asr_text)
        with open(run_dir / "preprocessed_transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)
        with open(run_dir / "ocr_segments.json", "w", encoding="utf-8") as f:
            json.dump(ocr_segments, f, ensure_ascii=False, indent=2)
        with open(run_dir / "ocr_context.txt", "w", encoding="utf-8") as f:
            f.write(ocr_context)

        report("initializing_client", progress_callback)

        service_type = config.service_type
        model = config.openai_model if service_type == "openai" else config.azure_model

        if service_type == "azure":
            client = initialize_client(
                service_type="azure",
                endpoint=config.azure_endpoint,
                key=config.azure_key,
                api_version=config.azure_api_version
            )
        else:
            client = initialize_client(
                service_type="openai",
                api_key=config.openai_api_key,
                base_url=config.openai_base_url
            )

        # Centralize context budgets per model
        MODEL_BUDGETS = {
            "gpt-4o": 128_000,
            "Meta-Llama-3.1-8B-Instruct": 128_000,
        }
        ctx_budget = MODEL_BUDGETS.get(model, 100_000)

        # ---------- MCQs ----------
        report("generating_mcqs", progress_callback)

        mcq_system_message = "你是一位資深教育專家，專門創建高質量、教育意義豐富的多選題。嚴格遵守輸出格式要求。"
        mcq_prompt_template_tokens = count_tokens_llama(build_mcq_prompt("", ocr_context, config.max_questions))
        mcq_budget = max(2_000, ctx_budget - mcq_prompt_template_tokens)
        mcq_transcript = truncate_text_by_tokens(transcript, mcq_budget)
        final_mcq_prompt = build_mcq_prompt(mcq_transcript, ocr_context, config.max_questions)

        logger.info(f"MCQ prompt approx tokens: {count_tokens_llama(final_mcq_prompt):,}")

        mcq_response = call_llm(
            service_type=service_type,
            client=client,
            system_message=mcq_system_message,
            user_message=final_mcq_prompt,
            model=model,
            max_tokens=4096,
            temperature=0.2,  # stabler JSON
            top_p=0.9
        )

        mcq_output = extract_text_from_response(mcq_response, service_type)
        mcqs = parse_mcq_response(mcq_output, force_traditional=config.force_traditional)

        # ---------- Lecture Notes ----------
        report("generating_notes", progress_callback)

        notes_system_message = "你是一位優秀的教學設計專家，專門創建結構清晰、教育價值高的講義筆記。嚴格遵守輸出格式要求。"
        notes_prompt_template_tokens = count_tokens_llama(build_lecture_notes_prompt("", ocr_context, config.max_notes_pages))
        notes_budget = max(2_000, ctx_budget - notes_prompt_template_tokens)
        notes_transcript = truncate_text_by_tokens(transcript, notes_budget)
        notes_prompt = build_lecture_notes_prompt(notes_transcript, ocr_context, config.max_notes_pages)

        notes_response = call_llm(
            service_type=service_type,
            client=client,
            system_message=notes_system_message,
            user_message=notes_prompt,
            model=model,
            max_tokens=4096,
            temperature=0.2,
            top_p=0.9
        )

        notes_output = extract_text_from_response(notes_response, service_type)
        lecture_sections, summary = parse_lecture_notes_response(notes_output, force_traditional=config.force_traditional)

        report("processing_results", progress_callback)

        result = EducationalContentResult(
            mcqs=mcqs,
            lecture_notes=lecture_sections,
            summary=summary
        )

        # ---------- Optional caching ----------
        if config.enable_cache:
            cache_dir = run_dir / "cache"
            cache_dir.mkdir(exist_ok=True)
            # cache keys reflect truncated transcripts actually sent
            mcq_key = get_content_hash(mcq_transcript, ocr_context, "mcq")
            notes_key = get_content_hash(notes_transcript, ocr_context, "notes")
            with open(cache_dir / f"{mcq_key}.json", "w", encoding="utf-8") as f:
                json.dump([vars(x) for x in mcqs], f, ensure_ascii=False, indent=2)
            with open(cache_dir / f"{notes_key}.json", "w", encoding="utf-8") as f:
                json.dump({"sections": [vars(s) for s in lecture_sections], "summary": summary}, f, ensure_ascii=False, indent=2)

        # persist raw LLM outputs & final results
        with open(run_dir / "mcq_response.txt", "w", encoding="utf-8") as f:
            f.write(mcq_output)
        with open(run_dir / "notes_response.txt", "w", encoding="utf-8") as f:
            f.write(notes_output)
        with open(run_dir / "final_result.json", "w", encoding="utf-8") as f:
            json.dump({
                "mcqs": [vars(mcq) for mcq in mcqs],
                "lecture_notes": [vars(section) for section in lecture_sections],
                "summary": summary
            }, f, ensure_ascii=False, indent=2)

        report("completed", progress_callback)
        logger.info(f"Successfully generated {len(mcqs)} MCQs and {len(lecture_sections)} lecture note sections")
        return result

    except Exception as e:
        logger.error(f"Educational content generation failed: {e}", exc_info=True)
        raise

# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    sample_transcript = """
    00:00:03: 今天我們來學習微積分的基本概念。首先，導數表示函數在某一點的瞬時變化率。
    00:01:05: 積分則是導數的逆運算，用來計算面積和累積量。
    """
    sample_ocr = [
        {"start": 0, "end": 10, "text": "導數定義: f'(x) = lim(h→0) [f(x+h)-f(x)]/h"},
        {"start": 60, "end": 70, "text": "積分符號: ∫ f(x) dx"}
    ]

    result = generate_educational_content(
        raw_asr_text=sample_transcript,
        ocr_segments=sample_ocr,
        video_id="calc_101"
    )

    print(f"Generated {len(result.mcqs)} MCQs and {len(result.lecture_notes)} lecture sections")
    print("Summary:", (result.summary or "")[:120], "…")
