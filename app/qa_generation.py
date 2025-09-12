# app/qa_generation.py
"""
Module for generating educational content (MCQs and Lecture Notes) from pre-processed ASR and OCR segments.
Designed for enhancing student learning and review of educational materials.

This version:
- ASR-first prompts for MCQs and Lecture Notes (OCR is auxiliary; conflict -> ASR wins)
- Bloom-structured MCQs; detailed, past-tense lecture notes; strict JSON outputs
- Simplified->Traditional conversion safety net (OpenCC if available, else fallback map)
- NEW: Post-processing helpers (shuffle options, regenerate explanations, enforce difficulty)
- NEW: Post-processing **controlled by function parameters**, not environment variables
"""

import hashlib
import json
import logging
import os
import re
import time
import random
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
    options: List[str]              # ["optA", "optB", "optC", "optD"]
    correct_answer: str             # "A" | "B" | "C" | "D"
    explanation: str
    difficulty: str                 # "easy" | "medium" | "hard"
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

def ocr_segments_to_raw_text(ocr_segments: List[Dict]) -> str:
    """Flatten OCR segments to raw lines (optionally timestamp-prefixed)."""
    lines: List[str] = []
    for seg in ocr_segments or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start")
        ts = f"[{sec_to_hms(int(start))}] " if isinstance(start, (int, float)) else ""
        lines.append(f"{ts}{text}")
    return "\n".join(lines)

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
    "价": "價", "优": "優", "儿": "兒", "动": "動", "华": "華", "发": "發",
    "复": "復", "国": "國", "广": "廣", "汉": "漢", "会": "會", "纪": "紀", "简": "簡",
    "经": "經", "历": "歷", "马": "馬", "门": "門", "面": "麵", "内": "內", "气": "氣",
    "权": "權", "确": "確", "实": "實", "术": "術", "云": "雲",
    "众": "眾", "为": "為", "从": "從", "冲": "衝", "读": "讀", "爱": "愛",
    "战": "戰", "钟": "鐘", "级": "級", "师": "師", "学": "學", "习": "習",
    "声": "聲", "观": "觀", "这": "這"
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

# ==================== PROMPT BUILDERS (V2, ASR-first) ====================
def build_mcq_prompt_v2(
    transcript: str,
    *,
    ocr_context: str = "",
    num_questions: int = 10,
    chapters: Optional[List[Dict]] = None,
    global_summary: str = "",
) -> str:
    """ASR-first MCQ prompt with Bloom structuring, global context, and practical constraints.
       Schema preserved: {"mcqs":[{question, options[A..D], correct_answer, explanation, difficulty, topic}]}.
    """
    base = num_questions // 3
    rem  = num_questions % 3
    recall_n      = base + (1 if rem >= 1 else 0)
    application_n = base + (1 if rem >= 2 else 0)
    analysis_n    = base

    chap_lines = []
    if chapters:
        for c in chapters[:18]:
            ts = c.get("ts") or c.get("timestamp") or ""
            title = c.get("title") or ""
            if ts or title:
                chap_lines.append(f"- {ts}：{title}")
    global_ctx = []
    if global_summary.strip():
        global_ctx.append(f"- 摘要：{global_summary.strip()}")
    if chap_lines:
        global_ctx.append("- 章節：\n" + "\n".join(chap_lines))
    global_ctx_block = "\n".join(global_ctx) if global_ctx else "（無）"

    ocr_block = ""
    if ocr_context.strip():
        ocr_block = f"## 螢幕文字（OCR，僅作輔助參考）\n{ocr_context}\n\n"

    prompt = f"""
你是一位資深的教育 AI，為學習者設計高品質的多選題（繁體中文）。請嚴格依照下列規則出題，並**僅**輸出 JSON。

### 資料來源優先序
1) **ASR 逐字稿（主要依據）**
2) **OCR 螢幕文字（輔助參考）**

### 全域脈絡（Global Context）
{global_ctx_block}

### 出題結構（Bloom；合計 {num_questions} 題）
- Recall：{recall_n} 題
- Application：{application_n} 題
- Analysis：{analysis_n} 題

### 指引
- 忽略行政/平台雜訊（連線、點名、會議 ID 等）。
- 平衡 What/How/Why，可加入角色情境。
- 每題 4 選項（A–D），具迷惑性；避免答案集中。
- 難度比例：30% easy / 40% medium / 30% hard。
- 每題需附解釋（正確原因 + 常見誤解）。

### 輸出格式（僅 JSON）
```json
{{
  "mcqs": [
    {{
      "question": "問題（繁體中文）",
      "options": ["選項A", "選項B", "選項C", "選項D"],
      "correct_answer": "A|B|C|D",
      "explanation": "為何正確＋常見誤解",
      "difficulty": "easy|medium|hard",
      "topic": "主題/概念"
    }}
  ]
}}
```

### 資料
## ASR 逐字稿（主要依據）
{transcript}

{ocr_block}
"""
    return prompt

def build_lecture_notes_prompt_v2(
    transcript: str,
    *,
    ocr_context: str = "",
    num_pages: int = 5,
    chapters: Optional[List[Dict]] = None,
    topics: Optional[List[Dict]] = None,
    global_summary: str = "",
) -> str:
    """ASR-first lecture notes prompt with strong structure and past-tense voice.
       Schema preserved: sections[{title, content, key_points[], examples[]}], summary
    """
    topics_snippet = ""
    if topics:
        lines = []
        for i, t in enumerate(topics, 1):
            tid   = t.get("id", str(i).zfill(2))
            title = t.get("title", "")
            summ  = t.get("summary", "")
            lines.append(f"{tid}. {title}：{summ}")
        topics_snippet = "\n".join(lines)

    chap_lines = []
    if chapters:
        for c in chapters[:18]:
            ts = c.get("ts") or c.get("timestamp") or ""
            title = c.get("title") or ""
            if ts or title:
                chap_lines.append(f"- {ts}：{title}")
    global_ctx = []
    if global_summary.strip():
        global_ctx.append(f"- 摘要：{global_summary.strip()}")
    if chap_lines:
        global_ctx.append("- 章節：\n" + "\n".join(chap_lines))
    if topics_snippet:
        global_ctx.append("- 主題大綱：\n" + topics_snippet)
    global_ctx_block = "\n".join(global_ctx) if global_ctx else "（無）"

    ocr_block = ""
    if ocr_context.strip():
        ocr_block = f"## 螢幕文字（OCR，僅作輔助參考）\n{ocr_context}\n\n"

    min_words = num_pages * 400
    max_words = (num_pages + 1) * 350

    prompt = f"""
你是一位專業的教學設計助手。請以**ASR 逐字稿**為主要依據撰寫講義（繁體中文，**過去式**）。OCR 僅輔助；衝突時以 ASR 為準。

### 全域脈絡（Global Context）
{global_ctx_block}

### 內容與語氣要求
- 覆蓋所有主要核心概念，提供可操作步驟與真實案例。
- 梳理教師的專業建議、常見錯誤、最佳做法。
- 忽略行政/平台雜訊；必要時一語帶過。
- 程式課程需提供可執行的程式碼區塊（```python / ```java / ```cpp / ```html 等）。
- 字數建議 **{min_words}–{max_words}**（軟限制）。

### 輸出格式（僅 JSON）
```json
{{
  "sections": [
    {{
      "title": "章節標題",
      "content": "Markdown、過去式、可含代碼區塊",
      "key_points": ["重點1", "重點2", "重點3"],
      "examples": ["案例1", "案例2"]
    }}
  ],
  "summary": "過去式總結與建議"
}}
```

### 資料
## ASR 逐字稿（主要依據）
{transcript}

{ocr_block}
"""
    return prompt

# ==================== SYSTEM MESSAGES (ASR-first) ====================
MCQ_SYSTEM_MESSAGE = (
    "你是一位專業的出題助手。以『ASR 逐字稿』為主要依據產生題目；"
    "『OCR 文字』僅作輔助參考，當兩者衝突時，一律以 ASR 為準。"
    "請輸出嚴格符合指定 JSON 架構的內容，且僅輸出 JSON。"
)

NOTES_SYSTEM_MESSAGE = (
    "你是一位專業的教學設計助手。以『ASR 逐字稿』為主要依據整理講義；"
    "『OCR 文字』僅作輔助參考，當兩者衝突時，一律以 ASR 為準。"
    "請嚴格輸出指定的 JSON 架構，且僅輸出 JSON。"
)

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
                 .replace("“", '\"').replace("”", '\"')
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

# ==================== ANSWER DISTRIBUTION & POST-PROCESSING ====================
def regenerate_explanation_with_llm(
    mcq: MCQ,
    *,
    service_type: str,
    client: Any,
    model: str,
    force_traditional: bool = True
) -> None:
    """Regenerate explanation text based on the (possibly shuffled) correct answer/option set."""
    labels = ["A", "B", "C", "D"]
    lines = [f"{labels[i]}. {opt}" for i, opt in enumerate(mcq.options[:4])]
    options_block = "\n".join(lines)

    sys = "你是一位出題老師，負責為測驗題提供解析（繁體中文）。請簡潔並指出常見誤解。"
    prompt = f"""
請根據以下題目與選項，提供中文解析說明為何正確答案是 {mcq.correct_answer}，並簡要說明其他選項為何不正確。
開頭請使用：「正確答案是 {mcq.correct_answer}，因為…」

題目：
{mcq.question}

選項：
{options_block}
""".strip()

    resp = call_llm(
        service_type=service_type,
        client=client,
        system_message=sys,
        user_message=prompt,
        model=model,
        max_tokens=400,
        temperature=0.2,
        top_p=0.9
    )
    text = extract_text_from_response(resp, service_type) or ""
    mcq.explanation = to_traditional(text.strip()) if force_traditional else text.strip()

def shuffle_mcq_options(
    mcqs: List[MCQ],
    *,
    seed: Optional[int] = None,
    regenerate_explanations: bool = False,
    regeneration_cb: Optional[Callable[[MCQ], None]] = None
) -> None:
    """Shuffle options for each MCQ, update correct_answer letter accordingly.
       If regenerate_explanations is True and the correct letter moved,
       call regeneration_cb(mcq) to rebuild the explanation.
    """
    rng = random.Random(seed) if seed is not None else random
    for mcq in mcqs:
        if not mcq.options:
            continue
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        old_idx = letter_to_idx.get(mcq.correct_answer, 0)
        correct_text = mcq.options[old_idx] if old_idx < len(mcq.options) else mcq.options[0]
        rng.shuffle(mcq.options)
        mcq.options = mcq.options[:4]
        new_idx = next((i for i, t in enumerate(mcq.options) if t == correct_text), 0)
        new_letter = "ABCD"[new_idx]
        moved = (new_letter != mcq.correct_answer)
        mcq.correct_answer = new_letter
        if regenerate_explanations and moved and regeneration_cb:
            try:
                regeneration_cb(mcq)
            except Exception as e:
                logger.warning(f"Explanation regeneration failed: {e}")
        else:
            mcq.explanation = re.sub(r"正確答案是\s+[A-D]", f"正確答案是 {mcq.correct_answer}", mcq.explanation or "")

def enforce_difficulty_distribution(
    mcqs: List[MCQ],
    target_ratio: Tuple[float, float, float] = (0.3, 0.4, 0.3)
) -> List[MCQ]:
    """Reassign difficulty labels to approximate target ratios using a simple proxy (explanation length)."""
    total = len(mcqs) or 1
    target_easy   = round(target_ratio[0] * total)
    target_medium = round(target_ratio[1] * total)
    target_hard   = total - target_easy - target_medium
    ranked = sorted(mcqs, key=lambda q: len(q.explanation or ""))
    for idx, q in enumerate(ranked):
        if idx < target_easy:
            q.difficulty = "easy"
        elif idx < target_easy + target_medium:
            q.difficulty = "medium"
        else:
            q.difficulty = "hard"
    return mcqs

def postprocess_mcqs(
    mcqs: List[MCQ],
    *,
    shuffle: bool,
    regenerate_explanations: bool,
    enforce_difficulty: bool,
    seed: Optional[int],
    service_type: Optional[str],
    client: Optional[Any],
    model: Optional[str],
    force_traditional: bool
) -> List[MCQ]:
    """Apply optional post-processing steps controlled by function parameters."""
    if enforce_difficulty:
        enforce_difficulty_distribution(mcqs)
    if shuffle:
        def regen_cb(m: MCQ):
            if regenerate_explanations and service_type and client and model:
                regenerate_explanation_with_llm(
                    m,
                    service_type=service_type,
                    client=client,
                    model=model,
                    force_traditional=force_traditional
                )
        shuffle_mcq_options(
            mcqs,
            seed=seed,
            regenerate_explanations=regenerate_explanations,
            regeneration_cb=regen_cb if (regenerate_explanations and service_type and client and model) else None
        )
    return mcqs

# ==================== OUTPUT ADAPTERS (Hook points for legacy formats) ====================
def result_to_simple_json(result: EducationalContentResult) -> dict:
    return {
        "mcqs": [
            {
                "question": m.question,
                "options": m.options,
                "correct_answer": m.correct_answer,
                "explanation": m.explanation,
                "difficulty": m.difficulty,
                "topic": m.topic,
            }
            for m in result.mcqs
        ],
        "lecture_notes": [
            {
                "title": s.title,
                "content": s.content,
                "key_points": s.key_points,
                "examples": s.examples,
            }
            for s in result.lecture_notes
        ],
        "summary": result.summary,
    }

def result_to_markdown(result: EducationalContentResult) -> str:
    lines = ["# 測驗題 (MCQs)", ""]
    for i, m in enumerate(result.mcqs, 1):
        lines.append(f"## Q{i}. {m.question}")
        for idx, opt in enumerate(m.options[:4]):
            lines.append(f"- {'ABCD'[idx]}. {opt}")
        lines.append(f"**正確答案**：{m.correct_answer}")
        if m.explanation:
            lines.append(f"**解析**：{m.explanation}")
        lines.append(f"**難度**：{m.difficulty}　**主題**：{m.topic}")
        lines.append("")
    lines.append("# 講義筆記")
    for s in result.lecture_notes:
        lines.append(f"## {s.title}")
        lines.append(s.content or "")
        if s.key_points:
            lines.append("**重點：**")
            for k in s.key_points:
                lines.append(f"- {k}")
        if s.examples:
            lines.append("**範例：**")
            for ex in s.examples:
                lines.append(f"- {ex}")
        lines.append("")
    lines.append("## 摘要")
    lines.append(result.summary or "")
    return "\n".join(lines)

def _mcqs_as_items(mcqs: List[MCQ]) -> List[dict]:
    items = []
    for i, m in enumerate(mcqs, 1):
        items.append({
            "QuestionId": f"Q{str(i).zfill(3)}",
            "QuestionText": m.question,
            "Options": [
                {"Label": "A", "Text": m.options[0] if len(m.options) > 0 else ""},
                {"Label": "B", "Text": m.options[1] if len(m.options) > 1 else ""},
                {"Label": "C", "Text": m.options[2] if len(m.options) > 2 else ""},
                {"Label": "D", "Text": m.options[3] if len(m.options) > 3 else ""},
            ],
            "CorrectAnswer": m.correct_answer,
            "Explanation": m.explanation,
            "Difficulty": {"easy": "簡單", "medium": "中等", "hard": "困難"}.get(m.difficulty, m.difficulty),
            "Topic": m.topic,
        })
    return items

def result_to_pipeline_like(
    result: EducationalContentResult,
    *,
    num_questions: int,
    num_pages: int,
    meta: Optional[dict] = None
) -> dict:
    meta = meta or {}
    return {
        "success": True,
        "qa_and_notes": {
            "questions": _mcqs_as_items(result.mcqs),
            "lecture_notes": {
                "sections": [
                    {
                        "title": s.title,
                        "content": s.content,
                        "key_points": s.key_points,
                        "examples": s.examples
                    } for s in result.lecture_notes
                ],
                "summary": result.summary
            }
        },
        "summary": {
            "questions_generated": len(result.mcqs),
            "lecture_notes_pages": num_pages,
        },
        "pipeline_info": {**meta}
    }


# ==================== LEGACY CLIENT ADAPTER ====================
def result_to_legacy_client_format(
    result: EducationalContentResult,
    *,
    id: str,
    team_id: str,
    section_no: int,
    created_at: str,
    chapters: Optional[List[Dict]] = None
) -> dict:
    """
    ADAPTER: Converts the new, rich EducationalContentResult into the
    legacy client format. This is a temporary bridge until the client
    can be updated to accept the new payload format.
    """
    # 1. Convert MCQs to the simple legacy question format
    legacy_questions = []
    for i, mcq in enumerate(result.mcqs, start=1):
        legacy_questions.append({
            "QuestionId": f"Q{str(i).zfill(3)}",
            "QuestionText": mcq.question,
            "Options": mcq.options,
            "CorrectAnswer": mcq.correct_answer,
            "Explanation": mcq.explanation,
            "Difficulty": mcq.difficulty,
            "Topic": mcq.topic
        })

    # 2. Convert Lecture Notes into a single Markdown string
    markdown_lines = []
    for section in result.lecture_notes:
        markdown_lines.append(f"## {section.title}")
        markdown_lines.append(section.content)
        if section.key_points:
            markdown_lines.append("### 重點")
            for point in section.key_points:
                markdown_lines.append(f"- {point}")
        if section.examples:
            markdown_lines.append("### 範例")
            for example in section.examples:
                markdown_lines.append(f"- {example}")
        markdown_lines.append("")

    markdown_lines.append("## 總結")
    markdown_lines.append(result.summary)
    combined_markdown = "\n".join(markdown_lines)

    # 3. Build the final payload in the legacy format
    legacy_payload = {
        "Id": id,
        "TeamId": team_id,
        "SectionNo": section_no,
        "CreatedAt": created_at,
        "Questions": legacy_questions,
        "CourseNote": combined_markdown.strip(),
        "chapters": chapters or []
    }

    logger.info(f"📦 Packaged {len(legacy_questions)} questions into legacy client JSON payload.")
    return legacy_payload

# ==================== ASR PREPROCESSING ====================
def preprocess_asr_text(raw_asr_text: str, min_chunk_duration: int = 60, max_gap: int = 10) -> str:
    """Preprocess raw ASR text by combining lines into meaningful chunks."""
    lines = raw_asr_text.strip().split('\n')
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_start_time: Optional[int] = None
    current_end_time: Optional[int] = None
    for line in lines:
        if not line.strip():
            continue
        if ':' in line and len(line.split(':', 1)) > 1:
            time_part, content = line.split(':', 1)
            time_part = time_part.strip()
            content = content.strip()
            if not content:
                continue
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
            if current_start_time is None:
                current_start_time = timestamp_sec
                current_end_time = timestamp_sec
                current_chunk = [content]
            elif timestamp_sec - current_end_time > max_gap:
                if current_chunk and current_end_time - current_start_time >= min_chunk_duration:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(f"[{sec_to_hms(current_start_time)}-{sec_to_hms(current_end_time)}] {chunk_text}")
                current_start_time = timestamp_sec
                current_end_time = timestamp_sec
                current_chunk = [content]
            else:
                current_chunk.append(content)
                current_end_time = timestamp_sec
        else:
            if current_chunk:
                current_chunk.append(line.strip())
    if current_chunk and current_end_time is not None and current_start_time is not None:
        dur = current_end_time - current_start_time
        if dur >= min_chunk_duration or not chunks:
            chunk_text = ' '.join(current_chunk)
            chunks.append(f"[{sec_to_hms(current_start_time)}-{sec_to_hms(current_end_time)}] {chunk_text}")
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
def initialize_and_get_client(config: EducationalContentConfig):
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
    return service_type, model, client

from typing import Union

def generate_educational_content(
    raw_asr_text: str,
    ocr_segments: Union[List[Dict], str],   # can be a string or list (backward compatible)
    video_id: str,
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    *,
    shuffle_options: bool = False,
    regenerate_explanations: bool = False,
    enforce_difficulty: bool = True,
    shuffle_seed: Optional[int] = None,
    ocr_text_override: Optional[str] = None,  # NEW: pass your OCR text directly
) -> EducationalContentResult:
    """
    Main function to generate educational content from pre-processed segments.
    Post-processing behavior is controlled by function parameters (no env vars).
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

        transcript = (raw_asr_text)
        if ocr_text_override is not None:
        # Use caller-provided OCR text AS-IS (no formatting, no truncation)
            ocr_context = ocr_text_override
        elif isinstance(ocr_segments, str):
            # If caller passed OCR as a string, also use it AS-IS
            ocr_context = ocr_segments
        else:
        # Legacy fallback: if we still receive segments, keep them raw (no bullets/timestamps)
        # Join only the 'text' fields in order, no extra formatting.
            ocr_context = "\n".join(
                (seg.get("text") or "").strip()
                for seg in (ocr_segments or [])
                if (seg.get("text") or "").strip()
            )
        logger.info(f"ASR-first policy active. Generating {config.max_questions} MCQs and {config.max_notes_pages}p notes.")
        logger.info(f"Preprocessed transcript chars: {len(transcript)}, OCR context chars: {len(ocr_context)}")

        with open(run_dir / "raw_asr_text.txt", "w", encoding="utf-8") as f:
            f.write(raw_asr_text)
        with open(run_dir / "preprocessed_transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)
        with open(run_dir / "ocr_segments.json", "w", encoding="utf-8") as f:
            json.dump(ocr_segments, f, ensure_ascii=False, indent=2)
        with open(run_dir / "ocr_context.txt", "w", encoding="utf-8") as f:
            f.write(ocr_context)

        report("initializing_client", progress_callback)
        service_type, model, client = initialize_and_get_client(config)

        # Centralize context budgets per model
        MODEL_BUDGETS = {
            "gpt-4o": 128_000,
            "Meta-Llama-3.1-8B-Instruct": 128_000,
        }
        ctx_budget = MODEL_BUDGETS.get(model, 100_000)

        # ---------- MCQs ----------
        report("generating_mcqs", progress_callback)

        mcq_prompt_template_tokens = count_tokens_llama(build_mcq_prompt_v2(
            transcript="",
            ocr_context=ocr_context,
            num_questions=config.max_questions,
            chapters=None,
            global_summary="",
        ))
        mcq_budget = max(2_000, ctx_budget - mcq_prompt_template_tokens)
        mcq_transcript = truncate_text_by_tokens(transcript, mcq_budget)
        final_mcq_prompt = build_mcq_prompt_v2(
            transcript=mcq_transcript,
            ocr_context=ocr_context,
            num_questions=config.max_questions,
            chapters=None,
            global_summary="",
        )

        logger.info(f"MCQ prompt approx tokens: {count_tokens_llama(final_mcq_prompt):,}")
        logger.info(f"📚 Generating {config.max_questions} MCQs with ASR-first policy")

        mcq_response = call_llm(
            service_type=service_type,
            client=client,
            system_message=MCQ_SYSTEM_MESSAGE,
            user_message=final_mcq_prompt,
            model=model,
            max_tokens=4096,
            temperature=0.2,
            top_p=0.9
        )
        mcq_output = extract_text_from_response(mcq_response, service_type)
        mcqs = parse_mcq_response(mcq_output, force_traditional=config.force_traditional)

        # 🔽 Post-processing (using function parameters)
        mcqs = postprocess_mcqs(
            mcqs,
            shuffle=shuffle_options,
            regenerate_explanations=regenerate_explanations,
            enforce_difficulty=enforce_difficulty,
            seed=shuffle_seed,
            service_type=service_type,
            client=client,
            model=model,
            force_traditional=config.force_traditional
        )

        # ---------- Lecture Notes ----------
        report("generating_notes", progress_callback)

        notes_prompt_template_tokens = count_tokens_llama(build_lecture_notes_prompt_v2(
            transcript="",
            ocr_context=ocr_context,
            num_pages=config.max_notes_pages,
            chapters=None,
            topics=None,
            global_summary="",
        ))
        notes_budget = max(2_000, ctx_budget - notes_prompt_template_tokens)
        notes_transcript = truncate_text_by_tokens(transcript, notes_budget)
        notes_prompt = build_lecture_notes_prompt_v2(
            transcript=notes_transcript,
            ocr_context=ocr_context,
            num_pages=config.max_notes_pages,
            chapters=None,
            topics=None,
            global_summary="",
        )

        logger.info(f"📘 Generating {config.max_notes_pages} pages of lecture notes with ASR-first policy")

        notes_response = call_llm(
            service_type=service_type,
            client=client,
            system_message=NOTES_SYSTEM_MESSAGE,
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

# ---- Adapter for tasks.py compatibility ----

# ---- Adapter for tasks.py compatibility ----

def process_text_for_qa_and_notes(
    *,
    # Prefer raw_asr_text (matches new tasks.py). Fallback to audio_segments if not provided.
    raw_asr_text: str = "",
    audio_segments: Optional[List[Dict]] = None,
    ocr_segments: Optional[List[Dict]] = None,
    num_questions: int = 10,
    num_pages: int = 3,
    id: str = "",
    team_id: str = "",
    section_no: int = 0,
    created_at: str = "",
) -> EducationalContentResult:
    """
    Adapter so tasks.py (and other callers) can invoke the generator.
    - If raw_asr_text is provided, we use it directly (ASR-first).
    - Else, we reconstruct a raw ASR string from audio_segments ("HH:MM:SS: text" per line).
    Returns the raw EducationalContentResult object (not the pipeline format).
    """
    ocr_segments = ocr_segments or []

    # 1) Choose ASR source
    if raw_asr_text and raw_asr_text.strip():
        asr_text_for_prompt = raw_asr_text
    else:
        audio_segments = audio_segments or []
        def _line(seg: Dict) -> str:
            ts = sec_to_hms(int(seg.get("start", 0)))
            txt = (seg.get("text") or "").strip()
            return f"{ts}: {txt}" if txt else ""
        asr_text_for_prompt = "\n".join(filter(None, (_line(s) for s in audio_segments)))

    # 2) Honor per-call counts by temporarily overriding env
    old_max_q = os.environ.get("MAX_QUESTIONS")
    old_max_p = os.environ.get("MAX_NOTES_PAGES")
    os.environ["MAX_QUESTIONS"] = str(num_questions)
    os.environ["MAX_NOTES_PAGES"] = str(num_pages)

    try:
        # Just return the raw result object, not the pipeline format
        result = generate_educational_content(
            raw_asr_text=asr_text_for_prompt,   # ← ASR-first (raw string)
            ocr_segments=ocr_segments,          # ← simple OCR (list or string)
            video_id=id or "video",
            run_dir=None,
            progress_callback=None,
            shuffle_options=False,
            regenerate_explanations=False,
            enforce_difficulty=True,
            shuffle_seed=None,
            ocr_text_override=None,
        )
        
        return result  # ← Return the EducationalContentResult object directly

    finally:
        if old_max_q is not None:
            os.environ["MAX_QUESTIONS"] = old_max_q
        else:
            os.environ.pop("MAX_QUESTIONS", None)
        if old_max_p is not None:
            os.environ["MAX_NOTES_PAGES"] = old_max_p
        else:
            os.environ.pop("MAX_NOTES_PAGES", None)

# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_transcript = """
    00:00:03: 今天我們來學習微積分的基本概念。首先，導數表示函數在某一點的瞬時變化率。
    00:01:05: 積分則是導數的逆運算，用來計算面積和累積量。
    """
    sample_ocr = [
        {"start": 0, "end": 10, "text": "導數定義: f'(x) = lim(h→0) [f(x+h)-f(x)]/h"},
        {"start": 60, "end": 70, "text": "積分符號: ∫ f(x) dx"}
    ]

    # Example: turn on shuffling + difficulty enforcement for this run
    result = generate_educational_content(
        raw_asr_text=sample_transcript,
        ocr_segments=sample_ocr,
        video_id="calc_101",
        shuffle_options=True,
        regenerate_explanations=False,
        enforce_difficulty=True,
        shuffle_seed=42
    )

    print(f"Generated {len(result.mcqs)} MCQs and {len(result.lecture_notes)} lecture sections")
    print("Summary:", (result.summary or "")[:120], "…")
