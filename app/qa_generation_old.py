#app/qa_generation.py

# qa_generation.py - Updated with Optimized Video Processing Integration
import json
import logging
import os
from dotenv import load_dotenv
import re
import time
from collections import Counter
import random
from app.file_saving import qa_text_to_json
from typing import List, Dict, Optional, Callable, Tuple, Any
import hashlib
from app.Providers.llm_provider import LLMProvider
from app.Providers.chapter_llama_provider import ChapterLlamaProvider
load_dotenv()  # <-- move up here, before reading envs

# --- place near top, after imports ---
try:
    from app.video_processing import sequential_video_processing, apply_ocr_filters, ASR_WINDOW_SEC
except Exception:
    from app.video_processing import sequential_video_processing, apply_ocr_filters
    ASR_WINDOW_SEC = int(os.getenv("ASR_WINDOW_SEC", "600"))

# caches hoisted to avoid name-before-definition edge cases
# NOTE: keep summarize() and compress() caches separate (string vs JSON)
_EMB_CACHE: Dict[str, List[float]] = {}
_TOPICS_CACHE: Dict[str, List[str]] = {}
_BIN_TS_CACHE: Dict[str, Dict[str, str]] = {}
_BIN_EMB_CACHE: Dict[str, List[float]] = {}

# --- Multi-bin/context & TOC knobs ---
BIN_TITLE_USE_CONTEXT   = os.getenv("BIN_TITLE_USE_CONTEXT", "1") == "1"
BIN_TITLE_MODULE_HINT   = os.getenv("BIN_TITLE_MODULE_HINT", "1") == "1"
TOC_CLUSTER_THR         = float(os.getenv("TOC_CLUSTER_THR", "0.78"))

# --- Bin title config (topic style) ---
BIN_TITLE_PREFIX = os.getenv("BIN_TITLE_PREFIX", "")   # e.g., "AutoCAD:", "Excel:", "", etc.
BIN_TITLE_MAX_CHARS = int(os.getenv("BIN_TITLE_MAX_CHARS", "45"))

# --- TOC-driven segmentation & topics knobs ---
USE_TOC_FOR_TOPICS     = os.getenv("USE_TOC_FOR_TOPICS", "1") == "1"  # prefer course_toc for topics
TOC_MERGE_THR          = float(os.getenv("TOC_MERGE_THR", "0.83"))     # module cosine merge threshold
TOC_ADJ_MERGE_GAP_SEC  = int(os.getenv("TOC_ADJ_MERGE_GAP_SEC", "10")) # join modules split by tiny timing gaps

# Comma-separated lists are supported via env; safe defaults below.
BIN_TITLE_BANLIST = set(
    (os.getenv("BIN_TITLE_BANLIST", "介紹,小結,示範,操作,畫面,這一段,Part,Chapter,概要,講解")
     .split(","))
)
# optional taxonomy: constrain to known topics if you have them per course
BIN_ALLOWED_TOPICS = [s.strip() for s in os.getenv("BIN_ALLOWED_TOPICS", "").split(",") if s.strip()]

# --- Chapter-Llama env knobs ---
CHAPTER_ENGINE = os.getenv("CHAPTER_ENGINE", "native").lower()   # native | chapter-llama
CHAPTER_LLAMA_URL = os.getenv("CHAPTER_LLAMA_URL", "http://localhost:8000/")
CHAPTER_LLAMA_WIN_SEC = int(os.getenv("CHAPTER_LLAMA_WIN_SEC", "240"))
CHAPTER_LLAMA_OVERLAP_SEC = int(os.getenv("CHAPTER_LLAMA_OVERLAP_SEC", "45"))
CHAPTER_LLAMA_TIMEOUT = int(os.getenv("CHAPTER_LLAMA_TIMEOUT", "60"))

CL = None
if CHAPTER_ENGINE == "chapter-llama":
    CL = ChapterLlamaProvider(CHAPTER_LLAMA_URL, timeout=CHAPTER_LLAMA_TIMEOUT)

# --- zh-Hant enforcement (post-processing) ---
import re

try:
    from opencc import OpenCC
    _OPENCC = OpenCC(os.getenv("OPENCC_PROFILE", "s2twp"))  # s2twp = Taiwan-friendly profile
except Exception:
    _OPENCC = None

# precompiled regex to quickly check for Han characters
_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

def _to_traditional(text: str) -> str:
    """Convert Simplified → Traditional if Han chars present, else return unchanged."""
    if not text or not isinstance(text, str):
        return text
    if _OPENCC is None or not _CHINESE_CHAR_RE.search(text):
        return text
    try:
        return _OPENCC.convert(text)
    except Exception:
        return text

def _to_traditional_json(x):
    """Recursively apply _to_traditional on all string values inside lists/dicts."""
    if isinstance(x, str):
        return _to_traditional(x)
    if isinstance(x, list):
        return [_to_traditional_json(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_traditional_json(v) for k, v in x.items()}
    return x

def _normalize_topic_title(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip())
    t = re.sub(r"[，。；、\s]+$", "", t)
    return t

def _embed_toc_item(module: Dict) -> Optional[List[float]]:
    txt = f"{module.get('module_title','')}. {module.get('module_summary','')}"
    key = _short_hash("TOC|" + txt)
    if key in _EMB_CACHE:
        return _EMB_CACHE[key]
    v = _safe_embed(txt)
    if v:
        _EMB_CACHE[key] = v
    return v

def _title_is_generic(t: str) -> bool:
    if not t: return True
    if any(bad in t for bad in BIN_TITLE_BANLIST): return True
    # ultra-short or purely decorative
    return len(t) < 2

def _maybe_prefix(title: str) -> str:
    if not BIN_TITLE_PREFIX:
        return title
    return title if title.startswith(BIN_TITLE_PREFIX) else f"{BIN_TITLE_PREFIX} {title}"

def _truncate_bytesafe(s: str, limit: int) -> str:
    # simple char-limit; swap if you need byte-aware truncation
    return s if len(s) <= limit else (s[:limit] + "…")

def build_course_toc(fusion_plan: List[Dict]) -> List[Dict]:
    """
    Placeholder: build a TOC-style list of modules from the fusion plan.
    Return empty list for now.
    """
    return []

def call_llm(prompt, role, max_tokens=12000, images=None):
    lang_rule = ("【語言要求】所有輸出一律使用繁體中文，不得使用簡體字；"
                 "程式碼、API 名稱與專有名詞可保留原文。")
    system_role = f"{(role or '').strip()}\n\n{lang_rule}"
    out = provider.chat(system=system_role, user=prompt, max_tokens=max_tokens, images=images)
    if isinstance(out, dict):
        out = out.get("content") or out.get("text") or json.dumps(out, ensure_ascii=False)
    # ⛑️ enforce zh-Hant
    out = _to_traditional(out or "")
    return out

# ==================== Setup Logging ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===== Fusion & context knobs =====
WINDOW_SEC = ASR_WINDOW_SEC        # 10-min bins
CTX_TOKEN_BUDGET = int(os.getenv("CTX_TOKEN_BUDGET", 2400))   # trimmed budget; raise if safe
W_FLOOR = float(os.getenv("W_FLOOR", 0.25))
W_CEIL = float(os.getenv("W_CEIL", 0.75))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

SEM_THRESHOLD       = float(os.getenv("SEM_THRESHOLD", 0.28))   # topic-shift sensitivity
COSINE_SPLIT_THRESHOLD = float(os.getenv("COSINE_SPLIT_THRESHOLD", 0.80))
GLOBAL_CTX_MAXTOK   = int(os.getenv("GLOBAL_CTX_MAXTOK", 2000)) # for global summary
GLOBAL_RELEVANCE_CUTOFF = float(os.getenv("GLOBAL_RELEVANCE_CUTOFF", "0.18"))
OCR_MERGE_THR = float(os.getenv("OCR_MERGE_THR", "0.65"))  # was 0.7; 0.60–0.70 works well for CJK
HEADER_GATE_THR = float(os.getenv("HEADER_GATE_THR", "0.65"))
HYBRID_SPIKE = float(os.getenv("HYBRID_SPIKE", "0.35"))

# NEW — relevance behavior toggles
GLOBAL_RELEVANCE_MODE = os.getenv("GLOBAL_RELEVANCE_MODE", "score").lower()  # score | filter
DROP_ADMIN_BLOCKS = os.getenv("DROP_ADMIN_BLOCKS", "0") == "1"               # 0 = tag only
RELEVANCE_KEEP_TOP = int(os.getenv("RELEVANCE_KEEP_TOP", "0"))               # safety keep N if filtering
RELEVANCE_SOFTFLOOR = float(os.getenv("RELEVANCE_SOFTFLOOR", "0.12"))        # gentler floor when filtering

logger.info(
    f"🔧 Knobs | WINDOW_SEC={WINDOW_SEC} CTX_TOKEN_BUDGET={CTX_TOKEN_BUDGET} "
    f"W_FLOOR={W_FLOOR} W_CEIL={W_CEIL} EMBED_MODEL={EMBED_MODEL} "
    f"SEM_THRESHOLD={SEM_THRESHOLD} GLOBAL_CTX_MAXTOK={GLOBAL_CTX_MAXTOK} "
    f"COSINE_SPLIT_THRESHOLD={COSINE_SPLIT_THRESHOLD} "
    f"GLOBAL_RELEVANCE_CUTOFF={GLOBAL_RELEVANCE_CUTOFF}"
)

def _coerce_obj_to_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, list) and x and isinstance(x[0], dict):
        return x[0]
    return {}

# ===== Canonical topics & alignment helpers =====
import unicodedata
from typing import Optional

def _extract_first_json_block(text: str) -> str:
    """
    Remove everything before the first '[' or '{' and after the last ']' or '}'.
    """
    if not text:
        return text
    # 1) drop anything before the first '[' or '{'
    start = min((text.find('['), text.find('{')), default=0)
    text = text[start:] if start >= 0 else text
    # 2) find balanced brackets
    stack, end = [], 0
    open_map = {']': '[', '}': '{'}
    for i, ch in enumerate(text):
        if ch in '[{':
            stack.append(ch)
        elif ch in ']}' and stack and stack[-1] == open_map[ch]:
            stack.pop()
            if not stack:
                end = i + 1
                break
    return text[:end] if end else text

def _normalize_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", (s or "").strip())
    s = re.sub(r"\s+", " ", s)
    return s

def canonicalize_topics(topics: List[Dict], max_topics: Optional[int] = None) -> List[Dict]:
    """
    De-duplicate, cap, and stamp stable IDs T01, T02... across the app.
    Input topics are [{title, summary}].
    """
    max_topics = max_topics or int(os.getenv("CANONICAL_TOPIC_MAX", "12"))
    seen = set()
    cleaned = []
    for t in (topics or []):
        title = _normalize_title(t.get("title") or "")
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"title": title, "summary": (t.get("summary") or "").strip()})
    cleaned = cleaned[:max_topics]
    for i, t in enumerate(cleaned, 1):
        t["id"] = f"T{str(i).zfill(2)}"
    return cleaned

def embed_text_for_align(s: str) -> Optional[List[float]]:
    try:
        return _safe_embed((s or "").strip())
    except Exception:
        return None

def _cos(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b: return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(y*y for y in b) ** 0.5
    return (dot / (na * nb)) if (na and nb) else 0.0

# --- Multi-view embedding helpers (summary + keyphrases) ---
def _l2(v):
    if not v: return None
    n = sum(x*x for x in v) ** 0.5
    return [x/n for x in v] if n else None

def convert_to_chapter_llama_payload(ocr_segments, audio_segments):
    """
    Converts legacy OCR/ASR lists into the strict schema Chapter-Llama needs.
    ocr_segments : list of dicts with keys: start, end, text
    audio_segments: list of dicts with keys: start, end, text
    Returns dict ready for POST /v1/chapter/boundaries
    """
    def _safe_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    audio_out = [
        {
            "start": _safe_float(seg.get("start")),
            "end":   _safe_float(seg.get("end")),
            "text":  str(seg.get("text", ""))
        }
        for seg in audio_segments
    ]

    ocr_out = [
        {
            "timestamp": _safe_float(seg.get("start", seg.get("timestamp", 0))),
            "text":      str(seg.get("text", ""))
        }
        for seg in ocr_segments
    ]

    return {
        "audio_segments": audio_out,
        "ocr_segments":   ocr_out,
        "win_sec":        240,      # or read from env / kwargs if you prefer
        "overlap_sec":    45,
        "language":       "zh-hant"
    }

def _mean_pool(vs):
    ok = [v for v in vs if v]
    if not ok: return None
    dim = len(ok[0])
    acc = [0.0]*dim
    for v in ok:
        if len(v) != dim:
            return None  # dimension mismatch → treat as failure
        for i, x in enumerate(v):
            acc[i] += x
    return [x/len(ok) for x in acc]

def compress_for_embedding(fused_prompt: str, video_context: dict) -> Dict[str, str]:
    return canonical_gist(video_context)

def _embed_views(summary: str, keyphrases: str) -> Optional[List[float]]:
    v1 = _l2(_safe_embed(summary))
    v2 = _l2(_safe_embed(keyphrases))
    return _mean_pool([v1, v2])

# ==================== Group_audio_by_time ====================

def group_audio_by_time(audio_segments, window_sec=WINDOW_SEC):
    if not audio_segments: return []
    grouped, current, block_start = [], [], audio_segments[0]['start']
    for seg in audio_segments:
        if seg['start'] - block_start >= window_sec and current:
            grouped.append(current); current = []; block_start = seg['start']
        current.append({ 'text': seg['text'], 'timestamp': seg['start'],
                         'start': seg['start'], 'end': seg['end'] })
    if current: grouped.append(current)
    return grouped

# ==================== Extracted Topics to Sentences ====================

def topics_to_sentence(topics: List[str]) -> str:
    """Turn ['旋轉','複製','標註'] -> '本段涵蓋了旋轉、複製，以及標註。'（繁體中文）"""
    seen, cleaned = set(), []
    for t in topics or []:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t); cleaned.append(t)
    n = len(cleaned)
    if n == 0: return ""
    if n == 1: return f"本段涵蓋了{cleaned[0]}。"
    if n == 2: return f"本段涵蓋了{cleaned[0]}與{cleaned[1]}。"
    body = "、".join(cleaned[:-1])
    return f"本段涵蓋了{body}，以及{cleaned[-1]}。"

# ==================== LLM Provider Configuration ====================
# Load .env file

from app.Providers.openai_provider import OpenAIProvider
from app.Providers.gemini_provider import GeminiProvider

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

if PROVIDER == "gemini":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise EnvironmentError("❌ GEMINI_API_KEY missing.")
    provider: LLMProvider = GeminiProvider(
        api_key=GEMINI_API_KEY,
        chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-pro"),
        embed_model=os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004"),
    )
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise EnvironmentError("❌ OPENAI_API_KEY missing.")
    provider: LLMProvider = OpenAIProvider(
        api_key=OPENAI_API_KEY,
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-large"),
    )

logger.info(
    f"🤖 Using provider={PROVIDER} | chat_model={getattr(provider, 'chat_model', 'n/a')} "
    f"| embed_model={getattr(provider, 'embed_model', 'n/a')}"
)


def _safe_embed(text: str):
    try:
        return provider.embed(text)
    except Exception as e:
        logger.warning(f"⚠️ Embedding failed: {e}")
        return None
    
# ==================== QandA JSON Parsing Helper ====================

def parse_qa_json(response_text):
    try:
        clean_text = (response_text or "").strip()
        # Remove ```json and ``` markers
        clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean_text, flags=re.MULTILINE)
        # Try to extract only the first JSON block
        match = re.search(r'(\{.*\}|\[.*\])', clean_text, re.DOTALL)
        if not match:
            logging.warning("⚠️ No JSON block found in response.")
            return None
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON parsing failed: {e}")
        logging.debug(f"📄 Raw response text:\n{response_text[:1000]}...")
        return None

# ---------- 1.  SINGLE GLOBAL GIST ----------
def canonical_gist(video_context: dict) -> dict:
    """
    Always return the same (global-summary, global-topics) pair
    so that every bin/chapter re-uses it instead of re-summarising.
    """
    summary = video_context.get("summary", "")
    topics  = video_context.get("main_topics", [])
    return {"summary": summary, "keyphrases": ", ".join(topics)}

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def _tfidf_vectors(texts: List[str]) -> List[List[float]]:
    """Return TF-IDF vectors for a list of texts; fallback if empty."""
    if not texts:
        return []
    vec = TfidfVectorizer(analyzer="word",
                          token_pattern=r"\b[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_]{3,}\b")
    return vec.fit_transform(texts).toarray().tolist()

# ==================== Generate Video-Level Summary ====================
def generate_video_summary_and_topics(combined_text: str):
    """
    Generates a high-level summary and definitive topic list for the entire video.
    Explicitly ignores admin/platform chatter (Zoom, meeting IDs, etc.).
    """
    logging.info("🎬 Generating a video-level summary and main topics...")

    prompt = f"""
你是一位資深的課程設計師。請先忽略所有行政性或平台設定相關的內容（例如：Zoom、會議 ID、麥克風/鏡頭、點名、請假、錄影/連線問題等），
只專注在「教學內容」本身，根據以下完整的 OCR + ASR 文本，提供：
1) 一段精煉且正確的**課程總結**（請勿包含行政事項），
2) 一份**主要學習主題**（純教學主題）的陣列。

課程原始內容（已過濾部分行政字詞）：
{combined_text}

請以 JSON 格式輸出：
{{
  "summary": "整體課程總結（只講教學內容，忽略行政）",
  "main_topics": ["主題1", "主題2", "..."]
}}
"""
    role = "你是一位課程內容分析專家，負責從大量文字中提取純教學的核心資訊。"

    # Let Gemini output more tokens by default; keep OpenAI moderate unless overridden by env
    max_tok = GLOBAL_CTX_MAXTOK
    if PROVIDER == "gemini":
        max_tok = int(os.getenv("GLOBAL_CTX_MAXTOK", "10000"))

    response = call_llm(prompt, role, max_tokens=max_tok)
    response = _extract_first_json_block(response)
    try:
        return parse_qa_json(response)
    except Exception as e:
        logging.error(f"❌ Failed to parse video summary JSON: {e}")
        return None
    
# Extract Global Topic #
def extract_global_topics_from_video(ocr_segments, audio_segments, max_topics=12):
    """
    Extract topics from the entire video content (ONCE per video)
    """
    # Build full corpus from all segments
    full_text = build_global_context_corpus(ocr_segments, audio_segments, max_chars=50000)
    
    prompt = f"""
    Analyze this entire course video and extract {max_topics} main learning topics.
    Focus on technical concepts, skills, and knowledge areas. Ignore administrative content.
    
    Video content:
    {full_text[:8000]}
    
    Return as JSON: {{"topics": ["topic1", "topic2", ...]}}
    """
    
    response = call_llm(prompt, "Educational topic extraction expert", max_tokens=1500)
    try:
        result = parse_qa_json(response)
        return result.get("topics", [])[:max_topics] if result else []
    except Exception:
        # Fallback: extract from OCR segments only
        ocr_text = clean_ocr_text("\n".join(seg.get("text", "") for seg in ocr_segments))
        return extract_main_topics(ocr_text, mode='combined', top_k=max_topics)

def assign_topics_to_chapters_lightweight(chapters, global_topics):
    """
    Enhanced lightweight topic assignment for better accuracy on long videos
    """
    for chapter in chapters:
        content = chapter.get("content", "").lower()
        
        # Enhanced scoring: consider term frequency and context
        best_topic = None
        best_score = 0
        
        for topic in global_topics:
            topic_lower = topic.lower()
            
            # Multi-factor scoring:
            # 1. Exact matches score higher
            exact_matches = content.count(topic_lower)
            
            # 2. Partial matches (for multi-word topics)
            topic_words = topic_lower.split()
            if len(topic_words) > 1:
                partial_score = sum(content.count(word) for word in topic_words) / len(topic_words)
            else:
                partial_score = exact_matches
                
            # 3. Length bonus (longer topics are more specific)
            length_bonus = len(topic_lower) * 0.1
            
            # 4. Position bonus (topics mentioned early are more important)
            first_mention = content.find(topic_lower)
            position_bonus = 10.0 / (first_mention + 1) if first_mention >= 0 else 0
            
            total_score = (exact_matches * 2.0) + partial_score + length_bonus + position_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_topic = topic
        
        # Only assign if we have a reasonable match
        chapter["assigned_topic"] = best_topic if best_score > 15 else None
    
    return chapters

# ### NEW FUNCTION: Jaccard Similarity Helper for Segmentation ###
def jaccard_similarity(set1: set, set2: set) -> float:
    """Computes the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

# ===== Minor/admin chatter detection & relevance scoring =====
MINOR_TOPIC_PATTERNS = [
    r"\bzoom\b", r"\bmeeting id\b", r"\bpasscode\b", r"\blog[\s-]?in\b",
    r"出勤", r"請假", r"排程", r"課程公告", r"麥克風",
    r"鏡頭", r"連線問題", r"投影", r"錄影設定"
]
MINOR_TOPIC_REGEX = re.compile("|".join(MINOR_TOPIC_PATTERNS), re.IGNORECASE)

def topic_relevance_score(global_topics: List[str], candidate_keywords: List[str]) -> float:
    """
    Returns a 0–1 overlap score between video-level topics and a block's keywords.
    """
    g = {t.strip().lower() for t in (global_topics or []) if t and isinstance(t, str)}
    c = {t.strip().lower() for t in (candidate_keywords or []) if t and isinstance(t, str)}
    if not g or not c:
        return 0.0
    j = len(g & c) / len(g | c)
    bonus = any(any(gg in cc or cc in gg for gg in g) for cc in c)
    return min(1.0, j + (0.1 if bonus else 0.0))

def looks_like_minor_admin(text: str) -> bool:
    """
    True when a block is likely admin/platform chatter instead of course content.
    """
    if not text:
        return False
    if MINOR_TOPIC_REGEX.search(text):
        return True
    admin_hits = len(re.findall(r"\b(id|passcode|join|record|mute|unmute|link|網址)\b", text, flags=re.I))
    domain_hits = len(re.findall(r"\b(class|function|api|dataset|變數|函式|圖層|製圖|建模|演算法|模型)\b", text, flags=re.I))
    return admin_hits >= 2 and domain_hits == 0

# ### NEW FUNCTION: Semantic Video Segmentation ###
def semantic_segment_video(fused_sections: List[Dict], threshold: float = 0.2):
    """
    Segments video content into logical blocks based on topic changes,
    using keywords extracted from each fused section.
    """
    if not fused_sections:
        return []
    
    # First, extract keywords for each fused section.
    keywords_per_section = []
    for section in fused_sections:
        keywords = extract_main_topics(section['fused_prompt'], mode='combined', top_k=10)
        keywords_per_section.append(set(keywords))

    semantic_blocks = []
    current_block = {
        'start': fused_sections[0]['start'],
        'end': fused_sections[0]['end'],
        'content': fused_sections[0]['fused_prompt'],
        'keywords': keywords_per_section[0]
    }

    for i in range(1, len(fused_sections)):
        prev_keywords = current_block['keywords']
        curr_keywords = keywords_per_section[i]
        similarity = jaccard_similarity(prev_keywords, curr_keywords)

        if similarity < threshold:
            # Topic shift detected, save the current block and start a new one
            semantic_blocks.append({'start': current_block['start'], 'end': current_block['end'], 'content': current_block['content']})
            current_block = {
                'start': fused_sections[i]['start'],
                'end': fused_sections[i]['end'],
                'content': fused_sections[i]['fused_prompt'],
                'keywords': curr_keywords
            }
        else:
            # Same topic, extend the current block
            current_block['end'] = fused_sections[i]['end']
            current_block['content'] += f"\n\n---\n\n{fused_sections[i]['fused_prompt']}"
            current_block['keywords'].update(curr_keywords) # Merge keywords

    # Append the last block
    semantic_blocks.append({'start': current_block['start'], 'end': current_block['end'], 'content': current_block['content']})

    return semantic_blocks
def _mean_std(xs):
    n = len(xs)
    if n == 0: return 0.0, 0.0
    mu = sum(xs) / n
    if n == 1: return mu, 0.0
    var = sum((x - mu) * (x - mu) for x in xs) / n
    return mu, var ** 0.5

from typing import Optional

def _similarity_debug(sim_scores: List[Optional[float]], used_threshold: Optional[float]):
    vals = [s for s in (sim_scores or []) if s is not None]
    mu, sd = _mean_std(vals) if vals else (0.0, 0.0)
    pct_below = (100.0 * sum(1 for s in vals if used_threshold is not None and s < used_threshold) / len(vals)) if (vals and used_threshold is not None) else None
    bins = [(-1.0,-0.5), (-0.5,0.0), (0.0,0.2), (0.2,0.4), (0.4,0.6), (0.6,0.8), (0.8,1.0)]
    hist = {f"{lo:.1f}..{hi:.1f}": 0 for (lo,hi) in bins}
    for s in vals:
        for lo,hi in bins:
            if s >= lo and s < hi:
                hist[f"{lo:.1f}..{hi:.1f}"] += 1
                break
    return {
        "mu": round(mu, 3),
        "sigma": round(sd, 3),
        "used_threshold": round(used_threshold, 3) if used_threshold is not None else None,
        "percent_below_threshold": round(pct_below, 1) if pct_below is not None else None,
        "histogram": hist,
        "pairs_total": len(sim_scores or []),
        "pairs_with_values": len(vals),
        "pairs_missing": (len(sim_scores or []) - len(vals))
    }


def semantic_segment_video_cosine(
    fused_sections: List[Dict],
    threshold: Optional[float] = None
):
    if not fused_sections:
        return []

    if len(fused_sections) == 1:
        s = fused_sections[0]
        return [{'start': s['start'], 'end': s['end'],
                 'content': s['fused_prompt'],
                 '_debug': {'similarity_scores': [], 'used_threshold': None,
                            'mode': 'single-section'}}]

    # Cheap TF-IDF vectors instead of LLM embeddings
    texts = [s["fused_prompt"] for s in fused_sections]
    embs = _tfidf_vectors(texts)

    similarity_scores = []
    for i in range(1, len(embs)):
        a, b = embs[i - 1], embs[i]
        dot = np.dot(a, b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        similarity_scores.append((dot / (na * nb)) if (na and nb) else None)

    sims_clean = [s for s in similarity_scores if s is not None]
    ADAPTIVE_SIGMA = float(os.getenv("ADAPTIVE_SIGMA", "2.0"))
    MIN_T = float(os.getenv("ADAPTIVE_MIN_T", "0.55"))
    MAX_T = float(os.getenv("ADAPTIVE_MAX_T", "0.92"))

    if threshold is None:
        if sims_clean:
            mu, sd = _mean_std(sims_clean)
            adaptive = mu - ADAPTIVE_SIGMA * sd
            used_threshold = max(MIN_T, min(MAX_T, adaptive))
        else:
            # No valid similarities at all → fall back to keyword/Jaccard segmentation
            logging.warning("⚠️ No valid cosine similarities; falling back to Jaccard segmentation.")
            return semantic_segment_video(fused_sections, threshold=0.20)
    else:
        used_threshold = threshold

    blocks = []
    cur = {
        'start': fused_sections[0]['start'],
        'end': fused_sections[0]['end'],
        'content': fused_sections[0]['fused_prompt']
    }

    for i in range(1, len(fused_sections)):
        sim = similarity_scores[i - 1]
        sec = fused_sections[i]
        # Conservative: if sim is None, we *do not split* here.
        if sim is not None and sim < used_threshold:
            blocks.append({
                'start': cur['start'], 'end': cur['end'], 'content': cur['content'],
                '_debug': {
                    'used_threshold': used_threshold,
                    'similarity_scores': similarity_scores if i == 1 else None
                }
            })
            cur = {'start': sec['start'], 'end': sec['end'], 'content': sec['fused_prompt']}
        else:
            cur['end'] = sec['end']
            cur['content'] += f"\n\n---\n\n{sec['fused_prompt']}"

    blocks.append({
        'start': cur['start'], 'end': cur['end'], 'content': cur['content'],
        '_debug': {'used_threshold': used_threshold, 'similarity_scores': None}
    })
    return blocks

def semantic_segment_video_hybrid(fused_sections, spike=0.35):
    """
    Hybrid: find cosine 'spikes' between consecutive bins and confirm with header-like cues.
    Falls back to cosine-only if data is too small.
    """
    if not fused_sections or len(fused_sections) == 1:
        return semantic_segment_video_cosine(fused_sections)

    # Reuse existing embeddings already attached by your pipeline when possible
    embs = [s.get("sum_emb") for s in fused_sections]

    def _cos(a, b):
        if not a or not b: return 0.0
        dot = sum(x*y for x,y in zip(a,b))
        na = sum(x*x for x in a) ** 0.5
        nb = sum(y*y for y in b) ** 0.5
        return (dot/(na*nb)) if (na and nb) else 0.0

    # Delta between neighbors: 1 - cosine
    deltas = []
    for i in range(1, len(embs)):
        deltas.append(1.0 - _cos(embs[i-1], embs[i]))

    # Candidate boundaries where delta >= spike
    split_pts = set(i for i, d in enumerate(deltas) if d is not None and d >= spike)  # boundary is between i and i+1

    # Light confirmation: if next bin started with OCR header flow, or looks like a header
    confirmed = set()
    for i in split_pts:
        nxt = fused_sections[i+1]["fused_prompt"]
        looks_header = nxt.startswith("[OCR_HEADER]") or (_title_likeness(nxt[:120]) >= 0.55)
        big_delta = deltas[i] >= (spike + 0.08)  # strong spike always splits
        if looks_header or big_delta:
            confirmed.add(i)

    if not confirmed:
        # fallback to cosine segmentation
        return semantic_segment_video_cosine(fused_sections)

    blocks = []
    cur = {"start": fused_sections[0]["start"], "end": fused_sections[0]["end"], "content": fused_sections[0]["fused_prompt"]}
    for i in range(1, len(fused_sections)):
        if (i-1) in confirmed:
            blocks.append(cur)
            cur = {"start": fused_sections[i]["start"], "end": fused_sections[i]["end"], "content": fused_sections[i]["fused_prompt"]}
        else:
            cur["end"] = fused_sections[i]["end"]
            cur["content"] += "\n\n---\n\n" + fused_sections[i]["fused_prompt"]
    blocks.append(cur)
    return blocks

def segment_bins_by_llm_titles(fusion_plan: List[Dict], global_topics: Optional[List[str]] = None) -> List[Dict]:
    """
    Build chapters by:
      (1) LLM distillation per bin -> title/summary,
      (2) cosine similarity on those embeddings to group consecutive bins.
    Returns blocks: [{start, end, content, meta:{title, summary}}]
    """
    if not fusion_plan:
        return []

    thr   = float(os.getenv("BIN_SIM_THR", "0.78"))
    hyst  = float(os.getenv("BIN_HYSTERESIS", "0.03"))
    min_s = int(os.getenv("BIN_MIN_SEC", "60"))
    gap_s = int(os.getenv("BIN_MERGE_GAP_SEC", "12"))

    # Enrich bins with distilled TS + embedding
    enriched = []
    for fp in fusion_plan:
        ts = title_summary_for_bin(fp["fused_prompt"])
        emb = _embed_for_title_summary(ts["title"], ts["summary"], global_topics)
        enriched.append({**fp,
                         "ts_title": ts["title"],
                         "ts_summary": ts["summary"],
                         "ts_emb": emb})

    if not enriched:
        return []

    # Group consecutive bins
    blocks = []
    cur = {
        "start": enriched[0]["start"],
        "end":   enriched[0]["end"],
        "content": enriched[0]["fused_prompt"],
        "meta": {"title": enriched[0]["ts_title"], "summary": enriched[0]["ts_summary"]}
    }
    prev_emb = enriched[0]["ts_emb"]

    for e in enriched[1:]:
        # Penalize generic titles unless summary is strong
        sim = _cos_sim(prev_emb, e["ts_emb"])
        if _is_generic_title(e["ts_title"]) and sim < (thr + 0.05):
            sim = 0.0

        if sim >= (thr + hyst):
            # extend current chapter
            cur["end"] = e["end"]
            cur["content"] += f"\n\n---\n\n{e['fused_prompt']}"
            # keep a concise/specific title & summary if better
            if len(e["ts_title"]) and len(e["ts_title"]) < len(cur["meta"]["title"]):
                cur["meta"]["title"] = e["ts_title"]
            if e["ts_summary"] and (not cur["meta"]["summary"] or len(e["ts_summary"]) < len(cur["meta"]["summary"])):
                cur["meta"]["summary"] = e["ts_summary"]
        else:
            # finalize previous if long enough
            if (cur["end"] - cur["start"]) >= min_s:
                blocks.append(cur)
            # start new
            cur = {
                "start": e["start"],
                "end":   e["end"],
                "content": e["fused_prompt"],
                "meta": {"title": e["ts_title"], "summary": e["ts_summary"]}
            }
        prev_emb = e["ts_emb"]

    # flush last
    if (cur["end"] - cur["start"]) >= min_s:
        blocks.append(cur)

    # Merge tiny gaps between adjacent chapters
    merged = []
    for b in blocks:
        if merged and (b["start"] - merged[-1]["end"]) <= gap_s:
            merged[-1]["end"] = max(merged[-1]["end"], b["end"])
            merged[-1]["content"] += "\n\n---\n\n" + b["content"]
            if b["meta"]["summary"] and b["meta"]["summary"] not in merged[-1]["meta"]["summary"]:
                merged[-1]["meta"]["summary"] += "；" + b["meta"]["summary"]
        else:
            merged.append(b)
    return merged

def rate_source_quality(ocr_txt: str, asr_txt: str) -> float:
    """Return a fused quality score [0,1] for a text block."""
    w_ocr, w_asr, sim = compute_weights(ocr_txt or "", asr_txt or "")
    # reward agreement (sim) and balanced weights
    return max(0.0, min(1.0, 0.5*sim + 0.25*w_ocr + 0.25*w_asr))

def topics_from_text_block(text: str, top_k: int = 8) -> List[str]:
    """
    Use LLM topic extraction with a regex fallback.
    Always returns a short list of strings.
    """
    try:
        toks = extract_main_topics(text, mode='combined', top_k=top_k)
        return [t for t in toks if isinstance(t, str) and t.strip()]
    except Exception:
        tokens = re.findall(r"[\u4e00-\u9fff]{2,}|\b[a-zA-Z0-9_]{4,}\b", (text or "").lower())
        counts = Counter(tokens)
        return [w for w, _ in counts.most_common(top_k)]

# ==================== This helper samples and formats OCR + BLIP + CLIP + transcript for GPT input ====================

def compute_weights(ocr_txt: str, asr_txt: str) -> Tuple[float, float, float]:
    # If one side is empty, snap weights to floor/ceil for clarity
    if not ocr_txt and asr_txt:
        return (W_FLOOR, 1.0 - W_FLOOR, 0.0)
    if not asr_txt and ocr_txt:
        return (W_CEIL, 1.0 - W_CEIL, 0.0)

    # Cheap path: if either side is very short, skip expensive summarize/embed
    if len(ocr_txt) < 200 or len(asr_txt) < 200:
        if ocr_txt and asr_txt:
            # shallow balance without sim
            q_ocr = score_ocr_quality(ocr_txt or "")
            q_asr = score_asr_quality(asr_txt or "")
            total = (q_ocr + q_asr) or 1.0
            w_ocr = max(W_FLOOR, min(W_CEIL, q_ocr / total))
            return (w_ocr, 1.0 - w_ocr, 0.0)
        return (W_CEIL, 1.0 - W_CEIL, 0.0) if ocr_txt else (W_FLOOR, 1.0 - W_FLOOR, 0.0)

    sim = compute_similarity_text(ocr_txt or "", asr_txt or "")

    q_ocr = score_ocr_quality(ocr_txt or "")
    q_asr = score_asr_quality(asr_txt or "")
    w_ocr = q_ocr * (0.5 + 0.5*sim)
    w_asr = q_asr * (0.5 + 0.5*sim)
    total = (w_ocr + w_asr) or 1.0
    w_ocr /= total; w_asr /= total
    w_ocr = min(W_CEIL, max(W_FLOOR, w_ocr))
    w_asr = 1.0 - w_ocr
    return w_ocr, w_asr, sim

def clean_ocr_text(s: str) -> str:
    if not s: return ""
    # remove typical overlays: timestamps, dates, temps, battery, watermark spam
    s = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", " ", s)
    s = re.sub(r"\b(?:\d{4}/\d{1,2}/\d{1,2}|202\d-\d{1,2}-\d{1,2})\b", " ", s)
    s = re.sub(r"[℃°]\s*\S*", " ", s)
    s = re.sub(r"(?:聯成電腦|Amara\.org|Google|msn)\S*", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def clean_asr_text(s: str) -> str:
    if not s: return ""
    # collapse repeated boilerplate
    s = re.sub(r"(字幕由\s*Amara\.org\s*社群提供\s*){2,}", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def get_total_duration(audio_segments, ocr_segments) -> float:
    a_end = max((seg.get("end", 0.0) for seg in (audio_segments or [])), default=0.0)
    v_end = max((seg.get("end", seg.get("timestamp", 0.0)) for seg in (ocr_segments or [])), default=0.0)
    return max(a_end, v_end)

def build_bins(total_duration: float, window_sec: int = WINDOW_SEC):
    bins = []
    t = 0.0
    while t < total_duration:
        bins.append((t, min(t + window_sec, total_duration)))
        t += window_sec
    return bins

def slice_by_time(segments, t0, t1, key_start="start", key_end="end"):
    if not segments:
        return []
    out = []
    for s in segments:
        s0 = s.get(key_start, s.get("timestamp", 0.0))
        s1 = s.get(key_end, s0)
        if s0 is None:
            s0 = 0.0
        if s1 is None:
            s1 = s0
        if s1 >= t0 and s0 <= t1:
            out.append(s)
    return out

def text_from_segments(segments, cleaner, key="text"):
    if not segments: return ""
    txt = "\n".join((s.get(key, "") or "").strip() for s in segments)
    return cleaner(txt)

def _token_like_code(s: str) -> float:
    patterns = [
        r"`[^`]+`", r":\s*$", r"\b(def|class|if|else|for|while|print)\b",
        r"\{|\}|\[|\]|\(|\)", r"^\s{2,}\S", r"^\s*#"
    ]
    hits = sum(bool(re.search(p, s, flags=re.MULTILINE)) for p in patterns)
    return min(1.0, hits / len(patterns))

def _junk_ratio(s: str) -> float:
    junk = re.findall(r"(?:\d{2}:\d{2}:\d{2})|(?:\d{4}/\d{1,2}/\d{1,2})|(?:℃)|(?:https?://\S+)", s)
    return min(1.0, len(junk) / max(1, len(s)//120))

def _uniq_line_ratio(s: str) -> float:
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", s) if ln.strip()]
    if not lines: return 0.0
    uniq = len(set(lines))
    return uniq / len(lines)

def score_ocr_quality(s: str) -> float:
    L = min(1.0, len(s)/2500.0)
    codey = _token_like_code(s)
    uniq = _uniq_line_ratio(s)
    junk = _junk_ratio(s)
    return max(0.0, min(1.0, 0.45*L + 0.25*codey + 0.20*uniq - 0.20*junk))

def score_asr_quality(s: str) -> float:
    L = min(1.0, len(s)/3000.0)
    punct = min(1.0, len(re.findall(r"[。！？.!?,，]", s))/max(1, len(s)//120))
    uniq = _uniq_line_ratio(s)
    boiler = 1.0 - uniq
    return max(0.0, min(1.0, 0.50*L + 0.25*punct + 0.20*uniq - 0.15*boiler))

# === Header-aware fusion helpers (insert below score_asr_quality) ===
TITLE_MAX_WORDS = 10  # used later by title validators

def _ratio_caps(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters: return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)

def _title_likeness(ocr: str) -> float:
    text = (ocr or "").strip()
    if not text: return 0.0
    words = re.findall(r"\b[\w\-]+\b", text)
    w = len(words)
    if w == 0: return 0.0
    punct_pen = 1.0 - (len(re.findall(r"[,:;.!?]", text)) / max(1, len(text)))
    len_score = 1.0 - min(abs(w - 6) / 10, 1.0)   # peak near ~6 words
    caps_bonus = 0.3 if _ratio_caps(text) > 0.35 else 0.0
    digit_pen = 0.15 if re.search(r"\b\d{1,2}\b", text) else 0.0
    return max(0.0, min(1.0, 0.55*punct_pen + 0.35*len_score + caps_bonus - digit_pen))

def _confidence_from_list(conf_list):
    if not conf_list: return 0.0
    return sum(conf_list) / len(conf_list)

def fuse_inputs(ocr_text, asr_text, ocr_confidence=None, asr_word_confidences=None, slide_changed=False):
    ocr_title_like = _title_likeness(ocr_text or "")
    asr_conf = _confidence_from_list(asr_word_confidences or [])
    ocr_conf = ocr_confidence if (ocr_confidence is not None) else 0.5  # default if unknown

    # Gate: if slide looks like a header and confidence is decent, prioritize OCR heavily
    gate = (0.6*ocr_title_like + 0.4*ocr_conf) + (0.1 if slide_changed else 0.0)

    if gate >= HEADER_GATE_THR:
        strategy = "OCR_DOMINANT"
        fused = f"[OCR_HEADER]\n{(ocr_text or '').strip()}\n\n[ASR_SUMMARY]\n{(asr_text or '').strip()[:500]}"
    else:
        strategy = "BALANCED"
        fused = f"[OCR_TEXT]\n{(ocr_text or '').strip()}\n\n[ASR_TEXT]\n{(asr_text or '').strip()}"

    meta = {
        "strategy": strategy,
        "scores": {
            "ocr_title_like": ocr_title_like,
            "ocr_conf": ocr_conf,
            "asr_conf": asr_conf,
            "gate": gate
        }
    }
    return fused, meta


def is_valid_transcript_entry(entry):
    return isinstance(entry, dict) and "start" in entry and "end" in entry and "text" in entry

def combine_for_prompt(audio_segments, ocr_segments):
    blocks = []
    for seg in ocr_segments:
        text = seg.get("text", "").strip()
        blip = seg.get("blip_caption", "") or "[BLIP caption missing]"
        tags = ", ".join(seg.get("clip_topics", [])) or "[no CLIP topics]"
        time = seg.get("timestamp", 0)

        block = (
            f"🕒 [{time:.1f}s]\n"
            f"📄 Slide Text (OCR):\n{text}\n"
            f"🖼️ Visual Caption (BLIP): {blip}\n"
            f"🏷️ Related Topics (CLIP): {tags}"
        )
        blocks.append(block)

    transcript_lines = []
    if len(ocr_segments) < 30 and audio_segments:
        step = max(len(audio_segments) // 50, 1)
        for i in range(0, len(audio_segments), step):
            seg = audio_segments[i]
            if is_valid_transcript_entry(seg):
                transcript_lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")
            else:
                logging.warning(f"⚠️ Skipping invalid transcript entry: {seg}")

    combined = "\n\n---\n\n".join(blocks)
    if transcript_lines:
        combined += "\n\n🎤 Sampled Audio Transcript:\n" + "\n".join(transcript_lines)

    return combined

def strip_minor_admin(text: str) -> str:
    """
    Aggressively downweight/remove admin/platform chatter from a big text blob.
    Uses MINOR_TOPIC_REGEX plus a few generic patterns.
    """
    if not text:
        return ""
    # Remove common admin lines
    text = MINOR_TOPIC_REGEX.sub(" ", text)
    # Extra scrub: repeated meeting/link phrases
    text = re.sub(r"(meeting id|passcode|zoom|mute|unmute|record|join|link|網址)[^\n]*", " ", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def build_global_context_corpus(ocr_segments, audio_segments, prefer_ocr=True, max_chars=300_000) -> str:
    """
    Build a single large-text corpus for 'what this class is about'.
    - Prefer OCR (slides) by default, then fold in ASR.
    - Strips admin chatter.
    - Caps size to avoid extreme costs.
    """
    ocr_txt = clean_ocr_text("\n".join((s.get("text") or "").strip() for s in (ocr_segments or [])))
    asr_txt = clean_asr_text("\n".join((s.get("text") or "").strip() for s in (audio_segments or [])))

    # Prefer OCR, then ASR
    both = f"【Slides(OCR)】\n{ocr_txt}\n\n【Audio(ASR)】\n{asr_txt}" if prefer_ocr else f"【Audio(ASR)】\n{asr_txt}\n\n【Slides(OCR)】\n{ocr_txt}"
    both = strip_minor_admin(both)

    # Hard cap by chars to be safe (Gemini can take a lot, but we still guard)
    if len(both) > max_chars:
        both = both[:max_chars]
    return both

def build_fused_window_prompt(ocr_txt, asr_txt, w_ocr, w_asr, t0, t1, token_budget=CTX_TOKEN_BUDGET):
    has_ocr = bool(ocr_txt.strip())
    has_asr = bool(asr_txt.strip())
    if has_ocr and has_asr:
        MIN_SHARE = 0.30
        ocr_share = max(MIN_SHARE, w_ocr)
        asr_share = max(MIN_SHARE, w_asr)
        norm = ocr_share + asr_share
        ocr_share /= norm; asr_share /= norm
    elif has_ocr:
        ocr_share, asr_share = 1.0, 0.0
    elif has_asr:
        ocr_share, asr_share = 0.0, 1.0
    else:
        ocr_share = asr_share = 0.5  # unreachable in practice

    char_budget = token_budget * 3
    ocr_budget = int(char_budget * ocr_share)
    asr_budget = int(char_budget * asr_share)

    def head(s, n):
        if len(s) <= n: return s
        lines, out, total = s.splitlines(), [], 0
        for ln in lines:
            if total + len(ln) + 1 > n: break
            out.append(ln); total += len(ln) + 1
        return "\n".join(out) or s[:n]

    code_snips = [m.group(0) for m in re.finditer(r"```[\s\S]*?```", ocr_txt)]
    code_blob = "\n\n".join(code_snips[:2])  # cap to 2 blocks
    ocr_take = head(ocr_txt, max(0, ocr_budget - len(code_blob)))
    asr_take = head(asr_txt, asr_budget)

    return (
        f"🕒 [{t0:.0f}-{t1:.0f}s]\n"
        f"📄 OCR重點（權重 {w_ocr:.2f}）:\n{code_blob}\n{ocr_take}\n\n"
        f"🎤 逐字稿重點（權重 {w_asr:.2f}）:\n{asr_take}\n"
    )

# ---- CJK-aware similarity utilities for OCR merging ----
_CJK_RE = re.compile(r"[\u3400-\u9fff]")  # Chinese Han range; extend if you need JP/KR

def _contains_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))

def _bigrams(s: str):
    s = (s or "").strip()
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else set()

def jaccard_chars(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    u = len(A | B) or 1
    return len(A & B) / u

def _jaccard_any(a: str, b: str) -> float:
    # Prefer char-bigram Jaccard if either side contains CJK
    if _contains_cjk(a) or _contains_cjk(b):
        return jaccard_chars(a, b)
    set_a = set((a or "").split())
    set_b = set((b or "").split())
    u = len(set_a | set_b) or 1
    return len(set_a & set_b) / u

# ==================== Merge Similar OCR Blocks ====================
def merge_similar_ocr_blocks(blocks, threshold=0.7):
    def jaccard(a, b):
        return _jaccard_any(a, b)

    merged = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        text = " ".join(seg['text'] for seg in block)
        j = i + 1
        while j < len(blocks):
            next_text = " ".join(seg['text'] for seg in blocks[j])
            if jaccard(text, next_text) > threshold:
                block.extend(blocks[j])
                j += 1
            else:
                break
        merged.append(block)
        i = j
    return merged

# ==================== Generate Topics ====================
def generate_topics_from_ocr_blocks(ocr_blocks, max_topics=12):
    # Step 1: Remove noisy/uninformative blocks
    def is_informative(block):
        block_text = " ".join(seg.get("text", "") for seg in block).strip()
        if len(block_text) < 40:
            return False
        if re.fullmatch(r"新竹小幫手|老師|助教", block_text):
            return False
        return True

    filtered_blocks = [b for b in ocr_blocks if is_informative(b)]
    if not filtered_blocks:
        logging.warning("⚠️ All blocks filtered out. Using original blocks.")
        filtered_blocks = ocr_blocks

    # Step 2: Downsample if too many blocks
    if len(filtered_blocks) > max_topics:
        step = len(filtered_blocks) // max_topics
        sampled_blocks = [filtered_blocks[i] for i in range(0, len(filtered_blocks), step)][:max_topics]
    else:
        sampled_blocks = filtered_blocks

    # Step 3: Prepare content for prompt
    content_blocks = []
    for block in sampled_blocks:
        merged = "\n".join(seg.get("text", "") for seg in block)
        content_blocks.append(merged)

    joined = "\n\n---\n\n".join(content_blocks)

    # Step 4: Prompt GPT
    prompt = f"""
這是一些依據投影片時間分組的課程模組資料，請幫我推測每組代表的課程主題名稱與簡介：

{joined}

請輸出格式：
[
  {{
    "title": "主題名稱",
    "summary": "這段課程大致涵蓋的教學內容"
  }}
]
"""
    role = "你是一位線上課程設計師，負責從課程投影片中提取主題名稱與摘要。"
    response = call_llm(prompt, role, max_tokens=4000)
    return parse_qa_json(response)

# ==================== QandA Structure Validator ====================
def validate_qa_structure(qa_list):
    required_keys = {"QuestionId", "QuestionText", "Options", "CorrectAnswer", "Explanation", "Tags", "Difficulty"}
    for i, item in enumerate(qa_list):
        if not isinstance(item, dict):
            logging.error(f"❌ Q{i+1} is not a dictionary.")
            return False
        missing = required_keys - item.keys()
        if missing:
            logging.error(f"❌ Q{i+1} is missing fields: {missing}")
            return False
    return True

def _sanitize_options(question):
    # Preserve the correct option's TEXT (labels are unreliable at this point)
    correct_label = (question.get("CorrectAnswer") or "").strip()
    opts = question.get("Options", []) or []

    # Try to locate the correct option text by label; if missing, fall back later
    correct_text = None
    for o in opts:
        if (o.get("Label") or "").strip() == correct_label:
            t = (o.get("Text") or "").strip()
            if t:
                correct_text = t
            break

    # Dedup and drop empties
    seen = set()
    cleaned = []
    for o in opts:
        t = (o.get("Text") or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        cleaned.append({"Label": (o.get("Label") or "").strip(), "Text": t})

    # If we lost the correct option during cleanup, re-add it (best-effort)
    if correct_text and correct_text not in seen:
        cleaned.insert(0, {"Label": "", "Text": correct_text})

    # Ensure exactly 4 options (pad)
    while len(cleaned) < 4:
        cleaned.append({"Label": "", "Text": f"（選項補齊）{len(cleaned)+1}"})
    cleaned = cleaned[:4]

    # Standardize labels A–D and update CorrectAnswer to a valid label
    label_map = ['A', 'B', 'C', 'D']
    for i, o in enumerate(cleaned):
        o["Label"] = label_map[i]

    # If we know the correct_text, point CorrectAnswer at the matching label; else default to A
    if correct_text:
        new_correct = next((o["Label"] for o in cleaned if o["Text"] == correct_text), 'A')
    else:
        new_correct = correct_label if correct_label in label_map else 'A'

    question["Options"] = cleaned
    question["CorrectAnswer"] = new_correct


# ==================== Answer Distribution Helpers ====================
def regenerate_explanation_with_llm(question):
    prompt = f"""
請根據以下題目與選項，提供一段中文解釋說明為何正確答案是 {question['CorrectAnswer']}。請簡要說明其他選項為何錯誤。

題目：
{question['QuestionText']}

選項：
{chr(10).join(f"{opt['Label']}. {opt['Text']}" for opt in question['Options'])}

請提供解析，開頭請寫「正確答案是 {question['CorrectAnswer']}，因為...」
"""
    explanation = call_llm(
        prompt,
        role="你是一位出題老師，負責為測驗題提供解析。",
        max_tokens=500
    )
    question['Explanation'] = explanation.strip()

def shuffle_question_options(qa_list, regenerate_explanation=False, seed: Optional[int] = None):
    # Deterministic shuffling if seed is provided (e.g., CI)
    rng = random.Random(seed) if seed is not None else random
    for question in qa_list:
        options = question.get("Options", []) or []

        old_correct = (question.get("CorrectAnswer") or "").strip()

        # Find correct option TEXT by current label; if missing, default to first option's text
        correct_text = next((opt.get("Text", "") for opt in options
                             if (opt.get("Label") or "").strip() == old_correct), None)
        if not correct_text and options:
            correct_text = options[0].get("Text", "")

        # Shuffle options (deterministically if rng has a seed)
        rng.shuffle(options)

        # Reassign labels A–D
        label_map = ['A', 'B', 'C', 'D']
        for idx, opt in enumerate(options[:4]):
            opt["Label"] = label_map[idx]
        options = options[:4]
        question["Options"] = options

        # Find new label for correct answer (by TEXT); if not found, default to A
        new_correct = next((opt["Label"] for opt in options if opt.get("Text", "") == correct_text), 'A')
        question["CorrectAnswer"] = new_correct

        moved = (new_correct != old_correct)

        # Regenerate or patch explanation
        if regenerate_explanation and moved:
            regenerate_explanation_with_llm(question)
        else:
            import re
            question["Explanation"] = re.sub(
                r"正確答案是\s+[A-D]",
                f"正確答案是 {question['CorrectAnswer']}",
                question.get("Explanation", "")
            )

def enforce_difficulty_distribution(qa_list, target_ratio=(0.3, 0.4, 0.3)):
    total = len(qa_list)
    target_counts = {
        '簡單': round(target_ratio[0] * total),
        '中等': round(target_ratio[1] * total),
        '困難': total - round(target_ratio[0] * total) - round(target_ratio[1] * total)
    }
    qa_sorted = sorted(qa_list, key=lambda q: len(q.get("Explanation", "")))
    for idx, q in enumerate(qa_sorted):
        if idx < target_counts['簡單']:
            q['Difficulty'] = '簡單'
        elif idx < target_counts['簡單'] + target_counts['中等']:
            q['Difficulty'] = '中等'
        else:
            q['Difficulty'] = '困難'
    return qa_sorted

# ==================== Key Concepts and Objectives Generation ====================
def generate_key_concepts_and_objectives(combined_text: str):
    """
    Analyzes video content to generate key concepts and learning objectives.
    """
    logging.info("🧠 Generating key concepts and learning objectives...")

    # === Prompt Construction ===
    prompt = f"""
    你是一位專業的教育工作者與教學設計師。你的任務是分析以下影片的逐字稿，並為學生生成一套結構化的關鍵概念與學習目標。

    影片內容：
    {combined_text}

    請輸出一個包含兩個鍵值的 JSON 物件：
    - "key_concepts": 一個以條列式摘要核心概念的列表。
    - "learning_objectives": 一個描述學生在觀看後應能完成的、以行動為導向的句子列表。

    範例格式：
    {{
      "key_concepts": [
        "核心概念1的簡要解釋。",
        "核心概念2及其重要性。"
      ],
      "learning_objectives": [
        "應用原則X來解決問題Y。",
        "分析方法A與方法B之間的差異。"
      ]
    }}

    請只輸出 JSON 物件，不要包含任何額外的文字或解釋。
    """

    role = "你是一位專業的教育工作者與教學設計師。"
    response = call_llm(prompt, role, max_tokens=2000)
    response = _extract_first_json_block(response) 
    try:
        return parse_qa_json(response)
    except Exception as e:
        logging.error(f"❌ Failed to parse key concepts JSON: {e}")
        return None
    
# ==================== Q&A Generation ====================


def generate_qa_from_text(text: str, topics: List[Dict], num_questions: int = 10, video_context: Optional[Dict] = None):
    """
    Generates Q&A based on Bloom's Taxonomy, using a mix of context and direct text.
    """
    # Turn your list of topic dicts into a bullet list
    topics_snippet = "\n".join(
        f"{t.get('id', str(i+1).zfill(2))}. {t['title']}：{t['summary']}"
        for i, t in enumerate(topics)
    )

    logging.info(f"📚 Generating {num_questions} course-based questions around topics: {topics_snippet!r}")

    # === Role Definition ===
    role = """
    你是一位專業的教育 AI，負責為不同領域（如 Python、AI、設計、AutoCAD、行銷、遊戲開發等）設計高品質的測驗題目。

    🔍 資料來源說明：
    - 📄 Slide Text（OCR）來自課程投影片，準確且專注於課程重點，請優先以此為出題依據。
    - 🎤 Audio Transcript 為語音辨識內容，可能包含閒聊、無關內容，僅在 OCR 不足時輔助參考。

    📌 出題目標：
    - 幫助學生理解與應用實際知識，而非僅記憶。
    - 題目須能反映實務邏輯、真實工作情境、設計思維或編程決策。
    - 鼓勵跨領域出題方式，自動識別所屬課程領域並標註 CourseType。
    """

    escaped_text = str(text or "").replace("{", "{{").replace("}", "}}")

    # --- Global context injection ---
    vc = video_context or {}
    gc_summary = (vc.get("summary") or "").strip()
    gc_topics = vc.get("main_topics") or []
    global_ctx_snippet = ""
    if gc_summary or gc_topics:
        global_ctx_snippet = "### 全域脈絡（Global Context）\n"
        if gc_summary:
            global_ctx_snippet += f"- 摘要：{gc_summary}\n"
        if gc_topics:
            global_ctx_snippet += f"- 主題列表：{', '.join(gc_topics[:12])}\n"

    # === Bloom's Taxonomy Prompts ===
    # Distribute the total number of questions across the three levels
    num_per_level = num_questions // 3
    remaining_questions = num_questions % 3

    bloom_prompts = {
        "Recall": {
            "prompt_text": f"""
            根據以下課程內容，請設計 **{num_per_level + (1 if remaining_questions > 0 else 0)}** 道多重選擇題，以測試學生回憶特定事實的能力。
            """,
            "count": num_per_level + (1 if remaining_questions > 0 else 0)
        },
        "Application": {
            "prompt_text": f"""
            根據以下課程內容，請設計 **{num_per_level + (1 if remaining_questions == 2 else 0)}** 道情境式問題，要求學生應用所討論的原則。
            """,
            "count": num_per_level + (1 if remaining_questions == 2 else 0)
        },
        "Analysis": {
            "prompt_text": f"""
            根據以下課程內容，請設計 **{num_per_level}** 道開放式問題，要求學生比較和對比內容中的概念。
            """,
            "count": num_per_level
        }
    }

    all_qa_json = []

    # === Iteration through Bloom's Levels ===
    for level, config in bloom_prompts.items():
        if config["count"] <= 0:
            continue
        
        logging.info(f"📚 Generating {config['count']} questions for Bloom's level: {level}...")
        
        # === Full Prompt Construction (using your original structure) ===

        full_prompt = f"""
<全域脈絡>
{global_ctx_snippet or '（無）'}
</全域脈絡>

<課程大綱>
{topics_snippet}
</課程大綱>

<背景說明>
{config['prompt_text']}
</背景說明>

<出題指引>
- 所有題目須根據課程內容設計，避免出現教材未涵蓋的主題。
- 請使用多樣化提問方式，避免所有題目皆以「在...時」開頭。
- 嚴格根據「全域脈絡」與「課程大綱」出題，忽略行政性或平台設定（如 Zoom 連線、會議 ID、麥克風/鏡頭設定、請假、點名等）。
- 題型建議涵蓋：
  - 概念理解與定義
  - 情境應用與錯誤辨識
  - 設計決策與最佳實踐
  - 實務案例與策略比較
- 問題風格請平衡 "做什麼（What）"、"怎麼做（How）" 與 "為什麼（Why）"。
- 每題包含 4 個具挑戰性的選項（A–D），須具迷惑性與真實性。
- 選項中應避免正確答案集中出現。
- 難度比例分配：
  - 30% 簡單：單一概念、直接應用
  - 40% 中等：邏輯推理、正確選擇
  - 30% 困難：跨章節整合、情境分析
</出題指引>

<情境化建議>
- 可加入角色背景（如開發者、設計師、行銷人員），提升真實感。
- 避免單純描述工具操作，應強調背後目的與設計意圖。
</情境化建議>

<解析要求>
- 每題需附解釋：
  - 正確答案的原因
  - 錯誤選項的常見誤解或問題
  - 與實作或案例的連結
</解析要求>

<輸出格式>
請以 JSON 陣列回傳，格式如下：
[
  {{
    "QuestionId": "Q001",
    "QuestionText": "...",
    "Options": [
      {{ "Label": "A", "Text": "..." }},
      {{ "Label": "B", "Text": "..." }},
      {{ "Label": "C", "Text": "..." }},
      {{ "Label": "D", "Text": "..." }}
    ],
    "CorrectAnswer": "B",
    "Explanation": "...",
    "Tags": ["..."],
    "Difficulty": "中等",
    "CourseType": "Python"
  }}
]
請勿加入說明文字，僅輸出 JSON 陣列。
</輸出格式>

<資料說明>
1. 📄 Slide Text：準確率高，請優先使用。
2. 🎤 Audio Transcript：可能有誤差，僅作輔助參考。
</資料說明>

<課程內容>
{escaped_text}
</課程內容>
"""
        qa_content = call_llm(full_prompt, role, max_tokens=12000)
        qa_content = _extract_first_json_block(qa_content)
        qa_json_part = parse_qa_json(qa_content)

        if qa_json_part:
            for item in qa_json_part:
                # Safe-append Bloom tag
                if 'Tags' not in item or not isinstance(item['Tags'], list):
                    item['Tags'] = []
                item['Tags'].append(f"Bloom's Taxonomy: {level}")
            all_qa_json.extend(qa_json_part)

    # Renumber QuestionIds to ensure uniqueness across batches
    for i, q in enumerate(all_qa_json, start=1):
        q['QuestionId'] = f"Q{str(i).zfill(3)}"

    # --- Final Validation and Return ---
    # Merge all questions and validate the final list
    if not all_qa_json or not validate_qa_structure(all_qa_json):
        logging.error("❌ Failed to parse or validate Q&A JSON from model output.")
        return None
        
    return all_qa_json

# ==================== Lecture Notes Generation ===================

def generate_lecture_notes(text: str, topics: List[Dict], num_pages: int = 3, video_context: Optional[Dict] = None):
    # Create a bullet list of topics again
    topics_snippet = "\n".join(
        f"{t.get('id', str(i+1).zfill(2))}. {t['title']}：{t['summary']}"
        for i, t in enumerate(topics)
    )
    # --- Global context injection ---
    vc = video_context or {}
    gc_summary = (vc.get("summary") or "").strip()
    gc_topics  = vc.get("main_topics") or []
    global_ctx_snippet = ""
    if gc_summary or gc_topics:
        global_ctx_snippet = "### 全域脈絡（Global Context）\n"
        if gc_summary:
            global_ctx_snippet += f"- 摘要：{gc_summary}\n"
        if gc_topics:
            global_ctx_snippet += f"- 主題列表：{', '.join(gc_topics[:12])}\n"

    """
    生成詳細且富有教學風格的講義筆記，包含：
    - **完整的核心概念 (Key Concepts)**
    - **詳細的步驟指南 (Step-by-Step Guide)**
    - **教師的專業建議與常見錯誤 (Teacher's Know-How)**
    - **有用的真實應用場景 (Real-World Applications)**
    - **語言風格符合「錄製的課程」，使用過去式描述內容**
    適用於各種課程，如 Python、AutoCAD、設計、AI 等。
    """
    min_words = num_pages * 400
    max_words = (num_pages + 1) * 350

    logging.info(f"正在生成 {num_pages} 頁的講義筆記 (~{min_words}-{max_words} 字)...")

    role = """
    你是一位經驗豐富的老師，負責整理**直播課程的講義筆記**，
    讓學生在課後能夠快速複習內容，並應用所學知識。

    🔎 資料使用準則：
    - 優先使用『📄 Slide Text』作為知識依據，這是從課程投影片 OCR 提取的資訊。
    - 僅當 OCR 不足時，才參考 🎤 音訊轉錄。

    你的目標是：
    - **確保涵蓋所有主要的核心概念，不遺漏任何重點。**
    - **為每個核心概念提供詳細的步驟指引，讓學生能夠按照指引操作。**
    - **在應用場景部分提供具體的案例分析，而不是抽象的應用描述。**
    - **整理老師的專業建議、常見錯誤與最佳做法。**
    - **使用「過去式」來描述內容，避免像直播教學一樣使用「今天我們來學…」，而改為「在這堂課中，我們學到了…」。**
    - **如果課程與程式編寫有關（例如 Python、Java、C、C++、web development、AI 等），請務必在每個相關段落中加入實際可執行的程式碼範例，而非僅以文字描述邏輯。**
    """

    prompt = f"""

    <全域脈絡>
    {global_ctx_snippet or '（無）'}
    </全域脈絡>

    <課程大綱>
    {topics_snippet}
    </課程大綱>

    <背景說明>
    這份講義是根據**錄製的線上課程**整理的，**請使用「過去式」來描述內容**，而不是像即時授課一樣使用「我們現在來學習…」。
    你的目標是確保這份筆記：
    - **幫助學生在課後復習**
    - **完整涵蓋直播課程的知識點**
    - **包含老師講解的操作細節**
    - **提供教師的專業建議與常見錯誤**
    - **強調真實應用場景**
    - 根據上方的「全域脈絡」判斷課程主軸，將行政性/平台設定視為次要內容，僅在必要時以一句話帶過，不得佔用主要篇幅。

    </背景說明>

    <講義結構>

    ## 1. 課程背景 (Course Background)
    - **這堂課的重點是什麼？**
    - 例如：「在這堂課中，我們學到了如何使用 AutoCAD 進行精確繪圖，這對於建築設計至關重要。」

    ## 2. 核心概念與應用 (Key Concepts & How They Are Used)
    - **完整列出所有課程中提到的重要概念**，確保不遺漏任何關鍵知識點。
    - **每個概念應該有詳細解釋，不只是定義，而是要「像老師講課一樣」描述。**
    - **💡 提供例子**：「例如，在 AutoCAD 中，我們可以用這個工具來…」
    - 每個概念前請使用如下格式：
      
### 🛠 概念 1：XXX 概念名稱


    ## 3. 操作步驟與實作指南 (Step-by-Step Guide & Implementation)
    - **對於每個核心概念，提供詳細的操作步驟，並清楚標註該步驟屬於哪個概念。**
    - **確保步驟足夠詳細，讓學生能夠跟著操作。**
    - **如果課程與程式設計相關，請使用適當的程式碼區塊（例如：python、java、AI、C++、網頁開發）來加入相關的程式碼範例。**
    - 每個概念的步驟示例：
      
### 🛠 概念 1：Python 變數
      在這堂課中，老師帶著我們完成了以下步驟：

      1. 開啟 Python 環境，輸入 `x = 10`  
      2. 嘗試執行 `print(x)`，觀察輸出
      3. 變數可以儲存各種類型的數據，例如 `y = "Hello"`  
      4. 使用 `type(y)` 來檢查變數的類型


    ## 4. 教師講解與專業建議 (Instructor Insights & Best Practices)
    - **整理老師口頭提到的重要觀念，常見錯誤與最佳做法：**
      - ❌ 常見錯誤：「老師提醒我們，很多學生會這樣做，但這樣可能會導致…」
      - ✅ 最佳做法：「老師建議最好的方法是…」
    - **提供教師的專業建議**：「老師提到，這個功能在實際工作中常用於…」


    ## 5. 應用場景與案例分析 (Real-World Scenarios & Case Studies)
    - **提供真實應用案例，幫助學生理解如何應用所學知識。**
    - 每個案例使用如下格式：
      
🎯 **案例 1：變數的應用**
      在這堂課中，老師舉了一個範例，說明變數如何應用於點餐系統：
      - 假設你正在開發一個點餐系統，你需要變數來儲存用戶選擇的餐點和價格。
      - 例如：
```python
        order_item = "漢堡"
        order_price = 120  # 單位：台幣
        total_price = order_price * 2  # 假設點了兩個
        print(f"您的訂單：order_item x2，共 total_price 元")
```


    ## 6. 總結與回顧 (Summary & Key Takeaways)
    - 用過去式總結：「在這堂課中，我們學到了…」
    - 可以提醒學生下一步可以做什麼：「如果你想更進一步，你可以試試…」

    <格式規則>
    - **使用 Markdown 格式** 來提高可讀性。
    - 確保層次分明，使用：
      - ## 為主要標題 (如 1. 課程背景, 2. 核心概念)
      - ### 為概念標題 (如 🛠 概念 1、概念 2)
      - 數字編號 (1., 2.) 為步驟指引
      - 正確使用代碼區塊和範例
    - **如果課程與程式設計相關，請使用適當的程式碼區塊（如 ```python、```java 等）來呈現程式碼，且所有程式碼範例應為可執行內容，並與老師的講解保持一致。**
    - ✅ 如果內容並非程式碼（例如設計概念、藝術技巧、講義總結），請使用：
      - ```text 或 ```note 來取代 ```python，避免不必要的語法標記

    **請根據以下課程內容撰寫講義，確保內容包含「所有主要的核心概念」，並提供「詳細的步驟指南」與「完整的應用案例」。**

{str(text or "").replace("{", "{{").replace("}", "}}")}
    """

    # Call OpenAI API
    course_note = call_llm(prompt, role, max_tokens=12000)

    # Debugging: Print Q&A
    logger.info(f"📘 Generated Lecture Notes:\n{course_note[:1000]}...\n")
    logger.info("-" * 80)

    return course_note

# ==================== Format_topic_timeline ====================
def format_topic_timeline(chapters):
    def seconds_to_hhmmss(seconds):
        hours = int(seconds) // 3600
        minutes = (int(seconds) % 3600) // 60
        secs = int(seconds) % 60
        return f"{hours:02}:{minutes:02}:{secs:02}"

    timeline = []
    for chapter in chapters:
        ts = seconds_to_hhmmss(chapter['start'])
        title = chapter.get("title", "Untitled")
        timeline.append(f"{ts} {title}")
    return timeline

# ==================== Main Topics ====================

# ── 1) Extract main topics from segments ──
def extract_main_topics(segments, mode="ocr", top_k=5):
    """
    segments: list of {'text': ..., 'timestamp': ...}
    mode: "ocr" or "audio"
    returns: List[str] of keywords
    """

    if isinstance(segments, str):
        full_text = segments
    else:
        full_text = "\n".join(seg.get("text", "") for seg in segments)

    if not full_text.strip():
        return []
    system_role = "你是一個主題關鍵詞擷取器。"
    label = "投影片文字 (OCR)" if mode == "ocr" else ("講師逐字稿" if mode == "audio" else "整合文本")
    user_prompt = f"""
    下面是一段{label}，請擷取出 **{top_k}** 個最能代表內容的「主題關鍵詞」，並以 JSON 陣列回傳：



```
{full_text}
```
""" 
    resp = call_llm(user_prompt, system_role, max_tokens=1000)
    try:
        clean = re.sub(r"^```(?:json)?|```$", "", resp.strip(), flags=re.MULTILINE)
        return json.loads(clean)
    except Exception:
        logging.warning("⚠️ 主題關鍵詞解析失敗，使用簡易分詞回退。")
        tokens = re.findall(r"[\u4e00-\u9fff]{2,}|\b[a-zA-Z0-9_]{4,}\b", full_text)
        freq = Counter(tokens)
        return [w for w, _ in freq.most_common(top_k)]

_GENERIC_TITLE_PATTERNS = [
    r"^介紹$", r"^引言$", r"^摘要$", r"^總結$", r"^回顧$",
    r"^說明$", r"^範例$", r"^示範$", r"^注意事項$"
]
_GENERIC_TITLE_RE = re.compile("|".join(_GENERIC_TITLE_PATTERNS))

def _cos_sim(a, b) -> float:
    if not a or not b: return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(y*y for y in b) ** 0.5
    return (dot/(na*nb)) if (na and nb) else 0.0
def _title_has_coverage(title: str, fused_text: str) -> bool:
    # simple coverage: at least 1 key term appears in title
    words = set(w.lower() for w in re.findall(r"\b[\w\-]+\b", title))
    # reuse your topics extractor as fallback if needed:
    key_terms = set(extract_main_topics(fused_text, mode='combined', top_k=10))
    return len(words & set(k.lower() for k in key_terms)) >= 1

def _validate_title_en(title: str) -> bool:
    # For English titles (≤10 words, no meta)
    words = re.findall(r"\b[\w\-]+\b", (title or ""))
    if len(words) > 10: return False
    if re.search(r"(slide|agenda|welcome|thanks|q\s*&\s*a|speaker)", title or "", re.I): return False
    return True

def _is_generic_title(t: str) -> bool:
    t = (t or "").strip()
    if not t: return True
    if any(bad in t for bad in BIN_TITLE_BANLIST): return True
    return bool(_GENERIC_TITLE_RE.search(t))

def title_summary_for_bin(fused_text: str, video_context: dict) -> Dict[str, str]:
    gist = canonical_gist(video_context)
    # pick a lightweight noun-phrase title
    title = "、".join(gist["keyphrases"].split(", ")[:2]) or "課程主題"
    return {"title": _maybe_prefix(title), "summary": gist["summary"]}

def title_summary_for_bin_ctx(fusion_plan: List[Dict], k: int, module_hint: Optional[str] = None) -> Dict[str, str]:
    """
    Tri-bin (prev/curr/next) contextual titling with optional module hint.
    Falls back to single-bin version on failure.
    """
    def _ctx(i):
        if i < 0 or i >= len(fusion_plan): return None
        fp = fusion_plan[i]
        comp = fp.get("_mv_comp") or {"summary": fp.get("sum_for_embed",""), "keyphrases": ""}
        return {
            "t0": fp["start"], "t1": fp["end"],
            "sum": comp.get("summary") or summarize_for_embedding(fp["fused_prompt"]),
            "kps": comp.get("keyphrases","")
        }

    prev, curr, nxt = _ctx(k-1), _ctx(k), _ctx(k+1)
    if not curr:  # safety
        return title_summary_for_bin(fusion_plan[k]["fused_prompt"])

    blocks = []
    if prev: blocks.append(f"【前一段 {prev['t0']:.0f}-{prev['t1']:.0f}s】\n{prev['sum']}\n關鍵詞：{prev['kps']}")
    blocks.append(f"【目前段落 {curr['t0']:.0f}-{curr['t1']:.0f}s】\n{curr['sum']}\n關鍵詞：{curr['kps']}")
    if nxt:  blocks.append(f"【下一段 {nxt['t0']:.0f}-{nxt['t1']:.0f}s】\n{nxt['sum']}\n關鍵詞：{nxt['kps']}")

    hint = f"\n模組脈絡提示：{module_hint}" if (module_hint and BIN_TITLE_MODULE_HINT) else ""
    prompt = f"""
你是課程分章器。僅針對「目前段落」產生主題式標題與1–2句摘要；前後段落僅供脈絡。{hint}

嚴格規則同現有版本（禁用：介紹/小結/示範…；選一個主軸；繁中；JSON only）。

{chr(10).join(blocks)}

JSON:
{{"title":"...", "summary":"...", "tags":["..."]}}
""".strip()

    raw = call_llm(prompt, role="章節命名助理", max_tokens=int(os.getenv("BIN_TITLE_MAXTOK","700")))
    data = parse_qa_json(raw) or {}
    if isinstance(data, list) and data: data = data[0]
    title = _normalize_topic_title(data.get("title",""))
    summary = (data.get("summary") or "").strip()

    # validations & fallback
    if _title_is_generic(title) or not _title_has_coverage(title, fusion_plan[k]["fused_prompt"]):
        return title_summary_for_bin(fusion_plan[k]["fused_prompt"])

    title = _truncate_bytesafe(title, BIN_TITLE_MAX_CHARS)
    return {"title": _maybe_prefix(title), "summary": summary}

def _embed_for_title_summary(title: str, summary: str, global_topics: Optional[List[str]] = None):
    """
    Embed compact semantics for grouping, with a gentle global-topic hint.
    Uses multi-view (summary+title) for stability.
    """
    hint = ""
    if global_topics:
        hint = "；主題提示：" + ", ".join([t for t in global_topics[:5] if isinstance(t, str)])

    title1 = (title or "").strip()
    summary2 = ((summary or "").strip() + (hint or "")).strip()

    cache_key = _short_hash("TS|" + title1 + "|" + summary2)
    if cache_key in _BIN_EMB_CACHE:
        return _BIN_EMB_CACHE[cache_key]

    emb = _embed_views(summary2 or title1, title1 or summary2)
    if emb:
        _BIN_EMB_CACHE[cache_key] = emb
    return emb


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]

def summarize_for_embedding(text: str, video_context: dict) -> str:
    return canonical_gist(video_context)["summary"]

def compute_similarity_text(doc1: str, doc2: str) -> float:
    summary1 = summarize_for_embedding(doc1)
    summary2 = summarize_for_embedding(doc2)
    if not summary1 or not summary2:
        logging.warning("⚠️ One or both summaries missing. Skipping similarity computation.")
        return 0.0

    emb1 = _safe_embed(summary1)
    emb2 = _safe_embed(summary2)
    if not emb1 or not emb2:
        return 0.0

    dot   = sum(a*b for a, b in zip(emb1, emb2))
    norm1 = sum(a*a for a in emb1) ** 0.5
    norm2 = sum(b*b for b in emb2) ** 0.5
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def _emb_for_snippet(snippet: str):
    """
    Returns a cached embedding for a snippet summary. Uses _short_hash key.
    """
    key = _short_hash(snippet)
    if key in _EMB_CACHE:
        return _EMB_CACHE[key]
    emb = _safe_embed(snippet)
    if emb:
        _EMB_CACHE[key] = emb
    return emb

# ===== Chrono-semantic grouping for course-level TOC =====
def _cos_local(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b: return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(y*y for y in b) ** 0.5
    return (dot/(na*nb)) if (na and nb) else 0.0

def chrono_semantic_groups(embs: List[Optional[List[float]]], thr: float) -> List[List[int]]:
    """
    One-pass chronological clustering: start a group, extend while consecutive
    bins are ≥ thr similar; otherwise start a new group.
    Returns list of index lists (e.g., [[0,1,2],[3,4],...]).
    """
    groups, cur = [], []
    for i, v in enumerate(embs):
        if not cur:
            cur = [i]; continue
        prev_v = embs[cur[-1]]
        sim = _cos_local(prev_v, v)
        if sim >= thr:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    if cur: groups.append(cur)
    return groups

def _module_times(module: Dict, fusion_plan: List[Dict]) -> Tuple[float, float]:
    idxs = sorted(module.get("bins") or [])
    if not idxs: return 0.0, 0.0
    start = float(fusion_plan[idxs[0]]["start"])
    end   = float(fusion_plan[idxs[-1]]["end"])
    return start, end

def merge_adjacent_toc_modules(course_toc: List[Dict],
                               fusion_plan: List[Dict],
                               sim_thr: Optional[float] = None,
                               gap_merge_sec: Optional[int] = None) -> List[Dict]:
    """
    Sequential pass over course_toc (already chronological by bins).
    Merge a module into the previous one if:
      - cosine(module_i, module_{i-1}) >= sim_thr, OR
      - time gap between them <= gap_merge_sec
    """
    if not course_toc: return []
    sim_thr = sim_thr if sim_thr is not None else TOC_MERGE_THR
    gap_merge_sec = gap_merge_sec if gap_merge_sec is not None else TOC_ADJ_MERGE_GAP_SEC

    merged: List[Dict] = []
    prev_vec: Optional[List[float]] = None
    prev_end: Optional[float] = None

    for m in course_toc:
        cur_vec = _embed_toc_item(m)
        cur_start, cur_end = _module_times(m, fusion_plan)

        if merged:
            sim = _cos(prev_vec, cur_vec) if (prev_vec and cur_vec) else 0.0
            gap_ok = (prev_end is not None) and (cur_start - prev_end <= gap_merge_sec)
            if sim >= sim_thr or gap_ok:
                # merge into last
                last = merged[-1]
                last["bins"].extend(m.get("bins") or [])
                # prefer shorter, more specific title if different
                t_old = (last.get("module_title") or "").strip()
                t_new = (m.get("module_title") or "").strip()
                if t_new and (not t_old or len(t_new) < len(t_old)):
                    last["module_title"] = t_new
                # concat summaries if distinct
                s_old = (last.get("module_summary") or "").strip()
                s_new = (m.get("module_summary") or "").strip()
                if s_new and s_new not in s_old:
                    last["module_summary"] = (s_old + "；" + s_new).strip("；")
                # update prev trackers
                prev_vec = _embed_toc_item(last)
                prev_end = _module_times(last, fusion_plan)[1]
                continue

        # start a new merged module
        merged.append({
            "bins": list(m.get("bins") or []),
            "module_title": (m.get("module_title") or "").strip() or "模組",
            "module_summary": (m.get("module_summary") or "").strip(),
        })
        prev_vec = cur_vec
        prev_end = cur_end

    # normalize bins ordering
    for m in merged:
        m["bins"] = sorted(set(m["bins"]))
    return merged


# ==== Chaptering (built on fused_sections + semantic_blocks) ====

CHAPTER_MIN_SEC       = int(os.getenv("CHAPTER_MIN_SEC", "60"))
CHAPTER_MAX_COUNT     = int(os.getenv("CHAPTER_MAX_COUNT", "20"))
CHAPTER_MERGE_GAP_SEC = int(os.getenv("CHAPTER_MERGE_GAP_SEC", "15"))
CHAPTER_TITLE_MAXTOK  = int(os.getenv("CHAPTER_TITLE_MAXTOK", "400"))
CHAPTER_SIM_MERGE_THR = float(os.getenv("CHAPTER_SIM_MERGE_THR", "0.82"))

# Backoff & forced-time chaptering knobs
MIN_CHAPTERS              = int(os.getenv("MIN_CHAPTERS", "3"))
CHAPTER_BACKOFF_STEP      = float(os.getenv("CHAPTER_BACKOFF_STEP", "0.05"))
CHAPTER_BACKOFF_TRIES     = int(os.getenv("CHAPTER_BACKOFF_TRIES", "6"))
ADAPTIVE_MIN_T            = float(os.getenv("ADAPTIVE_MIN_T", "0.55"))
FORCED_CHAPTER_EVERY_SEC  = int(os.getenv("FORCED_CHAPTER_EVERY_SEC", "600"))  # 10 min

def _force_time_segments(sections, every_sec=FORCED_CHAPTER_EVERY_SEC):
    """Force chapters every N seconds by concatenating fused sections until the bucket fills."""
    if not sections:
        return []
    forced = []
    cur = {
        "start": sections[0]["start"],
        "end": sections[0]["end"],
        "content": sections[0]["fused_prompt"],
    }
    bucket_start = cur["start"]
    for sec in sections[1:]:
        # if adding this section would exceed the bucket, cut a chapter
        if (sec["end"] - bucket_start) >= every_sec:
            forced.append({"start": cur["start"], "end": cur["end"], "content": cur["content"]})
            cur = {"start": sec["start"], "end": sec["end"], "content": sec["fused_prompt"]}
            bucket_start = cur["start"]
        else:
            cur["end"] = sec["end"]
            cur["content"] += f"\n\n---\n\n{sec['fused_prompt']}"
    forced.append({"start": cur["start"], "end": cur["end"], "content": cur["content"]})
    return forced


def _overlap(a0, a1, b0, b1):  # seconds overlap
    return max(0.0, min(a1, b1) - max(a0, b0))

def _avg_weights_for_block(block, fusion_plan):
    # Average OCR/ASR weights across all 10-min bins that overlap this block
    spans = [(fp["start"], fp["end"], fp["w_ocr"], fp["w_asr"], fp["sim"]) for fp in fusion_plan]
    total, o_sum, a_sum, s_sum = 0.0, 0.0, 0.0, 0.0
    for (b0,b1,w_ocr,w_asr,sim) in spans:
        w = _overlap(block["start"], block["end"], b0, b1)
        if w > 0:
            total += w; o_sum += w * w_ocr; a_sum += w * w_asr; s_sum += w * sim
    if total == 0:
        return 0.5, 0.5, 0.0
    return o_sum/total, a_sum/total, s_sum/total

def _trim_short_and_merge_nearby(blocks):
    # Drop very tiny blocks and merge short gaps
    kept = []
    for b in blocks:
        if (b["end"] - b["start"]) >= CHAPTER_MIN_SEC:
            kept.append(b)
    if not kept:  # fallback: keep longest 1
        return [max(blocks, key=lambda x: x["end"]-x["start"])] if blocks else []

    merged = [kept[0]]
    for b in kept[1:]:
        prev = merged[-1]
        if b["start"] - prev["end"] <= CHAPTER_MERGE_GAP_SEC:
            prev["end"] = max(prev["end"], b["end"])
            prev["content"] += "\n\n---\n\n" + b["content"]
        else:
            merged.append(b)
    return merged

def _title_block_with_llm(text, global_topics=None):
    # Let Gemini/OpenAI create a tight title + 1–2 sentence summary
    topics_hint = f"；全域主題提示：{', '.join(global_topics[:8])}" if global_topics else ""
    prompt = f"""
你是一位課程章節命名助理。請根據下面的內容，產生一個**簡短明確的章節標題**（最多14字），
以及一段**1–2 句中文摘要**（不含行政雜訊、Zoom/點名等）。

內容：
{text[:8000]}

請輸出 JSON：
{{"title":"...", "summary":"..."}}
{topics_hint}
"""
    role = "你專精於從課程內容中產生清晰的章節標題與摘要。"
    resp = call_llm(prompt, role, max_tokens=CHAPTER_TITLE_MAXTOK)
    obj_raw = parse_qa_json(resp)

    obj = _coerce_obj_to_dict(obj_raw)
    title = (obj.get("title") or "").strip() or "章節"
    summary = (obj.get("summary") or "").strip()

    # --- guardrail for generic/low-coverage titles ---
    try:
        if _is_generic_title(title) or not _title_has_coverage(title, text):
            nudged = call_llm(
                f"以下內容的章節標題過於籠統或缺乏關鍵詞覆蓋。"
                f"請重寫為更具體的 6–10 字標題（避免行政用語）：\n\n{title}\n\n內容：\n{text[:4000]}",
                role="課程章節命名助手", max_tokens=200
            )
            title = (nudged or title).strip()[:32]
        
    except Exception:
        pass
    return title, summary

def _similarity_title(a, b):
    # lightweight similarity to dedup near-identical titles
    a_set = set(re.findall(r"[\u4e00-\u9fff]{1,}|\b[a-zA-Z0-9_]+\b", a.lower()))
    b_set = set(re.findall(r"[\u4e00-\u9fff]{1,}|\b[a-zA-Z0-9_]+\b", b.lower()))
    if not a_set or not b_set: return 0.0
    return len(a_set & b_set) / len(a_set | b_set)

def build_chapters_from_semantic_blocks(
    semantic_blocks, fusion_plan, canonical_topics, video_context
):
    if not semantic_blocks:
        return []

    # 1) tidy blocks (length + gap merge)
    blocks = _trim_short_and_merge_nearby(semantic_blocks)

    # 2) cap count if needed (keep longest)
    if len(blocks) > CHAPTER_MAX_COUNT:
        blocks = sorted(blocks, key=lambda x: (x["end"]-x["start"]), reverse=True)[:CHAPTER_MAX_COUNT]
        blocks = sorted(blocks, key=lambda x: x["start"])

    # 3) title+summary per block (LLM) + compute weights/confidence/topic align
    chapters = []
    global_topics = (video_context or {}).get("main_topics") or []

    # Title/Summary (prefer pre-distilled titles if present; avoids extra LLM call)
    for i, b in enumerate(blocks, 1):
        # Source weights / agreement
        w_ocr, w_asr, sim = _avg_weights_for_block(b, fusion_plan)
        # Title/Summary
        if b.get("ts_title"):
            title = b["ts_title"]
            summary = b.get("ts_summary", "")
        else:
            title, summary = _title_block_with_llm(strip_minor_admin(b["content"]), global_topics)

        # Topic id (best-aligned from bin alignment heuristic)
        topic_id = None
        if b.get("assigned_topic") and canonical_topics:
            # Find the canonical ID for this topic
            for topic in canonical_topics:
                if topic["title"] == b["assigned_topic"]:
                    topic_id = topic["id"]
                    break


        confidence = max(0.0, min(1.0, 0.35*sim + 0.25*w_ocr + 0.25*w_asr + 0.15*(1.0 if topic_id else 0.0)))

        chapters.append({
            "index": i,
            "start": float(b["start"]),
            "end": float(b["end"]),
            "title": title,
            "summary": summary,
            "topic_id": topic_id,       # aligns with canonical_topics from QA
            "confidence": round(confidence, 3),
            "sources": {"ocr_weight": round(w_ocr,3), "asr_weight": round(w_asr,3)}
        })

    # 4) dedup near-identical adjacent titles
    deduped = []
    for ch in chapters:
        if deduped and _similarity_title(deduped[-1]["title"], ch["title"]) >= CHAPTER_SIM_MERGE_THR:
            # merge into previous
            prev = deduped[-1]
            prev["end"] = ch["end"]
            prev["summary"] = prev["summary"] or ch["summary"]
            prev["confidence"] = round(0.5*(prev["confidence"]+ch["confidence"]), 3)
        else:
            deduped.append(ch)

    # make sure starts are strictly increasing & non-negative
    for idx, ch in enumerate(deduped, 1):
        ch["index"] = idx
        if idx > 1 and ch["start"] <= deduped[idx-2]["end"]:
            ch["start"] = deduped[idx-2]["end"] + 0.001

    return deduped


# ==================== Enhanced Video Processing Functions ====================
def safe_video_processing(video_path: str, cache_dir: str = "./video_cache", **kwargs) -> Optional[dict]:
    """
    Video processing with graceful fallback if models fail
    """
    try:
        logging.info("🎬 Starting optimized video processing...")
        return sequential_video_processing(video_path=video_path, cache_dir=cache_dir, **kwargs)
    except Exception as e:
        logging.warning(f"⚠️ Optimized processing failed: {e}")
        logging.info("🔄 Falling back to basic processing...")
        
        # Fallback to basic processing without AI models
        try:
            return sequential_video_processing(
                video_path=video_path, 
                cache_dir=cache_dir
                # No BLIP/CLIP models = OCR + audio only
            )
        except Exception as fallback_error:
            logging.error(f"❌ Both optimized and fallback processing failed: {fallback_error}")
            return None

# ==================== Processing Text for Q&A and Lecture Notes ====================
def process_text_for_qa_and_notes(
    audio_segments,
    ocr_segments,
    num_questions=10,
    num_pages=3,
    id=None,
    team_id=None,
    section_no=None,
    created_at=None,
    progress_callback: Optional[Callable[[str, int], None]] = None
):
    """
    Process audio and OCR segments to generate Q&A and lecture notes
    """
    logging.info(f"🔄 Processing for {num_questions} Q&A and {num_pages} pages of lecture notes...")

    combined_text = ""  # ensure defined even if segmentation fails

    if progress_callback:
        progress_callback("Starting Q&A generation...", 0)

    if not audio_segments and not ocr_segments:
        logging.warning("⚠️ No audio or OCR content available for processing.")
        return None
    
    # Step 1: Topic-sanity check
    if progress_callback:
        progress_callback("Analyzing content quality...", 10)
    
    # --- Build a truly global context first (Gemini can take a lot) ---
    global_context_text = build_global_context_corpus(
        
        ocr_segments=ocr_segments,
        audio_segments=audio_segments,
        prefer_ocr=True,
        max_chars=int(os.getenv("GLOBAL_CTX_MAX_CHARS", "300000"))
    )
    if progress_callback:
        progress_callback("Building global context from full OCR+ASR...", 15)

    video_context = generate_video_summary_and_topics(global_context_text) or {}
    if not isinstance(video_context, dict):
        video_context = {}
    video_summary = (video_context.get("summary") or "").strip()
    video_topics  = video_context.get("main_topics") or []

        
    ocr_text       = "\n".join(seg.get("text", "") for seg in ocr_segments)
    ocr_char_count = len(ocr_text)

    ocr_topics   = extract_main_topics(ocr_segments, mode="ocr", top_k=10)
    audio_topics = extract_main_topics(audio_segments, mode="audio", top_k=10)

    # Double-check OCR filtering inside QA module
    if ocr_segments:
        total_duration = get_total_duration(audio_segments, ocr_segments)
        if total_duration <= 0:
            # minimal guard so filter doesn’t drop everything on unknown duration
            total_duration = max((s.get("end", 0) for s in (audio_segments or [])), default=0) or 1.0
        ocr_segments = apply_ocr_filters(ocr_segments, total_video_duration=total_duration)

    # ==== NEW: Fusion over fixed 10-min windows ====

    # ==== NEW: Fusion over fixed 10-min windows ====
    total_duration = get_total_duration(audio_segments, ocr_segments)
    bins = build_bins(total_duration, window_sec=WINDOW_SEC)

    fusion_plan: List[Dict] = []
    fused_sections: List[Dict] = []
        
    # === Course-level TOC (modules) for soft guidance ===
    course_toc = []
    try:
        course_toc = build_course_toc(fusion_plan)
    except Exception as _e:
        logging.warning(f"⚠️ build_course_toc failed: {_e}")
    # Track which path actually produced chapters
    used_mode = None


    # === Semantic segmentation ===

    if progress_callback:
        progress_callback("Segmenting content...", 25)
    
    semantic_blocks = []

    if CHAPTER_ENGINE == "chapter-llama" and CL:
    # 1) boundaries
        from app.Providers.chapter_llama import ChapterLlamaPayload, Segment
        payload = ChapterLlamaPayload(
            audio_segments=[Segment(**s) for s in audio_segments],
            ocr_segments=[Segment(**s) for s in ocr_segments]
        ).model_dump()

        logging.info(f"Payload for ChapterLlama: {json.dumps(payload,  ensure_ascii=False)}")
        cl_bounds = CL.detect_boundaries(**payload)

        if cl_bounds:
        # 2) materialize blocks
            blocks = []
            for ch in cl_bounds:
                t0, t1 = ch["start_s"], ch["end_s"]
                a_bin = slice_by_time(audio_segments, t0, t1, key_start="start", key_end="end")
                v_bin = slice_by_time(ocr_segments,   t0, t1, key_start="timestamp", key_end="end")
                asr_txt = text_from_segments(a_bin, clean_asr_text, key="text")
                ocr_txt = text_from_segments(v_bin, clean_ocr_text, key="text")
                fused   = (asr_txt or "") + (("\n[SLIDE]\n" + ocr_txt) if ocr_txt else "")
                blocks.append({"start": t0, "end": t1, "content": fused})

            # 🆕 LIGHTWEIGHT TOPIC ASSIGNMENT (ADD THIS)
            # Extract global topics first
            global_topic_list = extract_global_topics_from_video(ocr_segments, audio_segments, max_topics=12)
        
            # Assign topics to each block using lightweight keyword matching
            for block in blocks:
                content = block["content"].lower()
                best_topic = None
                best_score = 0
            
                for topic in global_topic_list:
                    topic_lower = topic.lower()
                    # Simple keyword frequency scoring
                    score = content.count(topic_lower) * len(topic_lower)
                
                    if score > best_score:
                        best_score = score
                        best_topic = topic
            
                # Only assign if we have a reasonable match
                block["assigned_topic"] = best_topic if best_score > 10 else None

            # 3) titles/summaries (continue with existing code)
            titled = CL.title_chapters(blocks, language="zh-hant") or []

            # 3) titles/summaries
            titled = CL.title_chapters(blocks, language="zh-hant") or []
            titled_map = {(round(x["start"],3), round(x["end"],3)): x for x in titled}

        semantic_blocks = []
        for b in blocks:
            key = (round(b["start"],3), round(b["end"],3))
            tb  = titled_map.get(key, {})
            semantic_blocks.append({
                "start": b["start"],
                "end": b["end"],
                "content": b["content"],
                "ts_title": tb.get("title", ""),
                "ts_summary": tb.get("summary", ""),
                "_debug": {"mode": "chapter-llama", "cl_score": tb.get("score")}
            })
        used_mode = "chapter-llama"

   

    # --- Fall back to your existing modes if CL disabled or failed
    if not semantic_blocks and fused_sections:
        MODE = os.getenv("CHAPTER_MODE", "llm_titles").lower()
        if MODE == "cosine":
            semantic_blocks = semantic_segment_video_cosine(
                fused_sections,
                threshold=float(os.getenv("COSINE_FALLBACK_THR", "0.80"))
            )
            used_mode = "cosine"
        elif MODE == "hybrid":
            semantic_blocks = semantic_segment_video_hybrid(
                fused_sections, spike=float(os.getenv("HYBRID_SPIKE", "0.35"))
            )
            used_mode = "hybrid"
        else:
            ts_blocks = segment_bins_by_llm_titles(fusion_plan, video_topics)
            if ts_blocks:
                semantic_blocks = [{
                    "start": b["start"], "end": b["end"], "content": b["content"],
                    "ts_title": b["meta"]["title"], "ts_summary": b["meta"]["summary"],
                    "_debug": {"mode": "llm-title-grouping"}
                } for b in ts_blocks]
                used_mode = "llm_titles"       
            
    if not semantic_blocks and fused_sections:
        semantic_blocks = semantic_segment_video_cosine(fused_sections)
        used_mode = "cosine"
    
    if not fused_sections:
        logging.warning("⚠️ No fused sections available for segmentation.")

    
    # Ensure a minimum number of chapters via backoff & forced-time fallback
    if semantic_blocks and len(semantic_blocks) < MIN_CHAPTERS:
        # Try lowering the threshold progressively to force more splits
        try:
            used_thr = semantic_blocks[0].get("_debug", {}).get("used_threshold", None)
        except Exception:
            used_thr = None
        if used_thr is None:
            used_thr = float(os.getenv("COSINE_SPLIT_THRESHOLD", "0.80"))
        
        attempt = 0
        while len(semantic_blocks) < MIN_CHAPTERS and attempt < CHAPTER_BACKOFF_TRIES and used_thr > ADAPTIVE_MIN_T:
            used_thr = max(ADAPTIVE_MIN_T, used_thr - CHAPTER_BACKOFF_STEP)
            logging.info(f"🔁 Chapter backoff: trying threshold={used_thr:.2f}")
            semantic_blocks = semantic_segment_video_cosine(fused_sections, threshold=used_thr)
            attempt += 1

        # Final safety net: time-based chapters
        if len(semantic_blocks) < MIN_CHAPTERS:
            logging.info(f"⏱️ Forcing time-based chapters every {FORCED_CHAPTER_EVERY_SEC}s")
            semantic_blocks = _force_time_segments(fused_sections, every_sec=FORCED_CHAPTER_EVERY_SEC)

    # --- Relevance filtering of semantic blocks
    if semantic_blocks:
        cutoff = GLOBAL_RELEVANCE_CUTOFF
        mode = GLOBAL_RELEVANCE_MODE
        # Only force score-only if the chapters truly came from course_toc
        if used_mode == "course_toc":
            mode = "score"

        scored_blocks = []
        # richer anchors: global topics + module titles
        g_anchors = (video_topics or []) + [ (m.get("module_title") or "") for m in (course_toc or []) ]

        for b in semantic_blocks:
            k = extract_main_topics(b["content"], mode='combined', top_k=10)
            rel = topic_relevance_score(g_anchors, k)
            is_admin = looks_like_minor_admin(b["content"])
            # attach telemetry; keep chronological order
            b["_rel"] = {"score": round(rel, 3), "is_admin": bool(is_admin), "keywords": k}
            scored_blocks.append(b)

        if mode == "filter":
            kept = []
            for b in scored_blocks:
                if b["_rel"]["is_admin"] and DROP_ADMIN_BLOCKS:
                    continue
                if b["_rel"]["score"] >= cutoff or b["_rel"]["score"] >= RELEVANCE_SOFTFLOOR:
                    kept.append(b)
            
            # safety rails: keep top-N by relevance or by length to avoid gutting the video
            if (RELEVANCE_KEEP_TOP > 0) and len(kept) < RELEVANCE_KEEP_TOP:
                extra = sorted(
                    [x for x in scored_blocks if x not in kept and not (x["_rel"]["is_admin"] and DROP_ADMIN_BLOCKS)],
                    key=lambda x: (x["_rel"]["score"], x["end"] - x["start"]),
                    reverse=True
                )[:RELEVANCE_KEEP_TOP - len(kept)]
                kept.extend(extra)

            # final floor: never return < MIN_CHAPTERS
            if len(kept) < MIN_CHAPTERS:
                logging.info("⚠️ Relevance filtering would undercut chapters; falling back to score-only.")
                kept = scored_blocks  # keep order
            
            semantic_blocks = kept
        else:
            # score-only: keep everything; nothing is dropped
            semantic_blocks = scored_blocks
    
    logger.info(
        f"📚 Chaptering mode used: {used_mode or 'none'} | "
        f"CHAPTER_ENGINE={CHAPTER_ENGINE} | "
        f"CHAPTER_MODE={os.getenv('CHAPTER_MODE','llm_titles')} | "
        f"blocks={len(semantic_blocks)}"
    )

    # --- Now finalize combined_text from filtered blocks (preferred path)
    if semantic_blocks:
        combined_text = "\n\n".join(
            f"## ⏱️ Block {i+1} [{int(b['start'])}-{int(b['end'])}s]\n{b['content']}"
            for i, b in enumerate(semantic_blocks)
        )
    else:
        # sensible fallbacks to avoid empty context
        if fused_sections:
            combined_text = "\n\n".join(
                f"## ⏱️ Bin {i+1} [{int(s['start'])}-{int(s['end'])}s]\n{s['fused_prompt']}"
                for i, s in enumerate(fused_sections)
            )
        elif 'global_context_text' in locals() and global_context_text:
            combined_text = global_context_text  # last resort

    # === Derive topics (prefer course_toc modules if available) ===

    # Extract global topics ONCE
    global_topic_list = extract_global_topics_from_video(ocr_segments, audio_segments, max_topics=12)

    # Convert to canonical format
    canonical_topics = canonicalize_topics(
        [{"title": topic, "summary": f"Content about {topic}"} for topic in global_topic_list]
    )

    # Lightweight assignment to chapters
    if semantic_blocks:
        semantic_blocks = assign_topics_to_chapters_lightweight(semantic_blocks, global_topic_list)
 
    else:
        topics = [{"title": t, "summary": t} for t in (video_topics[:8] if video_topics else [])]

    # Ensure non-empty fallback (keep your existing safety)
    if not topics:
        fallback_keys = topics_from_text_block(combined_text, top_k=6) if (combined_text or "").strip() else []
        if fallback_keys:
            topics = [{"title": ", ".join(fallback_keys[:3]), "summary": "、".join(fallback_keys)}]

    # === Canonicalize topics & align bins to topics (NEW) ===
    canonical_topics = canonicalize_topics(topics)
    # === Chapters (same fused data as Q&A/notes) ===
    if progress_callback:
        progress_callback("Building chapters...", 45)

    chapters = build_chapters_from_semantic_blocks(
    
        semantic_blocks=semantic_blocks,
        fusion_plan=fusion_plan,
        canonical_topics=canonical_topics,
        video_context=video_context
    )


    

    # Step 3: Q&A generation
    if progress_callback:
        progress_callback("Creating Q&A questions...", 50)

    if not (combined_text or "").strip():
        logging.error("❌ Combined text is empty after filtering; cannot generate Q&A.")
        return None
        
    qa_json = generate_qa_from_text(
        combined_text,
        canonical_topics,           # use canonical topics for consistent vocabulary
        num_questions,
        video_context=video_context
    )

    if not qa_json:
        logging.warning("❌ Q&A generation failed.")
        return None
    
    # NEW: sanitize options per question to ensure 4 unique, non-empty choices
    for q in qa_json:
        _sanitize_options(q)

    qa_json = enforce_difficulty_distribution(qa_json)
    # Optional determinism in CI: set QA_SHUFFLE_SEED in env (e.g., 42)
    # FIX
    _seed_val = None
    _seed_env = os.getenv("QA_SHUFFLE_SEED")
    if _seed_env is not None:
        try:
            _seed_val = int(_seed_env)
        except ValueError:
            _seed_val = None
    shuffle_question_options(qa_json, regenerate_explanation=True, seed=_seed_val)

    # Step 4: Lecture notes generation
    if progress_callback:
        progress_callback("Generating lecture notes...", 75)
        
    lecture_data = generate_lecture_notes(
    
        combined_text,
        canonical_topics,           # use canonical topics here too
        num_pages,
        video_context=video_context
    )

    if not lecture_data:
        logging.warning("❌ Lecture notes generation failed.")
        return None
    
    # ==================== INSERT NEW CODE HERE ====================
    # Step 4.5: Generate Key Concepts and Learning Objectives
    if progress_callback:
        progress_callback("Generating key concepts and learning objectives...", 85)

    key_concepts_and_objectives = generate_key_concepts_and_objectives(combined_text)
    if not key_concepts_and_objectives:
        logging.warning("❌ Key concepts and objectives generation failed. Continuing without them.")

    # Step 5: Final packaging
    if progress_callback:
        progress_callback("Finalizing results...", 90)
        
    final_payload = qa_text_to_json(
        qa_content=qa_json,
        id=id,
        team_id=team_id,
        section_no=section_no,
        created_at=created_at,
        course_note=lecture_data
    )

    # ✅ Now it's safe to attach chapters (final_payload exists)
    final_payload['chapters'] = chapters


    # Add the new data to the payload
    if key_concepts_and_objectives:
        final_payload['key_concepts_and_objectives'] = key_concepts_and_objectives
    

    # Optionally attach per-bin distilled titles for inspection (uses cache; cheap)

    bin_titles = []
    # make an index → module_title map for quick lookup
    _bin_to_module = {}
    for m in (course_toc or []):
        for i in (m.get("bins") or []):
            _bin_to_module[i] = m.get("module_title","")

    for fp in fusion_plan:
        k = fp["bin_index"]
        if BIN_TITLE_USE_CONTEXT:
            ts = title_summary_for_bin_ctx(
                fusion_plan, k,
                module_hint=_bin_to_module.get(k, "")
            )
        else:
            ts = title_summary_for_bin(fp["fused_prompt"])
        
        bin_titles.append({
            "bin_index": k,
            "start": fp["start"],
            "end": fp["end"],
            "title": ts["title"],
            "summary": ts["summary"]
         })
        
    # === Attach shared alignment truth for Chapters/Notes/QA (NEW) ===
    final_payload.setdefault("alignment", {})
    final_payload["alignment"].update({
        "video_context": video_context or {},
        "canonical_topics": canonical_topics,       # [{id,title,summary}]
        "fusion_plan": fusion_plan,                 # per 10-min bin details
        "window_sec": WINDOW_SEC,
        "chapters_timeline": format_topic_timeline(chapters),  # ← add this
        "chapter_mode_used": used_mode, 
    })
    final_payload["alignment"]["bin_titles"] = bin_titles
    final_payload["alignment"]["course_toc"] = course_toc  # [{bins, module_title, module_summary}]
    final_payload['bin_titles'] = bin_titles  # ← top-level mirror for downstream consumers


    if progress_callback:
        progress_callback("Complete!", 100)

    logging.info("✅ Q&A and Lecture Notes generation complete. Returning final payload.")
    return final_payload

# ==================== Complete Pipeline Integration ====================
def complete_video_to_qa_pipeline(
    video_path: str, 
    num_questions: int = 10, 
    num_pages: int = 3,
    id: Optional[str] = None, 
    team_id: Optional[str] = None, 
    section_no: Optional[int] = None, 
    created_at: Optional[str] = None,
    cache_dir: str = "./video_cache",
    progress_callback: Optional[Callable[[str, int], None]] = None,
    # Optional AI models for enhanced processing
    blip_processor=None,
    blip_model=None,
    clip_processor=None,
    clip_model=None,
    predefined_topics=None
) -> Optional[dict]:
    """
    Complete pipeline: Optimized Video Processing → Q&A Generation → Lecture Notes
    
    Args:
        video_path: Path to the video file
        num_questions: Number of Q&A questions to generate
        num_pages: Number of pages for lecture notes
        cache_dir: Directory for caching intermediate results
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Complete result dictionary with video processing, Q&A, and lecture notes
    """
    try:
        start_time = time.time()
        
        if progress_callback:
            progress_callback("Starting video processing...", 0)
        
        # Step 1: Process video with all optimizations
        logging.info("🎬 Starting complete video-to-Q&A pipeline...")
        video_result = safe_video_processing(
            video_path=video_path,
            cache_dir=cache_dir,
            blip_processor=blip_processor,
            blip_model=blip_model,
            clip_processor=clip_processor,
            clip_model=clip_model,
            predefined_topics=predefined_topics
        )
        
        if not video_result:
            logging.error("❌ Video processing failed")
            return None
        
        video_processing_time = video_result.get('processing_time', 0)
        logging.info(f"✅ Video processing completed in {video_processing_time:.1f}s")
        
        if progress_callback:
            progress_callback("Video processing complete, starting Q&A generation...", 30)
        
        # Step 2: Extract data for Q&A pipeline
        audio_segments = video_result['audio_transcription']
        ocr_segments = video_result['video_analysis']
        
        logging.info(f"📊 Processing {len(audio_segments)} audio blocks, {len(ocr_segments)} OCR segments")
        
        # Step 3: Generate Q&A and notes with progress tracking
        def qa_progress(message, percent):
            # Scale Q&A progress from 30% to 100%
            scaled_percent = 30 + (percent * 0.7)
            if progress_callback:
                progress_callback(message, int(scaled_percent))
        
        qa_result = process_text_for_qa_and_notes(
            audio_segments=audio_segments,
            ocr_segments=ocr_segments,
            num_questions=num_questions,
            num_pages=num_pages,
            id=id,
            team_id=team_id,
            section_no=section_no,
            created_at=created_at,
            progress_callback=qa_progress
        )
        
        if not qa_result:
            logging.error("❌ Q&A generation failed")
            return None
        
        total_time = time.time() - start_time

        def _count_questions(payload: dict) -> int:
            if not isinstance(payload, dict):
                return 0
            
            # Common keys we’ve seen in the wild
            for key in ("questions", "qa", "items", "data"):
                v = payload.get(key)
                if isinstance(v, list):
                    return len(v)
            # Nested fallback: sometimes payload nests under e.g. {"qa_and_notes": {...}}
            for v in payload.values():
                if isinstance(v, dict):
                    c = _count_questions(v)
                    if c:
                        return c
            return 0
        q_count = _count_questions(qa_result)    
        
        # Step 4: Combine all results
        final_result = {
            'success': True,
            'video_processing': video_result,
            'qa_and_notes': qa_result,
            'summary': {
                'total_processing_time': total_time,
                'video_processing_time': video_processing_time,
                'audio_blocks': len(audio_segments),
                'ocr_segments': len(ocr_segments),
                'questions_generated': q_count,
                'lecture_notes_pages': num_pages,
                'cache_used': video_processing_time < 10  # Likely cache hit if very fast
            },
            'pipeline_info': {
                'video_path': video_path,
                'optimizations_used': [
                    'sequential_chunked_asr',
                    'intelligent_caching', 
                    'smart_frame_selection',
                    'ocr_audio_balance'
                ]
            }
        }
        
        if progress_callback:
            progress_callback("Pipeline complete!", 100)
        
        logging.info(f"🎉 Complete pipeline finished successfully in {total_time:.1f}s!")
        logging.info(f"📈 Generated {q_count} questions and {num_pages} pages of notes")
        
        return final_result
        
    except Exception as e:
        logging.error(f"❌ Pipeline failed: {e}")
        if progress_callback:
            progress_callback(f"Pipeline failed: {str(e)}", -1)
        return None

# ==================== Backwards Compatibility Functions ====================
def process_video_for_qa(
    video_path: str,
    num_questions: int = 10,
    num_pages: int = 3,
    **kwargs
) -> Optional[dict]:
    """
    Backwards compatible function that maintains original interface
    """
    return complete_video_to_qa_pipeline(
        video_path=video_path,
        num_questions=num_questions,
        num_pages=num_pages,
        **kwargs
    )

# ==================== Pipeline Integration Ready ====================
if __name__ == "__main__":
    import sys

    # Paths to your JSON files
    audio_path = "D:/Codes/Class_Genius-best-safe/app/runs/input_video_20250827-192651_audio_10min.json"
    ocr_path = "D:/Codes/Class_Genius-best-safe/app/runs/input_video_20250814-173650_ocr_raw.json"

    # Load JSON files
    with open(audio_path, "r", encoding="utf-8") as f:
        audio_segments = json.load(f)
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_segments = json.load(f)

    # Make sure they’re lists of dicts
    if not isinstance(audio_segments, list) or not isinstance(ocr_segments, list):
        print("❌ audio_5min.json or ocr_filtered.json is not in expected list-of-dicts format.")
        sys.exit(1)

    # Run the processing
    result = process_text_for_qa_and_notes(
        audio_segments=audio_segments,
        ocr_segments=ocr_segments,
        num_questions=10,    # adjust as needed
        num_pages=3,         # adjust as needed
        id="test123",
        team_id="teamX",
        section_no=1,
        created_at="2025-08-10T12:00:00Z",
        progress_callback=lambda msg, pct: print(f"[{pct}%] {msg}")
    )

    # Save the result for inspection
    if result:
        with open("qa_and_notes_output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("✅ Q&A and lecture notes saved to qa_and_notes_output.json")
    else:
        print("❌ Processing failed.")
