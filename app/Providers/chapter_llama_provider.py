# app/Providers/chapter_llama_provider.py
import os, logging, json, gzip, time
from io import BytesIO
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Optional

def _clean_ocr_text(s: str) -> str:
    if not s: return ""
    import re
    s = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", " ", s)
    s = re.sub(r"\b(?:\d{4}/\d{1,2}/\d{1,2}|202\d-\d{1,2}-\d{1,2})\b", " ", s)
    s = re.sub(r"[℃°]\s*\S*", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def _clean_asr_text(s: str) -> str:
    if not s: return ""
    import re
    s = re.sub(r"(字幕由\s*Amara\.org\s*社群提供\s*){2,}", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def _slice_by_time(segments, t0, t1, key_start="start", key_end="end"):
    if not segments: return []
    out = []
    for s in segments:
        s0 = s.get(key_start, s.get("timestamp", 0.0)) or 0.0
        s1 = s.get(key_end, s0) or s0
        if s1 >= t0 and s0 <= t1:
            out.append(s)
    return out

def _text_from_segments(segments, cleaner, key="text"):
    if not segments: return ""
    txt = "\n".join((s.get(key, "") or "").strip() for s in segments)
    return cleaner(txt)

class ChapterLlamaProvider:
    """
    Same public API as before, but with hardened HTTP:
      - retries/backoff via requests.Session + Retry
      - gzip request bodies
      - optional Bearer auth
      - circuit breaker to avoid hammering a down server
      - /health probe
    Still does chunking/fusion of ASR+OCR windows.
    """
    def __init__(self, base_url: Optional[str] = None, timeout: int = 60,
                 api_key: Optional[str] = None, retries:int = 3, backoff: float = 0.3):
        self.base_url = (base_url or os.getenv("CHAPTER_LLAMA_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout
        self.api_key = api_key or os.getenv("CHAPTER_LLAMA_API_KEY") or None

        self.sess = requests.Session()
        retry = Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("POST", "GET"),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=64)
        self.sess.mount("http://", adapter)
        self.sess.mount("https://", adapter)

        self._fails = 0
        self._circuit_until = 0.0

    # ---------- HTTP ----------
    def _headers(self):
        h = {"User-Agent": "CG-CL-Provider/1.0", "Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, route: str, payload: dict) -> Optional[dict]:
        # circuit breaker
        if time.time() < self._circuit_until:
            logging.warning("Chapter-Llama circuit open; skipping call %s", route)
            return None

        # gzip the json payload
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        buf = BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(raw)
        data = buf.getvalue()
        headers = {
            **self._headers(),
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "Accept-Encoding": "gzip",
        }

        try:
            r = self.sess.post(self.base_url + route, data=data, headers=headers, timeout=(3, self.timeout))
            if r.status_code >= 500:
                self._register("server")
                logging.warning("Chapter-Llama %s -> %s", route, r.status_code)
                return None
            if r.status_code in (401, 403, 404):
                self._register("auth/404")
                logging.warning("Chapter-Llama %s -> %s", route, r.status_code)
                return None
            self._reset()
            return r.json()
        except Exception as e:
            self._register("net")
            logging.warning("⚠️ Chapter-Llama request failed %s: %s", route, e)
            return None

    def _register(self, reason: str):
        self._fails += 1
        if self._fails >= 5:
            self._circuit_until = time.time() + 120
            logging.error("Chapter-Llama circuit opened for 120s due to %s failures", reason)

    def _reset(self):
        self._fails = 0
        self._circuit_until = 0.0

    def health(self) -> bool:
        try:
            r = self.sess.get(self.base_url + "/health", headers=self._headers(), timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ---------- Chunking ----------
    def build_chunks(self, audio_segments: List[Dict], ocr_segments: List[Dict],
                     win_sec: int, overlap_sec: int) -> List[List]:
        total = self._total_duration(audio_segments, ocr_segments)
        if total <= 0: return []
        chunks, t0 = [], 0.0
        step = max(1, win_sec - overlap_sec)
        while t0 < total:
            t1 = min(t0 + win_sec, total)
            a_bin = _slice_by_time(audio_segments, t0, t1, key_start="start", key_end="end")
            v_bin = _slice_by_time(ocr_segments,   t0, t1, key_start="timestamp", key_end="end")
            asr_txt = _text_from_segments(a_bin, _clean_asr_text, key="text")
            ocr_txt = _text_from_segments(v_bin, _clean_ocr_text, key="text")
            fused   = self._fuse(asr_txt, ocr_txt)
            if fused.strip():
                chunks.append([round(t0,3), round(t1,3), fused])
            t0 += step
        return chunks

    @staticmethod
    def _fuse(asr_txt: str, ocr_txt: str) -> str:
        return (asr_txt or "") + (("\n[SLIDE]\n" + (ocr_txt or "")) if ocr_txt else "")

    @staticmethod
    def _total_duration(audio_segments, ocr_segments) -> float:
        a_end = max((seg.get("end", 0.0) for seg in (audio_segments or [])), default=0.0)
        v_end = max((seg.get("end", seg.get("timestamp", 0.0)) for seg in (ocr_segments or [])), default=0.0)
        return max(a_end, v_end)

    # ---------- Public API ----------
    def detect_boundaries(self, audio_segments: List[Dict], ocr_segments: List[Dict],
                          win_sec: int, overlap_sec: int, language: str = "zh-hant") -> Optional[List[Dict]]:
        chunks = self.build_chunks(audio_segments, ocr_segments, win_sec, overlap_sec)
        if not chunks:
            return None
        resp = self._post("/v1/chapter/boundaries", {"chunks": chunks, "language": language})
        if not resp or "chapters" not in resp:
            return None
        out = []
        for c in resp["chapters"]:
            try:
                s, e = float(c["start_s"]), float(c["end_s"])
                if e > s:
                    out.append({
                        "start_s": s,
                        "end_s": e,
                        "reason": (c.get("reason") or "").strip(),
                        "score": float(c.get("score", 0.0))
                    })
            except Exception:
                pass
        out.sort(key=lambda x: (x["start_s"], x["end_s"]))
        dedup = []
        for c in out:
            if not dedup or (c["start_s"] != dedup[-1]["start_s"] or c["end_s"] != dedup[-1]["end_s"]):
                dedup.append(c)
        return dedup

    def title_chapters(self, blocks: List[Dict], language: str = "zh-hant",
                       max_chars_per_block: int = 2000) -> Optional[List[Dict]]:
        compact = []
        for b in blocks:
            txt = (b.get("content") or "").strip()
            if len(txt) > max_chars_per_block:
                txt = txt[:max_chars_per_block]
            compact.append({"start_s": float(b["start"]), "end_s": float(b["end"]), "text": txt})

        resp = self._post("/v1/chapter/titles", {"chapters": compact, "language": language})
        if not resp or "chapters" not in resp:
            return None
        out = []
        for c in resp["chapters"]:
            out.append({
                "start": float(c.get("start_s", 0.0)),
                "end": float(c.get("end_s", 0.0)),
                "title": (c.get("title") or "").strip(),
                "summary": (c.get("summary") or "").strip(),
                "score": float(c.get("score", 0.0))
            })
        return out
