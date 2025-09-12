# app/Providers/gemini_provider.py
from typing import List, Optional
import google.generativeai as genai

from .llm_provider import LLMProvider



class GeminiProvider(LLMProvider):
    """
    Google Gemini-based LLM provider.
    Implements chat and embedding capabilities using the google-generativeai SDK.
    """

    def __init__(
        self,
        api_key: str,
        chat_model: str = "gemini-1.5-pro",
        embed_model: str = "text-embedding-004",
    ):
        genai.configure(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.model = genai.GenerativeModel(chat_model)

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        images: Optional[List[str]] = None
    ) -> str:
        """
        Generate a chat response from Gemini.
        Images are ignored for now; system prompt is prepended to the user prompt.
        """
        parts = [
            {"text": f"[SYSTEM]\n{system}"},
            {"text": user}
        ]
        resp = self.model.generate_content(
            parts,
            generation_config={
                "temperature": 0.25,
                "max_output_tokens": max_tokens,
            },
        )
        return (resp.text or "").strip()

    def embed(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding vector using Gemini.
        """
        try:
            out = genai.embed_content(model=self.embed_model, content=text)
            return out.get("embedding")
        except Exception:
            return None
