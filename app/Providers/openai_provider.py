# app/Providers/openai_provider.py
from typing import List, Optional
from openai import OpenAI

from .llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    OpenAI-based LLM provider (chat + embeddings) using the openai>=1.x SDK.
    """

    def __init__(
        self,
        api_key: str,
        chat_model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-large",
    ):
        # The client will also read OPENAI_API_KEY from env; passing explicitly is fine.
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        images: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a chat completion. (Images are ignored in this text-only path.)
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.25,
                max_tokens=max_tokens,
            )
            # Defensive access in case of empty choices
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            # Return an empty string rather than raising to keep pipelines alive
            return f""

    def embed(self, text: str) -> Optional[List[float]]:
        """
        Create a single embedding vector for `text`. Returns None on failure.
        """
        try:
            r = self.client.embeddings.create(model=self.embed_model, input=text)
            return r.data[0].embedding
        except Exception:
            return None
