# app/Providers/llm_provider.py
"""
Base abstract class for Large Language Model (LLM) providers.
All providers (e.g., OpenAI, Gemini) should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class LLMProvider(ABC):
    """
    Abstract base class for LLM provider implementations.
    """

    @abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        images: Optional[List[str]] = None
    ) -> str:
        """
        Generate a chat response from the LLM.

        Args:
            system: System-level instructions for the LLM.
            user: The user prompt.
            max_tokens: The maximum number of tokens for the response.
            images: Optional list of image URLs/paths for multimodal input.

        Returns:
            The generated response text.
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding vector for the given text.

        Args:
            text: Input text to embed.

        Returns:
            A list of floats representing the embedding vector,
            or None if embedding fails.
        """
        raise NotImplementedError
