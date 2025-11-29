"""Helper for interacting with Google Gemini models."""

from __future__ import annotations

import logging
from typing import Optional

try:  # pragma: no cover - optional dependency checked at runtime
    import google.generativeai as genai
except ImportError:  # pragma: no cover - handled by raising at runtime
    genai = None


class GeminiProcessorError(RuntimeError):
    """Raised when Gemini processing fails."""


class GeminiProcessor:
    """Wrapper around the Google Gemini SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if genai is None:
            raise GeminiProcessorError(
                "google-generativeai is not installed. Please install it to use Gemini features."
            )

        if not api_key:
            raise GeminiProcessorError("Gemini API key is required")

        self.model_name = model_name or "gemini-1.5-flash"
        self._configure(api_key)

    def _configure(self, api_key: str) -> None:
        """Configure SDK and initialize model."""

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as exc:  # pragma: no cover - network failure
            logging.error("Failed to initialize Gemini model: %s", exc, exc_info=True)
            raise GeminiProcessorError(f"Failed to initialize Gemini model: {exc}") from exc

    def generate_text(self, prompt: str) -> str:
        """Send prompt to Gemini and return the text response."""

        if not prompt.strip():
            raise GeminiProcessorError("Prompt must not be empty")

        try:
            response = self.model.generate_content(prompt)
        except Exception as exc:  # pragma: no cover - network failure
            logging.error("Gemini API error: %s", exc, exc_info=True)
            raise GeminiProcessorError(f"Gemini API error: {exc}") from exc

        text = self._extract_text(response)
        if not text:
            raise GeminiProcessorError("Gemini response did not contain any text")

        return text.strip()

    @staticmethod
    def _extract_text(response) -> Optional[str]:
        """Extract text from Gemini response, handling different payloads."""

        text = getattr(response, "text", None)
        if text:
            return text

        candidates = getattr(response, "candidates", None) or []
        parts = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)

        if parts:
            return "\n\n".join(parts)

        return None

    @classmethod
    def list_available_models(cls, api_key: str) -> list[str]:
        """Return list of Gemini models that support text generation."""

        if genai is None:
            raise GeminiProcessorError(
                "google-generativeai is not installed. Please install it to use Gemini features."
            )

        if not api_key:
            raise GeminiProcessorError("Gemini API key is required")

        try:
            genai.configure(api_key=api_key)
            models = genai.list_models()
        except Exception as exc:  # pragma: no cover - network failure
            logging.error("Failed to list Gemini models: %s", exc, exc_info=True)
            raise GeminiProcessorError(f"Failed to list Gemini models: {exc}") from exc

        available = []
        for model in models:
            supported = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in supported:
                available.append(model.name)

        if not available:
            raise GeminiProcessorError("No Gemini models supporting text generation were found")

        return available
