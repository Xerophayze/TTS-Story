"""Helper for interacting with Google Gemini models."""

from __future__ import annotations

import logging
from typing import Optional

# Try new SDK first, fall back to deprecated one
try:  # pragma: no cover - optional dependency checked at runtime
    from google import genai
    USING_NEW_SDK = True
except ImportError:  # pragma: no cover - try legacy SDK
    try:
        import google.generativeai as genai
        USING_NEW_SDK = False
    except ImportError:
        genai = None
        USING_NEW_SDK = False


class GeminiProcessorError(RuntimeError):
    """Raised when Gemini processing fails."""


class GeminiProcessor:
    """Wrapper around the Google Gemini SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if genai is None:
            raise GeminiProcessorError(
                "google-genai is not installed. Please install it to use Gemini features: pip install google-genai"
            )

        if not api_key:
            raise GeminiProcessorError("Gemini API key is required")

        self.model_name = model_name or "gemini-1.5-flash"
        self.api_key = api_key
        self._configure(api_key)

    def _configure(self, api_key: str) -> None:
        """Configure SDK and initialize client."""
        try:
            if USING_NEW_SDK:
                self.client = genai.Client(api_key=api_key)
            else:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(self.model_name)
        except Exception as exc:  # pragma: no cover - network failure
            logging.error("Failed to initialize Gemini: %s", exc, exc_info=True)
            raise GeminiProcessorError(f"Failed to initialize Gemini: {exc}") from exc

    def generate_text(self, prompt: str) -> str:
        """Send prompt to Gemini and return the text response."""

        if not prompt.strip():
            raise GeminiProcessorError("Prompt must not be empty")

        try:
            if USING_NEW_SDK:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
            else:
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
                "google-genai is not installed. Please install it to use Gemini features: pip install google-genai"
            )

        if not api_key:
            raise GeminiProcessorError("Gemini API key is required")

        try:
            if USING_NEW_SDK:
                client = genai.Client(api_key=api_key)
                models = client.models.list()
                available = []
                for model in models:
                    # New SDK returns model objects with name attribute
                    name = getattr(model, "name", None)
                    if name:
                        available.append(name)
            else:
                genai.configure(api_key=api_key)
                models = genai.list_models()
                available = []
                for model in models:
                    supported = getattr(model, "supported_generation_methods", []) or []
                    if "generateContent" in supported:
                        available.append(model.name)
        except Exception as exc:  # pragma: no cover - network failure
            logging.error("Failed to list Gemini models: %s", exc, exc_info=True)
            raise GeminiProcessorError(f"Failed to list Gemini models: {exc}") from exc

        if not available:
            raise GeminiProcessorError("No Gemini models supporting text generation were found")

        return available
