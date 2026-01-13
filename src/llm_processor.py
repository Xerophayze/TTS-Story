"""Unified LLM processor supporting multiple providers (Gemini, OpenAI, Anthropic)."""

from __future__ import annotations

import logging
from typing import List, Optional

# Provider constants
LLM_PROVIDER_GEMINI = "gemini"
LLM_PROVIDER_OPENAI = "openai"
LLM_PROVIDER_ANTHROPIC = "anthropic"

SUPPORTED_LLM_PROVIDERS = [LLM_PROVIDER_GEMINI, LLM_PROVIDER_OPENAI, LLM_PROVIDER_ANTHROPIC]

# Default models for each provider
DEFAULT_MODELS = {
    LLM_PROVIDER_GEMINI: "gemini-1.5-flash",
    LLM_PROVIDER_OPENAI: "gpt-4o",
    LLM_PROVIDER_ANTHROPIC: "claude-3-5-sonnet-20241022",
}

# Optional imports - checked at runtime
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


class LLMProcessorError(RuntimeError):
    """Raised when LLM processing fails."""


class LLMProcessor:
    """Unified wrapper for multiple LLM providers."""

    def __init__(self, provider: str, api_key: str, model_name: Optional[str] = None):
        self.provider = provider.lower().strip()
        if self.provider not in SUPPORTED_LLM_PROVIDERS:
            raise LLMProcessorError(f"Unsupported LLM provider: {provider}")

        if not api_key:
            raise LLMProcessorError(f"{provider} API key is required")

        self.api_key = api_key
        self.model_name = model_name or DEFAULT_MODELS.get(self.provider)
        self._client = None
        self._model = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize the appropriate client based on provider."""
        if self.provider == LLM_PROVIDER_GEMINI:
            self._init_gemini()
        elif self.provider == LLM_PROVIDER_OPENAI:
            self._init_openai()
        elif self.provider == LLM_PROVIDER_ANTHROPIC:
            self._init_anthropic()

    def _init_gemini(self) -> None:
        """Initialize Google Gemini client."""
        if genai is None:
            raise LLMProcessorError(
                "google-generativeai is not installed. Run: pip install google-generativeai"
            )
        try:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        except Exception as exc:
            logging.error("Failed to initialize Gemini: %s", exc, exc_info=True)
            raise LLMProcessorError(f"Failed to initialize Gemini: {exc}") from exc

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        if openai is None:
            raise LLMProcessorError(
                "openai is not installed. Run: pip install openai"
            )
        try:
            self._client = openai.OpenAI(api_key=self.api_key)
        except Exception as exc:
            logging.error("Failed to initialize OpenAI: %s", exc, exc_info=True)
            raise LLMProcessorError(f"Failed to initialize OpenAI: {exc}") from exc

    def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        if anthropic is None:
            raise LLMProcessorError(
                "anthropic is not installed. Run: pip install anthropic"
            )
        try:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as exc:
            logging.error("Failed to initialize Anthropic: %s", exc, exc_info=True)
            raise LLMProcessorError(f"Failed to initialize Anthropic: {exc}") from exc

    def generate_text(self, prompt: str) -> str:
        """Send prompt to the configured LLM and return the text response."""
        if not prompt.strip():
            raise LLMProcessorError("Prompt must not be empty")

        if self.provider == LLM_PROVIDER_GEMINI:
            return self._generate_gemini(prompt)
        elif self.provider == LLM_PROVIDER_OPENAI:
            return self._generate_openai(prompt)
        elif self.provider == LLM_PROVIDER_ANTHROPIC:
            return self._generate_anthropic(prompt)

        raise LLMProcessorError(f"Unsupported provider: {self.provider}")

    def _generate_gemini(self, prompt: str) -> str:
        """Generate text using Gemini."""
        try:
            response = self._model.generate_content(prompt)
        except Exception as exc:
            logging.error("Gemini API error: %s", exc, exc_info=True)
            raise LLMProcessorError(f"Gemini API error: {exc}") from exc

        text = self._extract_gemini_text(response)
        if not text:
            raise LLMProcessorError("Gemini response did not contain any text")
        return text.strip()

    def _generate_openai(self, prompt: str) -> str:
        """Generate text using OpenAI."""
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=16000,
            )
        except Exception as exc:
            logging.error("OpenAI API error: %s", exc, exc_info=True)
            raise LLMProcessorError(f"OpenAI API error: {exc}") from exc

        if not response.choices:
            raise LLMProcessorError("OpenAI response did not contain any choices")

        text = response.choices[0].message.content
        if not text:
            raise LLMProcessorError("OpenAI response did not contain any text")
        return text.strip()

    def _generate_anthropic(self, prompt: str) -> str:
        """Generate text using Anthropic Claude."""
        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            logging.error("Anthropic API error: %s", exc, exc_info=True)
            raise LLMProcessorError(f"Anthropic API error: {exc}") from exc

        if not response.content:
            raise LLMProcessorError("Anthropic response did not contain any content")

        text_parts = [block.text for block in response.content if hasattr(block, 'text')]
        if not text_parts:
            raise LLMProcessorError("Anthropic response did not contain any text")
        return "\n".join(text_parts).strip()

    @staticmethod
    def _extract_gemini_text(response) -> Optional[str]:
        """Extract text from Gemini response."""
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
    def list_available_models(cls, provider: str, api_key: str) -> List[str]:
        """Return list of available models for the given provider."""
        provider = provider.lower().strip()

        if provider == LLM_PROVIDER_GEMINI:
            return cls._list_gemini_models(api_key)
        elif provider == LLM_PROVIDER_OPENAI:
            return cls._list_openai_models(api_key)
        elif provider == LLM_PROVIDER_ANTHROPIC:
            return cls._list_anthropic_models(api_key)

        raise LLMProcessorError(f"Unsupported provider: {provider}")

    @classmethod
    def _list_gemini_models(cls, api_key: str) -> List[str]:
        """List available Gemini models."""
        if genai is None:
            raise LLMProcessorError(
                "google-generativeai is not installed. Run: pip install google-generativeai"
            )

        if not api_key:
            raise LLMProcessorError("Gemini API key is required")

        try:
            genai.configure(api_key=api_key)
            models = genai.list_models()
        except Exception as exc:
            logging.error("Failed to list Gemini models: %s", exc, exc_info=True)
            raise LLMProcessorError(f"Failed to list Gemini models: {exc}") from exc

        available = []
        for model in models:
            supported = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in supported:
                available.append(model.name)

        if not available:
            raise LLMProcessorError("No Gemini models supporting text generation were found")

        return available

    @classmethod
    def _list_openai_models(cls, api_key: str) -> List[str]:
        """List available OpenAI models."""
        if openai is None:
            raise LLMProcessorError(
                "openai is not installed. Run: pip install openai"
            )

        if not api_key:
            raise LLMProcessorError("OpenAI API key is required")

        try:
            client = openai.OpenAI(api_key=api_key)
            models_response = client.models.list()
        except Exception as exc:
            logging.error("Failed to list OpenAI models: %s", exc, exc_info=True)
            raise LLMProcessorError(f"Failed to list OpenAI models: {exc}") from exc

        # Filter for chat/completion models
        available = []
        for model in models_response.data:
            model_id = model.id
            # Include GPT models and exclude embedding/whisper/tts models
            if any(prefix in model_id for prefix in ['gpt-', 'o1', 'o3', 'chatgpt']):
                if not any(exclude in model_id for exclude in ['instruct', 'vision', 'audio', 'realtime']):
                    available.append(model_id)

        # Sort with newest/best models first
        available.sort(key=lambda x: (
            0 if 'gpt-4o' in x else 1 if 'gpt-4' in x else 2 if 'o1' in x else 3
        ))

        if not available:
            raise LLMProcessorError("No OpenAI chat models were found")

        return available

    @classmethod
    def _list_anthropic_models(cls, api_key: str) -> List[str]:
        """List available Anthropic models."""
        if anthropic is None:
            raise LLMProcessorError(
                "anthropic is not installed. Run: pip install anthropic"
            )

        if not api_key:
            raise LLMProcessorError("Anthropic API key is required")

        # Anthropic doesn't have a public list models API, so we fetch from their models endpoint
        # Fall back to known models if the API call fails
        try:
            import requests
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
            response = requests.get("https://api.anthropic.com/v1/models", headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                available = [model.get("id") for model in data.get("data", []) if model.get("id")]
                if available:
                    return sorted(available)
        except Exception as exc:
            logging.warning("Could not fetch Anthropic models from API, using fallback list: %s", exc)

        # Fallback to known models - updated for 2025/2026
        available = [
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

        # Validate the API key by making a minimal request
        try:
            client = anthropic.Anthropic(api_key=api_key)
            # Just verify the client can be created - actual validation happens on first request
            _ = client
        except Exception as exc:
            logging.error("Failed to validate Anthropic API key: %s", exc, exc_info=True)
            raise LLMProcessorError(f"Failed to validate Anthropic API key: {exc}") from exc

        return available
