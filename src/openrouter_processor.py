"""OpenRouter LLM integration using its OpenAI-compatible HTTP API."""

from __future__ import annotations

import re
from typing import Any, List, Optional

import requests


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openrouter/auto"


class OpenRouterProcessorError(RuntimeError):
    """Raised when OpenRouter model discovery or generation fails."""


def _unique_models(models: List[Optional[str]]) -> List[str]:
    cleaned = {str(model).strip() for model in models if model and str(model).strip()}
    return sorted(cleaned, key=str.casefold)


class OpenRouterProcessor:
    """List user-available OpenRouter models and submit chat completions."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: int = 120,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        disable_reasoning: bool = False,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model_name = (model_name or "").strip()
        self.base_url = self.normalize_base_url(base_url)
        self.timeout = max(10, int(timeout or 120))
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.disable_reasoning = disable_reasoning

        if not self.api_key:
            raise OpenRouterProcessorError("OpenRouter API key is required")
        if not self.model_name:
            raise OpenRouterProcessorError("OpenRouter model is required")
        if not self.base_url:
            raise OpenRouterProcessorError("OpenRouter base URL is required")

    @staticmethod
    def normalize_base_url(base_url: str) -> str:
        base = (base_url or DEFAULT_OPENROUTER_BASE_URL).strip().rstrip("/")
        if base and not re.search(r"/(?:api/)?v1$", base, re.IGNORECASE):
            base = f"{base}/api/v1"
        return base

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "TTS-Story",
        }

    @staticmethod
    def _response_error(response: Any, action: str) -> OpenRouterProcessorError:
        detail = ""
        try:
            payload = response.json()
            raw_error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(raw_error, dict):
                detail = str(raw_error.get("message") or "").strip()
            elif raw_error:
                detail = str(raw_error).strip()
            if not detail and isinstance(payload, dict):
                detail = str(payload.get("message") or "").strip()
        except Exception:
            raw_text = str(getattr(response, "text", "") or "").strip()
            if not re.search(r"<(?:!doctype|html|head|body)\b", raw_text, re.IGNORECASE):
                detail = re.sub(r"\s+", " ", raw_text)[:500]
        suffix = f": {detail[:500]}" if detail else ""
        return OpenRouterProcessorError(
            f"OpenRouter {action} failed (HTTP {response.status_code}){suffix}"
        )

    @classmethod
    def list_available_models(
        cls,
        api_key: str,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: int = 30,
        current_model: str = DEFAULT_OPENROUTER_MODEL,
    ) -> List[str]:
        api_key = (api_key or "").strip()
        if not api_key:
            raise OpenRouterProcessorError("OpenRouter API key is required")

        try:
            response = requests.get(
                f"{cls.normalize_base_url(base_url)}/models/user",
                headers=cls._headers(api_key),
                timeout=min(max(10, int(timeout or 30)), 60),
            )
        except Exception as exc:
            raise OpenRouterProcessorError(
                f"OpenRouter model API request failed: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise cls._response_error(response, "model discovery")

        try:
            payload = response.json() if response.content else {}
        except Exception as exc:
            raise OpenRouterProcessorError(
                "OpenRouter model API returned invalid JSON"
            ) from exc

        entries = payload.get("data", []) if isinstance(payload, dict) else []
        model_ids: List[Optional[str]] = []
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, str):
                model_ids.append(entry)
                continue
            if not isinstance(entry, dict):
                continue
            architecture = entry.get("architecture") or {}
            output_modalities = architecture.get("output_modalities") or []
            if output_modalities and "text" not in output_modalities:
                continue
            model_ids.append(entry.get("id") or entry.get("canonical_slug"))

        discovered = _unique_models(model_ids)
        if not discovered:
            raise OpenRouterProcessorError(
                "No text-output models are available for this OpenRouter API key"
            )
        return _unique_models([*discovered, current_model])

    @staticmethod
    def _extract_content(payload: Any) -> str:
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
            return "".join(parts).strip()
        return ""

    def generate_text(self, prompt: str) -> str:
        if not (prompt or "").strip():
            raise OpenRouterProcessorError("Prompt must not be empty")

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        if self.top_k is not None and int(self.top_k) > 0:
            payload["top_k"] = int(self.top_k)
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = float(self.repetition_penalty)
        if self.max_tokens is not None and int(self.max_tokens) > 0:
            payload["max_tokens"] = int(self.max_tokens)
        if self.disable_reasoning:
            payload["reasoning"] = {"effort": "none"}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(self.api_key),
                timeout=self.timeout,
            )
        except Exception as exc:
            raise OpenRouterProcessorError(
                f"OpenRouter generation request failed: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise self._response_error(response, "generation")

        try:
            result = response.json()
        except Exception as exc:
            raise OpenRouterProcessorError(
                "OpenRouter generation returned invalid JSON"
            ) from exc
        content = self._extract_content(result)
        if not content:
            raise OpenRouterProcessorError(
                "OpenRouter response did not contain any text"
            )
        return content
