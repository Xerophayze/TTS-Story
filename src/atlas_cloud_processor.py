"""Atlas Cloud LLM integration using its OpenAI-compatible HTTP API."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, List, Optional
from urllib.parse import unquote

import requests


DEFAULT_ATLAS_CLOUD_BASE_URL = "https://api.atlascloud.ai/v1"
DEFAULT_ATLAS_CLOUD_MODEL = "deepseek-v3"
ATLAS_LLM_CATALOG_URL = "https://www.atlascloud.ai/models/list/llm"


class AtlasCloudProcessorError(RuntimeError):
    """Raised when Atlas Cloud model discovery or generation fails."""


@dataclass(frozen=True)
class AtlasCloudModelCatalog:
    models: List[str]
    warnings: List[str]


def _unique_models(models: List[Optional[str]]) -> List[str]:
    cleaned = {str(model).strip() for model in models if model and str(model).strip()}
    return sorted(cleaned, key=str.casefold)


def _is_image_model(model_id: str) -> bool:
    return bool(
        re.search(
            r"(?:^|[-_/])(image|imagen|banana|flux|seedream|ideogram|recraft|hidream|text-to-image|dall-e)(?:$|[-_/])",
            model_id,
            re.IGNORECASE,
        )
    )


class AtlasCloudProcessor:
    """List Atlas Cloud LLMs and submit non-streaming chat completions."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = DEFAULT_ATLAS_CLOUD_BASE_URL,
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
            raise AtlasCloudProcessorError("Atlas Cloud API key is required")
        if not self.model_name:
            raise AtlasCloudProcessorError("Atlas Cloud model is required")
        if not self.base_url:
            raise AtlasCloudProcessorError("Atlas Cloud base URL is required")

    @staticmethod
    def normalize_base_url(base_url: str) -> str:
        base = (base_url or DEFAULT_ATLAS_CLOUD_BASE_URL).strip().rstrip("/")
        if base and not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _response_error(response: Any, action: str) -> AtlasCloudProcessorError:
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
        return AtlasCloudProcessorError(
            f"Atlas Cloud {action} failed (HTTP {response.status_code}){suffix}"
        )

    @classmethod
    def _fetch_api_models(
        cls,
        api_key: str,
        base_url: str,
        timeout: int,
    ) -> List[str]:
        try:
            response = requests.get(
                f"{cls.normalize_base_url(base_url)}/models",
                headers=cls._headers(api_key),
                timeout=min(max(10, int(timeout or 30)), 60),
            )
        except Exception as exc:
            raise AtlasCloudProcessorError(
                f"Atlas Cloud model API request failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise cls._response_error(response, "model discovery")

        try:
            payload = response.json() if response.content else {}
        except Exception as exc:
            raise AtlasCloudProcessorError(
                "Atlas Cloud model API returned invalid JSON"
            ) from exc

        entries = payload.get("data", []) if isinstance(payload, dict) else []
        if not entries and isinstance(payload, dict):
            entries = payload.get("models", [])
        model_ids: List[Optional[str]] = []
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, str):
                model_ids.append(entry)
            elif isinstance(entry, dict):
                model_ids.append(entry.get("id") or entry.get("name"))
        return _unique_models(model_ids)

    @staticmethod
    def _fetch_public_catalog(timeout: int) -> List[str]:
        try:
            response = requests.get(
                ATLAS_LLM_CATALOG_URL,
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "User-Agent": "TTS-Story Atlas model catalog",
                },
                timeout=min(max(10, int(timeout or 20)), 30),
            )
        except Exception as exc:
            raise AtlasCloudProcessorError(
                f"Atlas Cloud public catalog request failed: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise AtlasCloudProcessorError(
                f"Atlas Cloud public catalog failed (HTTP {response.status_code})"
            )

        page = html.unescape(response.text or "")
        models = [
            unquote(match)
            for match in re.findall(r'href=["\']/models/([^"\'?#]+)', page)
            if "/" in match and not re.match(r"^(?:all|explore|list/)", match)
        ]
        return _unique_models([model for model in models if not _is_image_model(model)])

    @classmethod
    def list_available_models(
        cls,
        api_key: str,
        base_url: str = DEFAULT_ATLAS_CLOUD_BASE_URL,
        timeout: int = 30,
        current_model: str = DEFAULT_ATLAS_CLOUD_MODEL,
    ) -> AtlasCloudModelCatalog:
        api_key = (api_key or "").strip()
        if not api_key:
            raise AtlasCloudProcessorError("Atlas Cloud API key is required")

        warnings: List[str] = []
        api_models: List[str] = []
        public_models: List[str] = []
        try:
            api_models = cls._fetch_api_models(api_key, base_url, timeout)
        except AtlasCloudProcessorError as exc:
            warnings.append(f"{exc}; showing the public Atlas LLM catalog when available.")

        try:
            public_models = cls._fetch_public_catalog(timeout)
        except AtlasCloudProcessorError:
            warnings.append(
                "The public Atlas LLM catalog could not be loaded; showing API results and saved defaults."
            )

        models = _unique_models(
            [
                *[model for model in api_models if not _is_image_model(model)],
                *public_models,
                current_model,
                DEFAULT_ATLAS_CLOUD_MODEL,
            ]
        )
        if not models:
            raise AtlasCloudProcessorError("No Atlas Cloud LLM models were returned")
        return AtlasCloudModelCatalog(models=models, warnings=warnings)

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
            raise AtlasCloudProcessorError("Prompt must not be empty")

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
            payload["thinking"] = {"type": "disabled"}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(self.api_key),
                timeout=self.timeout,
            )
        except Exception as exc:
            raise AtlasCloudProcessorError(
                f"Atlas Cloud generation request failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise self._response_error(response, "generation")
        try:
            result = response.json()
        except Exception as exc:
            raise AtlasCloudProcessorError(
                "Atlas Cloud generation returned invalid JSON"
            ) from exc
        content = self._extract_content(result)
        if not content:
            raise AtlasCloudProcessorError(
                "Atlas Cloud response did not contain any text"
            )
        return content
