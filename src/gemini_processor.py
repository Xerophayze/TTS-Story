"""Helper for interacting with Google Gemini models."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Dict, Optional

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

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: Optional[int] = None,
        reason: Optional[str] = None,
        attempts: int = 1,
        max_retries: int = 0,
        retry_after_seconds: Optional[float] = None,
    ):
        super().__init__(message)
        self.retryable = bool(retryable)
        self.status_code = status_code
        self.reason = reason
        self.attempts = max(1, int(attempts or 1))
        self.max_retries = max(0, int(max_retries or 0))
        self.retry_after_seconds = retry_after_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": str(self),
            "retryable": self.retryable,
            "status_code": self.status_code,
            "reason": self.reason,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "retry_after_seconds": self.retry_after_seconds,
        }


class GeminiProcessor:
    """Wrapper around the Google Gemini SDK."""

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    RETRYABLE_REASONS = {
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "DEADLINE_EXCEEDED",
        "INTERNAL",
    }
    RETRYABLE_MESSAGE_FRAGMENTS = (
        " 429",
        " 500",
        " 502",
        " 503",
        " 504",
        "unavailable",
        "temporarily unavailable",
        "overloaded",
        "overload",
        "capacity",
        "resource exhausted",
        "rate limit",
        "try again later",
    )
    NON_RETRYABLE_MESSAGE_FRAGMENTS = (
        "api key",
        "permission denied",
        "permission",
        "unauthorized",
        "invalid argument",
        "invalid request",
        "malformed",
        "authentication",
        "forbidden",
    )

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 8.0,
        jitter: float = 0.35,
    ):
        if genai is None:
            raise GeminiProcessorError(
                "google-genai is not installed. Please install it to use Gemini features: pip install google-genai"
            )

        if not api_key:
            raise GeminiProcessorError("Gemini API key is required")

        self.model_name = model_name or "gemini-1.5-flash"
        self.api_key = api_key
        self.max_retries = max(0, int(max_retries or 0))
        self.base_delay = max(0.1, float(base_delay or 1.0))
        self.max_delay = max(self.base_delay, float(max_delay or self.base_delay))
        self.jitter = max(0.0, float(jitter or 0.0))
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
            raise self._classify_exception(exc, prefix="Failed to initialize Gemini") from exc

    def generate_text(
        self,
        prompt: str,
        *,
        on_retry: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        """Send prompt to Gemini and return the text response."""

        if not prompt.strip():
            raise GeminiProcessorError("Prompt must not be empty")

        attempt = 1
        while True:
            if attempt > 1 and on_retry:
                on_retry(
                    {
                        "status": "retrying",
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                    }
                )

            try:
                if USING_NEW_SDK:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                    )
                else:
                    response = self.model.generate_content(prompt)
            except Exception as exc:  # pragma: no cover - network failure
                error = self._classify_exception(exc, prefix="Gemini API error")
                error.attempts = attempt
                error.max_retries = self.max_retries

                if error.retryable and attempt <= self.max_retries:
                    delay = self._compute_backoff_seconds(attempt)
                    error.retry_after_seconds = delay
                    logging.warning(
                        "Gemini API retry %s/%s in %.2fs: %s",
                        attempt,
                        self.max_retries,
                        delay,
                        error,
                    )
                    if on_retry:
                        on_retry(
                            {
                                "status": "waiting_to_retry",
                                "attempt": attempt,
                                "max_retries": self.max_retries,
                                "delay_seconds": delay,
                                "message": str(error),
                            }
                        )
                    time.sleep(delay)
                    attempt += 1
                    continue

                logging.error("Gemini API error: %s", error, exc_info=True)
                raise error from exc

            text = self._extract_text(response)
            if not text:
                raise GeminiProcessorError("Gemini response did not contain any text")

            return text.strip()

    def _compute_backoff_seconds(self, attempt: int) -> float:
        base_delay = min(self.max_delay, self.base_delay * (2 ** max(attempt - 1, 0)))
        if self.jitter <= 0:
            return base_delay
        jitter_window = min(self.jitter, base_delay * 0.35)
        return min(self.max_delay, base_delay + random.uniform(0.0, jitter_window))

    @classmethod
    def _coerce_status_code(cls, exc: Exception) -> Optional[int]:
        for attr_name in ("status_code", "code", "status", "http_status"):
            value = getattr(exc, attr_name, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        match = None
        message = str(exc)
        for pattern in (r"\b(429|500|502|503|504)\b",):
            import re

            match = re.search(pattern, message)
            if match:
                break
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return None
        return None

    @classmethod
    def _extract_reason(cls, exc: Exception) -> Optional[str]:
        for attr_name in ("reason", "status", "code_name"):
            value = getattr(exc, attr_name, None)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
        message = str(exc)
        for candidate in cls.RETRYABLE_REASONS:
            if candidate.lower() in message.lower():
                return candidate
        return None

    @classmethod
    def _classify_exception(cls, exc: Exception, *, prefix: str) -> GeminiProcessorError:
        status_code = cls._coerce_status_code(exc)
        reason = cls._extract_reason(exc)
        message = f"{prefix}: {exc}".strip()
        lowered = message.lower()

        retryable = False
        if status_code in cls.RETRYABLE_STATUS_CODES:
            retryable = True
        elif reason in cls.RETRYABLE_REASONS:
            retryable = True
        elif any(fragment in lowered for fragment in cls.RETRYABLE_MESSAGE_FRAGMENTS):
            retryable = True

        if any(fragment in lowered for fragment in cls.NON_RETRYABLE_MESSAGE_FRAGMENTS):
            retryable = False

        return GeminiProcessorError(
            message,
            retryable=retryable,
            status_code=status_code,
            reason=reason,
        )

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
