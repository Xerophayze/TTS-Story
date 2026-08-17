"""LocalAI text-to-speech adapter built on its OpenAI-compatible endpoint."""
from __future__ import annotations

import threading
from typing import Callable, Dict

from .openai_tts_engine import OpenAITTSEngine
from .base import VoiceAssignment
from ..localai_tts_client import normalize_localai_urls


DEFAULT_LOCALAI_TTS_BASE_URL = "http://localhost:8080/v1"


class LocalAITTSEngine(OpenAITTSEngine):
    """Render speech with a LocalAI TTS model and optional voice profile URI."""

    name = "localai_tts"

    def __init__(
        self,
        api_key: str = "",
        *,
        base_url: str = DEFAULT_LOCALAI_TTS_BASE_URL,
        model_id: str,
        default_voice: str = "",
        instructions: str = "",
        timeout: int = 180,
        max_parallel: int = 1,
        max_retries: int = 4,
        voice_resolver: Callable[[str], str] | None = None,
        **kwargs,
    ) -> None:
        _, v1_root = normalize_localai_urls(base_url)
        self._voice_resolver = voice_resolver
        self._resolved_voice_cache: Dict[str, str] = {}
        self._voice_resolver_lock = threading.Lock()
        super().__init__(
            api_key=api_key,
            base_url=v1_root,
            model_id=model_id,
            default_voice=default_voice,
            instructions=instructions,
            timeout=timeout,
            max_parallel=max_parallel,
            max_retries=max_retries,
            voice_required=False,
            **kwargs,
        )

    def _synthesize(self, text: str, assignment: VoiceAssignment, *, fallback_speed: float) -> bytes:
        voice = str(assignment.voice or self.default_voice or "").strip()
        if voice and self._voice_resolver:
            with self._voice_resolver_lock:
                resolved = self._resolved_voice_cache.get(voice)
                if not resolved:
                    resolved = self._voice_resolver(voice)
                    self._resolved_voice_cache[voice] = resolved
            assignment = VoiceAssignment(
                voice=resolved,
                lang_code=assignment.lang_code,
                audio_prompt_path=assignment.audio_prompt_path,
                fx_payload=assignment.fx_payload,
                speed_override=assignment.speed_override,
                extra=assignment.extra,
            )
        return super()._synthesize(text, assignment, fallback_speed=fallback_speed)


__all__ = ["DEFAULT_LOCALAI_TTS_BASE_URL", "LocalAITTSEngine"]
