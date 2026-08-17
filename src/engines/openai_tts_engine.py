"""OpenAI and OpenAI-compatible text-to-speech REST adapter."""
from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from ..audio_effects import VoiceFXSettings
from .base import EngineCapabilities, TtsEngineBase, VoiceAssignment
from .cloud_audio import CloudAudioError, apply_wav_effects, audio_bytes_to_wav


DEFAULT_OPENAI_TTS_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_OPENAI_TTS_VOICE = "coral"
OPENAI_TTS_VOICES = (
    "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx",
    "nova", "sage", "shimmer", "verse", "marin", "cedar",
)


class OpenAITTSError(RuntimeError):
    """Raised when an OpenAI-compatible speech endpoint fails."""


class OpenAITTSEngine(TtsEngineBase):
    """Render chunks through the OpenAI-compatible ``audio/speech`` API."""

    name = "openai_tts"
    capabilities = EngineCapabilities(False, False, None)

    def __init__(
        self,
        api_key: str = "",
        *,
        base_url: str = DEFAULT_OPENAI_TTS_BASE_URL,
        model_id: str = DEFAULT_OPENAI_TTS_MODEL,
        default_voice: str = DEFAULT_OPENAI_TTS_VOICE,
        instructions: str = "",
        timeout: int = 120,
        max_parallel: int = 2,
        max_retries: int = 4,
        voice_required: bool = True,
        request_func: Optional[Callable[..., Any]] = None,
        sleep_func: Callable[[float], None] = time.sleep,
        audio_converter: Callable[..., bytes] = audio_bytes_to_wav,
    ) -> None:
        super().__init__(device="cloud")
        self.api_key = str(api_key or "").strip()
        raw_url = str(base_url or DEFAULT_OPENAI_TTS_BASE_URL).strip().rstrip("/")
        if not raw_url.startswith(("https://", "http://")):
            raise OpenAITTSError("The OpenAI-compatible endpoint must begin with http:// or https://.")
        if raw_url.endswith("/audio/speech"):
            self.speech_url = raw_url
            self.root_url = raw_url[:-13].rstrip("/")
        else:
            self.root_url = raw_url
            self.speech_url = f"{raw_url}/audio/speech"
        self.model_id = self._required(model_id, "model")
        self.voice_required = bool(voice_required)
        self.default_voice = (
            self._required(default_voice, "voice")
            if self.voice_required else str(default_voice or "").strip()
        )
        self.instructions = str(instructions or "").strip()
        self.timeout = max(10, min(int(timeout or 120), 600))
        self.max_parallel = max(1, min(int(max_parallel or 1), 8))
        self.max_retries = max(0, min(int(max_retries or 0), 8))
        self._request = request_func or requests.request
        self._sleep = sleep_func
        self._audio_converter = audio_converter

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "audio/wav"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        lang_code: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: Optional[int] = None,
        fx_settings: Optional[VoiceFXSettings] = None,
        voice_options: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> bytes:
        fx_payload = None
        if fx_settings:
            fx_payload = {
                "enabled": True,
                "pitch": fx_settings.pitch_semitones,
                "tone": fx_settings.tone,
                "speed": fx_settings.speed,
            }
        return self._synthesize(
            text,
            VoiceAssignment(
                voice=voice or self.default_voice,
                lang_code=lang_code,
                fx_payload=fx_payload,
                speed_override=speed,
                extra=voice_options or {},
            ),
            fallback_speed=speed,
        )

    def generate_batch(
        self,
        segments: List[Dict],
        voice_config: Dict[str, Dict],
        output_dir: Path,
        speed: float = 1.0,
        sample_rate: Optional[int] = None,
        progress_cb=None,
        chunk_cb=None,
        parallel_workers: int = 1,
        start_index: int = 0,
    ) -> List[str]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        items = self._work_items(segments, voice_config)

        def render(item: Dict[str, Any]) -> tuple[int, str]:
            wav = self._synthesize(item["text"], item["assignment"], fallback_speed=speed)
            path = destination / f"{self.name}_chunk_{start_index + item['order_index']:06d}.wav"
            path.write_bytes(wav)
            return item["order_index"], str(path)

        results: Dict[int, str] = {}
        workers = max(1, min(int(parallel_workers or 1), self.max_parallel, len(items) or 1))
        if workers == 1:
            for item in items:
                order, path = render(item)
                results[order] = path
                self._notify(item, path, progress_cb, chunk_cb)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="openai-tts") as pool:
                pending = {}
                next_submit = 0
                next_callback = 0

                def fill_workers() -> None:
                    nonlocal next_submit
                    while next_submit < len(items) and len(pending) < workers:
                        item = items[next_submit]
                        pending[pool.submit(render, item)] = item
                        next_submit += 1

                fill_workers()
                try:
                    while pending:
                        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            pending.pop(future)
                            order, path = future.result()
                            results[order] = path
                        while next_callback in results:
                            self._notify(items[next_callback], results[next_callback], progress_cb, chunk_cb)
                            next_callback += 1
                        fill_workers()
                except Exception:
                    for future in pending:
                        future.cancel()
                    raise
        return [results[index] for index in range(len(items))]

    def cleanup(self) -> None:
        """The REST adapter keeps no local model resources."""

    def _synthesize(self, text: str, assignment: VoiceAssignment, *, fallback_speed: float) -> bytes:
        clean_text = str(text or "").strip()
        if not clean_text:
            raise OpenAITTSError("The OpenAI-compatible endpoint cannot synthesize empty text.")
        extra = assignment.extra or {}
        voice = str(assignment.voice or self.default_voice or "").strip()
        if self.voice_required:
            voice = self._required(voice, "voice")
        model = self._required(extra.get("model_id") or self.model_id, "model")
        speed = self._clamp(
            assignment.speed_override if assignment.speed_override is not None else fallback_speed,
            0.25, 4.0, 1.0,
        )
        payload: Dict[str, Any] = {
            "model": model,
            "input": clean_text,
            "response_format": "wav",
            "speed": speed,
        }
        if voice:
            payload["voice"] = {"id": voice} if voice.startswith("voice_") else voice
        instructions = str(extra.get("instructions") or self.instructions).strip()
        if instructions:
            payload["instructions"] = instructions

        response = self._request_with_retries("POST", self.speech_url, headers=self.headers, json=payload)
        try:
            wav = self._audio_converter(
                bytes(response.content or b""), input_format="wav", sample_rate=None, channels=1
            )
            original_fx = VoiceFXSettings.from_payload(assignment.fx_payload)
            # Speed is native to the endpoint; only apply pitch/tone locally.
            fx = VoiceFXSettings(
                pitch_semitones=original_fx.pitch_semitones if original_fx else 0.0,
                speed=1.0,
                tone=original_fx.tone if original_fx else "neutral",
            )
            return apply_wav_effects(wav, None if fx.is_identity() else fx)
        except CloudAudioError as exc:
            raise OpenAITTSError(str(exc)) from exc

    def _request_with_retries(self, method: str, url: str, **kwargs: Any) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep(min(2 ** attempt, 20))
                continue
            status = int(getattr(response, "status_code", 0) or 0)
            if 200 <= status < 300:
                return response
            if status in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                retry_after = str((getattr(response, "headers", {}) or {}).get("Retry-After") or "")
                try:
                    delay = float(retry_after) if retry_after else min(2 ** attempt, 30)
                except ValueError:
                    delay = min(2 ** attempt, 30)
                self._sleep(max(0.0, min(delay, 120.0)))
                continue
            raise OpenAITTSError(self._response_error(response))
        if last_error:
            raise OpenAITTSError(f"Unable to reach the OpenAI-compatible TTS endpoint: {last_error}") from last_error
        raise OpenAITTSError("The OpenAI-compatible TTS request failed after retries.")

    def _work_items(self, segments: List[Dict], voice_config: Dict[str, Dict]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for segment_index, segment in enumerate(segments):
            speaker = segment.get("speaker") or "default"
            config = voice_config.get(speaker) or voice_config.get("default") or {}
            assignment = VoiceAssignment(
                voice=config.get("voice") or self.default_voice,
                lang_code=config.get("lang_code"),
                fx_payload=config.get("fx"),
                speed_override=config.get("speed"),
                extra=config.get("extra") or {},
            )
            for chunk_index, chunk_text in enumerate(segment.get("chunks") or []):
                items.append({
                    "order_index": len(items), "segment_index": segment_index,
                    "chunk_index": chunk_index, "speaker": speaker,
                    "text": chunk_text, "assignment": assignment,
                })
        return items

    @staticmethod
    def _notify(item: Dict[str, Any], path: str, progress_cb, chunk_cb) -> None:
        if callable(progress_cb):
            progress_cb()
        if callable(chunk_cb):
            chunk_cb(item["order_index"], {key: item[key] for key in (
                "speaker", "text", "segment_index", "chunk_index", "order_index"
            )}, path)

    @staticmethod
    def _required(value: Any, label: str) -> str:
        parsed = str(value or "").strip()
        if not parsed:
            raise OpenAITTSError(f"An OpenAI-compatible TTS {label} is required.")
        return parsed

    @staticmethod
    def _clamp(value: Any, minimum: float, maximum: float, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = fallback
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _response_error(response: Any) -> str:
        status = int(getattr(response, "status_code", 0) or 0)
        messages = {
            401: "The TTS endpoint rejected the API key (401).",
            403: "The TTS endpoint denied access to the selected model or voice (403).",
            429: "The TTS endpoint rate or quota limit was reached (429).",
        }
        detail = str(getattr(response, "text", "") or "").strip().replace("\n", " ")[:300]
        return messages.get(status) or f"OpenAI-compatible TTS request failed ({status}){': ' + detail if detail else '.'}"


__all__ = [
    "DEFAULT_OPENAI_TTS_BASE_URL", "DEFAULT_OPENAI_TTS_MODEL",
    "DEFAULT_OPENAI_TTS_VOICE", "OPENAI_TTS_VOICES",
    "OpenAITTSEngine", "OpenAITTSError",
]
