"""ElevenLabs text-to-speech REST adapter."""
from __future__ import annotations

import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import requests

from ..audio_effects import VoiceFXSettings
from .base import EngineCapabilities, TtsEngineBase, VoiceAssignment
from .cloud_audio import CloudAudioError, apply_wav_effects, audio_bytes_to_wav


DEFAULT_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_VOICE = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"
ELEVENLABS_OUTPUT_FORMATS = {
    "mp3_44100_128": ("mp3", 44100),
    "mp3_44100_192": ("mp3", 44100),
    "wav_44100": ("wav", 44100),
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class ElevenLabsError(RuntimeError):
    """Raised when ElevenLabs cannot complete a request."""


class ElevenLabsEngine(TtsEngineBase):
    """Render TTS-Story chunks using ElevenLabs voices and models."""

    name = "elevenlabs"
    capabilities = EngineCapabilities(False, False, None)

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_ELEVENLABS_BASE_URL,
        model_id: str = DEFAULT_ELEVENLABS_MODEL,
        default_voice: str = DEFAULT_ELEVENLABS_VOICE,
        output_format: str = DEFAULT_ELEVENLABS_OUTPUT_FORMAT,
        timeout: int = 120,
        max_parallel: int = 2,
        max_retries: int = 4,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
        request_func: Optional[Callable[..., Any]] = None,
        sleep_func: Callable[[float], None] = time.sleep,
        audio_converter: Callable[..., bytes] = audio_bytes_to_wav,
    ) -> None:
        super().__init__(device="cloud")
        self.api_key = str(api_key or "").strip()
        if not self.api_key:
            raise ElevenLabsError("An ElevenLabs API key is required.")
        raw_url = str(base_url or DEFAULT_ELEVENLABS_BASE_URL).strip().rstrip("/")
        self.root_url = raw_url[:-3] if raw_url.endswith("/v1") else raw_url
        if not self.root_url.startswith(("https://", "http://")):
            raise ElevenLabsError("The ElevenLabs base URL must begin with http:// or https://.")
        self.model_id = self._identifier(model_id, "model")
        self.default_voice = self._identifier(default_voice, "voice")
        if output_format not in ELEVENLABS_OUTPUT_FORMATS:
            raise ElevenLabsError(f"Unsupported ElevenLabs output format: {output_format}")
        self.output_format = output_format
        self.timeout = max(10, min(int(timeout or 120), 600))
        self.max_parallel = max(1, min(int(max_parallel or 1), 8))
        self.max_retries = max(0, min(int(max_retries or 0), 8))
        self.stability = self._clamp_float(stability, 0, 1, 0.5)
        self.similarity_boost = self._clamp_float(similarity_boost, 0, 1, 0.75)
        self.style = self._clamp_float(style, 0, 1, 0)
        self.use_speaker_boost = bool(use_speaker_boost)
        self._request = request_func or requests.request
        self._sleep = sleep_func
        self._audio_converter = audio_converter

    @property
    def sample_rate(self) -> int:
        return ELEVENLABS_OUTPUT_FORMATS[self.output_format][1]

    @property
    def headers(self) -> Dict[str, str]:
        return {"xi-api-key": self.api_key, "Accept": "application/json"}

    def list_models(self) -> List[Dict[str, Any]]:
        payload = self._json_request("GET", f"{self.root_url}/v1/models")
        if not isinstance(payload, list):
            raise ElevenLabsError("ElevenLabs returned an unexpected model catalog response.")
        models = []
        for item in payload:
            if not isinstance(item, dict) or not item.get("can_do_text_to_speech"):
                continue
            model_id = str(item.get("model_id") or "").strip()
            if not model_id:
                continue
            models.append(
                {
                    "model_id": model_id,
                    "name": str(item.get("name") or model_id).strip(),
                    "description": str(item.get("description") or "").strip(),
                    "max_characters_request_free_user": item.get("max_characters_request_free_user"),
                    "max_characters_request_subscribed_user": item.get("max_characters_request_subscribed_user"),
                    "can_use_style": bool(item.get("can_use_style")),
                    "can_use_speaker_boost": bool(item.get("can_use_speaker_boost")),
                    "languages": item.get("languages") if isinstance(item.get("languages"), list) else [],
                }
            )
        models.sort(key=lambda model: model["name"].casefold())
        return models

    def list_voices(self) -> List[Dict[str, Any]]:
        voices: List[Dict[str, Any]] = []
        token = ""
        for _ in range(20):
            params: Dict[str, Any] = {"page_size": 100}
            if token:
                params["next_page_token"] = token
            payload = self._json_request("GET", f"{self.root_url}/v2/voices", params=params)
            if not isinstance(payload, dict):
                raise ElevenLabsError("ElevenLabs returned an unexpected voice catalog response.")
            for item in payload.get("voices") or []:
                if not isinstance(item, dict):
                    continue
                voice_id = str(item.get("voice_id") or "").strip()
                if not voice_id:
                    continue
                labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
                locale = str(labels.get("language") or labels.get("locale") or "").strip()
                voices.append(
                    {
                        "short_name": voice_id,
                        "voice_id": voice_id,
                        "display_name": str(item.get("name") or voice_id).strip(),
                        "local_name": str(item.get("name") or voice_id).strip(),
                        "gender": str(labels.get("gender") or "Unknown").strip(),
                        "locale": locale,
                        "locale_name": locale,
                        "category": str(item.get("category") or "").strip(),
                        "description": str(item.get("description") or "").strip(),
                        "preview_url": str(item.get("preview_url") or "").strip(),
                        "labels": {str(k): str(v) for k, v in labels.items()},
                    }
                )
            token = str(payload.get("next_page_token") or "").strip()
            if not token or not payload.get("has_more"):
                break
        voices.sort(key=lambda voice: (voice["display_name"].casefold(), voice["short_name"]))
        return voices

    def get_subscription(self) -> Dict[str, Any]:
        payload = self._json_request("GET", f"{self.root_url}/v1/user/subscription")
        if not isinstance(payload, dict):
            raise ElevenLabsError("ElevenLabs returned an unexpected subscription response.")
        return {
            "tier": str(payload.get("tier") or "").strip(),
            "character_count": self._optional_int(payload.get("character_count")),
            "character_limit": self._optional_int(payload.get("character_limit")),
            "next_character_count_reset_unix": self._optional_int(
                payload.get("next_character_count_reset_unix")
            ),
            "can_extend_character_limit": bool(payload.get("can_extend_character_limit")),
            "status": str(payload.get("status") or "").strip(),
        }

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
        assignment = VoiceAssignment(
            voice=voice or self.default_voice,
            lang_code=lang_code,
            fx_payload=fx_payload,
            speed_override=speed,
            extra=voice_options or {},
        )
        return self._synthesize(text, assignment, fallback_speed=speed)

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
        work_items = self._work_items(segments, voice_config)
        if not work_items:
            return []

        def render(item: Dict[str, Any]) -> tuple[int, str]:
            wav = self._synthesize(
                item["text"],
                item["assignment"],
                fallback_speed=speed,
                previous_text=item.get("previous_text"),
                next_text=item.get("next_text"),
            )
            path = destination / f"elevenlabs_chunk_{start_index + item['order_index']:06d}.wav"
            path.write_bytes(wav)
            return item["order_index"], str(path)

        results: Dict[int, str] = {}
        workers = max(1, min(int(parallel_workers or 1), self.max_parallel, len(work_items)))
        if workers == 1:
            for item in work_items:
                order, path = render(item)
                results[order] = path
                self._notify_callbacks(item, path, progress_cb, chunk_cb)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="elevenlabs") as executor:
                pending = {}
                next_submit = 0
                next_callback = 0

                def fill_workers() -> None:
                    nonlocal next_submit
                    while next_submit < len(work_items) and len(pending) < workers:
                        item = work_items[next_submit]
                        pending[executor.submit(render, item)] = item
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
                            self._notify_callbacks(
                                work_items[next_callback], results[next_callback], progress_cb, chunk_cb
                            )
                            next_callback += 1
                        fill_workers()
                except Exception:
                    for future in pending:
                        future.cancel()
                    raise
        return [results[index] for index in range(len(work_items))]

    def cleanup(self) -> None:
        """ElevenLabs REST keeps no local model resources."""

    def _synthesize(
        self,
        text: str,
        assignment: VoiceAssignment,
        *,
        fallback_speed: float,
        previous_text: Optional[str] = None,
        next_text: Optional[str] = None,
    ) -> bytes:
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ElevenLabsError("ElevenLabs cannot synthesize empty text.")
        voice = self._identifier(assignment.voice or self.default_voice, "voice")
        extra = assignment.extra or {}
        model_id = self._identifier(extra.get("model_id") or self.model_id, "model")
        requested_speed = self._clamp_float(
            assignment.speed_override if assignment.speed_override is not None else fallback_speed,
            0.5,
            2.0,
            1.0,
        )
        native_speed = max(0.7, min(1.2, requested_speed))
        payload: Dict[str, Any] = {
            "text": clean_text,
            "model_id": model_id,
            "voice_settings": {
                "stability": self._clamp_float(extra.get("stability"), 0, 1, self.stability),
                "similarity_boost": self._clamp_float(
                    extra.get("similarity_boost"), 0, 1, self.similarity_boost
                ),
                "style": self._clamp_float(extra.get("style"), 0, 1, self.style),
                "use_speaker_boost": self._bool_value(
                    extra.get("use_speaker_boost"), self.use_speaker_boost
                ),
                "speed": native_speed,
            },
        }
        language_code = str(extra.get("language_code") or assignment.lang_code or "").strip()
        if language_code:
            payload["language_code"] = language_code
        if previous_text:
            payload["previous_text"] = previous_text
        if next_text:
            payload["next_text"] = next_text

        response = self._request_with_retries(
            "POST",
            f"{self.root_url}/v1/text-to-speech/{quote(voice, safe='')}",
            headers={**self.headers, "Content-Type": "application/json", "Accept": "audio/mpeg"},
            params={"output_format": self.output_format},
            json=payload,
        )
        raw = bytes(response.content or b"")
        source_format, output_rate = ELEVENLABS_OUTPUT_FORMATS[self.output_format]
        try:
            wav = self._audio_converter(
                raw,
                input_format=source_format,
                sample_rate=output_rate,
                channels=1,
            )
            original_fx = VoiceFXSettings.from_payload(assignment.fx_payload)
            residual_speed = requested_speed / native_speed
            fx = VoiceFXSettings(
                pitch_semitones=original_fx.pitch_semitones if original_fx else 0.0,
                speed=max(0.5, min(2.0, residual_speed)),
                tone=original_fx.tone if original_fx else "neutral",
            )
            return apply_wav_effects(wav, None if fx.is_identity() else fx)
        except CloudAudioError as exc:
            raise ElevenLabsError(str(exc)) from exc

    def _json_request(self, method: str, url: str, *, params: Optional[Dict] = None) -> Any:
        response = self._request_with_retries(method, url, headers=self.headers, params=params)
        try:
            return response.json()
        except ValueError as exc:
            raise ElevenLabsError("ElevenLabs returned invalid JSON.") from exc

    def _request_with_retries(self, method: str, url: str, **kwargs: Any) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep(min(2**attempt, 20))
                continue
            if 200 <= int(response.status_code) < 300:
                return response
            if int(response.status_code) in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                self._sleep(self._retry_delay(response, attempt))
                continue
            raise ElevenLabsError(self._response_error(response))
        if last_error:
            raise ElevenLabsError(f"Unable to reach ElevenLabs: {last_error}") from last_error
        raise ElevenLabsError("ElevenLabs request failed after retries.")

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
                items.append(
                    {
                        "order_index": len(items),
                        "segment_index": segment_index,
                        "chunk_index": chunk_index,
                        "speaker": speaker,
                        "text": chunk_text,
                        "assignment": assignment,
                    }
                )
        for index, item in enumerate(items):
            if index and items[index - 1]["speaker"] == item["speaker"]:
                item["previous_text"] = items[index - 1]["text"]
            if index + 1 < len(items) and items[index + 1]["speaker"] == item["speaker"]:
                item["next_text"] = items[index + 1]["text"]
        return items

    @staticmethod
    def _notify_callbacks(item: Dict[str, Any], path: str, progress_cb, chunk_cb) -> None:
        if callable(progress_cb):
            progress_cb()
        if callable(chunk_cb):
            chunk_cb(item["order_index"], {key: item[key] for key in (
                "speaker", "text", "segment_index", "chunk_index", "order_index"
            )}, path)

    @staticmethod
    def _identifier(value: Any, label: str) -> str:
        parsed = str(value or "").strip()
        if not parsed or not _IDENTIFIER_RE.fullmatch(parsed):
            raise ElevenLabsError(f"A valid ElevenLabs {label} ID is required.")
        return parsed

    @staticmethod
    def _retry_delay(response: Any, attempt: int) -> float:
        value = str((getattr(response, "headers", {}) or {}).get("Retry-After") or "").strip()
        try:
            return max(0.0, min(float(value), 120.0)) if value else float(min(2**attempt, 30))
        except ValueError:
            return float(min(2**attempt, 30))

    @staticmethod
    def _response_error(response: Any) -> str:
        status = int(getattr(response, "status_code", 0) or 0)
        messages = {
            401: "ElevenLabs rejected the API key (401). Check the key in Settings.",
            402: "ElevenLabs reports that the account quota or billing allowance is exhausted (402).",
            403: "ElevenLabs denied this request (403). Check the selected voice and account access.",
            422: "ElevenLabs rejected the voice, model, text, or voice settings (422).",
            429: "ElevenLabs concurrency or rate limit reached (429). Reduce parallel requests or retry later.",
        }
        if status in messages:
            return messages[status]
        detail = str(getattr(response, "text", "") or "").strip().replace("\n", " ")[:300]
        return f"ElevenLabs request failed ({status}){': ' + detail if detail else '.'}"

    @staticmethod
    def _clamp_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = fallback
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _bool_value(value: Any, fallback: bool) -> bool:
        if value is None:
            return fallback
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


__all__ = [
    "DEFAULT_ELEVENLABS_BASE_URL",
    "DEFAULT_ELEVENLABS_MODEL",
    "DEFAULT_ELEVENLABS_OUTPUT_FORMAT",
    "DEFAULT_ELEVENLABS_VOICE",
    "ELEVENLABS_OUTPUT_FORMATS",
    "ElevenLabsEngine",
    "ElevenLabsError",
]
