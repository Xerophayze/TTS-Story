"""Microsoft Azure AI Speech text-to-speech engine.

The adapter intentionally uses Azure's REST surface instead of the native
Speech SDK.  TTS-Story already depends on ``requests`` and its engine contract
is file-oriented, so REST keeps the provider cross-platform while still
supporting regional voice discovery and the SSML controls Azure exposes.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from xml.sax.saxutils import escape, quoteattr

import requests

from ..audio_effects import VoiceFXSettings
from .base import EngineCapabilities, TtsEngineBase, VoiceAssignment


logger = logging.getLogger(__name__)

AZURE_SPEECH_OUTPUT_FORMATS = {
    "riff-24khz-16bit-mono-pcm": 24000,
    "riff-48khz-16bit-mono-pcm": 48000,
}
DEFAULT_AZURE_SPEECH_OUTPUT_FORMAT = "riff-24khz-16bit-mono-pcm"
DEFAULT_AZURE_SPEECH_VOICE = "en-US-AvaMultilingualNeural"
DEFAULT_AZURE_SPEECH_REQUESTS_PER_MINUTE = 20

_REGION_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LOCALE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})+$")
_SSML_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class AzureSpeechError(RuntimeError):
    """Raised when Azure Speech cannot complete a request."""


class AzureSpeechEngine(TtsEngineBase):
    """Render TTS-Story chunks with Azure neural voices."""

    name = "azure_speech"
    capabilities = EngineCapabilities(
        supports_voice_cloning=False,
        supports_emotion_tags=False,
        supported_languages=None,
    )

    def __init__(
        self,
        subscription_key: str,
        region: str,
        *,
        output_format: str = DEFAULT_AZURE_SPEECH_OUTPUT_FORMAT,
        timeout: int = 60,
        max_parallel: int = 2,
        requests_per_minute: int = DEFAULT_AZURE_SPEECH_REQUESTS_PER_MINUTE,
        default_voice: str = DEFAULT_AZURE_SPEECH_VOICE,
        default_style: str = "",
        default_role: str = "",
        default_style_degree: float = 1.0,
        max_retries: int = 4,
        request_func: Optional[Callable[..., Any]] = None,
        sleep_func: Callable[[float], None] = time.sleep,
        monotonic_func: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(device="cloud")
        self.subscription_key = (subscription_key or "").strip()
        self.region = (region or "").strip().lower()
        if not self.subscription_key:
            raise AzureSpeechError("Azure Speech resource key is required.")
        if not self.region or not _REGION_RE.fullmatch(self.region):
            raise AzureSpeechError(
                "Azure Speech region is required and must look like 'eastus' or 'west-europe'."
            )
        if output_format not in AZURE_SPEECH_OUTPUT_FORMATS:
            raise AzureSpeechError(f"Unsupported Azure Speech output format: {output_format}")

        self.output_format = output_format
        self.timeout = max(10, min(int(timeout or 60), 600))
        self.max_parallel = max(1, min(int(max_parallel or 1), 8))
        self.requests_per_minute = max(0, min(int(requests_per_minute or 0), 60000))
        self.default_voice = (default_voice or DEFAULT_AZURE_SPEECH_VOICE).strip()
        self.default_style = self._safe_ssml_token(default_style)
        self.default_role = self._safe_ssml_token(default_role)
        self.default_style_degree = self._clamp_float(default_style_degree, 0.01, 2.0, 1.0)
        self.max_retries = max(0, min(int(max_retries), 8))
        self._request = request_func or requests.request
        self._sleep = sleep_func
        self._monotonic = monotonic_func
        self._rate_lock = threading.Lock()
        self._request_times: deque[float] = deque()

    @property
    def sample_rate(self) -> int:
        return AZURE_SPEECH_OUTPUT_FORMATS[self.output_format]

    @property
    def synthesis_url(self) -> str:
        return f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"

    @property
    def voices_url(self) -> str:
        return f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/voices/list"

    def list_voices(self) -> List[Dict[str, Any]]:
        """Return a normalized, region-specific Azure voice catalog."""
        response = self._request_with_retries(
            "GET",
            self.voices_url,
            headers={"Ocp-Apim-Subscription-Key": self.subscription_key},
            rate_limited=False,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise AzureSpeechError("Azure Speech returned an invalid voice catalog.") from exc
        if not isinstance(payload, list):
            raise AzureSpeechError("Azure Speech returned an unexpected voice catalog response.")

        voices: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            short_name = str(item.get("ShortName") or "").strip()
            locale = str(item.get("Locale") or "").strip()
            if not short_name or not locale:
                continue
            voices.append(
                {
                    "short_name": short_name,
                    "display_name": str(item.get("DisplayName") or short_name).strip(),
                    "local_name": str(item.get("LocalName") or item.get("DisplayName") or short_name).strip(),
                    "gender": str(item.get("Gender") or "Unknown").strip(),
                    "locale": locale,
                    "locale_name": str(item.get("LocaleName") or locale).strip(),
                    "styles": self._string_list(item.get("StyleList")),
                    "roles": self._string_list(item.get("RolePlayList")),
                    "secondary_locales": self._string_list(item.get("SecondaryLocaleList")),
                    "sample_rate_hertz": self._optional_int(item.get("SampleRateHertz")),
                    "voice_type": str(item.get("VoiceType") or "").strip(),
                    "status": str(item.get("Status") or "").strip(),
                    "words_per_minute": self._optional_int(item.get("WordsPerMinute")),
                }
            )
        voices.sort(key=lambda voice: (voice["locale_name"].casefold(), voice["display_name"].casefold()))
        return voices

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
        """Generate a ready-to-serve RIFF/WAV payload for preview requests."""
        options = voice_options or {}
        assignment = VoiceAssignment(
            voice=voice or self.default_voice,
            lang_code=lang_code,
            fx_payload=(
                {"pitch": fx_settings.pitch_semitones}
                if fx_settings and abs(fx_settings.pitch_semitones) > 1e-3
                else None
            ),
            speed_override=speed,
            extra=options,
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
        """Render all chunks and return their WAV paths in chronological order."""
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        work_items: List[Dict[str, Any]] = []
        for segment_index, segment in enumerate(segments):
            speaker = segment.get("speaker") or "default"
            assignment = self._voice_assignment_for(voice_config, speaker)
            for chunk_index, chunk_text in enumerate(segment.get("chunks") or []):
                work_items.append(
                    {
                        "order_index": len(work_items),
                        "segment_index": segment_index,
                        "chunk_index": chunk_index,
                        "speaker": speaker,
                        "text": chunk_text,
                        "assignment": assignment,
                    }
                )

        if not work_items:
            return []

        def render(item: Dict[str, Any]) -> tuple[int, str]:
            audio_bytes = self._synthesize(item["text"], item["assignment"], fallback_speed=speed)
            output_path = destination / f"azure_chunk_{start_index + item['order_index']:06d}.wav"
            output_path.write_bytes(audio_bytes)
            return item["order_index"], str(output_path)

        results: Dict[int, str] = {}
        workers = max(1, min(int(parallel_workers or 1), self.max_parallel, len(work_items)))
        if workers == 1:
            completed = ((item, render(item)) for item in work_items)
            for item, (order_index, file_path) in completed:
                results[order_index] = file_path
                self._notify_callbacks(item, file_path, progress_cb, chunk_cb)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="azure-tts") as executor:
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
                            order_index, file_path = future.result()
                            results[order_index] = file_path
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
        """Azure REST keeps no model or device resources resident."""

    def build_ssml(
        self,
        text: str,
        assignment: VoiceAssignment,
        *,
        fallback_speed: float = 1.0,
    ) -> str:
        """Build escaped SSML from a normalized TTS-Story voice assignment."""
        clean_text = (text or "").strip()
        if not clean_text:
            raise AzureSpeechError("Azure Speech cannot synthesize empty text.")

        extra = assignment.extra or {}
        voice = (assignment.voice or self.default_voice).strip()
        if not voice:
            raise AzureSpeechError("An Azure Speech voice is required.")
        locale = self._resolve_locale(assignment.lang_code, extra.get("locale"), voice)
        rate = self._clamp_float(
            assignment.speed_override if assignment.speed_override is not None else fallback_speed,
            0.5,
            2.0,
            1.0,
        )
        fx = VoiceFXSettings.from_payload(assignment.fx_payload)
        pitch = self._clamp_float(fx.pitch_semitones if fx else 0.0, -12.0, 12.0, 0.0)
        volume = self._clamp_float(extra.get("volume", 0.0), -100.0, 100.0, 0.0)
        style = self._safe_ssml_token(extra.get("style")) or self.default_style
        role = self._safe_ssml_token(extra.get("role")) or self.default_role
        style_degree = self._clamp_float(
            extra.get("style_degree", self.default_style_degree),
            0.01,
            2.0,
            self.default_style_degree,
        )

        body = escape(clean_text)
        prosody_attributes = []
        if abs(rate - 1.0) > 1e-3:
            prosody_attributes.append(f'rate={quoteattr(f"{rate:.3f}")}')
        if abs(pitch) > 1e-3:
            prosody_attributes.append(f'pitch={quoteattr(f"{pitch:+.2f}st")}')
        if abs(volume) > 1e-3:
            prosody_attributes.append(f'volume={quoteattr(f"{volume:+.2f}%")}')
        if prosody_attributes:
            body = f"<prosody {' '.join(prosody_attributes)}>{body}</prosody>"

        express_attributes = []
        if style:
            express_attributes.append(f"style={quoteattr(style)}")
            express_attributes.append(f'styledegree={quoteattr(f"{style_degree:.2f}")}')
        if role:
            express_attributes.append(f"role={quoteattr(role)}")
        if express_attributes:
            body = f"<mstts:express-as {' '.join(express_attributes)}>{body}</mstts:express-as>"

        return (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xmlns:mstts="https://www.w3.org/2001/mstts" '
            f'xml:lang={quoteattr(locale)}>'
            f'<voice name={quoteattr(voice)}>{body}</voice>'
            "</speak>"
        )

    def _synthesize(self, text: str, assignment: VoiceAssignment, *, fallback_speed: float) -> bytes:
        ssml = self.build_ssml(text, assignment, fallback_speed=fallback_speed)
        response = self._request_with_retries(
            "POST",
            self.synthesis_url,
            headers={
                "Ocp-Apim-Subscription-Key": self.subscription_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": self.output_format,
                "User-Agent": "TTS-Story",
            },
            data=ssml.encode("utf-8"),
            rate_limited=True,
        )
        audio = bytes(response.content or b"")
        if len(audio) < 12 or audio[:4] != b"RIFF":
            raise AzureSpeechError("Azure Speech returned an invalid WAV response.")
        return audio

    def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        headers: Dict[str, str],
        data: Optional[bytes] = None,
        rate_limited: bool,
    ):
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            if rate_limited:
                self._wait_for_rate_limit()
            try:
                response = self._request(
                    method,
                    url,
                    headers=headers,
                    data=data,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep(min(2**attempt, 20))
                continue

            if 200 <= response.status_code < 300:
                return response
            if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                self._sleep(self._retry_delay(response, attempt))
                continue
            raise AzureSpeechError(self._response_error(response))

        if last_error is not None:
            raise AzureSpeechError(f"Unable to reach Azure Speech: {last_error}") from last_error
        raise AzureSpeechError("Azure Speech request failed after retries.")

    def _wait_for_rate_limit(self) -> None:
        if self.requests_per_minute <= 0:
            return
        while True:
            wait_seconds = 0.0
            with self._rate_lock:
                now = self._monotonic()
                while self._request_times and now - self._request_times[0] >= 60.0:
                    self._request_times.popleft()
                if len(self._request_times) < self.requests_per_minute:
                    self._request_times.append(now)
                    return
                wait_seconds = max(0.01, 60.0 - (now - self._request_times[0]))
            logger.info("Azure Speech rate limit reached; waiting %.1f seconds.", wait_seconds)
            self._sleep(wait_seconds)

    @staticmethod
    def _retry_delay(response: Any, attempt: int) -> float:
        retry_after = str((getattr(response, "headers", {}) or {}).get("Retry-After") or "").strip()
        if retry_after:
            try:
                return max(0.0, min(float(retry_after), 120.0))
            except ValueError:
                pass
        return float(min(2**attempt, 30))

    @staticmethod
    def _response_error(response: Any) -> str:
        status = int(getattr(response, "status_code", 0) or 0)
        if status == 401:
            return "Azure Speech rejected the resource key or region (401). Check both settings."
        if status == 403:
            return "Azure Speech denied this request (403). Check the Speech resource and selected voice."
        if status == 429:
            return "Azure Speech rate limit reached (429). Reduce parallel requests or try again later."
        detail = str(getattr(response, "text", "") or "").strip().replace("\n", " ")[:300]
        return f"Azure Speech request failed ({status}){': ' + detail if detail else '.'}"

    def _voice_assignment_for(self, voice_config: Dict[str, Dict], speaker: str) -> VoiceAssignment:
        payload = voice_config.get(speaker) or voice_config.get("default") or {}
        return VoiceAssignment(
            voice=payload.get("voice") or self.default_voice,
            lang_code=payload.get("lang_code"),
            fx_payload=payload.get("fx"),
            speed_override=payload.get("speed"),
            extra=payload.get("extra") or {},
        )

    @staticmethod
    def _notify_callbacks(item: Dict[str, Any], file_path: str, progress_cb, chunk_cb) -> None:
        if callable(progress_cb):
            progress_cb()
        if callable(chunk_cb):
            chunk_cb(
                item["order_index"],
                {
                    "speaker": item["speaker"],
                    "text": item["text"],
                    "segment_index": item["segment_index"],
                    "chunk_index": item["chunk_index"],
                    "order_index": item["order_index"],
                },
                file_path,
            )

    @staticmethod
    def _resolve_locale(lang_code: Any, extra_locale: Any, voice: str) -> str:
        for candidate in (extra_locale, lang_code):
            value = str(candidate or "").strip()
            if value and _LOCALE_RE.fullmatch(value):
                return value
        parts = voice.split("-")
        inferred = "-".join(parts[:2]) if len(parts) >= 2 else "en-US"
        return inferred if _LOCALE_RE.fullmatch(inferred) else "en-US"

    @staticmethod
    def _safe_ssml_token(value: Any) -> str:
        token = str(value or "").strip()
        if not token or token.lower() in {"default", "neutral", "none"}:
            return ""
        return token if _SSML_TOKEN_RE.fullmatch(token) else ""

    @staticmethod
    def _clamp_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = fallback
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _string_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


__all__ = [
    "AzureSpeechEngine",
    "AzureSpeechError",
    "AZURE_SPEECH_OUTPUT_FORMATS",
    "DEFAULT_AZURE_SPEECH_OUTPUT_FORMAT",
    "DEFAULT_AZURE_SPEECH_REQUESTS_PER_MINUTE",
    "DEFAULT_AZURE_SPEECH_VOICE",
]
