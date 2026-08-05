"""Experimental Microsoft Edge online text-to-speech adapter."""
from __future__ import annotations

import asyncio
import io
import inspect
import logging
import queue
import threading
import time
import wave
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..audio_effects import VoiceFXSettings
from .base import EngineCapabilities, TtsEngineBase, VoiceAssignment
from .cloud_audio import CloudAudioError, apply_wav_effects, audio_bytes_to_wav

try:  # The app remains importable before install/update has installed edge-tts.
    import edge_tts

    EDGE_TTS_AVAILABLE = True
    EDGE_TTS_UNAVAILABLE_REASON = ""
except Exception as exc:  # pragma: no cover - depends on optional installation state
    edge_tts = None
    EDGE_TTS_AVAILABLE = False
    EDGE_TTS_UNAVAILABLE_REASON = str(exc)


DEFAULT_EDGE_TTS_VOICE = "en-US-AriaNeural"
DEFAULT_EDGE_TTS_SAMPLE_RATE = 24000
logger = logging.getLogger(__name__)


class EdgeTTSError(RuntimeError):
    """Raised when the Edge speech service cannot complete a request."""


def _run_awaitable(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="edge-async") as executor:
        return executor.submit(asyncio.run, value).result()


class EdgeTTSEngine(TtsEngineBase):
    """Render TTS-Story chunks through the consumer Edge speech endpoint."""

    name = "edge_tts"
    capabilities = EngineCapabilities(False, False, None)

    def __init__(
        self,
        *,
        default_voice: str = DEFAULT_EDGE_TTS_VOICE,
        timeout: int = 60,
        max_parallel: int = 2,
        max_retries: int = 2,
        default_volume: int = 0,
        communicate_factory: Optional[Callable[..., Any]] = None,
        list_voices_func: Optional[Callable[..., Any]] = None,
        sleep_func: Callable[[float], None] = time.sleep,
        audio_converter: Callable[..., bytes] = audio_bytes_to_wav,
    ) -> None:
        super().__init__(device="cloud")
        if communicate_factory is None and not EDGE_TTS_AVAILABLE:
            raise EdgeTTSError(
                "Edge TTS is not installed. Run install-update.bat, then restart TTS-Story."
            )
        self.default_voice = str(default_voice or DEFAULT_EDGE_TTS_VOICE).strip()
        self.timeout = max(10, min(int(timeout or 60), 300))
        self.max_parallel = max(1, min(int(max_parallel or 1), 8))
        self.max_retries = max(0, min(int(max_retries or 0), 6))
        self.default_volume = max(-100, min(int(default_volume or 0), 100))
        self._communicate = communicate_factory or edge_tts.Communicate
        self._list_voices = list_voices_func or edge_tts.list_voices
        self._sleep = sleep_func
        self._audio_converter = audio_converter

    @property
    def sample_rate(self) -> int:
        return DEFAULT_EDGE_TTS_SAMPLE_RATE

    def list_voices(self) -> List[Dict[str, Any]]:
        try:
            payload = _run_awaitable(self._list_voices())
        except Exception as exc:
            raise EdgeTTSError(f"Unable to retrieve Edge TTS voices: {exc}") from exc
        voices: List[Dict[str, Any]] = []
        for item in payload or []:
            if not isinstance(item, dict):
                continue
            short_name = str(item.get("ShortName") or item.get("Name") or "").strip()
            locale = str(item.get("Locale") or "").strip()
            if not short_name or not locale:
                continue
            voice_tag = item.get("VoiceTag") if isinstance(item.get("VoiceTag"), dict) else {}
            voices.append(
                {
                    "short_name": short_name,
                    "display_name": str(item.get("FriendlyName") or short_name).strip(),
                    "local_name": str(item.get("FriendlyName") or short_name).strip(),
                    "gender": str(item.get("Gender") or "Unknown").strip(),
                    "locale": locale,
                    "locale_name": locale,
                    "content_categories": self._string_list(voice_tag.get("ContentCategories")),
                    "personalities": self._string_list(voice_tag.get("VoicePersonalities")),
                }
            )
        voices.sort(key=lambda voice: (voice["locale"].casefold(), voice["display_name"].casefold()))
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
        fx_payload = None
        if fx_settings:
            fx_payload = {
                "enabled": True,
                "pitch": fx_settings.pitch_semitones,
                "tone": fx_settings.tone,
                "speed": 1.0,
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
            wav = self._synthesize(item["text"], item["assignment"], fallback_speed=speed)
            path = destination / f"edge_chunk_{start_index + item['order_index']:06d}.wav"
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
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="edge-tts") as executor:
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
                            item = pending.pop(future)
                            order, path = future.result()
                            results[order] = path
                        while next_callback in results:
                            self._notify_callbacks(
                                work_items[next_callback],
                                results[next_callback],
                                progress_cb,
                                chunk_cb,
                            )
                            next_callback += 1
                        fill_workers()
                except Exception:
                    for future in pending:
                        future.cancel()
                    raise
        return [results[index] for index in range(len(work_items))]

    def cleanup(self) -> None:
        """The remote Edge service keeps no local model resources."""

    def _synthesize(self, text: str, assignment: VoiceAssignment, *, fallback_speed: float) -> bytes:
        clean_text = str(text or "").strip()
        if not clean_text:
            raise EdgeTTSError("Edge TTS cannot synthesize empty text.")
        if not any(character.isalnum() for character in clean_text):
            logger.info(
                "Edge TTS converted a symbol-only separator (%d chars) to silence.",
                len(clean_text),
            )
            return self._silence_wav()
        voice = str(assignment.voice or self.default_voice).strip()
        if not voice:
            raise EdgeTTSError("An Edge TTS voice is required.")
        speed = self._clamp_float(
            assignment.speed_override if assignment.speed_override is not None else fallback_speed,
            0.5,
            2.0,
            1.0,
        )
        volume = self._clamp_float((assignment.extra or {}).get("volume"), -100, 100, self.default_volume)
        rate_percent = max(-50, min(100, round((speed - 1.0) * 100)))
        kwargs = {
            "voice": voice,
            "rate": f"{rate_percent:+d}%",
            "volume": f"{round(volume):+d}%",
            "connect_timeout": self.timeout,
            "receive_timeout": self.timeout,
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                communicator = self._communicate(clean_text, **kwargs)
                mp3 = b"".join(self._audio_chunks(communicator))
                wav = self._audio_converter(
                    mp3,
                    input_format="mp3",
                    sample_rate=self.sample_rate,
                    channels=1,
                )
                fx = VoiceFXSettings.from_payload(assignment.fx_payload)
                if fx:
                    fx.speed = 1.0
                return apply_wav_effects(wav, fx)
            except (CloudAudioError, ValueError) as exc:
                raise EdgeTTSError(str(exc)) from exc
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = min(2**attempt, 10)
                logger.warning(
                    "Edge TTS request failed for voice %s (%d chars), retrying in %ss "
                    "(attempt %d/%d): %s",
                    voice,
                    len(clean_text),
                    delay,
                    attempt + 2,
                    self.max_retries + 1,
                    exc,
                )
                self._sleep(delay)
        raise EdgeTTSError(f"Edge TTS synthesis failed: {last_error}") from last_error

    def _audio_chunks(self, communicator: Any) -> Iterable[bytes]:
        if hasattr(communicator, "stream"):
            async def collect() -> List[Dict[str, Any]]:
                return [message async for message in communicator.stream()]

            try:
                messages = _run_awaitable(asyncio.wait_for(collect(), timeout=self.timeout))
            except asyncio.TimeoutError as exc:
                raise EdgeTTSError(
                    f"Edge TTS streaming timed out after {self.timeout} seconds."
                ) from exc
        elif hasattr(communicator, "stream_sync"):
            messages = self._collect_sync_with_timeout(communicator)
        else:
            raise EdgeTTSError("The installed edge-tts package does not provide a streaming API.")
        found = False
        for message in messages:
            if isinstance(message, dict) and message.get("type") == "audio":
                data = bytes(message.get("data") or b"")
                if data:
                    found = True
                    yield data
        if not found:
            raise EdgeTTSError("Edge TTS returned no audio data.")

    def _collect_sync_with_timeout(self, communicator: Any) -> List[Dict[str, Any]]:
        """Collect a legacy synchronous stream without allowing it to block the job forever."""
        outcome: queue.Queue = queue.Queue(maxsize=1)

        def collect() -> None:
            try:
                outcome.put((True, list(communicator.stream_sync())))
            except BaseException as exc:  # delivered back to the caller thread
                outcome.put((False, exc))

        worker = threading.Thread(target=collect, name="edge-stream-timeout", daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise EdgeTTSError(f"Edge TTS streaming timed out after {self.timeout} seconds.")
        try:
            succeeded, value = outcome.get_nowait()
        except queue.Empty as exc:
            raise EdgeTTSError("Edge TTS streaming stopped without returning a result.") from exc
        if not succeeded:
            raise value
        return value

    def _silence_wav(self, duration_ms: int = 250) -> bytes:
        """Return a short valid WAV for scene-break markers such as ``***``."""
        frames = max(1, int(self.sample_rate * max(1, duration_ms) / 1000))
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(b"\x00\x00" * frames)
        return output.getvalue()

    def _work_items(self, segments: List[Dict], voice_config: Dict[str, Dict]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for segment_index, segment in enumerate(segments):
            speaker = segment.get("speaker") or "default"
            payload = voice_config.get(speaker) or voice_config.get("default") or {}
            assignment = VoiceAssignment(
                voice=payload.get("voice") or self.default_voice,
                lang_code=payload.get("lang_code"),
                fx_payload=payload.get("fx"),
                speed_override=payload.get("speed"),
                extra=payload.get("extra") or {},
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


__all__ = [
    "DEFAULT_EDGE_TTS_SAMPLE_RATE",
    "DEFAULT_EDGE_TTS_VOICE",
    "EDGE_TTS_AVAILABLE",
    "EDGE_TTS_UNAVAILABLE_REASON",
    "EdgeTTSEngine",
    "EdgeTTSError",
]
