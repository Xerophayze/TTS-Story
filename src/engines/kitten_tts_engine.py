"""KittenTTS engine adapter.

KittenTTS is a lightweight CPU-optimized TTS engine. The recommended mini
model is about 80 MB; the experimental nano-int8 option is about 25 MB.
Install: pip install https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from .base import EngineCapabilities, TtsEngineBase, VoiceAssignment
from ..audio_effects import AudioPostProcessor, VoiceFXSettings

logger = logging.getLogger(__name__)

KITTEN_TTS_BUILTIN_VOICES = {"Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"}
KITTEN_TTS_DEFAULT_MODEL = "KittenML/kitten-tts-mini-0.8"
KITTEN_TTS_SAMPLE_RATE = 24000
KITTEN_TTS_CACHE_DIR = Path(__file__).resolve().parents[2] / "models" / "kitten_tts"


def _configure_huggingface_downloads() -> None:
    """Keep first-use model downloads bounded and Pinokio-friendly."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("KITTEN_TTS_CACHE_DIR", str(KITTEN_TTS_CACHE_DIR))


_configure_huggingface_downloads()
try:
    from kittentts import KittenTTS as _KittenTTS  # type: ignore

    KITTEN_TTS_AVAILABLE = True
except ImportError:
    _KittenTTS = None  # type: ignore[assignment]
    KITTEN_TTS_AVAILABLE = False


class KittenTTSEngine(TtsEngineBase):
    """Local KittenTTS engine — CPU-optimized, no GPU required."""

    name = "kitten_tts"
    capabilities = EngineCapabilities(
        supports_voice_cloning=False,
        supports_emotion_tags=False,
        supported_languages=["en"],
    )

    def __init__(
        self,
        *,
        model_id: str = KITTEN_TTS_DEFAULT_MODEL,
        default_voice: str = "Jasper",
        **_kwargs,
    ) -> None:
        if not KITTEN_TTS_AVAILABLE:
            raise ImportError(
                "kittentts is not installed. Run: "
                "pip install https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl"
            )

        self.model_id = model_id
        self.default_voice = default_voice if default_voice in KITTEN_TTS_BUILTIN_VOICES else "Jasper"
        self.post_processor = AudioPostProcessor()
        cache_dir = Path(os.environ.get("KITTEN_TTS_CACHE_DIR") or KITTEN_TTS_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading KittenTTS model: %s (cache: %s)", model_id, cache_dir)
        try:
            self._model = _KittenTTS(model_id, cache_dir=str(cache_dir))
        except Exception as exc:
            raise RuntimeError(
                "KittenTTS model download/load failed. Check network access to Hugging Face, "
                "then rerun setup.bat to prefetch the model cache. "
                "Set PREFETCH_KITTEN_TTS_MODEL=0 to skip setup prefetch if needed. "
                f"Model: {model_id}; cache: {cache_dir}; error: {exc}"
            ) from exc
        logger.info("KittenTTS model loaded (default voice: %s)", self.default_voice)

    @property
    def sample_rate(self) -> int:
        return KITTEN_TTS_SAMPLE_RATE

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
        pause_cb=None,
    ) -> List[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files: List[str] = []
        chunk_index = 0

        for seg_idx, segment in enumerate(segments):
            speaker = segment.get("speaker")
            chunks = segment.get("chunks") or []
            assignment = self._voice_assignment_for(voice_config, speaker)
            voice = self._resolve_voice(assignment)
            fx_settings = VoiceFXSettings.from_payload(assignment.fx_payload)

            for chunk_idx, chunk_text in enumerate(chunks):
                if callable(pause_cb) and pause_cb():
                    return files

                output_path = output_dir / f"chunk_{chunk_index:04d}.wav"
                try:
                    audio_np = self._synthesize(chunk_text, voice, fx_settings)
                    sf.write(str(output_path), audio_np, self.sample_rate)
                    files.append(str(output_path))
                except Exception as exc:
                    logger.error(
                        "KittenTTS failed on chunk %d (speaker=%s): %s",
                        chunk_index, speaker, exc,
                    )
                    raise

                if callable(progress_cb):
                    try:
                        progress_cb()
                    except Exception:
                        # JobPaused or other cancellation exception - re-raise so caller handles it
                        raise
                if callable(chunk_cb):
                    chunk_cb(chunk_idx, {
                        "speaker": speaker,
                        "text": chunk_text,
                        "segment_index": seg_idx,
                        "chunk_index": chunk_idx,
                    }, str(output_path))

                chunk_index += 1

        return files

    def generate_audio(
        self,
        *,
        text: str,
        voice: str,
        lang_code: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: Optional[int] = None,
        fx_settings: Optional[VoiceFXSettings] = None,
    ) -> np.ndarray:
        resolved_voice = voice if voice in KITTEN_TTS_BUILTIN_VOICES else self.default_voice
        return self._synthesize(text, resolved_voice, fx_settings)

    def cleanup(self) -> None:
        logger.info("Cleaning up KittenTTS engine resources")
        try:
            if hasattr(self, "_model") and self._model is not None:
                del self._model
                self._model = None
        except Exception:
            pass

    def _synthesize(
        self,
        text: str,
        voice: str,
        fx_settings: Optional[VoiceFXSettings] = None,
    ) -> np.ndarray:
        audio = self._model.generate(text, voice=voice)
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        audio = audio.astype(np.float32, copy=False)
        return self.post_processor.apply_post_pipeline(audio, self.sample_rate, fx_settings)

    def _resolve_voice(self, assignment: VoiceAssignment) -> str:
        candidate = (assignment.voice or "").strip()
        if candidate in KITTEN_TTS_BUILTIN_VOICES:
            return candidate
        return self.default_voice

    def _voice_assignment_for(self, voice_config: Dict[str, Dict], speaker: Optional[str]) -> VoiceAssignment:
        speaker_key = (speaker or "").strip().lower()
        payload = (
            voice_config.get(speaker)
            or voice_config.get(speaker_key)
            or voice_config.get("default")
            or {}
        )
        return VoiceAssignment(
            voice=payload.get("voice"),
            lang_code=payload.get("lang_code"),
            audio_prompt_path=payload.get("audio_prompt_path"),
            fx_payload=payload.get("fx"),
            speed_override=payload.get("speed"),
            extra=payload.get("extra") or {},
        )


__all__ = [
    "KittenTTSEngine",
    "KITTEN_TTS_AVAILABLE",
    "KITTEN_TTS_BUILTIN_VOICES",
    "KITTEN_TTS_DEFAULT_MODEL",
]
