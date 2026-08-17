"""Lazy local transcription for reusable reference voice samples."""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Callable, Optional


class VoiceTranscriptionError(RuntimeError):
    """Raised when a reference clip cannot be transcribed."""


class VoiceTranscriptGenerator:
    """Transcribe short reference clips with a lazily loaded SenseVoice model."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        model_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.device = str(device or "cpu")
        self._model_factory = model_factory
        self._model = None
        self._lock = threading.RLock()

    def _load_model(self):
        if self._model is not None:
            return self._model
        factory = self._model_factory
        if factory is None:
            try:
                from funasr import AutoModel as factory
            except ImportError as exc:
                raise VoiceTranscriptionError(
                    "Automatic transcription is unavailable because funasr is not installed. "
                    "Run setup or install-update, then restart TTS-Story."
                ) from exc
        try:
            self._model = factory(
                model="iic/SenseVoiceSmall",
                trust_remote_code=True,
                device=self.device,
                disable_update=True,
            )
        except Exception as exc:
            raise VoiceTranscriptionError(
                f"SenseVoice could not be loaded: {exc}"
            ) from exc
        return self._model

    def transcribe(self, audio_path: Path | str) -> str:
        path = Path(audio_path)
        if not path.is_file():
            raise VoiceTranscriptionError("The selected voice sample file is missing.")
        with self._lock:
            model = self._load_model()
            try:
                result = model.generate(input=str(path), batch_size_s=0)
            except Exception as exc:
                raise VoiceTranscriptionError(
                    f"SenseVoice could not transcribe this sample: {exc}"
                ) from exc
        transcript = ""
        if isinstance(result, list) and result and isinstance(result[0], dict):
            transcript = str(result[0].get("text") or "")
        elif isinstance(result, dict):
            transcript = str(result.get("text") or "")
        transcript = re.sub(r"<\|[^|]+\|>", "", transcript)
        transcript = re.sub(r"\s+", " ", transcript).strip()
        if not transcript:
            raise VoiceTranscriptionError(
                "No speech was detected in this sample. Check the recording or enter the transcript manually."
            )
        return transcript


__all__ = ["VoiceTranscriptionError", "VoiceTranscriptGenerator"]
