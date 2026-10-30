"""Lazy local transcription for reusable reference voice samples."""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "FunAudioLLM/SenseVoiceSmall"
MODEL_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
TRANSCRIPT_EVENT = "TTS_STORY_TRANSCRIPT "


def transcription_runtime_root() -> Path:
    return ROOT / "engines" / "voice_transcription"


def transcription_python() -> Path:
    return transcription_runtime_root() / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def transcription_available() -> bool:
    return (transcription_python().is_file()
            and (transcription_runtime_root() / ".ready").is_file()
            and (ROOT / "engines" / "transcription_worker.py").is_file())


class VoiceTranscriptionError(RuntimeError):
    """Raised when a reference clip cannot be transcribed."""


class VoiceTranscriptionUnavailable(VoiceTranscriptionError):
    """The optional, isolated transcription tool has not been installed."""


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
        self._busy = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._busy.is_set()

    def _transcribe_isolated(self, path: Path) -> str:
        if not transcription_available():
            raise VoiceTranscriptionUnavailable(
                "Automatic transcription needs its separate local tool. Open Settings → Engines → "
                "Utilities → Voice transcription and click Install tool. Reinstalling a TTS engine "
                "does not install this tool. You can also enter the transcript manually."
            )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["HF_HUB_OFFLINE"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            result = subprocess.run(
                [str(transcription_python()), str(ROOT / "engines" / "transcription_worker.py"),
                 "--audio", str(path.resolve())],
                cwd=str(ROOT), env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", timeout=180,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise VoiceTranscriptionError("Local transcription timed out after three minutes; its worker was stopped. Try a shorter reference clip.") from exc
        except OSError as exc:
            raise VoiceTranscriptionError("The transcription worker could not start. Repair Voice transcription in Settings → Engines → Utilities.") from exc
        event = None
        for line in (result.stdout or "").splitlines():
            if line.startswith(TRANSCRIPT_EVENT):
                try:
                    event = json.loads(line[len(TRANSCRIPT_EVENT):])
                except (ValueError, TypeError):
                    continue
        if result.returncode or not isinstance(event, dict) or not event.get("success"):
            detail = event.get("error") if isinstance(event, dict) else None
            raise VoiceTranscriptionError(detail or "Local transcription failed. Repair Voice transcription in Settings → Engines → Utilities if this persists.")
        return self._clean_transcript(event.get("transcript"))

    def _load_model(self):
        if self._model is not None:
            return self._model
        factory = self._model_factory
        if factory is None:
            raise VoiceTranscriptionError("SenseVoice models must be loaded in the isolated transcription worker.")
        try:
            self._model = factory(
                model="iic/SenseVoiceSmall",
                trust_remote_code=False,
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
            self._busy.set()
            try:
                if self._model_factory is None:
                    return self._transcribe_isolated(path)
                # Only the isolated worker (or an injected test factory) loads
                # FunASR; the normal web process never imports its model stack.
                model = self._load_model()
                try:
                    result = model.generate(input=str(path), batch_size_s=0)
                except Exception as exc:
                    raise VoiceTranscriptionError(
                        f"SenseVoice could not transcribe this sample: {exc}"
                    ) from exc
            finally:
                self._busy.clear()
        transcript = ""
        if isinstance(result, list) and result and isinstance(result[0], dict):
            transcript = str(result[0].get("text") or "")
        elif isinstance(result, dict):
            transcript = str(result.get("text") or "")
        return self._clean_transcript(transcript)

    @staticmethod
    def _clean_transcript(transcript: Any) -> str:
        transcript = re.sub(r"<\|[^|]+\|>", "", str(transcript or ""))
        transcript = re.sub(r"\s+", " ", transcript).strip()
        if not transcript:
            raise VoiceTranscriptionError(
                "No speech was detected in this sample. Check the recording or enter the transcript manually."
            )
        return transcript


__all__ = ["VoiceTranscriptionError", "VoiceTranscriptionUnavailable", "VoiceTranscriptGenerator", "transcription_available"]
