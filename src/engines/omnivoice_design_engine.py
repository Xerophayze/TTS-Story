"""OmniVoice voice-design engine adapter (subprocess-isolated).

OmniVoice requires torch==2.8 and transformers==5.3, which conflict with the
main project dependencies.  This adapter runs OmniVoice in an isolated venv
under engines/omnivoice/ via subprocess.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from .base import EngineCapabilities, TtsEngineBase, VoiceAssignment
from ..audio_effects import AudioPostProcessor, VoiceFXSettings
from .omnivoice_clone_engine import (
    OMNIVOICE_AVAILABLE,
    _OMNIVOICE_UNAVAILABLE_REASON,
    _ENGINE_ROOT,
    _WORKER,
    OMNIVOICE_SAMPLE_RATE,
    _find_venv_python,
)

logger = logging.getLogger(__name__)


class OmniVoiceDesignEngine(TtsEngineBase):
    """OmniVoice voice-design engine (subprocess-isolated).

    Generates speech from a free-text *instruct* string describing speaker
    attributes (e.g. "female, low pitch, british accent") — no reference audio
    needed.
    """

    name = "omnivoice_design"
    capabilities = EngineCapabilities(
        supports_voice_cloning=False,
        supports_emotion_tags=True,
    )

    def __init__(
        self,
        *,
        device: str = "auto",
        model_id: str = "k2-fsa/OmniVoice",
        dtype: str = "float16",
        num_step: int = 32,
        default_instruct: Optional[str] = None,
        post_process: bool = True,
        duration_safety_margin: float = 0.25,
    ):
        if not OMNIVOICE_AVAILABLE:
            raise ImportError(
                f"OmniVoice is not set up. {_OMNIVOICE_UNAVAILABLE_REASON}"
            )

        self._python = _find_venv_python(_ENGINE_ROOT)
        self._worker = _WORKER
        self.device = device
        self.model_id = model_id
        self.dtype = dtype
        self.num_step = num_step
        self.default_instruct = default_instruct
        self.post_process = post_process
        self.duration_safety_margin = max(0.0, min(float(duration_safety_margin), 2.0))
        self.post_processor = AudioPostProcessor()

        logger.info(
            "OmniVoiceDesignEngine ready (subprocess) model=%s device=%s",
            model_id, device,
        )

    @property
    def sample_rate(self) -> int:
        return OMNIVOICE_SAMPLE_RATE

    def generate_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        lang_code: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: Optional[int] = None,
        fx_settings=None,
        **_kwargs,
    ) -> np.ndarray:
        """Single-clip synthesis via subprocess worker."""
        instruct = voice or self.default_instruct or ""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = str(Path(tmp_dir) / "preview.wav")
            job = {
                "mode": "design",
                "model_id": self.model_id,
                "device": self.device,
                "dtype": self.dtype,
                "num_step": self.num_step,
                "speed": speed,
                "post_process": self.post_process,
                "duration_safety_margin": self.duration_safety_margin,
                "text": text,
                "instruct": instruct,
                "output_path": output_path,
            }
            self._run_worker(job)
            audio, sr = sf.read(output_path, dtype="float32")
        return self.post_processor.apply_post_pipeline(audio, self.sample_rate, None)

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
        cancel_cb=None,
    ) -> List[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files: List[str] = []
        chunk_index = 0

        for seg_idx, segment in enumerate(segments):
            speaker = segment["speaker"]
            chunks = segment["chunks"]
            assignment = self._voice_assignment_for(voice_config, speaker)
            fx_settings = VoiceFXSettings.from_payload(assignment.fx_payload)
            spd = assignment.speed_override or speed

            segment_emotion = segment.get("emotion")
            user_instruct = (
                assignment.extra.get("instruct") if assignment.extra else None
            ) or assignment.voice or self.default_instruct

            instruct = (
                f"{segment_emotion}; {user_instruct}" if user_instruct else segment_emotion
            ) if segment_emotion else (user_instruct or "")

            for chunk_idx, chunk_text in enumerate(chunks):
                output_path = output_dir / f"chunk_{chunk_index:04d}.wav"
                job = {
                    "mode": "design",
                    "model_id": self.model_id,
                    "device": self.device,
                    "dtype": self.dtype,
                    "num_step": self.num_step,
                    "speed": spd,
                    "post_process": self.post_process,
                    "duration_safety_margin": self.duration_safety_margin,
                    "text": chunk_text,
                    "instruct": instruct,
                    "output_path": str(output_path),
                }
                self._run_worker(job)
                audio, sr = sf.read(str(output_path), dtype="float32")
                audio = self.post_processor.apply_post_pipeline(
                    audio, self.sample_rate, fx_settings
                )
                sf.write(str(output_path), audio, self.sample_rate)
                files.append(str(output_path))
                chunk_index += 1
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

        return files

    def cleanup(self) -> None:
        pass  # No persistent model in adapter process

    def _run_worker(self, job: Dict) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tf:
            json.dump(job, tf)
            job_file = tf.name

        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            result = subprocess.run(
                [str(self._python), str(self._worker), "--job-file", job_file],
                capture_output=True,
                text=True,
                cwd=str(_ENGINE_ROOT),
                env=env,
            )
            for line in result.stderr.splitlines():
                logger.info("[omnivoice worker] %s", line)
            if result.returncode != 0:
                err = result.stderr[-2000:] if result.stderr else "(no stderr)"
                raise RuntimeError(
                    f"OmniVoice worker failed (exit {result.returncode}):\n{err}"
                )
        finally:
            Path(job_file).unlink(missing_ok=True)

    def _voice_assignment_for(
        self, voice_config: Dict[str, Dict], speaker: str
    ) -> VoiceAssignment:
        payload = voice_config.get(speaker) or voice_config.get("default") or {}
        return VoiceAssignment(
            voice=payload.get("voice"),
            lang_code=payload.get("lang_code"),
            audio_prompt_path=payload.get("audio_prompt_path"),
            fx_payload=payload.get("fx"),
            speed_override=payload.get("speed"),
            extra=payload.get("extra") or {},
        )
