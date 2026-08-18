"""Chatterbox Turbo adapter backed by an isolated engine environment."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from .base import EngineCapabilities, TtsEngineBase, VoiceAssignment
from ..audio_effects import AudioPostProcessor, VoiceFXSettings, convert_mp3_to_wav_if_needed


logger = logging.getLogger(__name__)
CHATTERBOX_TURBO_SAMPLE_RATE = 24000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHATTERBOX_ENGINE_ROOT = PROJECT_ROOT / "engines" / "chatterbox"
CHATTERBOX_WORKER = CHATTERBOX_ENGINE_ROOT / "chatterbox_worker.py"
CHATTERBOX_PYTHON = CHATTERBOX_ENGINE_ROOT / ".venv" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)
CHATTERBOX_READY_MARKER = CHATTERBOX_ENGINE_ROOT / ".chatterbox_ready"
CHATTERBOX_TURBO_AVAILABLE = (
    CHATTERBOX_PYTHON.is_file()
    and CHATTERBOX_WORKER.is_file()
    and CHATTERBOX_READY_MARKER.is_file()
)
CHATTERBOX_TURBO_UNAVAILABLE_REASON = "" if CHATTERBOX_TURBO_AVAILABLE else (
    "isolated Chatterbox environment is not installed"
)


class ChatterboxTurboLocalEngine(TtsEngineBase):
    name = "chatterbox_turbo_local"
    capabilities = EngineCapabilities(
        supports_voice_cloning=True,
        supports_emotion_tags=True,
        supported_languages=["en"],
    )

    def __init__(
        self,
        *,
        device: str = "auto",
        default_prompt: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 1000,
        repetition_penalty: float = 1.2,
        cfg_weight: float = 0.0,
        exaggeration: float = 0.0,
        norm_loudness: bool = True,
        prompt_norm_loudness: bool = True,
    ) -> None:
        if not CHATTERBOX_TURBO_AVAILABLE:
            raise ImportError(
                "Chatterbox Turbo's isolated environment is not installed. "
                "Install Chatterbox Local from Settings → Engine Settings."
            )
        self.device = (device or "auto").strip().lower()
        self.default_prompt = (default_prompt or "").strip() or None
        self.defaults = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "cfg_weight": float(cfg_weight),
            "exaggeration": float(exaggeration),
            "norm_loudness": bool(norm_loudness),
            "prompt_norm_loudness": bool(prompt_norm_loudness),
        }
        self.post_processor = AudioPostProcessor()

    @property
    def sample_rate(self) -> int:
        return CHATTERBOX_TURBO_SAMPLE_RATE

    def _resolve_prompt_path(self, path_value: Optional[str]) -> Path:
        candidate = Path(path_value or self.default_prompt or "")
        if candidate.is_file():
            return candidate.resolve()
        fallback = PROJECT_ROOT / "data" / "voice_prompts" / str(path_value or self.default_prompt or "")
        if fallback.is_file():
            return fallback.resolve()
        raise FileNotFoundError(
            f"Chatterbox reference audio was not found: {path_value or self.default_prompt or '(empty)'}. "
            "Assign a voice sample at least five seconds long."
        )

    @staticmethod
    def _assignment(voice_config: Dict[str, Dict], speaker: str) -> VoiceAssignment:
        payload = voice_config.get(speaker) or voice_config.get("default") or {}
        return VoiceAssignment(
            voice=payload.get("voice"),
            lang_code=payload.get("lang_code"),
            audio_prompt_path=payload.get("audio_prompt_path"),
            fx_payload=payload.get("fx"),
            speed_override=payload.get("speed"),
            extra=payload.get("extra") or {},
        )

    def _prepare_item(self, item: Dict[str, Any], assignment: VoiceAssignment, temporary: List[Path]) -> None:
        prompt, converted = convert_mp3_to_wav_if_needed(
            self._resolve_prompt_path(assignment.audio_prompt_path)
        )
        if converted:
            temporary.append(Path(converted))
        fx_settings = (
            assignment.fx_payload
            if isinstance(assignment.fx_payload, VoiceFXSettings)
            else VoiceFXSettings.from_payload(assignment.fx_payload)
        )
        output_fx = fx_settings
        if fx_settings:
            prepared = self.post_processor.prepare_prompt_audio(str(prompt), fx_settings)
            if prepared:
                prompt = Path(prepared)
                temporary.append(prompt)
                output_fx = (
                    VoiceFXSettings(pitch_semitones=0.0, speed=1.0, tone=fx_settings.tone)
                    if fx_settings.tone != "neutral"
                    else None
                )
        item["audio_prompt_path"] = str(prompt)
        item["extra"] = assignment.extra or {}
        item["output_fx"] = output_fx

    def _run_worker(self, items: List[Dict[str, Any]], progress_cb=None, chunk_cb=None) -> List[str]:
        temporary: List[Path] = []
        job_file: Optional[Path] = None
        process: Optional[subprocess.Popen] = None
        completed: set[int] = set()
        output_lines: List[str] = []
        try:
            for item in items:
                self._prepare_item(item, item.pop("assignment"), temporary)
            job_payload = {
                "device": self.device,
                "defaults": self.defaults,
                "items": [
                    {key: value for key, value in item.items() if key not in {"output_fx", "chunk_meta"}}
                    for item in items
                ],
            }
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="tts_story_chatterbox_",
                encoding="utf-8",
                delete=False,
            )
            job_file = Path(handle.name)
            with handle:
                json.dump(job_payload, handle, ensure_ascii=False)
            process = subprocess.Popen(
                [str(CHATTERBOX_PYTHON), str(CHATTERBOX_WORKER), "--job-file", str(job_file)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for raw_line in process.stdout or []:
                line = raw_line.rstrip()
                output_lines.append(line)
                if not line.startswith("TTS_STORY_EVENT "):
                    logger.info("[chatterbox worker] %s", line)
                    continue
                event = json.loads(line[len("TTS_STORY_EVENT "):])
                if event.get("event") != "chunk":
                    continue
                index = int(event["index"])
                item = items[index]
                output_path = Path(event["path"])
                output_fx = item.get("output_fx")
                if output_fx:
                    audio, output_rate = sf.read(output_path, dtype="float32")
                    audio = self.post_processor.apply_post_pipeline(audio, output_rate, output_fx)
                    sf.write(output_path, audio, output_rate)
                completed.add(index)
                if callable(progress_cb):
                    progress_cb()
                if callable(chunk_cb):
                    chunk_cb(item["chunk_index"], item["chunk_meta"], str(output_path))
            return_code = process.wait()
            if return_code != 0:
                details = "\n".join(output_lines[-30:])
                raise RuntimeError(f"Chatterbox isolated worker failed (code {return_code}).\n{details}")
            if len(completed) != len(items):
                raise RuntimeError(
                    f"Chatterbox generated {len(completed)} of {len(items)} requested chunks."
                )
            ordered = sorted(items, key=lambda item: item["original_order_index"])
            return [str(item["output_path"]) for item in ordered]
        except BaseException:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
        finally:
            if job_file:
                job_file.unlink(missing_ok=True)
            for path in temporary:
                path.unlink(missing_ok=True)

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
        group_by_speaker: bool = False,
    ) -> List[str]:
        del speed, sample_rate, parallel_workers
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        items: List[Dict[str, Any]] = []
        for segment_index, segment in enumerate(segments):
            speaker = segment.get("speaker")
            for chunk_index, text in enumerate(segment.get("chunks") or []):
                order_index = len(items)
                items.append({
                    "speaker": speaker,
                    "text": text,
                    "segment_index": segment_index,
                    "chunk_index": chunk_index,
                    "order_index": order_index,
                    "original_order_index": order_index,
                    "output_path": str(output_dir / f"chunk_{order_index:04d}.wav"),
                    "assignment": self._assignment(voice_config, speaker),
                    "chunk_meta": {
                        "speaker": speaker,
                        "text": text,
                        "segment_index": segment_index,
                        "chunk_index": chunk_index,
                        "order_index": order_index,
                    },
                })
        if group_by_speaker:
            items = sorted(items, key=lambda item: (str(item["speaker"]), item["original_order_index"]))
        for worker_index, item in enumerate(items):
            item["order_index"] = worker_index
        return self._run_worker(items, progress_cb=progress_cb, chunk_cb=chunk_cb)

    def generate_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        lang_code: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: Optional[int] = None,
        audio_prompt_path: Optional[str] = None,
        fx_settings=None,
        **_kwargs,
    ) -> np.ndarray:
        del voice, lang_code, speed, sample_rate
        with tempfile.TemporaryDirectory(prefix="tts_story_chatterbox_preview_") as directory:
            output = Path(directory) / "preview.wav"
            item = {
                "speaker": "preview",
                "text": text,
                "segment_index": 0,
                "chunk_index": 0,
                "order_index": 0,
                "original_order_index": 0,
                "output_path": str(output),
                "assignment": VoiceAssignment(
                    voice="",
                    audio_prompt_path=audio_prompt_path or self.default_prompt,
                    fx_payload=fx_settings,
                    extra={},
                ),
                "chunk_meta": {},
            }
            self._run_worker([item])
            audio, _ = sf.read(output, dtype="float32")
            return np.asarray(audio, dtype=np.float32)

    def cleanup(self) -> None:
        """Each worker exits after its batch, releasing model and GPU memory."""


__all__ = [
    "ChatterboxTurboLocalEngine",
    "CHATTERBOX_TURBO_AVAILABLE",
    "CHATTERBOX_TURBO_UNAVAILABLE_REASON",
    "CHATTERBOX_TURBO_SAMPLE_RATE",
]
