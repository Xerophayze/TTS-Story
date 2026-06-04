"""OmniVoice voice-cloning engine adapter (subprocess-isolated).

OmniVoice requires torch==2.8 and transformers==5.3, which conflict with the
main project dependencies.  This adapter runs OmniVoice in an isolated venv
under engines/omnivoice/ via subprocess, exactly like the IndexTTS adapter.

Setup:
    Run setup.bat — it creates engines/omnivoice/.venv and installs omnivoice
    with its own torch/transformers versions there.
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

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent / "engines" / "omnivoice"
_WORKER = _ENGINE_ROOT / "omnivoice_worker.py"
OMNIVOICE_SAMPLE_RATE = 24000


def _find_venv_python(engine_root: Path) -> Optional[Path]:
    candidates = [
        engine_root / ".venv" / "Scripts" / "python.exe",
        engine_root / ".venv" / "bin" / "python",
        engine_root / ".venv" / "bin" / "python3",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _check_omnivoice_available(engine_root: Path) -> tuple[bool, str]:
    if not engine_root.exists():
        return False, f"OmniVoice engine directory not found: {engine_root}. Run setup.bat."
    if not (engine_root / "omnivoice_worker.py").exists():
        return False, f"OmniVoice worker script missing: {engine_root / 'omnivoice_worker.py'}. Run setup.bat."
    python = _find_venv_python(engine_root)
    if python is None:
        return False, f"OmniVoice isolated venv not found under {engine_root}. Run setup.bat."
    return True, ""


OMNIVOICE_AVAILABLE, _OMNIVOICE_UNAVAILABLE_REASON = _check_omnivoice_available(_ENGINE_ROOT)


class OmniVoiceCloneEngine(TtsEngineBase):
    """OmniVoice voice-cloning engine (subprocess-isolated).

    Runs the OmniVoice model in an isolated venv under engines/omnivoice/
    via subprocess to avoid torch/transformers version conflicts.
    """

    name = "omnivoice_clone"
    capabilities = EngineCapabilities(
        supports_voice_cloning=True,
        supports_emotion_tags=False,
    )

    def __init__(
        self,
        *,
        device: str = "auto",
        model_id: str = "k2-fsa/OmniVoice",
        dtype: str = "float16",
        num_step: int = 32,
        default_prompt: Optional[str] = None,
        default_prompt_text: Optional[str] = None,
        post_process: bool = True,
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
        self.default_prompt = default_prompt
        self.default_prompt_text = default_prompt_text
        self.post_process = post_process
        self.post_processor = AudioPostProcessor()

        self._transcript_cache: Dict[str, str] = {}
        self._transcripts_file = (
            Path(__file__).parent.parent.parent
            / "data"
            / "voice_prompts"
            / "omnivoice_transcripts.json"
        )
        self._load_persistent_transcripts()

        logger.info(
            "OmniVoiceCloneEngine ready (subprocess) model=%s device=%s",
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
        audio_prompt_path: Optional[str] = None,
        fx_settings=None,
        **_kwargs,
    ) -> np.ndarray:
        """Single-clip synthesis via subprocess worker."""
        prompt_path = audio_prompt_path or self.default_prompt
        if not prompt_path:
            raise ValueError("OmniVoice Clone requires a reference audio prompt.")
        prompt_path = self._resolve_prompt_path(prompt_path)
        if not prompt_path:
            raise ValueError("Reference audio file not found.")
        # Convert MP3 to WAV to prevent artifacts
        from ..audio_effects import convert_mp3_to_wav_if_needed
        prompt_path, temp_mp3_conv = convert_mp3_to_wav_if_needed(prompt_path)
        ref_text = self._get_ref_text(prompt_path)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_path = str(Path(tmp_dir) / "preview.wav")
                job = {
                    "mode": "clone",
                    "model_id": self.model_id,
                    "device": self.device,
                    "dtype": self.dtype,
                    "num_step": self.num_step,
                    "speed": speed,
                    "post_process": self.post_process,
                    "chunks": [{
                        "text": text,
                        "ref_audio": prompt_path,
                        "ref_text": ref_text,
                        "output_path": output_path,
                        "_order_index": 0,
                    }],
                }
                self._run_worker(job)
                audio, sr = sf.read(output_path, dtype="float32")
        finally:
            if temp_mp3_conv:
                temp_mp3_conv.unlink(missing_ok=True)
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
        group_by_speaker: bool = False,
        pause_cb=None,
        cancel_cb=None,
    ) -> List[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate all prompts exist first
        unique_speakers = list(dict.fromkeys(
            seg.get("speaker") or "default" for seg in segments
        ))
        missing_prompt: List[str] = []
        for _spk in unique_speakers:
            _asgn = self._voice_assignment_for(voice_config, _spk)
            _path = _asgn.audio_prompt_path or self.default_prompt
            if _path:
                _path = self._resolve_prompt_path(_path)
            if not _path:
                missing_prompt.append(_spk)
        if missing_prompt:
            raise ValueError(
                "OmniVoice Clone requires a reference audio prompt for every speaker. "
                f"Missing: {', '.join(missing_prompt)}"
            )

        # Assign chunk order indices
        _order = 0
        for seg in segments:
            seg["_chunk_order_start"] = _order
            _order += len(seg.get("chunks") or [])
        total_chunks_count = _order

        if group_by_speaker and len(segments) > 1:
            seen: List[str] = []
            by_spk: Dict[str, List] = {}
            for seg in segments:
                spk = seg.get("speaker") or "default"
                if spk not in by_spk:
                    seen.append(spk)
                    by_spk[spk] = []
                by_spk[spk].append(seg)
            segments = [seg for spk in seen for seg in by_spk[spk]]

        # Build flat worker chunk list
        worker_chunks: List[Dict] = []
        chunk_meta_list: List[Dict] = []
        for seg_idx, segment in enumerate(segments):
            speaker = segment["speaker"]
            chunks = segment["chunks"]
            assignment = self._voice_assignment_for(voice_config, speaker)
            prompt_path = self._resolve_prompt_path(
                assignment.audio_prompt_path or self.default_prompt
            )
            ref_text = self._get_ref_text(prompt_path) if prompt_path else None
            fx_settings = VoiceFXSettings.from_payload(assignment.fx_payload)
            spd = assignment.speed_override or speed

            temp_prompt = None
            if prompt_path and fx_settings:
                temp_prompt = self.post_processor.prepare_prompt_audio(
                    prompt_path, fx_settings
                )
                if temp_prompt:
                    prompt_path = str(temp_prompt)
                    fx_settings = VoiceFXSettings(
                        pitch_semitones=0.0, speed=1.0, tone=fx_settings.tone
                    ) if fx_settings.tone != "neutral" else None

            for chunk_idx, chunk_text in enumerate(chunks):
                order_index = segment["_chunk_order_start"] + chunk_idx
                output_path = str(output_dir / f"chunk_{order_index:04d}.wav")
                worker_chunks.append({
                    "text": chunk_text,
                    "ref_audio": prompt_path,
                    "ref_text": ref_text,
                    "output_path": output_path,
                    "_order_index": order_index,
                })
                chunk_meta_list.append({
                    "speaker": speaker,
                    "text": chunk_text,
                    "segment_index": seg_idx,
                    "chunk_index": chunk_idx,
                    "order_index": order_index,
                    "fx_settings": fx_settings,
                    "temp_prompt": temp_prompt,
                })

        if not worker_chunks:
            return []

        job = {
            "mode": "clone",
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "num_step": self.num_step,
            "speed": speed,
            "post_process": self.post_process,
            "chunks": worker_chunks,
        }

        completed_paths: List[str] = []
        _pending_exception: List[BaseException] = []  # shared across threads

        def _on_chunk_done(done_path: str) -> None:
            idx = len(completed_paths)
            completed_paths.append(done_path)
            if idx < len(chunk_meta_list):
                meta = chunk_meta_list[idx]
                fx = meta.get("fx_settings")
                if fx:
                    audio, sr = sf.read(done_path, dtype="float32")
                    audio = self.post_processor.apply_post_pipeline(audio, self.sample_rate, fx)
                    sf.write(done_path, audio, self.sample_rate)
                if callable(progress_cb):
                    try:
                        progress_cb()
                    except Exception as _exc:
                        # JobPaused or other cancellation - capture and signal subprocess to stop
                        _pending_exception.append(_exc)
                        raise  # propagates up into _read_stderr which will exit its loop
                if callable(chunk_cb):
                    chunk_cb(
                        meta["chunk_index"],
                        {"speaker": meta["speaker"], "text": meta["text"],
                         "segment_index": meta["segment_index"],
                         "chunk_index": meta["chunk_index"]},
                        done_path,
                    )

        def _cancel_or_pause_cb() -> bool:
            return bool(_pending_exception) or (callable(cancel_cb) and cancel_cb())

        try:
            self._run_worker(job, chunk_done_cb=_on_chunk_done, cancel_cb=_cancel_or_pause_cb)
        except RuntimeError:
            # If we have a pending exception (pause/cancel), ignore the worker exit error
            if not _pending_exception:
                raise

        # Clean up temp prompt files
        seen_temps: set = set()
        for meta in chunk_meta_list:
            tp = meta.get("temp_prompt")
            if tp and tp not in seen_temps:
                seen_temps.add(tp)
                Path(tp).unlink(missing_ok=True)

        # Re-raise any exception captured from the stderr thread (e.g. JobPaused)
        if _pending_exception:
            raise _pending_exception[0]

        files: List[Optional[str]] = [None] * total_chunks_count
        for path in completed_paths:
            name = Path(path).stem  # chunk_0000
            try:
                idx = int(name.split("_")[1])
                files[idx] = path
            except (IndexError, ValueError):
                pass
        return [p for p in files if p]

    def cleanup(self) -> None:
        pass  # No persistent model in adapter process

    def _run_worker(
        self,
        job: Dict,
        chunk_done_cb=None,
        cancel_cb=None,
    ) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tf:
            json.dump(job, tf)
            job_file = tf.name

        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            proc = subprocess.Popen(
                [str(self._python), str(self._worker), "--job-file", job_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(_ENGINE_ROOT),
                env=env,
            )
            stderr_lines: List[str] = []

            def _read_stderr() -> None:
                assert proc.stderr is not None
                try:
                    for line in proc.stderr:
                        line = line.rstrip()
                        stderr_lines.append(line)
                        logger.info("[omnivoice worker] %s", line)
                        if line.startswith("[CHUNK_DONE] ") and callable(chunk_done_cb):
                            chunk_done_cb(line[len("[CHUNK_DONE] "):])
                except Exception:
                    # JobPaused or other exceptions - let the main thread handle it
                    pass

            def _poll_cancel() -> None:
                while proc.poll() is None:
                    if callable(cancel_cb) and cancel_cb():
                        proc.terminate()
                        return
                    threading.Event().wait(0.5)

            stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
            cancel_thread = threading.Thread(target=_poll_cancel, daemon=True)
            stderr_thread.start()
            cancel_thread.start()
            stdout_data = proc.stdout.read() if proc.stdout else ""
            proc.wait()
            stderr_thread.join(timeout=5)

            if proc.returncode != 0:
                err = "\n".join(stderr_lines[-20:])
                raise RuntimeError(
                    f"OmniVoice worker failed (exit {proc.returncode}):\n{err}"
                )
        finally:
            Path(job_file).unlink(missing_ok=True)

    def _get_ref_text(self, prompt_path: str) -> Optional[str]:
        import hashlib
        path = Path(prompt_path)
        stat = path.stat()
        key_data = f"{path.name}:{stat.st_size}:{stat.st_mtime}"
        cache_key = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return self._transcript_cache.get(cache_key)

    def _load_persistent_transcripts(self) -> None:
        if self._transcripts_file.exists():
            try:
                with open(self._transcripts_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._transcript_cache = data.get("transcripts", {})
            except Exception as exc:
                logger.warning("Failed to load OmniVoice transcripts: %s", exc)
                self._transcript_cache = {}

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

    def _resolve_prompt_path(self, prompt_path: Optional[str]) -> Optional[str]:
        if not prompt_path:
            return None
        resolved = Path(prompt_path)
        if resolved.is_file():
            return str(resolved)
        fallback = (
            Path(__file__).parent.parent.parent / "data" / "voice_prompts" / prompt_path
        )
        if fallback.is_file():
            return str(fallback)
        return None
