"""Dot.TTS engine adapter (subprocess-isolated)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .base import EngineCapabilities, TtsEngineBase, VoiceAssignment
from ..audio_effects import AudioPostProcessor, VoiceFXSettings

logger = logging.getLogger(__name__)

DOTS_TTS_SAMPLE_RATE = 48000
DOTS_TTS_DEFAULT_MODEL_ID = "rednote-hilab/dots.tts-soar"

_ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent / "engines" / "dots-tts"
_WORKER = _ENGINE_ROOT / "dots_tts_worker.py"
_TRANSCRIPTS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "voice_prompts" / "transcripts.json"

try:
    from funasr import AutoModel as FunASRAutoModel

    SENSEVOICE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    FunASRAutoModel = None  # type: ignore[assignment]
    SENSEVOICE_AVAILABLE = False


def _find_venv_python(engine_root: Path) -> Optional[Path]:
    candidates = [
        engine_root / ".venv" / "Scripts" / "python.exe",
        engine_root / ".venv" / "bin" / "python",
        engine_root / ".venv" / "bin" / "python3",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _check_dots_tts_available(engine_root: Path) -> Tuple[bool, str]:
    if not engine_root.exists():
        return False, f"Dot.TTS engine directory not found: {engine_root}. Run setup.bat."
    if not _WORKER.is_file():
        return False, f"Dot.TTS worker script missing: {_WORKER}. Run setup.bat to repair."
    python = _find_venv_python(engine_root)
    if python is None:
        return False, f"Dot.TTS isolated venv not found under {engine_root}. Run setup.bat."
    try:
        check = subprocess.run(
            [
                str(python),
                str(_WORKER),
                "--check-env",
            ],
            cwd=str(engine_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return False, f"Dot.TTS isolated venv dependency check failed: {exc}. Run setup.bat."
    if check.returncode != 0:
        detail = (check.stderr or check.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else "unknown import error"
        return False, f"Dot.TTS isolated venv is incomplete ({reason}). Run setup.bat."
    return True, ""


DOTS_TTS_AVAILABLE, DOTS_TTS_UNAVAILABLE_REASON = _check_dots_tts_available(_ENGINE_ROOT)


class DotsTTSEngine(TtsEngineBase):
    """Dot.TTS voice-cloning engine via an isolated subprocess worker."""

    name = "dots_tts"
    capabilities = EngineCapabilities(
        supports_voice_cloning=True,
        supports_emotion_tags=False,
        supported_languages=None,
    )

    def __init__(
        self,
        *,
        device: str = "auto",
        model_id: str = DOTS_TTS_DEFAULT_MODEL_ID,
        precision: str = "auto",
        optimize: bool = False,
        num_steps: int = 10,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        seed: Optional[int] = 42,
        language: str = "none",
        normalize_text: bool = False,
        default_prompt: Optional[str] = None,
        default_prompt_text: Optional[str] = None,
        allow_xvector_only: bool = False,
    ) -> None:
        available, reason = _check_dots_tts_available(_ENGINE_ROOT)
        if not available:
            raise ImportError(f"Dot.TTS is not available: {reason}")

        self._python = _find_venv_python(_ENGINE_ROOT)
        self._worker = _WORKER
        self.device = device
        self.model_id = model_id
        self.precision = precision
        self.optimize = bool(optimize)
        self.num_steps = max(1, int(num_steps))
        self.guidance_scale = float(guidance_scale)
        self.speaker_scale = float(speaker_scale)
        self.seed = None if seed in (None, "") else int(seed)
        self.language = language or "none"
        self.normalize_text = bool(normalize_text)
        self.default_prompt = (default_prompt or "").strip() or None
        self.default_prompt_text = (default_prompt_text or "").strip() or None
        self.allow_xvector_only = bool(allow_xvector_only)
        self.post_processor = AudioPostProcessor()
        self._asr_model = None
        self._transcript_cache = self._load_persistent_transcripts()

        logger.info(
            "DotsTTSEngine ready model=%s precision=%s steps=%s guidance=%s optimize=%s",
            self.model_id,
            self.precision,
            self.num_steps,
            self.guidance_scale,
            self.optimize,
        )

    @property
    def sample_rate(self) -> int:
        return DOTS_TTS_SAMPLE_RATE

    def generate_audio(
        self,
        *,
        text: str,
        voice: str = "",
        lang_code: Optional[str] = None,
        speed: float = 1.0,
        sample_rate: Optional[int] = None,
        fx_settings: Optional[VoiceFXSettings] = None,
        audio_prompt_path: Optional[str] = None,
        **_kwargs,
    ) -> np.ndarray:
        prompt_audio, prompt_text, temp_mp3_conv = self._resolve_prompt_and_text(
            VoiceAssignment(audio_prompt_path=audio_prompt_path or voice or None)
        )
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                output_path = tf.name
            job = self._build_job(
                [{
                    "text": text,
                    "prompt_audio_path": prompt_audio,
                    "prompt_text": prompt_text,
                    "output_path": output_path,
                }]
            )
            self._run_worker(job)
            audio, sr = sf.read(output_path, dtype="float32")
        finally:
            if temp_mp3_conv:
                temp_mp3_conv.unlink(missing_ok=True)
            if "output_path" in locals():
                Path(output_path).unlink(missing_ok=True)

        if fx_settings and not fx_settings.is_identity():
            audio = self.post_processor.apply_post_pipeline(audio, sr, fx_settings)
        return np.asarray(audio, dtype=np.float32)

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
        group_by_speaker: bool = False,
    ) -> List[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        worker_chunks: List[Dict] = []
        chunk_meta: List[Dict] = []
        temp_files: List[Path] = []
        order = 0
        for seg_idx, segment in enumerate(segments):
            speaker = segment.get("speaker")
            assignment = self._voice_assignment_for(voice_config, speaker)
            for chunk_idx, chunk_text in enumerate(segment.get("chunks") or []):
                output_path = output_dir / f"chunk_{order:04d}.wav"
                worker_chunk, temp_prompt = self._build_worker_chunk(
                    chunk_text,
                    str(output_path),
                    assignment,
                    order,
                )
                if temp_prompt:
                    temp_files.append(temp_prompt)
                worker_chunks.append(worker_chunk)
                chunk_meta.append({
                    "speaker": speaker,
                    "text": chunk_text,
                    "segment_index": seg_idx,
                    "chunk_index": chunk_idx,
                    "output_path": str(output_path),
                    "assignment": assignment,
                    "_order_index": order,
                })
                order += 1

        try:
            return self.generate_batch_prebuilt(
                worker_chunks,
                chunk_meta,
                progress_cb=progress_cb,
                chunk_cb=chunk_cb,
                pause_cb=pause_cb,
                cancel_cb=cancel_cb,
                group_by_speaker=group_by_speaker,
            )
        finally:
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)

    def generate_batch_prebuilt(
        self,
        worker_chunks: List[Dict],
        chunk_meta: List[Dict],
        progress_cb=None,
        chunk_cb=None,
        pause_cb=None,
        cancel_cb=None,
        group_by_speaker: bool = False,
    ) -> List[str]:
        if not worker_chunks:
            return []

        if group_by_speaker and len(worker_chunks) > 1:
            seen_prompts: List[str] = []
            by_prompt: Dict[str, List[Dict]] = {}
            for item in worker_chunks:
                key = f"{item.get('prompt_audio_path') or ''}\n{item.get('prompt_text') or ''}"
                if key not in by_prompt:
                    seen_prompts.append(key)
                    by_prompt[key] = []
                by_prompt[key].append(item)
            meta_by_order = {m["_order_index"]: m for m in chunk_meta}
            worker_chunks = [item for key in seen_prompts for item in by_prompt[key]]
            chunk_meta = [meta_by_order[item["_order_index"]] for item in worker_chunks]

        job = self._build_job([{k: v for k, v in chunk.items() if not k.startswith("_")} for chunk in worker_chunks])
        files, completed_paths, paused, cancelled = self._run_worker_with_progress(
            job,
            chunk_meta,
            progress_cb=progress_cb,
            chunk_cb=chunk_cb,
            pause_cb=pause_cb,
            cancel_cb=cancel_cb,
        )

        if cancelled or paused:
            return [path for path in files if Path(path).exists()]

        already_reported = set(completed_paths)
        for idx, file_path in enumerate(files):
            if idx >= len(chunk_meta):
                break
            assignment: VoiceAssignment = chunk_meta[idx]["assignment"]
            fx_settings = VoiceFXSettings.from_payload(assignment.fx_payload)
            if fx_settings and not fx_settings.is_identity():
                try:
                    audio, sr = sf.read(file_path, dtype="float32")
                    audio = self.post_processor.apply_post_pipeline(audio, sr, fx_settings)
                    sf.write(file_path, audio, sr)
                except Exception as exc:
                    logger.warning("Dot.TTS FX post-processing failed for %s: %s", file_path, exc)

            if callable(pause_cb) and pause_cb():
                return files
            if file_path not in already_reported:
                if callable(progress_cb):
                    progress_cb()
                if callable(chunk_cb):
                    meta = chunk_meta[idx]
                    chunk_cb(
                        meta["chunk_index"],
                        {
                            "speaker": meta["speaker"],
                            "text": meta["text"],
                            "segment_index": meta["segment_index"],
                            "chunk_index": meta["chunk_index"],
                            "chapter_index": meta.get("chapter_index", 0),
                        },
                        file_path,
                    )

        files.sort(key=lambda path: path)
        return files

    def cleanup(self) -> None:
        logger.info("DotsTTSEngine cleanup (subprocess-based, nothing to unload)")

    def _build_worker_chunk(
        self,
        text: str,
        output_path: str,
        assignment: VoiceAssignment,
        order_index: int,
    ) -> Tuple[Dict, Optional[Path]]:
        prompt_audio, prompt_text, temp_mp3_conv = self._resolve_prompt_and_text(assignment)
        return (
            {
                "text": self._prepare_synthesis_text(text),
                "prompt_audio_path": prompt_audio,
                "prompt_text": prompt_text,
                "output_path": output_path,
                "_order_index": order_index,
            },
            temp_mp3_conv,
        )

    @staticmethod
    def _prepare_synthesis_text(text: str) -> str:
        prepared = (text or "").strip()
        if not prepared:
            return prepared

        prepared = re.sub(r"^[,;:]\s*", "", prepared)
        if not prepared:
            return prepared

        closing = ""
        while prepared and prepared[-1] in "\"')]":
            closing = prepared[-1] + closing
            prepared = prepared[:-1].rstrip()

        if prepared and prepared[-1] in ",;:":
            prepared = prepared[:-1].rstrip()

        if prepared and prepared[-1] not in ".!?…":
            prepared = f"{prepared}."

        return f"{prepared}{closing}"

    def _build_job(self, chunks: List[Dict]) -> Dict:
        return {
            "model_id": self.model_id,
            "precision": self.precision,
            "optimize": self.optimize,
            "num_steps": self.num_steps,
            "guidance_scale": self.guidance_scale,
            "speaker_scale": self.speaker_scale,
            "seed": self.seed,
            "language": self.language,
            "normalize_text": self.normalize_text,
            "allow_xvector_only": self.allow_xvector_only,
            "chunks": chunks,
        }

    def _run_worker(self, job: Dict) -> None:
        files, _, _, _ = self._run_worker_with_progress(job, [])
        if not files and job.get("chunks"):
            raise RuntimeError("Dot.TTS worker produced no output files.")

    def _run_worker_with_progress(
        self,
        job: Dict,
        chunk_meta: List[Dict],
        progress_cb=None,
        chunk_cb=None,
        pause_cb=None,
        cancel_cb=None,
    ) -> Tuple[List[str], List[str], bool, bool]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(job, tf)
            job_file = tf.name

        paused_early = threading.Event()
        cancelled_early = threading.Event()
        stderr_lines: List[str] = []
        stdout_chunks: List[str] = []
        completed_paths: List[str] = []
        try:
            env = os.environ.copy()
            env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            proc = subprocess.Popen(
                [str(self._python), str(self._worker), "--job-file", job_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(_ENGINE_ROOT),
                env=env,
            )

            def _stream_stderr() -> None:
                assert proc.stderr is not None
                for line in proc.stderr:
                    line = line.rstrip()
                    stderr_lines.append(line)
                    logger.info("[dots-tts worker] %s", line)
                    if not line.startswith("[CHUNK_DONE] "):
                        continue
                    done_path = line[len("[CHUNK_DONE] "):]
                    completed_paths.append(done_path)
                    idx = len(completed_paths) - 1
                    if idx < len(chunk_meta):
                        meta = chunk_meta[idx]
                        if callable(cancel_cb) and cancel_cb():
                            cancelled_early.set()
                            proc.terminate()
                            return
                        if callable(pause_cb) and pause_cb():
                            paused_early.set()
                            proc.terminate()
                            return
                        if callable(progress_cb):
                            try:
                                progress_cb()
                            except Exception:
                                paused_early.set()
                                proc.terminate()
                                return
                        if callable(chunk_cb):
                            chunk_cb(
                                meta["chunk_index"],
                                {
                                    "speaker": meta["speaker"],
                                    "text": meta["text"],
                                    "segment_index": meta["segment_index"],
                                    "chunk_index": meta["chunk_index"],
                                    "chapter_index": meta.get("chapter_index", 0),
                                },
                                done_path,
                            )

            def _read_stdout() -> None:
                assert proc.stdout is not None
                stdout_chunks.append(proc.stdout.read())

            def _poll_cancel() -> None:
                while proc.poll() is None:
                    if callable(cancel_cb) and cancel_cb():
                        cancelled_early.set()
                        proc.terminate()
                        return
                    if callable(pause_cb) and pause_cb():
                        paused_early.set()
                        proc.terminate()
                        return
                    threading.Event().wait(0.5)

            stderr_thread = threading.Thread(target=_stream_stderr, daemon=True)
            stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
            poll_thread = threading.Thread(target=_poll_cancel, daemon=True)
            stderr_thread.start()
            stdout_thread.start()
            poll_thread.start()
            proc.wait()
            stderr_thread.join(timeout=5)
            stdout_thread.join(timeout=5)

            if cancelled_early.is_set() or paused_early.is_set():
                return (
                    [path for path in completed_paths if Path(path).exists()],
                    completed_paths,
                    paused_early.is_set(),
                    cancelled_early.is_set(),
                )

            stdout_data = "".join(stdout_chunks).strip()
            if not stdout_data:
                raise RuntimeError(
                    f"Dot.TTS worker produced no output (exit code {proc.returncode}).\n"
                    f"stderr:\n{chr(10).join(stderr_lines[-30:]) or '(empty)'}"
                )
            response = json.loads(stdout_data.splitlines()[-1])
            if not response.get("success"):
                raise RuntimeError(f"Dot.TTS worker failed:\n{response.get('error', 'unknown error')}")
            return response.get("files", []), completed_paths, False, False
        finally:
            Path(job_file).unlink(missing_ok=True)

    def _resolve_prompt_path(self, prompt_path: Optional[str]) -> Optional[str]:
        if not prompt_path:
            return None
        candidate = Path(prompt_path)
        if candidate.is_file():
            return str(candidate.resolve())
        project_root = _ENGINE_ROOT.parent.parent
        fallback = project_root / "data" / "voice_prompts" / prompt_path
        if fallback.is_file():
            return str(fallback.resolve())
        fallback_name = project_root / "data" / "voice_prompts" / candidate.name
        if fallback_name.is_file():
            return str(fallback_name.resolve())
        return None

    def _resolve_prompt_and_text(self, assignment: VoiceAssignment) -> Tuple[str, Optional[str], Optional[Path]]:
        prompt_path = assignment.audio_prompt_path or self.default_prompt
        resolved = self._resolve_prompt_path(prompt_path)
        if not resolved:
            raise ValueError("Dot.TTS requires a reference audio prompt for voice cloning.")

        prompt_text = (
            (assignment.extra or {}).get("prompt_text")
            or self.default_prompt_text
            or self._get_cached_transcript(resolved)
        )

        temp_mp3_conv = None
        from ..audio_effects import convert_mp3_to_wav_if_needed
        resolved, temp_mp3_conv = convert_mp3_to_wav_if_needed(resolved)

        if not prompt_text:
            prompt_text = self._transcribe_audio(resolved)

        if not prompt_text and not self.allow_xvector_only:
            raise ValueError(
                "Dot.TTS requires transcript text for the selected reference audio. "
                "Add prompt text to the voice prompt, provide a default transcript in settings, "
                "or enable x-vector-only fallback."
            )

        return resolved, prompt_text, temp_mp3_conv

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

    def _audio_hash(self, audio_path: str) -> str:
        path = Path(audio_path)
        stat = path.stat()
        key_data = f"{path.name}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def _load_persistent_transcripts(self) -> Dict[str, str]:
        if not _TRANSCRIPTS_FILE.exists():
            return {}
        try:
            with _TRANSCRIPTS_FILE.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data.get("transcripts", {}) if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("Failed to load voice prompt transcripts: %s", exc)
            return {}

    def _save_persistent_transcripts(self) -> None:
        try:
            _TRANSCRIPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with _TRANSCRIPTS_FILE.open("w", encoding="utf-8") as handle:
                json.dump({"transcripts": self._transcript_cache}, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to save voice prompt transcripts: %s", exc)

    def _get_cached_transcript(self, audio_path: str) -> Optional[str]:
        key = self._audio_hash(audio_path)
        return self._transcript_cache.get(key)

    def _transcribe_audio(self, audio_path: str) -> Optional[str]:
        key = self._audio_hash(audio_path)
        if key in self._transcript_cache:
            return self._transcript_cache[key]
        if not SENSEVOICE_AVAILABLE:
            logger.warning("SenseVoice is not available for Dot.TTS prompt transcription.")
            return None
        try:
            if self._asr_model is None:
                asr_device = self.device if self.device not in ("", "auto") else "cpu"
                self._asr_model = FunASRAutoModel(
                    model="iic/SenseVoiceSmall",
                    trust_remote_code=True,
                    device=asr_device,
                    disable_update=True,
                )
            result = self._asr_model.generate(input=audio_path, batch_size_s=0)
            if result and len(result) > 0:
                transcript = result[0].get("text", "").strip()
                transcript = re.sub(r"<\|[^|]+\|>", "", transcript).strip()
                if transcript:
                    self._transcript_cache[key] = transcript
                    self._save_persistent_transcripts()
                    return transcript
        except Exception as exc:
            logger.warning("Failed to auto-transcribe Dot.TTS reference audio: %s", exc)
        return None


__all__ = [
    "DOTS_TTS_AVAILABLE",
    "DOTS_TTS_UNAVAILABLE_REASON",
    "DOTS_TTS_SAMPLE_RATE",
    "DotsTTSEngine",
]
