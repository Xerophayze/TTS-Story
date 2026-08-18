"""Pocket TTS engine adapter."""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
try:
    import torch
except ImportError:  # Core-only installs do not include local engine runtimes.
    torch = None  # type: ignore[assignment]

from .base import EngineCapabilities, TtsEngineBase, VoiceAssignment
from ..audio_effects import AudioPostProcessor, VoiceFXSettings

logger = logging.getLogger(__name__)

try:
    from pocket_tts import TTSModel  # type: ignore

    POCKET_TTS_AVAILABLE = torch is not None
except ImportError:  # pragma: no cover - optional dependency
    TTSModel = None  # type: ignore[assignment]
    POCKET_TTS_AVAILABLE = False


class PocketTTSEngine(TtsEngineBase):
    """Local Pocket TTS engine with voice prompt cloning support."""

    name = "pocket_tts"
    capabilities = EngineCapabilities(
        supports_voice_cloning=True,
        supports_emotion_tags=False,
        supported_languages=["en"],
    )

    def __init__(
        self,
        *,
        device: str = "cpu",
        model_variant: str = "b6369a24",
        temp: float = 0.7,
        lsd_decode_steps: int = 1,
        noise_clamp: Optional[float] = None,
        eos_threshold: float = -4.0,
        default_prompt: Optional[str] = None,
        prompt_truncate: bool = False,
        num_threads: Optional[int] = None,
        interop_threads: Optional[int] = None,
    ) -> None:
        if not POCKET_TTS_AVAILABLE:
            raise ImportError("pocket-tts is not installed. Run setup to enable Pocket TTS.")

        # Set HF_TOKEN environment variable for voice cloning model download if available
        # This is required for Pocket TTS voice cloning models on HuggingFace
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            logger.info("HF_TOKEN environment variable is set for Pocket TTS voice cloning")
        else:
            logger.warning(
                "HF_TOKEN not set. Voice cloning may fail. Set HF_TOKEN environment variable "
                "or accept terms at https://huggingface.co/kyutai/pocket-tts and run 'huggingface-cli login'"
            )

        resolved_device = (device or "cpu").strip().lower()
        if resolved_device not in {"cpu", "auto"}:
            logger.info("Pocket TTS runs on CPU; ignoring device=%s", resolved_device)
            resolved_device = "cpu"

        logger.info(
            "Loading Pocket TTS model variant=%s temp=%s steps=%s",
            model_variant,
            temp,
            lsd_decode_steps,
        )
        load_kwargs = {
            "temp": float(temp),
            "lsd_decode_steps": int(lsd_decode_steps),
            "noise_clamp": noise_clamp,
            "eos_threshold": float(eos_threshold),
        }
        try:
            import inspect

            signature = inspect.signature(TTSModel.load_model)
            # Use variant parameter if available, as config expects a YAML file path
            if "variant" in signature.parameters:
                load_kwargs["variant"] = model_variant
            elif "config" in signature.parameters and str(model_variant).lower().endswith(".yaml"):
                load_kwargs["config"] = model_variant
        except Exception:
            load_kwargs.setdefault("variant", model_variant)

        self._explicit_num_threads = num_threads is not None
        if num_threads is not None:
            try:
                torch.set_num_threads(int(num_threads))
            except Exception:
                logger.warning("Unable to set torch num threads to %s", num_threads)
        if interop_threads:
            try:
                torch.set_num_interop_threads(int(interop_threads))
            except Exception:
                logger.warning("Unable to set torch interop threads to %s", interop_threads)

        self.model = TTSModel.load_model(**load_kwargs)

        self.device = resolved_device
        self.model_variant = model_variant
        self.default_prompt = default_prompt
        self.prompt_truncate = bool(prompt_truncate)
        self.post_processor = AudioPostProcessor()
        self._voice_cache: Dict[str, Dict] = {}
        self._prompt_wav_cache: Dict[str, str] = {}

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.model, "sample_rate", 24000))

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
        group_by_speaker: bool = False,
    ) -> List[str]:
        if sample_rate and sample_rate != self.sample_rate:
            logger.warning(
                "Pocket TTS outputs at %s Hz. Requested sample rate %s will be resampled during merge.",
                self.sample_rate,
                sample_rate,
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group non-consecutive same-speaker segments together so the voice
        # state cache stays hot — all chunks for one speaker run back-to-back
        # before switching to the next voice.
        # Stamp each segment with its narrative order_index before any grouping
        # so filenames always reflect playback order regardless of processing order.
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

        files: List[Optional[str]] = [None] * total_chunks_count
        chunk_index = 0
        effective_workers = max(1, min(8, int(parallel_workers or 1)))
        if effective_workers > 1 and not self._explicit_num_threads:
            cpu_count = os.cpu_count() or 1
            threads_per_worker = max(1, min(4, cpu_count // effective_workers))
            try:
                torch.set_num_threads(threads_per_worker)
                logger.info(
                    "Pocket TTS parallel workers=%s; torch num threads=%s",
                    effective_workers,
                    threads_per_worker,
                )
            except Exception:
                logger.warning("Unable to adjust torch num threads for parallel Pocket TTS")
        tasks = []

        for seg_idx, segment in enumerate(segments):
            speaker = segment.get("speaker")
            chunks = segment.get("chunks") or []
            assignment = self._voice_assignment_for(voice_config, speaker)
            prompt_source = assignment.audio_prompt_path or assignment.voice or self.default_prompt
            if not prompt_source:
                raise ValueError("Pocket TTS requires a voice sample prompt or a built-in voice name.")

            prompt_path = self._resolve_prompt_path(prompt_source)
            # Convert MP3 to WAV to prevent artifacts
            from ..audio_effects import convert_mp3_to_wav_if_needed
            prompt_path, temp_mp3_conv = convert_mp3_to_wav_if_needed(prompt_path)
            voice_state = self._get_voice_state(prompt_path, allow_predefined=True)
            fx_settings = VoiceFXSettings.from_payload(assignment.fx_payload)

            for chunk_idx, chunk_text in enumerate(chunks):
                order_index = segment["_chunk_order_start"] + chunk_idx
                output_path = output_dir / f"chunk_{order_index:04d}.wav"
                tasks.append({
                    "chunk_index": order_index,
                    "segment_index": seg_idx,
                    "speaker": speaker,
                    "chunk_text": chunk_text,
                    "output_path": output_path,
                    "voice_state": voice_state,
                    "fx_settings": fx_settings,
                    "speaker_chunk_index": chunk_idx,
                })
                chunk_index += 1

        if not tasks:
            return []

        files = [None] * total_chunks_count

        if effective_workers == 1:
            for task in tasks:
                if callable(pause_cb) and pause_cb():
                    break
                chunk_order, output_path = self._render_chunk(task, progress_cb, chunk_cb)
                files[chunk_order] = output_path
            return [path for path in files if path]

        pause_requested = False
        next_index = 0
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {}
            while futures or next_index < len(tasks):
                if callable(pause_cb) and pause_cb():
                    pause_requested = True
                while not pause_requested and next_index < len(tasks) and len(futures) < effective_workers:
                    task = tasks[next_index]
                    future = executor.submit(self._render_chunk, task, progress_cb, chunk_cb)
                    futures[future] = task
                    next_index += 1
                if not futures:
                    break
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures.pop(future)
                    chunk_order, output_path = future.result()
                    files[chunk_order] = output_path

            if pause_requested:
                for future in futures:
                    future.cancel()

        return [path for path in files if path]

    def _render_chunk(self, task: Dict, progress_cb, chunk_cb) -> tuple[int, str]:
        output_path = task["output_path"]
        audio_tensor = self.model.generate_audio(
            task["voice_state"],
            task["chunk_text"],
            frames_after_eos=None,
            copy_state=True,
        )
        audio = audio_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        audio = self.post_processor.apply_post_pipeline(
            audio,
            self.sample_rate,
            task["fx_settings"],
        )
        sf.write(str(output_path), audio, self.sample_rate)
        output_path_str = str(output_path)
        if callable(progress_cb):
            try:
                progress_cb()
            except Exception:
                # JobPaused or other cancellation exception - re-raise so caller handles it
                raise
        if callable(chunk_cb):
            chunk_meta = {
                "speaker": task["speaker"],
                "text": task["chunk_text"],
                "segment_index": task["segment_index"],
                "chunk_index": task["speaker_chunk_index"],
            }
            chunk_cb(task["speaker_chunk_index"], chunk_meta, output_path_str)
        return task["chunk_index"], output_path_str

    def cleanup(self) -> None:  # pragma: no cover
        logger.info("Cleaning up Pocket TTS engine resources")
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
                self.model = None
        except Exception:
            pass
        self._voice_cache.clear()

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
        prompt_path = self._resolve_prompt_path(voice)
        voice_state = self._get_voice_state(prompt_path)
        audio_tensor = self.model.generate_audio(voice_state, text)
        audio = audio_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        return self.post_processor.apply_post_pipeline(audio, self.sample_rate, fx_settings)

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

    def _resolve_prompt_path(self, prompt_path: str) -> str:
        trimmed = prompt_path.strip()
        if trimmed in self._builtin_voices():
            return trimmed
        if trimmed.startswith("hf://") or trimmed.startswith("http://") or trimmed.startswith("https://"):
            return trimmed
        candidate = Path(trimmed)
        if candidate.is_file():
            return str(candidate)
        fallback = Path(__file__).parent.parent.parent / "data" / "voice_prompts" / trimmed
        if fallback.is_file():
            return str(fallback)
        return trimmed

    def _get_voice_state(self, prompt_path: str, allow_predefined: bool = True) -> Dict:
        prompt_path = self._ensure_wav_prompt(prompt_path)
        cache_key = prompt_path
        cached = self._voice_cache.get(cache_key)
        if cached is not None:
            return cached
        if allow_predefined and prompt_path in self._builtin_voices():
            voice_state = self.model.get_state_for_audio_prompt(prompt_path, truncate=False)
            self._voice_cache[cache_key] = voice_state
            return voice_state
        try:
            voice_state = self.model.get_state_for_audio_prompt(
                prompt_path,
                truncate=self.prompt_truncate,
            )
        except ValueError as e:
            if "voice cloning" in str(e).lower():
                raise ValueError(
                    f"Pocket TTS voice cloning requires HuggingFace authentication. "
                    f"Please: 1) Accept terms at https://huggingface.co/kyutai/pocket-tts "
                    f"2) Run 'huggingface-cli login' or set HF_TOKEN environment variable. "
                    f"Alternatively, use a built-in voice: {list(self._builtin_voices())}. "
                    f"Original error: {e}"
                ) from e
            raise
        self._voice_cache[cache_key] = voice_state
        return voice_state

    def _ensure_wav_prompt(self, prompt_path: str) -> str:
        if prompt_path in self._builtin_voices():
            return prompt_path
        if prompt_path.startswith("hf://") or prompt_path.startswith("http://") or prompt_path.startswith("https://"):
            return prompt_path
        source_path = Path(prompt_path)
        if not source_path.is_file():
            return prompt_path
        if source_path.suffix.lower() == ".wav":
            return str(source_path)
        cached = self._prompt_wav_cache.get(prompt_path)
        if cached and Path(cached).is_file():
            return cached
        try:
            from pydub import AudioSegment
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Pocket TTS requires WAV prompts; pydub unavailable: %s", exc)
            return prompt_path
        try:
            stat = source_path.stat()
            cache_key = f"{source_path.resolve()}:{stat.st_size}:{stat.st_mtime}"
            digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
            temp_dir = Path(tempfile.gettempdir()) / "tts_story_pocket_prompts"
            temp_dir.mkdir(parents=True, exist_ok=True)
            target_path = temp_dir / f"{source_path.stem}_{digest}.wav"
            if not target_path.exists():
                audio = AudioSegment.from_file(str(source_path))
                audio.export(target_path, format="wav")
            self._prompt_wav_cache[prompt_path] = str(target_path)
            return str(target_path)
        except Exception as exc:
            logger.warning("Failed to convert prompt %s to WAV: %s", source_path, exc)
            return prompt_path

    @staticmethod
    def _builtin_voices() -> set:
        return {"alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"}


__all__ = [
    "PocketTTSEngine",
    "POCKET_TTS_AVAILABLE",
]
