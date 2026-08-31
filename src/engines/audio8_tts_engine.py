"""Audio8 TTS Preview adapter for isolated zero-shot voice cloning."""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

try:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoProcessor
    AUDIO8_AVAILABLE = True
except ImportError:  # pragma: no cover - optional isolated dependency
    torch = None  # type: ignore[assignment]
    snapshot_download = None  # type: ignore[assignment]
    AutoModel = AutoProcessor = None  # type: ignore[assignment]
    AUDIO8_AVAILABLE = False

from .base import EngineCapabilities, TtsEngineBase, VoiceAssignment
from ..audio_effects import AudioPostProcessor, VoiceFXSettings, convert_mp3_to_wav_if_needed

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
VOICE_PROMPT_DIR = ROOT / "data" / "voice_prompts"
TRANSCRIPTS_FILE = VOICE_PROMPT_DIR / "transcripts.json"
SUPPORTED_LANGUAGES = ["yue", "zh", "nl", "en", "fr", "de", "it", "ja", "ko", "pl", "es"]


class Audio8TTSEngine(TtsEngineBase):
    """Audio8 0.6B local engine with optional zero-shot reference conditioning."""

    name = "audio8_tts"
    capabilities = EngineCapabilities(
        supports_voice_cloning=True,
        supports_emotion_tags=False,
        supported_languages=SUPPORTED_LANGUAGES,
    )

    def __init__(
        self,
        *,
        device: str = "auto",
        model_id: str = "Audio8/Audio8-TTS-Preview-0.6b",
        dtype: str = "auto",
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_new_tokens: int = 1024,
        retry_max_new_tokens: int = 2000,
        max_input_chars: int = 400,
        seed: int = 42,
        default_prompt: Optional[str] = None,
        default_prompt_text: Optional[str] = None,
    ) -> None:
        if not AUDIO8_AVAILABLE:
            raise ImportError("Audio8 TTS is not installed. Install it from Settings → Engine Settings.")
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.model_id = model_id
        self.temperature = max(0.01, float(temperature))
        self.top_p = min(1.0, max(0.01, float(top_p)))
        self.top_k = max(0, int(top_k))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.retry_max_new_tokens = max(self.max_new_tokens, int(retry_max_new_tokens))
        self.max_input_chars = max(150, int(max_input_chars))
        self.seed = int(seed)
        self.default_prompt = default_prompt
        self.default_prompt_text = (default_prompt_text or "").strip() or None
        self.post_processor = AudioPostProcessor()
        self._transcripts = self._load_transcripts()
        self._reference_codes_cache: Dict[
            str, tuple[torch.Tensor, Optional[VoiceFXSettings]]
        ] = {}

        load_started = time.perf_counter()
        model_path = self._ensure_model(model_id)
        logger.info("Loading Audio8 TTS model=%s device=%s dtype=%s", model_id, self.device, self.dtype)
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            str(model_path), trust_remote_code=True, dtype=self.dtype
        ).eval().to(self.device)
        self._sample_rate = int(getattr(self.model.config, "codec_sample_rate", 44100))
        logger.info(
            "Audio8 ready device=%s dtype=%s load_seconds=%.2f",
            self.device, self.dtype, time.perf_counter() - load_started,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def generate_audio(self, text: str, audio_prompt_path: Optional[str] = None,
                       prompt_text: Optional[str] = None, fx_settings=None, **kwargs) -> np.ndarray:
        assignment = VoiceAssignment(
            audio_prompt_path=audio_prompt_path,
            fx_payload=fx_settings,
            extra={"prompt_text": prompt_text or kwargs.get("reference_text") or ""},
        )
        return self._synthesize(self._validate_text(text), assignment, self.seed)

    def generate_batch(self, segments: List[Dict], voice_config: Dict[str, Dict], output_dir: Path,
                       speed: float = 1.0, sample_rate: Optional[int] = None, progress_cb=None,
                       chunk_cb=None, parallel_workers: int = 1,
                       group_by_speaker: bool = False) -> List[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        order = 0
        for segment in segments:
            segment["_chunk_order_start"] = order
            order += len(segment.get("chunks") or [])
        files: List[Optional[str]] = [None] * order

        if group_by_speaker and len(segments) > 1:
            speakers: Dict[str, List[Dict]] = {}
            for segment in segments:
                speakers.setdefault(segment.get("speaker") or "default", []).append(segment)
            segments = [segment for group in speakers.values() for segment in group]

        self._preflight(segments, voice_config)
        for segment_index, segment in enumerate(segments):
            speaker = segment.get("speaker") or "default"
            assignment = self._voice_assignment_for(voice_config, speaker)
            for local_index, text in enumerate(segment.get("chunks") or []):
                order_index = int(segment["_chunk_order_start"]) + local_index
                output_path = output_dir / f"chunk_{order_index:04d}.wav"
                audio = self._synthesize(self._validate_text(text), assignment, self.seed + order_index)
                sf.write(str(output_path), audio, self.sample_rate)
                files[order_index] = str(output_path)
                if callable(progress_cb):
                    progress_cb()
                if callable(chunk_cb):
                    chunk_cb(local_index, {
                        "speaker": speaker, "text": text, "segment_index": segment_index,
                        "chunk_index": local_index, "order_index": order_index,
                    }, str(output_path))
        return [path for path in files if path]

    def _synthesize(self, text: str, assignment: VoiceAssignment, seed: int) -> np.ndarray:
        total_started = time.perf_counter()
        prompt = assignment.audio_prompt_path or self.default_prompt
        transcript = str((assignment.extra or {}).get("prompt_text") or self.default_prompt_text or "").strip()
        temp_conversion = None
        temp_prompt = None
        fx = VoiceFXSettings.from_payload(assignment.fx_payload)
        output_fx = fx
        try:
            processor_args = {"text": [text], "return_tensors": "pt"}
            reference_seconds = 0.0
            if prompt:
                prompt = self._resolve_prompt(prompt)
                if not transcript:
                    transcript = self._transcript_for(Path(prompt))
                if not transcript:
                    raise ValueError(
                        f"Audio8 requires an exact transcript for reference voice '{Path(prompt).name}'. "
                        "Add or generate it in Available Voices first."
                    )
                cache_key = self._reference_cache_key(prompt, transcript, fx)
                cached_reference = self._reference_codes_cache.get(cache_key)
                if cached_reference is None:
                    prompt, temp_conversion = convert_mp3_to_wav_if_needed(prompt)
                    if fx:
                        temp_prompt = self.post_processor.prepare_prompt_audio(prompt, fx)
                        if temp_prompt:
                            prompt = str(temp_prompt)
                            output_fx = (
                                VoiceFXSettings(pitch_semitones=0.0, speed=1.0, tone=fx.tone)
                                if fx.tone != "neutral" else None
                            )
                    reference_started = time.perf_counter()
                    reference_inputs = self.processor(
                        text=[text], reference_audio=[prompt], reference_text=[transcript],
                        return_tensors="pt",
                    )
                    reference_codes_batch, reference_lengths = self.model.encode_audio(
                        reference_inputs["reference_audio_values"].to(self.device),
                        reference_inputs["reference_audio_lengths"].to(self.device),
                    )
                    reference_length = int(reference_lengths[0])
                    reference_codes = reference_codes_batch[0, :, :reference_length].detach().cpu()
                    self._reference_codes_cache[cache_key] = (reference_codes, output_fx)
                    reference_seconds = time.perf_counter() - reference_started
                    logger.info(
                        "Audio8 reference encoded frames=%d seconds=%.2f cache_entries=%d",
                        reference_length, reference_seconds, len(self._reference_codes_cache),
                    )
                else:
                    reference_codes, output_fx = cached_reference
                    logger.info("Audio8 reference cache hit cache_entries=%d", len(self._reference_codes_cache))
                processor_args.update(reference_codes=[reference_codes], reference_text=[transcript])

            inputs = self.processor(**processor_args)
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            generation_started = time.perf_counter()
            result = self._generate(inputs, self.max_new_tokens, seed)
            if not bool(result.finished[0]) and self.retry_max_new_tokens > self.max_new_tokens:
                logger.warning("Audio8 did not emit EOS; retrying with %d tokens", self.retry_max_new_tokens)
                result = self._generate(inputs, self.retry_max_new_tokens, seed)
            generation_seconds = time.perf_counter() - generation_started
            if not bool(result.finished[0]):
                raise RuntimeError("Audio8 did not finish this chunk after its output-length retry.")
            decode_started = time.perf_counter()
            waveforms, lengths = self.model.decode_audio(result.codes)
            audio = waveforms[0, :int(lengths[0])].float().cpu().numpy().astype(np.float32)
            decode_seconds = time.perf_counter() - decode_started
            processed = self.post_processor.apply_post_pipeline(audio, self.sample_rate, output_fx)
            logger.info(
                "Audio8 synthesis chars=%d audio_seconds=%.2f reference_seconds=%.2f "
                "generation_seconds=%.2f decode_seconds=%.2f total_seconds=%.2f",
                len(text), len(processed) / float(self.sample_rate), reference_seconds,
                generation_seconds, decode_seconds, time.perf_counter() - total_started,
            )
            return processed
        finally:
            if temp_prompt:
                Path(temp_prompt).unlink(missing_ok=True)
            if temp_conversion:
                Path(temp_conversion).unlink(missing_ok=True)

    def _generate(self, inputs: Dict, max_tokens: int, seed: int):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            return self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=self.temperature,
                top_p=self.top_p, top_k=self.top_k, do_sample=True,
                generator=generator, return_dict_in_generate=True,
            )

    def _preflight(self, segments: List[Dict], voice_config: Dict[str, Dict]) -> None:
        errors = []
        for speaker in dict.fromkeys(segment.get("speaker") or "default" for segment in segments):
            assignment = self._voice_assignment_for(voice_config, speaker)
            prompt = assignment.audio_prompt_path or self.default_prompt
            if not prompt:
                continue  # Audio8 supports unconditioned generation.
            resolved = self._resolve_prompt(prompt)
            transcript = str((assignment.extra or {}).get("prompt_text") or self.default_prompt_text or "").strip()
            if not transcript and not self._transcript_for(Path(resolved)):
                errors.append(speaker)
        if errors:
            raise ValueError("Audio8 requires reference transcripts for: " + ", ".join(errors))

    def _validate_text(self, text: str) -> str:
        value = re.sub(r"\s+", " ", str(text or "")).strip()
        if not value:
            raise ValueError("Audio8 cannot synthesize empty text.")
        if len(value) > self.max_input_chars:
            raise ValueError(
                f"Audio8 input is {len(value)} characters; the configured sentence hard limit is "
                f"{self.max_input_chars}. Increase the Audio8 hard limit or split the sentence."
            )
        return value

    @staticmethod
    def _reference_cache_key(prompt: str, transcript: str, fx: Optional[VoiceFXSettings]) -> str:
        path = Path(prompt)
        stat = path.stat()
        fx_payload = vars(fx) if fx is not None else {}
        payload = json.dumps(
            {
                "path": str(path.resolve()), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "transcript": transcript,
                "fx": fx_payload,
            },
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _voice_assignment_for(config: Dict[str, Dict], speaker: str) -> VoiceAssignment:
        payload = config.get(speaker) or config.get("default") or {}
        return VoiceAssignment(
            voice=payload.get("voice"), lang_code=payload.get("lang_code"),
            audio_prompt_path=payload.get("audio_prompt_path"), fx_payload=payload.get("fx"),
            speed_override=payload.get("speed"), extra=payload.get("extra") or {},
        )

    @staticmethod
    def _resolve_prompt(value: str) -> str:
        path = Path(value)
        if not path.is_file():
            path = VOICE_PROMPT_DIR / Path(value).name
        if not path.is_file():
            raise FileNotFoundError(f"Audio8 reference voice not found: {value}")
        return str(path)

    def _transcript_for(self, path: Path) -> str:
        try:
            stat = path.stat()
        except OSError:
            return ""
        key = hashlib.md5(f"{path.name}:{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()[:16]
        return str(self._transcripts.get(key) or "").strip()

    @staticmethod
    def _load_transcripts() -> Dict[str, str]:
        try:
            data = json.loads(TRANSCRIPTS_FILE.read_text(encoding="utf-8"))
            return data.get("transcripts", {}) if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _resolve_device(value: str) -> str:
        value = (value or "auto").lower()
        if value == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if value.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Audio8 was configured for CUDA, but CUDA is unavailable.")
        return value

    def _resolve_dtype(self, value: str):
        value = (value or "auto").lower()
        if self.device == "cpu":
            return torch.float32
        return {"auto": torch.bfloat16, "bfloat16": torch.bfloat16,
                "float16": torch.float16, "float32": torch.float32}.get(value, torch.bfloat16)

    @staticmethod
    def _ensure_model(model_id: str) -> Path:
        model_root = Path(os.environ.get("TTS_STORY_ENGINE_MODEL_ROOT") or ROOT / "engines" / "audio8_tts" / "models")
        destination = model_root / model_id.replace("/", "_")
        if not destination.exists() or not any(destination.iterdir()):
            destination.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=model_id, local_dir=str(destination))
        return destination

    def cleanup(self) -> None:  # pragma: no cover
        self._reference_codes_cache.clear()
        for name in ("model", "processor"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
