"""Chatterbox Turbo engine that streams through Replicate's hosted model."""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import soundfile as sf

from replicate import Client

from .base import EngineCapabilities, TtsEngineBase, VoiceAssignment
from ..audio_effects import AudioPostProcessor, VoiceFXSettings

logger = logging.getLogger(__name__)

CHATTERBOX_TURBO_REPLICATE_SAMPLE_RATE = 24000
DEFAULT_CHATTERBOX_TURBO_REPLICATE_MODEL = (
    "resemble-ai/chatterbox-turbo:95c87b883ff3e842a1643044dff67f9d204f70a80228f24ff64bffe4a4b917d4"
)
DEFAULT_CHATTERBOX_TURBO_REPLICATE_VOICE = "Andy"


class ChatterboxTurboReplicateEngine(TtsEngineBase):
    """Hosted Chatterbox Turbo inference on Replicate."""

    name = "chatterbox_turbo_replicate"
    capabilities = EngineCapabilities(
        supports_voice_cloning=True,
        supports_emotion_tags=True,
        supported_languages=["en"],
    )

    def __init__(
        self,
        api_token: str,
        *,
        model_version: str = DEFAULT_CHATTERBOX_TURBO_REPLICATE_MODEL,
        default_voice: str = DEFAULT_CHATTERBOX_TURBO_REPLICATE_VOICE,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 1000,
        repetition_penalty: float = 1.2,
        seed: Optional[int] = None,
    ):
        if not api_token:
            raise ValueError("Replicate API token is required for Chatterbox Turbo (Replicate).")

        super().__init__(device="cpu")
        self.client = Client(api_token=api_token)
        self.model_ref = model_version or DEFAULT_CHATTERBOX_TURBO_REPLICATE_MODEL
        self.default_voice = default_voice or DEFAULT_CHATTERBOX_TURBO_REPLICATE_VOICE
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.seed = seed
        self.prompt_upload_cache: Dict[str, str] = {}
        self.post_processor = AudioPostProcessor()

    # ------------------------------------------------------------------ #
    @property
    def sample_rate(self) -> int:
        return CHATTERBOX_TURBO_REPLICATE_SAMPLE_RATE

    # ------------------------------------------------------------------ #
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
    ) -> List[str]:
        if sample_rate and sample_rate != self.sample_rate:
            logger.warning(
                "Replicate Turbo outputs at %s Hz. Requested sample rate %s will be resampled later.",
                self.sample_rate,
                sample_rate,
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build flat list of all chunks with their metadata
        all_chunks: List[Dict] = []
        for seg_idx, segment in enumerate(segments):
            speaker = segment["speaker"]
            chunks = segment["chunks"]
            assignment = self._voice_assignment_for(voice_config, speaker)
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "global_index": len(all_chunks),
                    "seg_idx": seg_idx,
                    "chunk_idx": chunk_idx,
                    "speaker": speaker,
                    "text": chunk_text,
                    "assignment": assignment,
                })

        if not all_chunks:
            return []

        effective_workers = max(1, min(10, parallel_workers))
        
        # For parallel processing, submit all predictions at once, then poll
        if effective_workers > 1:
            return self._generate_batch_async(
                all_chunks, output_dir, effective_workers, progress_cb, chunk_cb
            )
        else:
            # Sequential processing
            return self._generate_batch_sequential(
                all_chunks, output_dir, progress_cb, chunk_cb
            )

    def _generate_batch_async(
        self,
        all_chunks: List[Dict],
        output_dir: Path,
        parallel_workers: int,
        progress_cb,
        chunk_cb,
    ) -> List[str]:
        """Submit predictions in batches of parallel_workers, then poll for results."""
        
        logger.info("Processing %d chunks with %d parallel workers", len(all_chunks), parallel_workers)
        
        # Parse model string once
        if ":" in self.model_ref:
            model_name, version = self.model_ref.split(":", 1)
        else:
            model_name = self.model_ref
            version = None
        
        results: Dict[int, str] = {}
        chunk_queue = list(all_chunks)  # Chunks waiting to be submitted
        active_predictions = {}  # global_idx -> {prediction, chunk_info}
        
        while chunk_queue or active_predictions:
            # Submit new predictions up to parallel_workers limit
            while chunk_queue and len(active_predictions) < parallel_workers:
                chunk_info = chunk_queue.pop(0)
                global_idx = chunk_info["global_index"]
                
                logger.info(
                    "Submitting chunk %s/%s: speaker=%s (active: %d)",
                    global_idx + 1,
                    len(all_chunks),
                    chunk_info["speaker"],
                    len(active_predictions) + 1,
                )
                
                try:
                    # Build params and submit async
                    reference_url = None
                    assignment = chunk_info["assignment"]
                    if assignment.audio_prompt_path:
                        reference_url = self._upload_reference_audio(assignment.audio_prompt_path)
                    
                    params = self._build_payload(chunk_info["text"], assignment, reference_url)
                    
                    create_kwargs = {"input": params}
                    if version:
                        create_kwargs["version"] = version
                    else:
                        create_kwargs["model"] = model_name
                    
                    prediction = self.client.predictions.create(**create_kwargs)
                    active_predictions[global_idx] = {
                        "prediction": prediction,
                        "chunk_info": chunk_info,
                    }
                except Exception as e:
                    logger.error("Failed to submit chunk %d: %s", global_idx, e)
                    raise
            
            # Poll active predictions
            for global_idx in list(active_predictions.keys()):
                pred_info = active_predictions[global_idx]
                prediction = pred_info["prediction"]
                chunk_info = pred_info["chunk_info"]
                
                # Reload prediction status
                prediction.reload()
                
                if prediction.status == "succeeded":
                    output_path = output_dir / f"chunk_{global_idx:04d}.wav"
                    audio_url = prediction.output
                    
                    # Download and process audio
                    audio = self._download_audio(audio_url)
                    
                    # Apply FX if needed
                    fx_settings = VoiceFXSettings.from_payload(chunk_info["assignment"].fx_payload)
                    if fx_settings:
                        audio = self.post_processor.apply(audio, self.sample_rate, fx_settings)
                    
                    sf.write(str(output_path), audio, self.sample_rate)
                    results[global_idx] = str(output_path)
                    del active_predictions[global_idx]
                    
                    logger.info("Chunk %s/%s completed", global_idx + 1, len(all_chunks))
                    
                    # Callbacks
                    if callable(progress_cb):
                        progress_cb()
                    if callable(chunk_cb):
                        chunk_meta = {
                            "speaker": chunk_info["speaker"],
                            "text": chunk_info["text"],
                            "segment_index": chunk_info["seg_idx"],
                            "chunk_index": chunk_info["chunk_idx"],
                        }
                        chunk_cb(global_idx, chunk_meta, str(output_path))
                        
                elif prediction.status == "failed":
                    error_msg = getattr(prediction, "error", "Unknown error")
                    logger.error("Chunk %d failed: %s", global_idx, error_msg)
                    del active_predictions[global_idx]
                    raise RuntimeError(f"Prediction failed for chunk {global_idx}: {error_msg}")
                    
                elif prediction.status == "canceled":
                    logger.error("Chunk %d was canceled", global_idx)
                    del active_predictions[global_idx]
                    raise RuntimeError(f"Prediction canceled for chunk {global_idx}")
            
            # Small delay before next poll cycle if still have active predictions
            if active_predictions:
                time.sleep(0.5)
        
        # Return files in order
        return [results[i] for i in range(len(all_chunks))]

    def _generate_batch_sequential(
        self,
        all_chunks: List[Dict],
        output_dir: Path,
        progress_cb,
        chunk_cb,
    ) -> List[str]:
        """Process chunks one at a time using blocking API."""
        
        logger.info("Processing %d chunks sequentially", len(all_chunks))
        results: Dict[int, str] = {}
        
        for chunk_info in all_chunks:
            global_idx = chunk_info["global_index"]
            output_path = output_dir / f"chunk_{global_idx:04d}.wav"
            
            logger.info(
                "Chatterbox Turbo (Replicate) chunk %s/%s speaker=%s",
                global_idx + 1,
                len(all_chunks),
                chunk_info["speaker"],
            )
            
            audio, sr = self._synthesize(chunk_info["text"], chunk_info["assignment"])
            sf.write(str(output_path), audio, sr)
            
            results[global_idx] = str(output_path)
            
            if callable(progress_cb):
                progress_cb()
            if callable(chunk_cb):
                chunk_meta = {
                    "speaker": chunk_info["speaker"],
                    "text": chunk_info["text"],
                    "segment_index": chunk_info["seg_idx"],
                    "chunk_index": chunk_info["chunk_idx"],
                }
                chunk_cb(global_idx, chunk_meta, str(output_path))
        
        return [results[i] for i in range(len(all_chunks))]

    # ------------------------------------------------------------------ #
    def cleanup(self) -> None:  # pragma: no cover - trivial
        """No persistent resources to release."""

    # ------------------------------------------------------------------ #
    def _voice_assignment_for(self, voice_config: Dict[str, Dict], speaker: str) -> VoiceAssignment:
        payload = voice_config.get(speaker) or voice_config.get("default") or {}
        return VoiceAssignment(
            voice=payload.get("voice"),
            lang_code=payload.get("lang_code"),
            audio_prompt_path=payload.get("audio_prompt_path"),
            fx_payload=payload.get("fx"),
            speed_override=payload.get("speed"),
            extra=payload.get("extra") or {},
        )

    # ------------------------------------------------------------------ #
    def _synthesize(self, text: str, assignment: VoiceAssignment) -> Tuple[np.ndarray, int]:
        reference_url = None
        if assignment.audio_prompt_path:
            reference_url = self._upload_reference_audio(assignment.audio_prompt_path)

        params = self._build_payload(text, assignment, reference_url)
        try:
            output_url = self.client.run(self.model_ref, input=params)
        except Exception as exc:  # pragma: no cover - API failure
            raise RuntimeError(f"Chatterbox Turbo (Replicate) request failed: {exc}") from exc

        audio_array = self._download_audio(output_url)
        fx_settings = VoiceFXSettings.from_payload(assignment.fx_payload)
        if fx_settings:
            audio_array = self.post_processor.apply(audio_array, self.sample_rate, fx_settings)
        return audio_array, self.sample_rate

    # ------------------------------------------------------------------ #
    def _build_payload(
        self,
        text: str,
        assignment: VoiceAssignment,
        reference_url: Optional[str],
    ) -> Dict:
        params = {
            "text": text,
            "temperature": self._resolve_numeric(assignment.extra, "temperature", self.temperature),
            "top_p": self._resolve_numeric(assignment.extra, "top_p", self.top_p),
            "top_k": int(self._resolve_numeric(assignment.extra, "top_k", self.top_k)),
            "repetition_penalty": self._resolve_numeric(
                assignment.extra, "repetition_penalty", self.repetition_penalty
            ),
        }
        if self.seed is not None:
            params["seed"] = int(self.seed)

        if reference_url:
            params["reference_audio"] = reference_url
        else:
            params["voice"] = assignment.voice or self.default_voice

        return params

    # ------------------------------------------------------------------ #
    def _resolve_numeric(self, extra: Dict, key: str, default_value: float) -> float:
        value = extra.get(key) if extra else None
        if value is None:
            return default_value
        try:
            return float(value)
        except (TypeError, ValueError):
            return default_value

    # ------------------------------------------------------------------ #
    def _upload_reference_audio(self, path_str: str) -> str:
        resolved = self._resolve_prompt_path(path_str)
        key = str(resolved.resolve())
        cached = self.prompt_upload_cache.get(key)
        if cached:
            return cached

        file_resource = self.client.files.create(str(resolved))
        url = (
            file_resource.urls.get("get")
            or file_resource.urls.get("download")
            or file_resource.urls.get("web")
        )
        if not url:
            raise RuntimeError("Replicate did not return a download URL for uploaded prompt.")
        self.prompt_upload_cache[key] = url
        return url

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_prompt_path(path_str: str) -> Path:
        candidate = Path(path_str)
        if candidate.is_file():
            return candidate
        alt = Path("data/voice_prompts") / path_str
        if alt.is_file():
            return alt
        raise FileNotFoundError(
            f"Reference audio not found: {path_str}. Place files in data/voice_prompts or provide an absolute path."
        )

    # ------------------------------------------------------------------ #
    def _download_audio(self, url: str) -> np.ndarray:
        if not url:
            raise RuntimeError("Replicate response did not include an audio URL.")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        buffer = io.BytesIO(response.content)
        audio_array, sr = sf.read(buffer, dtype="float32")
        if sr != self.sample_rate:
            audio_array = self._resample_audio(audio_array, sr)
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        return audio_array.astype("float32")

    # ------------------------------------------------------------------ #
    def _resample_audio(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        if original_sr == self.sample_rate:
            return audio
        try:
            import librosa
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "librosa is required to resample Replicate audio output. Install via requirements."
            ) from exc
        return librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)


__all__ = [
    "ChatterboxTurboReplicateEngine",
    "CHATTERBOX_TURBO_REPLICATE_SAMPLE_RATE",
    "DEFAULT_CHATTERBOX_TURBO_REPLICATE_MODEL",
    "DEFAULT_CHATTERBOX_TURBO_REPLICATE_VOICE",
]
