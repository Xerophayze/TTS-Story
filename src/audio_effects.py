"""
Lightweight post-processing utilities for TTS-Story audio output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf

from .system_tools import find_system_tool

try:
    import librosa
    from librosa import util as librosa_util
except ImportError:  # pragma: no cover - optional dependency
    librosa = None
    librosa_util = None

try:  # Optional dependency that librosa uses for resampling-heavy effects
    import resampy  # noqa: F401
except ImportError:  # pragma: no cover
    resampy = None

try:  # Optional, higher-quality transformations
    import pyrubberband as pyrb  # noqa: F401
except ImportError:  # pragma: no cover
    pyrb = None

@dataclass
class VoiceFXSettings:
    """Container for user-defined post-processing controls."""

    pitch_semitones: float = 0.0
    speed: float = 1.0  # 0.5 to 2.0 (1.0 = normal)
    tone: str = "neutral"  # neutral | warm | bright

    def is_identity(self) -> bool:
        """Return True when the settings would not alter the audio."""
        return (
            abs(self.pitch_semitones) < 1e-3
            and abs(self.speed - 1.0) < 1e-3
            and self.tone == "neutral"
        )

    @classmethod
    def from_payload(cls, payload: Optional[Dict]) -> Optional["VoiceFXSettings"]:
        """
        Build a VoiceFXSettings instance from a JSON payload.
        Returns None when effects are effectively disabled.
        """
        if not payload or payload.get("enabled") is False:
            return None

        pitch = float(payload.get("pitch", 0.0) or 0.0)
        speed = float(payload.get("speed", 1.0) or 1.0)
        tone = (payload.get("tone") or "neutral").strip().lower()
        if tone not in {"neutral", "warm", "bright"}:
            tone = "neutral"

        pitch = max(-12.0, min(pitch, 12.0))
        speed = max(0.5, min(speed, 2.0))

        if abs(pitch) < 1e-3 and abs(speed - 1.0) < 1e-3 and tone == "neutral":
            return None

        return cls(pitch_semitones=pitch, speed=speed, tone=tone)


logger = logging.getLogger(__name__)


def convert_mp3_to_wav_if_needed(prompt_path: str) -> tuple[str, Optional[Path]]:
    """
    Convert MP3 voice prompts to WAV to prevent artifacts from lossy compression.
    
    Returns:
        tuple: (prompt_path_to_use, temp_file_to_cleanup)
        If conversion is not needed or fails, returns (original_path, None)
    """
    if not prompt_path:
        return prompt_path, None
    
    prompt_ext = Path(prompt_path).suffix.lower()
    if prompt_ext != ".mp3":
        return prompt_path, None
    
    try:
        # Find FFmpeg in local tools folder
        script_dir = Path(__file__).resolve().parent.parent  # src/ -> TTS-Story root
        ffmpeg_path = script_dir / "tools" / "ffmpeg" / "ffmpeg.exe"
        
        if not ffmpeg_path.exists():
            # Try relative path calculation
            ffmpeg_path = Path(__file__).resolve().parents[2] / "tools" / "ffmpeg" / "ffmpeg.exe"
        
        if ffmpeg_path.exists():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
                temp_mp3_conv = Path(temp_out.name)
            
            result = subprocess.run(
                [str(ffmpeg_path), "-y", "-i", str(prompt_path), str(temp_mp3_conv)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info("Converted MP3 voice prompt to WAV for better quality: %s", Path(prompt_path).name)
                return str(temp_mp3_conv), temp_mp3_conv
            else:
                temp_mp3_conv.unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Failed to convert MP3 voice prompt to WAV: %s", e)
    
    return prompt_path, None


class AudioPostProcessor:
    """Applies pitch, speed, and tonal shaping to generated audio arrays."""

    @staticmethod
    def _find_sox() -> Optional[Path]:
        """Find a platform-appropriate SoX executable."""

        return find_system_tool("sox")

    SOX_PATH = _find_sox.__func__()

    def apply_sox_post(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        normalize: bool = True,
        fade_seconds: float = 0.01,
    ) -> np.ndarray:
        """Apply a lightweight SoX post-processing pass to audio output."""
        if audio is None:
            return audio
        if not self.SOX_PATH or not self.SOX_PATH.exists():
            logger.warning("SoX not found - skipping post-processing")
            return audio

        logger.info("Applying SoX post-processing (normalize=%s, fade=%.2fs)", normalize, fade_seconds)

        is_stereo = audio.ndim == 2 and audio.shape[1] == 2
        if is_stereo:
            audio = np.mean(audio, axis=1)

        input_path = None
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
                input_path = Path(temp_in.name)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
                output_path = Path(temp_out.name)

            sf.write(str(input_path), audio, int(sample_rate))
            command = [str(self.SOX_PATH), str(input_path), str(output_path)]
            if normalize:
                command += ["gain", "-n"]
            if fade_seconds and fade_seconds > 0:
                command += ["fade", f"{fade_seconds:.2f}"]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "SoX post-processing failed")
            processed, _ = sf.read(str(output_path), dtype="float32")
            return processed.astype(np.float32, copy=False)
        except Exception as exc:  # pragma: no cover - fallback to raw audio
            logger.warning("SoX post-processing failed: %s", exc)
            return audio
        finally:
            if input_path:
                input_path.unlink(missing_ok=True)
            if output_path:
                output_path.unlink(missing_ok=True)

    def apply(self, audio: np.ndarray, sample_rate: int, fx: Optional[VoiceFXSettings], blend_override: Optional[float] = None) -> np.ndarray:
        """
        Apply audio effects to the input audio.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate of the audio
            fx: Voice FX settings to apply
            blend_override: If provided, overrides the computed blend mix. Set to 0.0 to disable blending.
        """
        if audio is None or fx is None:
            return audio

        # Handle stereo audio - convert to mono for processing
        is_stereo = audio.ndim == 2 and audio.shape[1] == 2
        if is_stereo:
            # Average the channels to mono
            audio = np.mean(audio, axis=1)

        base_audio = audio.astype(np.float32, copy=False)
        processed = base_audio.copy()

        if self._can_use_sox(fx):
            try:
                processed = self._apply_speed_pitch_sox(processed, sample_rate, fx.speed, fx.pitch_semitones)
            except Exception as exc:  # pragma: no cover - fallback for local installs
                logger.warning("SoX FX failed (%s); falling back to librosa pipeline.", exc)
                processed = self._apply_speed_pitch_librosa(processed, sample_rate, fx)
        else:
            processed = self._apply_speed_pitch_librosa(processed, sample_rate, fx)

        if fx.tone and fx.tone != "neutral":
            processed = self._apply_tone(processed, sample_rate, fx.tone)

        # Use blend_override if provided, otherwise compute blend mix
        if blend_override is not None:
            blend_mix = blend_override
        else:
            blend_mix = self._compute_blend_mix(fx)
        
        if blend_mix > 0.0:
            processed = self._blend_with_original(base_audio, processed, mix=blend_mix)

        return np.clip(processed, -1.0, 1.0)

    def apply_post_pipeline(
        self,
        audio: np.ndarray,
        sample_rate: int,
        fx: Optional[VoiceFXSettings],
        *,
        normalize: bool = True,
        fade_seconds: float = 0.01,
        blend_override: Optional[float] = None,
    ) -> np.ndarray:
        """Apply optional VFX first, then SoX post-processing."""
        processed = audio
        if fx is not None:
            processed = self.apply(processed, sample_rate, fx, blend_override=blend_override)
        return self.apply_sox_post(
            processed,
            sample_rate,
            normalize=normalize,
            fade_seconds=fade_seconds,
        )

    def prepare_prompt_audio(self, prompt_path: str, fx: Optional[VoiceFXSettings]) -> Optional[Path]:
        """Apply pitch/speed FX to a prompt audio file and return a temp WAV path."""
        if fx is None:
            return None
        if abs(fx.speed - 1.0) < 1e-3 and abs(fx.pitch_semitones) < 1e-3:
            return None

        fd, temp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        output_path = Path(temp_name)
        prompt = Path(prompt_path)
        try:
            logger.info(
                "Preparing temporary reference audio %s (pitch=%+.2f, speed=%.2f)",
                prompt.name,
                fx.pitch_semitones,
                fx.speed,
            )
            if self._can_use_sox(fx):
                command = [str(self.SOX_PATH), str(prompt), str(output_path)]
                if abs(fx.pitch_semitones) > 1e-3:
                    command += ["pitch", f"{fx.pitch_semitones * 100:.2f}"]
                if abs(fx.speed - 1.0) > 1e-3:
                    command += ["tempo", "-s", f"{fx.speed:.3f}"]
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or "SoX failed")
            else:
                audio, sr = sf.read(str(prompt), dtype='float32')
                processed = self._apply_speed_pitch_librosa(audio, sr, fx)
                sf.write(str(output_path), processed, sr)
            return output_path
        except Exception:
            output_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _compute_blend_mix(fx: VoiceFXSettings) -> float:
        if fx is None:
            return 0.0
        severity = min(abs(fx.pitch_semitones) / 3.0, 1.0)
        if abs(fx.speed - 1.0) > 0.1:
            severity = min(1.0, severity + 0.1)
        if fx.tone != "neutral":
            severity = min(1.0, severity + 0.15)
        return max(0.0, 0.2 * (1.0 - severity))

    @staticmethod
    def _apply_speed(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
        """
        Apply speed/tempo change without affecting pitch.
        speed > 1.0 = faster (shorter duration)
        speed < 1.0 = slower (longer duration)
        """
        AudioPostProcessor._require_librosa("speed")
        if pyrb is not None:
            try:
                # Use high-quality settings for Rubber Band
                return pyrb.time_stretch(audio, sample_rate, speed).astype(np.float32)
            except Exception as exc:  # pragma: no cover - graceful degradation
                logger.warning("Rubber Band time_stretch failed (%s); falling back to librosa", exc)

        # Use larger hop_length and n_fft for better quality (reduces metallic artifacts)
        # Default librosa uses n_fft=2048, hop_length=512 which can sound robotic
        n_fft = 4096
        hop_length = 1024
        
        # Compute STFT with better parameters
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        
        # Apply phase vocoder time stretch
        stft_stretched = librosa.phase_vocoder(stft, rate=speed, hop_length=hop_length)
        
        # Reconstruct audio
        stretched = librosa.istft(stft_stretched, hop_length=hop_length)
        
        return stretched.astype(np.float32, copy=False)

    @staticmethod
    def _apply_pitch(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
        AudioPostProcessor._require_librosa("pitch")
        if pyrb is not None:
            try:
                return pyrb.pitch_shift(audio, sample_rate, semitones).astype(np.float32)
            except Exception as exc:  # pragma: no cover - graceful degradation
                logger.warning("Rubber Band pitch_shift failed (%s); falling back to librosa", exc)

        # Use librosa's built-in pitch_shift with better parameters
        # n_fft=4096 provides smoother frequency resolution
        # Using larger values reduces metallic/robotic artifacts
        n_fft = 4096
        hop_length = 1024
        
        # Try high-quality resampling, fall back to kaiser_best if soxr not available
        try:
            shifted = librosa.effects.pitch_shift(
                audio,
                sr=sample_rate,
                n_steps=semitones,
                n_fft=n_fft,
                hop_length=hop_length,
                res_type='soxr_hq'
            )
        except Exception:
            shifted = librosa.effects.pitch_shift(
                audio,
                sr=sample_rate,
                n_steps=semitones,
                n_fft=n_fft,
                hop_length=hop_length,
                res_type='kaiser_best'
            )
        
        return shifted.astype(np.float32, copy=False)

    @staticmethod
    def _apply_speed_pitch_librosa(audio: np.ndarray, sample_rate: int, fx: VoiceFXSettings) -> np.ndarray:
        processed = audio
        if math.isfinite(fx.speed) and abs(fx.speed - 1.0) > 1e-3:
            processed = AudioPostProcessor._apply_speed(processed, sample_rate, fx.speed)
        if math.isfinite(fx.pitch_semitones) and abs(fx.pitch_semitones) > 1e-3:
            processed = AudioPostProcessor._apply_pitch(processed, sample_rate, fx.pitch_semitones)
        return processed

    @classmethod
    def _can_use_sox(cls, fx: VoiceFXSettings) -> bool:
        if fx is None:
            return False
        if not (math.isfinite(fx.speed) or math.isfinite(fx.pitch_semitones)):
            return False
        if abs(fx.speed - 1.0) < 1e-3 and abs(fx.pitch_semitones) < 1e-3:
            return False
        return bool(cls.SOX_PATH and cls.SOX_PATH.exists())

    @classmethod
    def _apply_speed_pitch_sox(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        speed: float,
        pitch_semitones: float,
    ) -> np.ndarray:
        speed = max(0.5, min(2.0, speed))
        pitch_semitones = max(-12.0, min(12.0, pitch_semitones))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
            input_path = Path(temp_in.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
            output_path = Path(temp_out.name)

        try:
            sf.write(str(input_path), audio, sample_rate)
            command = [str(cls.SOX_PATH), str(input_path), str(output_path)]
            if abs(pitch_semitones) > 1e-3:
                command += ["pitch", f"{pitch_semitones * 100:.2f}"]
            if abs(speed - 1.0) > 1e-3:
                command += ["tempo", "-s", f"{speed:.3f}"]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "SoX failed")
            processed, _ = sf.read(str(output_path), dtype='float32')
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        return processed.astype(np.float32, copy=False)

    @staticmethod
    def _apply_tone(audio: np.ndarray, sample_rate: int, profile: str) -> np.ndarray:
        spectrum = np.fft.rfft(audio)
        if spectrum.size == 0:
            return audio

        freqs = np.fft.rfftfreq(audio.shape[0], d=1.0 / sample_rate)
        last_freq = freqs[-1] or 1.0
        norm = freqs / last_freq

        strength = 0.18
        if profile == "warm":
            gain = 1.0 - strength * norm
        else:  # bright
            gain = 1.0 + strength * norm

        gain = np.clip(gain, 0.2, 1.8)
        spectrum *= gain
        shaped = np.fft.irfft(spectrum, n=audio.shape[0])
        return shaped.astype(np.float32)

    @staticmethod
    def _require_librosa(feature: str):
        if librosa is None:
            raise ImportError(
                "librosa is required for audio post-processing "
                f"({feature}). Run `pip install -r requirements.txt` "
                "to install the dependency."
            )

    @staticmethod
    def _blend_with_original(original: np.ndarray, processed: np.ndarray, mix: float = 0.15) -> np.ndarray:
        if processed is None or original is None or mix <= 0.0:
            return processed
        if processed.shape[0] != original.shape[0]:
            if librosa_util is not None:
                original = librosa_util.fix_length(original, size=processed.shape[0])
            else:
                original = np.interp(
                    np.linspace(0, 1, num=processed.shape[0], endpoint=False),
                    np.linspace(0, 1, num=original.shape[0], endpoint=False),
                    original
                ).astype(np.float32)
        mix = max(0.0, min(mix, 0.4))
        return (1.0 - mix) * processed + mix * original
