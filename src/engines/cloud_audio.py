"""Audio helpers shared by cloud TTS engine adapters."""
from __future__ import annotations

from io import BytesIO
import subprocess
from typing import Optional

import soundfile as sf

from ..audio_effects import AudioPostProcessor, VoiceFXSettings
from ..system_tools import find_system_tool


class CloudAudioError(RuntimeError):
    """Raised when a cloud provider returns audio that cannot be used."""


def audio_bytes_to_wav(
    payload: bytes,
    *,
    input_format: Optional[str] = None,
    sample_rate: Optional[int] = None,
    channels: int = 1,
) -> bytes:
    """Decode provider audio and return a validated PCM RIFF/WAV payload."""
    raw = bytes(payload or b"")
    if not raw:
        raise CloudAudioError("The speech provider returned an empty audio response.")

    try:
        ffmpeg = find_system_tool("ffmpeg")
        if ffmpeg:
            command = [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
            ]
            source_format = "wav" if raw[:4] == b"RIFF" else str(input_format or "mp3")
            command.extend(["-f", source_format, "-i", "pipe:0"])
            if channels:
                command.extend(["-ac", str(max(1, min(int(channels), 2)))])
            if sample_rate:
                command.extend(["-ar", str(int(sample_rate))])
            command.extend(["-acodec", "pcm_s16le", "-f", "wav", "pipe:1"])
            result = subprocess.run(
                command,
                input=raw,
                capture_output=True,
                check=False,
                timeout=120,
            )
            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="replace").strip()[:300]
                raise CloudAudioError(detail or "FFmpeg could not decode the provider audio.")
            wav = _finalize_streamed_wav(bytes(result.stdout or b""))
        elif raw[:4] == b"RIFF":
            # A WAV response is already in the engine contract. This fallback
            # keeps plan-gated ElevenLabs WAV output usable even without FFmpeg.
            wav = raw
        else:
            raise CloudAudioError("FFmpeg is required to decode MP3 speech responses.")
    except CloudAudioError:
        raise
    except Exception as exc:
        raise CloudAudioError(
            "Unable to decode the speech provider's audio. Verify that FFmpeg is installed."
        ) from exc

    validate_wav_bytes(wav)
    return wav


def _finalize_streamed_wav(payload: bytes) -> bytes:
    """Replace unknown RIFF/data sizes emitted when FFmpeg writes to a pipe."""
    if len(payload) < 44 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        return payload
    data_offset = -1
    offset = 12
    while offset + 8 <= len(payload):
        chunk_id = payload[offset:offset + 4]
        if chunk_id == b"data":
            data_offset = offset
            break
        chunk_size = int.from_bytes(payload[offset + 4:offset + 8], "little")
        if chunk_size > len(payload) - offset - 8:
            break
        offset += 8 + chunk_size + (chunk_size % 2)
    if data_offset < 0:
        return payload
    finalized = bytearray(payload)
    finalized[4:8] = min(len(finalized) - 8, 0xFFFFFFFF).to_bytes(4, "little")
    finalized[data_offset + 4:data_offset + 8] = min(
        len(finalized) - data_offset - 8, 0xFFFFFFFF
    ).to_bytes(4, "little")
    return bytes(finalized)


def apply_wav_effects(payload: bytes, fx: Optional[VoiceFXSettings]) -> bytes:
    """Apply TTS-Story voice effects to WAV bytes without changing the contract."""
    validate_wav_bytes(payload)
    if fx is None or fx.is_identity():
        return payload
    try:
        audio, rate = sf.read(BytesIO(payload), dtype="float32", always_2d=False)
        processed = AudioPostProcessor().apply(audio, int(rate), fx, blend_override=0.0)
        output = BytesIO()
        sf.write(output, processed, int(rate), format="WAV", subtype="PCM_16")
        wav = output.getvalue()
    except Exception as exc:
        raise CloudAudioError("Unable to apply the selected voice effects.") from exc
    validate_wav_bytes(wav)
    return wav


def validate_wav_bytes(payload: bytes) -> None:
    """Reject empty, truncated, or mislabeled cloud audio responses."""
    if len(payload or b"") < 44 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise CloudAudioError("The speech provider returned an invalid WAV response.")


__all__ = ["CloudAudioError", "apply_wav_effects", "audio_bytes_to_wav", "validate_wav_bytes"]
