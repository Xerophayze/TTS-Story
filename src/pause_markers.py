"""Engine-independent pause markers used by text processing and rendering."""

from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import List, Tuple


# Pause controls can be placed on their own line or directly after punctuation,
# for example ``The room fell silent.***``. Requiring either whitespace or
# punctuation before the marker keeps ordinary Markdown emphasis such as
# ``important***`` from being interpreted as silence.
PAUSE_MARKER_PATTERN = re.compile(
    r"(?<![\w*])(\*{3}(?:\*{3})*)(?!\*)(?=\s|$)",
)
# Speaker-tagged audiobook text is an explicit TTS control context. Be more
# permissive there so an LLM-produced heading such as ``CHAPTER ONE******``
# is treated as speech followed by silence rather than literal asterisks.
TAGGED_PAUSE_MARKER_PATTERN = re.compile(
    r"(?<!\*)(\*{3}(?:\*{3})*)(?!\*)(?=\s|$)",
)
TRAILING_PAUSE_MARKER_PATTERN = re.compile(r"\s*\*{3}(?:\*{3})*\s*$")


def pause_seconds_for_text(text: str) -> float | None:
    """Return the pause represented by a 3-star group."""
    candidate = str(text or "").strip()
    if not re.fullmatch(r"\*{3}(?:\*{3})*", candidate):
        return None
    return (len(candidate) // 3) * 0.25


def sanitize_display_title(value: str | None) -> str:
    """Remove only recognized trailing pause controls from public titles."""
    title = str(value or "")
    return TRAILING_PAUSE_MARKER_PATTERN.sub("", title).strip()


def split_text_and_pause_markers(
    text: str,
    *,
    allow_attached: bool = False,
) -> List[Tuple[str, str]]:
    """Split prose from standalone or inline pause markers, preserving order."""
    source = str(text or "")
    parts: List[Tuple[str, str]] = []
    cursor = 0
    pattern = TAGGED_PAUSE_MARKER_PATTERN if allow_attached else PAUSE_MARKER_PATTERN
    for match in pattern.finditer(source):
        prose = source[cursor:match.start()]
        if prose.strip():
            parts.append(("text", prose))
        parts.append(("pause", match.group(1)))
        cursor = match.end()
    remainder = source[cursor:]
    if remainder.strip():
        parts.append(("text", remainder))
    return parts


def write_silence_wav(path: Path, duration_seconds: float, sample_rate: int = 24000) -> Path:
    """Create a mono 16-bit PCM WAV containing only digital silence."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rate = max(int(sample_rate or 24000), 8000)
    frame_count = max(1, int(round(max(float(duration_seconds), 0.0) * rate)))
    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return destination
