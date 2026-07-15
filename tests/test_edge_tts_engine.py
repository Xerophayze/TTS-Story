from __future__ import annotations

import io
import wave

import pytest

from src.engines.edge_tts_engine import EdgeTTSEngine, EdgeTTSError


def _wav_bytes(rate: int = 24000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * 240)
    return output.getvalue()


class FakeCommunicate:
    calls = []

    def __init__(self, text, **kwargs):
        self.text = text
        self.kwargs = kwargs
        self.calls.append((text, kwargs))

    def stream_sync(self):
        yield {"type": "WordBoundary", "offset": 0}
        yield {"type": "audio", "data": b"mp3-a"}
        yield {"type": "audio", "data": b"mp3-b"}


def _converter(payload, **kwargs):
    assert payload == b"mp3-amp3-b"
    assert kwargs == {"input_format": "mp3", "sample_rate": 24000, "channels": 1}
    return _wav_bytes()


def test_voice_catalog_is_dynamic_and_normalized():
    async def voices():
        return [
            {
                "ShortName": "en-US-TestNeural",
                "FriendlyName": "Test Voice",
                "Gender": "Female",
                "Locale": "en-US",
                "VoiceTag": {
                    "ContentCategories": ["General"],
                    "VoicePersonalities": ["Friendly"],
                },
            }
        ]

    engine = EdgeTTSEngine(
        communicate_factory=FakeCommunicate,
        list_voices_func=voices,
        audio_converter=_converter,
    )
    assert engine.list_voices() == [
        {
            "short_name": "en-US-TestNeural",
            "display_name": "Test Voice",
            "local_name": "Test Voice",
            "gender": "Female",
            "locale": "en-US",
            "locale_name": "en-US",
            "content_categories": ["General"],
            "personalities": ["Friendly"],
        }
    ]


def test_preview_maps_speed_and_returns_wav():
    FakeCommunicate.calls.clear()
    engine = EdgeTTSEngine(
        default_voice="en-US-TestNeural",
        default_volume=5,
        communicate_factory=FakeCommunicate,
        list_voices_func=lambda: [],
        audio_converter=_converter,
    )
    audio = engine.generate_audio("Hello", speed=1.25)
    assert audio[:4] == b"RIFF"
    assert FakeCommunicate.calls[-1] == (
        "Hello",
        {
            "voice": "en-US-TestNeural",
            "rate": "+25%",
            "volume": "+5%",
            "connect_timeout": 60,
            "receive_timeout": 60,
        },
    )


def test_batch_preserves_chronological_order_and_callbacks(tmp_path):
    engine = EdgeTTSEngine(
        communicate_factory=FakeCommunicate,
        list_voices_func=lambda: [],
        audio_converter=_converter,
        max_parallel=2,
    )
    seen = []
    paths = engine.generate_batch(
        [{"speaker": "Narrator", "chunks": ["One", "Two"]}],
        {"Narrator": {"voice": "en-US-TestNeural"}},
        tmp_path,
        chunk_cb=lambda index, metadata, path: seen.append((index, metadata["text"], path)),
        parallel_workers=8,
    )
    assert [path.rsplit("_", 1)[-1] for path in paths] == ["000000.wav", "000001.wav"]
    assert sorted((index, text) for index, text, _ in seen) == [(0, "One"), (1, "Two")]
    assert all(open(path, "rb").read(4) == b"RIFF" for path in paths)


def test_empty_text_is_rejected():
    engine = EdgeTTSEngine(
        communicate_factory=FakeCommunicate,
        list_voices_func=lambda: [],
        audio_converter=_converter,
    )
    with pytest.raises(EdgeTTSError, match="empty"):
        engine.generate_audio("  ")
