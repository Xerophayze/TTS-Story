from __future__ import annotations

import io
import asyncio
import time
import wave
from pathlib import Path

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


def test_symbol_only_scene_break_becomes_local_silence_without_provider_call():
    calls = []

    class RecordingCommunicate:
        def __init__(self, text, **kwargs):
            calls.append(text)

    engine = EdgeTTSEngine(
        communicate_factory=RecordingCommunicate,
        list_voices_func=lambda: [],
        audio_converter=_converter,
    )

    audio = engine.generate_audio("***")

    assert audio[:4] == b"RIFF"
    assert len(audio) > 44
    assert calls == []


def test_six_star_marker_becomes_half_second_of_local_silence():
    calls = []

    class RecordingCommunicate:
        def __init__(self, text, **kwargs):
            calls.append(text)

    engine = EdgeTTSEngine(
        communicate_factory=RecordingCommunicate,
        list_voices_func=lambda: [],
        audio_converter=_converter,
    )

    audio = engine.generate_audio("******")

    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        assert wav_file.getnframes() / wav_file.getframerate() == pytest.approx(0.5)
    assert calls == []


def test_hung_async_stream_hits_hard_timeout_and_retries():
    calls = []
    delays = []

    class HangingCommunicate:
        def __init__(self, text, **kwargs):
            calls.append((text, kwargs))

        async def stream(self):
            await asyncio.sleep(1)
            if False:
                yield {}

    engine = EdgeTTSEngine(
        communicate_factory=HangingCommunicate,
        list_voices_func=lambda: [],
        audio_converter=_converter,
        max_retries=1,
        sleep_func=delays.append,
    )
    engine.timeout = 0.01

    with pytest.raises(EdgeTTSError, match="timed out"):
        engine.generate_audio("A request that never finishes")

    assert len(calls) == 2
    assert delays == [1]


def test_parallel_callbacks_are_committed_in_order_for_safe_resume(tmp_path):
    delays = {"One": 0.04, "Two": 0.001, "Three": 0.002}

    class VariableCommunicate:
        def __init__(self, text, **kwargs):
            self.text = text

        def stream_sync(self):
            time.sleep(delays[self.text])
            yield {"type": "audio", "data": self.text.encode("utf-8")}

    engine = EdgeTTSEngine(
        communicate_factory=VariableCommunicate,
        list_voices_func=lambda: [],
        audio_converter=lambda payload, **kwargs: _wav_bytes(),
        max_parallel=3,
    )
    seen = []
    paths = engine.generate_batch(
        [{"speaker": "Narrator", "chunks": ["One", "Two", "Three"]}],
        {"Narrator": {"voice": "en-US-TestNeural"}},
        tmp_path,
        parallel_workers=3,
        chunk_cb=lambda index, metadata, path: seen.append((index, metadata["text"])),
        start_index=7,
    )

    assert seen == [(0, "One"), (1, "Two"), (2, "Three")]
    assert [Path(path).name for path in paths] == [
        "edge_chunk_000007.wav",
        "edge_chunk_000008.wav",
        "edge_chunk_000009.wav",
    ]
