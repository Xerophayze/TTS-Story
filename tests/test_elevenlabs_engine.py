from __future__ import annotations

import io
import wave

import pytest

from src.engines.elevenlabs_engine import ElevenLabsEngine, ElevenLabsError


def _wav_bytes(rate: int = 44100) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * 441)
    return output.getvalue()


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b"mp3", headers=None, text=""):
        self.status_code = status
        self._payload = payload
        self.content = content
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


def _converter(payload, **kwargs):
    assert payload == b"mp3"
    assert kwargs == {"input_format": "mp3", "sample_rate": 44100, "channels": 1}
    return _wav_bytes()


def test_catalogs_and_subscription_are_normalized():
    responses = [
        FakeResponse(payload=[{
            "model_id": "eleven_multilingual_v2",
            "name": "Multilingual v2",
            "can_do_text_to_speech": True,
            "can_use_style": True,
            "can_use_speaker_boost": True,
        }]),
        FakeResponse(payload={
            "voices": [{
                "voice_id": "voice_1",
                "name": "Narrator",
                "category": "premade",
                "labels": {"gender": "female", "language": "en"},
            }],
            "has_more": False,
        }),
        FakeResponse(payload={"tier": "creator", "character_count": 100, "character_limit": 10000}),
    ]
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return responses.pop(0)

    engine = ElevenLabsEngine(api_key="secret", request_func=request, audio_converter=_converter)
    assert engine.list_models()[0]["model_id"] == "eleven_multilingual_v2"
    assert engine.list_voices()[0]["short_name"] == "voice_1"
    assert engine.get_subscription()["character_limit"] == 10000
    assert all(call[2]["headers"]["xi-api-key"] == "secret" for call in calls)


def test_preview_posts_voice_settings_and_returns_wav():
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return FakeResponse(content=b"mp3")

    engine = ElevenLabsEngine(
        api_key="secret",
        default_voice="voice_1",
        request_func=request,
        audio_converter=_converter,
        stability=0.6,
        similarity_boost=0.8,
    )
    audio = engine.generate_audio("Hello", speed=1.1, voice_options={"style": 0.2})
    assert audio[:4] == b"RIFF"
    method, url, kwargs = calls[-1]
    assert method == "POST"
    assert url.endswith("/v1/text-to-speech/voice_1")
    assert kwargs["params"] == {"output_format": "mp3_44100_128"}
    assert kwargs["json"]["voice_settings"] == {
        "stability": 0.6,
        "similarity_boost": 0.8,
        "style": 0.2,
        "use_speaker_boost": True,
        "speed": 1.1,
    }


def test_batch_adds_same_speaker_continuity_and_keeps_order(tmp_path):
    payloads = []

    def request(method, url, **kwargs):
        payloads.append(kwargs["json"])
        return FakeResponse(content=b"mp3")

    engine = ElevenLabsEngine(
        api_key="secret",
        request_func=request,
        audio_converter=_converter,
        max_parallel=1,
    )
    paths = engine.generate_batch(
        [{"speaker": "A", "chunks": ["One", "Two"]}, {"speaker": "B", "chunks": ["Three"]}],
        {"A": {"voice": "voice_a"}, "B": {"voice": "voice_b"}},
        tmp_path,
    )
    assert payloads[0]["next_text"] == "Two"
    assert payloads[1]["previous_text"] == "One"
    assert "next_text" not in payloads[1]
    assert "previous_text" not in payloads[2]
    assert [path.rsplit("_", 1)[-1] for path in paths] == [
        "000000.wav", "000001.wav", "000002.wav"
    ]


def test_transient_rate_limit_honors_retry_after():
    responses = [FakeResponse(status=429, headers={"Retry-After": "0"}), FakeResponse(content=b"mp3")]
    sleeps = []
    engine = ElevenLabsEngine(
        api_key="secret",
        request_func=lambda *args, **kwargs: responses.pop(0),
        sleep_func=sleeps.append,
        audio_converter=_converter,
    )
    assert engine.generate_audio("Hello")[:4] == b"RIFF"
    assert sleeps == [0.0]


def test_authentication_error_is_actionable():
    engine = ElevenLabsEngine(
        api_key="secret",
        request_func=lambda *args, **kwargs: FakeResponse(status=401),
        audio_converter=_converter,
    )
    with pytest.raises(ElevenLabsError, match="API key"):
        engine.generate_audio("Hello")
