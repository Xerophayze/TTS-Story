from __future__ import annotations

import io
import wave

import pytest

from src.engines.openai_tts_engine import OpenAITTSEngine, OpenAITTSError


def _wav_bytes(rate: int = 24000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * 240)
    return output.getvalue()


class FakeResponse:
    def __init__(self, status=200, content=None, text="", headers=None):
        self.status_code = status
        self.content = _wav_bytes() if content is None else content
        self.text = text
        self.headers = headers or {}


def _converter(payload, **kwargs):
    assert payload[:4] == b"RIFF"
    assert kwargs == {"input_format": "wav", "sample_rate": None, "channels": 1}
    return payload


def test_official_speech_request_uses_voice_speed_and_instructions():
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return FakeResponse()

    engine = OpenAITTSEngine(
        api_key="secret",
        default_voice="coral",
        instructions="Speak warmly.",
        request_func=request,
        audio_converter=_converter,
    )
    assert engine.generate_audio("Hello", speed=0.72)[:4] == b"RIFF"
    method, url, kwargs = calls[-1]
    assert method == "POST"
    assert url == "https://api.openai.com/v1/audio/speech"
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["json"] == {
        "model": "gpt-4o-mini-tts",
        "input": "Hello",
        "voice": "coral",
        "response_format": "wav",
        "speed": 0.72,
        "instructions": "Speak warmly.",
    }


def test_custom_full_endpoint_and_custom_voice_object_need_no_key():
    calls = []
    engine = OpenAITTSEngine(
        base_url="http://localhost:8080/v1/audio/speech",
        default_voice="voice_123",
        request_func=lambda method, url, **kwargs: calls.append((method, url, kwargs)) or FakeResponse(),
        audio_converter=_converter,
    )
    engine.generate_audio("Hello")
    assert calls[0][1] == "http://localhost:8080/v1/audio/speech"
    assert "Authorization" not in calls[0][2]["headers"]
    assert calls[0][2]["json"]["voice"] == {"id": "voice_123"}


def test_error_is_actionable():
    engine = OpenAITTSEngine(
        api_key="bad",
        request_func=lambda *args, **kwargs: FakeResponse(status=401),
        audio_converter=_converter,
    )
    with pytest.raises(OpenAITTSError, match="API key"):
        engine.generate_audio("Hello")
