from __future__ import annotations

import pytest

from src.voice_transcription import VoiceTranscriptionError, VoiceTranscriptGenerator


class FakeSenseVoice:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_voice_transcript_generator_cleans_sensevoice_tags_and_reuses_model(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-test")
    model = FakeSenseVoice([{
        "text": "<|en|><|NEUTRAL|><|Speech|>  These are the exact words.  "
    }])
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return model

    generator = VoiceTranscriptGenerator(device="cpu", model_factory=factory)
    assert generator.transcribe(audio) == "These are the exact words."
    assert generator.transcribe(audio) == "These are the exact words."
    assert len(factory_calls) == 1
    assert factory_calls[0]["model"] == "iic/SenseVoiceSmall"
    assert len(model.calls) == 2


def test_voice_transcript_generator_rejects_missing_or_silent_sample(tmp_path):
    generator = VoiceTranscriptGenerator(
        model_factory=lambda **kwargs: FakeSenseVoice([{"text": "<|en|>  "}])
    )
    with pytest.raises(VoiceTranscriptionError, match="file is missing"):
        generator.transcribe(tmp_path / "missing.wav")
    audio = tmp_path / "silent.wav"
    audio.write_bytes(b"RIFF-test")
    with pytest.raises(VoiceTranscriptionError, match="No speech was detected"):
        generator.transcribe(audio)
