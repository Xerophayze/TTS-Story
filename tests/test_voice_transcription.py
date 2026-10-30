from __future__ import annotations

import pytest
import subprocess
from types import SimpleNamespace

import src.voice_transcription as transcription

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


def test_missing_isolated_tool_has_actionable_error_without_importing_funasr(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.touch()
    monkeypatch.setattr(transcription, "transcription_available", lambda: False)
    generator = VoiceTranscriptGenerator()
    with pytest.raises(transcription.VoiceTranscriptionUnavailable, match="Utilities.*Install tool"):
        generator.transcribe(audio)
    assert generator._model is None
    assert not generator.is_running


def test_runtime_probe_requires_python_marker_and_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(transcription, "ROOT", tmp_path)
    python = transcription.transcription_python()
    python.parent.mkdir(parents=True)
    python.touch()
    assert not transcription.transcription_available()
    (transcription.transcription_runtime_root() / ".ready").touch()
    assert not transcription.transcription_available()
    (tmp_path / "engines" / "transcription_worker.py").touch()
    assert transcription.transcription_available()


def test_isolated_request_uses_cpu_worker_and_cleans_protocol_response(tmp_path, monkeypatch):
    audio = tmp_path / "sample with spaces.wav"
    audio.touch()
    monkeypatch.setattr(transcription, "transcription_available", lambda: True)
    generator = VoiceTranscriptGenerator()
    def run(command, **kwargs):
        assert generator.is_running
        assert command[2:] == ["--audio", str(audio.resolve())]
        assert command[1].endswith("transcription_worker.py")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["timeout"] == 180
        return SimpleNamespace(returncode=0, stdout='ordinary loader output\nTTS_STORY_TRANSCRIPT {"success":true,"transcript":"<|en|> Exact words."}\n')
    monkeypatch.setattr(transcription.subprocess, "run", run)
    assert generator.transcribe(audio) == "Exact words."
    assert generator._model is None
    assert not generator.is_running


@pytest.mark.parametrize("result", [
    SimpleNamespace(returncode=0, stdout="no event"),
    SimpleNamespace(returncode=0, stdout='TTS_STORY_TRANSCRIPT {invalid json}\n'),
    SimpleNamespace(returncode=1, stdout='TTS_STORY_TRANSCRIPT {"success":false,"error":"Repair the model cache."}\n'),
    SimpleNamespace(returncode=0, stdout='TTS_STORY_TRANSCRIPT {"success":true,"transcript":"<|en|>"}\n'),
])
def test_isolated_failures_do_not_return_fake_success(tmp_path, monkeypatch, result):
    audio = tmp_path / "sample.wav"
    audio.touch()
    monkeypatch.setattr(transcription, "transcription_available", lambda: True)
    monkeypatch.setattr(transcription.subprocess, "run", lambda *args, **kwargs: result)
    generator = VoiceTranscriptGenerator()
    with pytest.raises(VoiceTranscriptionError):
        generator.transcribe(audio)
    assert not generator.is_running


def test_worker_timeout_releases_busy_state(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.touch()
    monkeypatch.setattr(transcription, "transcription_available", lambda: True)
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 180)
    monkeypatch.setattr(transcription.subprocess, "run", run)
    generator = VoiceTranscriptGenerator()
    with pytest.raises(VoiceTranscriptionError, match="timed out"):
        generator.transcribe(audio)
    assert not generator.is_running
