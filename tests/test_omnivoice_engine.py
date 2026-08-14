from unittest.mock import Mock

from src.audio_effects import VoiceFXSettings
from src.engines.omnivoice_clone_engine import OmniVoiceCloneEngine


def _bare_engine():
    engine = object.__new__(OmniVoiceCloneEngine)
    engine._get_ref_text = Mock(return_value="cached transcript")
    return engine


def test_omnivoice_prefers_explicit_prompt_transcript():
    engine = _bare_engine()

    result = engine._resolve_ref_text("reference.wav", "  exact transcript  ")

    assert result == "exact transcript"
    engine._get_ref_text.assert_not_called()


def test_omnivoice_falls_back_to_cached_transcript():
    engine = _bare_engine()

    result = engine._resolve_ref_text("reference.wav", "   ")

    assert result == "cached transcript"
    engine._get_ref_text.assert_called_once_with("reference.wav")


def test_omnivoice_routes_pitch_and_speaker_speed_to_reference_prompt():
    prompt_fx, output_fx = OmniVoiceCloneEngine._split_reference_and_output_fx(
        VoiceFXSettings(pitch_semitones=2.5, speed=1.0, tone="warm"),
        speed_override=0.84,
    )

    assert prompt_fx is not None
    assert prompt_fx.pitch_semitones == 2.5
    assert prompt_fx.speed == 0.84
    assert prompt_fx.tone == "neutral"
    assert output_fx is not None
    assert output_fx.pitch_semitones == 0.0
    assert output_fx.speed == 0.84
    assert output_fx.tone == "warm"


def test_omnivoice_prepares_shared_reference_only_once(tmp_path):
    engine = object.__new__(OmniVoiceCloneEngine)
    engine.default_prompt = None
    engine.default_prompt_text = None
    engine.model_id = "test-model"
    engine.device = "cpu"
    engine.dtype = "float32"
    engine.num_step = 1
    engine.post_process = False
    engine.duration_safety_margin = 0.4
    engine.post_processor = Mock()
    engine.post_processor.apply_post_pipeline.side_effect = lambda audio, *_args: audio
    engine._resolve_ref_text = Mock(return_value="reference transcript")

    source_prompt = tmp_path / "reference.wav"
    source_prompt.write_bytes(b"reference")
    prepared_prompt = tmp_path / "prepared.wav"
    prepared_prompt.write_bytes(b"prepared")
    engine.post_processor.prepare_prompt_audio.return_value = prepared_prompt

    captured_job = {}

    def fake_run_worker(job, chunk_done_cb=None, cancel_cb=None):
        import numpy as np
        import soundfile as sf
        captured_job.update(job)
        for chunk in job["chunks"]:
            sf.write(chunk["output_path"], np.zeros(2400, dtype="float32"), 24000)
            if chunk_done_cb:
                chunk_done_cb(chunk["output_path"])

    engine._run_worker = fake_run_worker
    segments = [
        {"speaker": "narrator", "chunks": ["First chunk."]},
        {"speaker": "narrator", "chunks": ["Second chunk."]},
    ]
    voice_config = {
        "narrator": {
            "audio_prompt_path": str(source_prompt),
            "speed": 0.8,
            "fx": {"pitch": -1.5},
        }
    }

    files = engine.generate_batch(
        segments=segments,
        voice_config=voice_config,
        output_dir=tmp_path / "output",
        speed=1.0,
    )

    assert len(files) == 2
    engine.post_processor.prepare_prompt_audio.assert_called_once()
    prepared_fx = engine.post_processor.prepare_prompt_audio.call_args.args[1]
    assert prepared_fx.pitch_semitones == -1.5
    assert prepared_fx.speed == 0.8
    assert {chunk["ref_audio"] for chunk in captured_job["chunks"]} == {
        str(prepared_prompt)
    }
    assert captured_job["duration_safety_margin"] == 0.4
    assert not prepared_prompt.exists()
