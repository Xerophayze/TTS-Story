import ast
from pathlib import Path
from unittest.mock import Mock

from src.audio_effects import VoiceFXSettings
from src.engines.omnivoice_clone_engine import OmniVoiceCloneEngine
from app import DEFAULT_CONFIG, _normalize_omnivoice_clone_options


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
    engine.batch_size = 2
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
    assert captured_job["batch_size"] == 2
    assert not prepared_prompt.exists()


def test_omnivoice_worker_reuses_encoded_clone_prompts():
    worker_source = (
        Path(__file__).resolve().parents[1]
        / "engines"
        / "omnivoice"
        / "omnivoice_worker.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(worker_source)
    keyword_names = {
        keyword.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    }

    assert "create_voice_clone_prompt" in worker_source
    assert "voice_clone_prompt" in keyword_names
    assert "ref_audio=ref_audio" not in worker_source.split('if mode == "clone":', 1)[1]
    assert "VOICE_PROMPT_CACHE_DIR" in worker_source


def test_omnivoice_settings_clamp_batch_and_preserve_decimal_buffer():
    normalized = _normalize_omnivoice_clone_options({
        "omnivoice_batch_size": 99,
        "omnivoice_duration_safety_margin": 0.5,
    })

    assert normalized["omnivoice_batch_size"] == 8
    assert normalized["omnivoice_duration_safety_margin"] == 0.5


def test_all_omnivoice_form_settings_are_persistable():
    expected = {
        "omnivoice_clone_model_id",
        "omnivoice_clone_device",
        "omnivoice_clone_dtype",
        "omnivoice_clone_num_step",
        "omnivoice_clone_default_prompt",
        "omnivoice_clone_default_prompt_text",
        "omnivoice_design_model_id",
        "omnivoice_design_device",
        "omnivoice_design_dtype",
        "omnivoice_design_num_step",
        "omnivoice_design_default_instruct",
        "omnivoice_chunk_size",
        "omnivoice_post_process",
        "omnivoice_duration_safety_margin",
        "omnivoice_batch_size",
    }

    assert expected <= DEFAULT_CONFIG.keys()


def _pause_test_engine(tmp_path):
    engine = object.__new__(OmniVoiceCloneEngine)
    engine.default_prompt = None
    engine.default_prompt_text = None
    engine.model_id = "test-model"
    engine.device = "cpu"
    engine.dtype = "float32"
    engine.num_step = 1
    engine.batch_size = 1
    engine.post_process = False
    engine.duration_safety_margin = 0.25
    engine.post_processor = Mock()
    engine.post_processor.apply_post_pipeline.side_effect = lambda audio, *_args: audio
    engine._resolve_ref_text = Mock(return_value="reference transcript")
    prompt = tmp_path / "reference.wav"
    prompt.write_bytes(b"reference")
    segments = [{"speaker": "narrator", "chunks": ["First chunk."]}]
    voices = {"narrator": {"audio_prompt_path": str(prompt)}}
    return engine, segments, voices


def test_omnivoice_pause_poll_terminates_before_next_chunk(tmp_path):
    engine, segments, voices = _pause_test_engine(tmp_path)

    class PauseSignal(Exception):
        pass

    def fake_run_worker(_job, chunk_done_cb=None, cancel_cb=None):
        assert cancel_cb is not None and cancel_cb() is True
        raise RuntimeError("worker terminated")

    def progress(increment=1):
        if increment == 0:
            raise PauseSignal()

    engine._run_worker = fake_run_worker

    import pytest
    with pytest.raises(PauseSignal):
        engine.generate_batch(
            segments,
            voices,
            tmp_path / "output",
            progress_cb=progress,
            pause_cb=lambda: True,
            cancel_cb=lambda: False,
        )


def test_omnivoice_records_pause_raised_by_chunk_callback(tmp_path):
    engine, segments, voices = _pause_test_engine(tmp_path)

    class PauseSignal(Exception):
        pass

    def fake_run_worker(job, chunk_done_cb=None, cancel_cb=None):
        import numpy as np
        import soundfile as sf
        output_path = job["chunks"][0]["output_path"]
        sf.write(output_path, np.zeros(240, dtype="float32"), 24000)
        try:
            chunk_done_cb(output_path)
        except PauseSignal:
            pass  # Mirrors the worker stderr reader handing control back.

    engine._run_worker = fake_run_worker

    import pytest
    with pytest.raises(PauseSignal):
        engine.generate_batch(
            segments,
            voices,
            tmp_path / "output",
            progress_cb=lambda: None,
            chunk_cb=lambda *_args: (_ for _ in ()).throw(PauseSignal()),
            pause_cb=lambda: True,
            cancel_cb=lambda: False,
        )
