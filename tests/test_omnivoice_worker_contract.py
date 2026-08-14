from pathlib import Path


WORKER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "engines"
    / "omnivoice"
    / "omnivoice_worker.py"
).read_text(encoding="utf-8")


def test_worker_uses_supported_omnivoice_postprocess_option():
    assert "postprocess_output=bool(post_process)" in WORKER_SOURCE
    assert 'kwargs["post_process"]' not in WORKER_SOURCE
    assert 'design_kwargs["post_process"]' not in WORKER_SOURCE


def test_worker_preserves_final_audio_and_keeps_silence_padding():
    assert "def _fade_in_and_pad_audio(" in WORKER_SOURCE
    assert "processed[..., :fade_length] *= fade_in" in WORKER_SOURCE
    assert "fade_out" not in WORKER_SOURCE
    assert "processed = torch.cat([silence, processed, silence], dim=-1)" in WORKER_SOURCE
    assert (
        "omnivoice_model_module.fade_and_pad_audio = _fade_in_and_pad_audio"
        in WORKER_SOURCE
    )


def test_worker_adds_configurable_duration_headroom_without_rewriting_text():
    assert "def _apply_duration_safety_margin(" in WORKER_SOURCE
    assert 'job.get("duration_safety_margin", 0.25)' in WORKER_SOURCE
    assert "original_estimate(*args, **kwargs) + margin_tokens" in WORKER_SOURCE
