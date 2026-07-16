from unittest.mock import Mock

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
