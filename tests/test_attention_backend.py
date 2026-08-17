from __future__ import annotations

from unittest.mock import patch

from src.attention_backend import resolve_qwen_attention_backend


def test_flash_attention_request_uses_validated_backend():
    with patch("src.attention_backend.validate_flash_attention", return_value=(True, "2.8.3")):
        assert resolve_qwen_attention_backend("flash_attention_2") == "flash_attention_2"


def test_flash_attention_request_falls_back_to_sdpa():
    with (
        patch("src.attention_backend.validate_flash_attention", return_value=(False, "not installed")),
        patch("src.attention_backend.has_torch_sdpa", return_value=True),
    ):
        assert resolve_qwen_attention_backend("flash_attention_2") == "sdpa"


def test_flash_attention_request_falls_back_to_eager_without_sdpa():
    with (
        patch("src.attention_backend.validate_flash_attention", return_value=(False, "not installed")),
        patch("src.attention_backend.has_torch_sdpa", return_value=False),
    ):
        assert resolve_qwen_attention_backend("flash") == "eager"


def test_explicit_backend_is_preserved():
    assert resolve_qwen_attention_backend("eager") == "eager"
    assert resolve_qwen_attention_backend("sdpa") == "sdpa"
    assert resolve_qwen_attention_backend("auto") is None


def test_flash_validation_can_be_forced_after_an_install_attempt():
    import src.attention_backend as backend

    backend._flash_attention_check = (False, "old failure")
    with patch.dict("sys.modules", {"flash_attn": None}):
        available, _detail = backend.validate_flash_attention(force=True)
    assert available is False
    assert backend._flash_attention_check != (False, "old failure")
