"""Shared attention-backend selection for Qwen3-TTS engines."""

from __future__ import annotations

import logging
from typing import Optional


_flash_attention_check: Optional[tuple[bool, str]] = None


def has_torch_sdpa() -> bool:
    """Return whether this PyTorch build exposes scaled-dot-product attention."""
    try:
        import torch.nn.functional as functional

        return callable(getattr(functional, "scaled_dot_product_attention", None))
    except Exception:
        return False


def validate_flash_attention(*, force: bool = False) -> tuple[bool, str]:
    """Import FlashAttention and run a tiny CUDA forward pass once per process."""
    global _flash_attention_check
    if force:
        _flash_attention_check = None
    if _flash_attention_check is not None:
        return _flash_attention_check

    try:
        import torch
        import flash_attn  # type: ignore
        from flash_attn import flash_attn_func  # type: ignore

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        query = torch.randn((1, 4, 1, 32), device="cuda", dtype=torch.float16)
        result = flash_attn_func(query, query, query, causal=False)
        torch.cuda.synchronize()
        if result.shape != query.shape:
            raise RuntimeError(f"unexpected output shape {tuple(result.shape)}")
        version = getattr(flash_attn, "__version__", "unknown")
        _flash_attention_check = (True, str(version))
    except Exception as exc:
        _flash_attention_check = (False, str(exc))
    return _flash_attention_check


def resolve_qwen_attention_backend(
    requested: Optional[str],
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Resolve a requested Transformers attention backend with safe fallbacks."""
    normalized = (requested or "").strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return None
    if normalized not in {"flash_attention_2", "flash_attention2", "flash"}:
        return normalized

    available, detail = validate_flash_attention()
    if available:
        if logger:
            logger.info("FlashAttention 2 validated successfully (version=%s)", detail)
        return "flash_attention_2"

    fallback = "sdpa" if has_torch_sdpa() else "eager"
    if logger:
        logger.warning(
            "FlashAttention 2 is unavailable or failed validation (%s); "
            "falling back to %s attention for Qwen3",
            detail,
            fallback,
        )
    return fallback


__all__ = [
    "has_torch_sdpa",
    "resolve_qwen_attention_backend",
    "validate_flash_attention",
]
