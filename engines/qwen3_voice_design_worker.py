"""Persistent Qwen3 VoiceDesign worker for the isolated Qwen environment."""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = ROOT / "engines" / "qwen3"
MODEL_ROOT = ENGINE_ROOT / "models"
CACHE_ROOT = ENGINE_ROOT / "cache"
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
os.environ["PATH"] = os.pathsep.join([
    str(ROOT / "tools" / "ffmpeg"),
    str(ROOT / "tools" / "sox"),
    str(ROOT / "tools" / "rubberband"),
    os.environ.get("PATH", ""),
])
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
MODEL = None
MODEL_SIGNATURE = None


def resolve_device(value: str) -> str:
    import torch
    value = (value or "auto").lower()
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(value: str):
    import torch
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get((value or "bfloat16").lower(), torch.bfloat16)


def resolve_attention(value: str):
    requested = (value or "").strip()
    if requested != "flash_attention_2":
        return requested or None
    try:
        import flash_attn  # noqa: F401
        return requested
    except Exception:
        return "eager"


def ensure_model(model_id: str) -> Path:
    from huggingface_hub import snapshot_download
    target = MODEL_ROOT / model_id.replace("/", "_")
    if not target.exists() or not any(target.iterdir()):
        snapshot_download(repo_id=model_id, local_dir=str(target))
    return target


def get_model(request: dict):
    global MODEL, MODEL_SIGNATURE
    from qwen_tts import Qwen3TTSModel
    model_id = request["model_id"]
    device = resolve_device(request.get("device", "auto"))
    dtype_name = request.get("dtype", "bfloat16")
    attention = resolve_attention(request.get("attention", "flash_attention_2"))
    signature = (model_id, device, dtype_name, attention)
    if MODEL is not None and signature == MODEL_SIGNATURE:
        return MODEL
    MODEL = None
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    MODEL = Qwen3TTSModel.from_pretrained(
        str(ensure_model(model_id)),
        device_map=device,
        dtype=resolve_dtype(dtype_name),
        attn_implementation=attention,
    )
    MODEL_SIGNATURE = signature
    return MODEL


def generate(request: dict) -> dict:
    import soundfile as sf
    import torch
    model = get_model(request)
    seed = int(request["seed"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    started = time.perf_counter()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        wavs, sample_rate = model.generate_voice_design(
            text=request["text"],
            instruct=request.get("instruct", ""),
            language=request.get("language", "English"),
            non_streaming_mode=True,
            **(request.get("generation_kwargs") or {}),
        )
    if not wavs:
        raise RuntimeError("No audio produced for preview")
    output = Path(request["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, wavs[0], int(sample_rate))
    result = {
        "path": str(output),
        "sample_rate": int(sample_rate),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if torch.cuda.is_available():
        result.update({
            "cuda_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "cuda_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
            "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        })
    gc.collect()
    return result


def main() -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            result = generate(request)
            response = {"id": request["id"], "success": True, **result}
        except Exception as exc:
            response = {
                "id": request.get("id") if "request" in locals() else None,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print("TTS_STORY_QWEN_RESULT " + json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
