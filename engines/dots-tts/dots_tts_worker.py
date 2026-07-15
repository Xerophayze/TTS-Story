"""Dot.TTS isolated worker.

Called by the main app via subprocess. The worker keeps Dot.TTS dependencies in
engines/dots-tts/.venv and loads the model once per job file.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _install_wetext_fallback_modules() -> None:
    """Let Dot.TTS import on Windows when WeTextProcessing/pynini is unavailable."""
    try:
        import tn.chinese.normalizer  # type: ignore  # noqa: F401
        import tn.english.normalizer  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    class _IdentityNormalizer:
        def __init__(self, *args, **kwargs):
            pass

        def normalize(self, text: str) -> str:
            return text

    tn_mod = sys.modules.setdefault("tn", types.ModuleType("tn"))
    chinese_mod = sys.modules.setdefault("tn.chinese", types.ModuleType("tn.chinese"))
    english_mod = sys.modules.setdefault("tn.english", types.ModuleType("tn.english"))
    zh_norm_mod = types.ModuleType("tn.chinese.normalizer")
    en_norm_mod = types.ModuleType("tn.english.normalizer")
    zh_norm_mod.Normalizer = _IdentityNormalizer
    en_norm_mod.Normalizer = _IdentityNormalizer
    sys.modules["tn.chinese.normalizer"] = zh_norm_mod
    sys.modules["tn.english.normalizer"] = en_norm_mod
    setattr(tn_mod, "chinese", chinese_mod)
    setattr(tn_mod, "english", english_mod)
    setattr(chinese_mod, "normalizer", zh_norm_mod)
    setattr(english_mod, "normalizer", en_norm_mod)


DEFAULT_MODEL_ID = "rednote-hilab/dots.tts-soar"
ENGINE_ROOT = Path(__file__).resolve().parent
MODEL_CACHE_ROOT = ENGINE_ROOT / "models"
HF_CACHE_ROOT = ENGINE_ROOT / "hf-cache"


def _check_required_modules() -> list[str]:
    required = [
        "numpy",
        "soundfile",
        "torch",
        "huggingface_hub",
        "dots_tts.runtime",
    ]
    return [name for name in required if importlib.util.find_spec(name) is None]


def _safe_model_dir_name(model_id: str) -> str:
    return model_id.replace("\\", "_").replace("/", "--").replace(":", "_")


def _model_cache_ready(local_dir: Path) -> bool:
    return (
        (local_dir / "config.json").is_file()
        and (local_dir / "model.safetensors").is_file()
        and any(local_dir.glob("*tokenizer*"))
    )


def _resolve_model_path(model_id: str) -> str:
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    local_dir = MODEL_CACHE_ROOT / _safe_model_dir_name(model_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    if _model_cache_ready(local_dir):
        print(
            f"[dots-tts worker] Using local model cache for {model_id}: {local_dir}",
            file=sys.stderr,
        )
        return str(local_dir.resolve())
    print(
        f"[dots-tts worker] Ensuring local model cache for {model_id}: {local_dir}",
        file=sys.stderr,
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        cache_dir=str(HF_CACHE_ROOT),
    )
    return str(local_dir.resolve())


def _load_runtime(job: dict):
    _install_wetext_fallback_modules()
    from dots_tts.runtime import DotsTtsRuntime  # type: ignore

    model_id = _resolve_model_path(job.get("model_id") or DEFAULT_MODEL_ID)
    precision = _resolve_precision(job.get("precision") or "auto")
    optimize = bool(job.get("optimize", False))
    print(
        f"[dots-tts worker] Loading {model_id} precision={precision} optimize={optimize}",
        file=sys.stderr,
    )
    return DotsTtsRuntime.from_pretrained(
        model_id,
        precision=precision,
        optimize=optimize,
    )


def _resolve_precision(requested: str) -> str:
    import torch

    requested = (requested or "auto").strip().lower()
    cuda_available = torch.cuda.is_available()
    if requested in {"", "auto"}:
        return "bfloat16" if cuda_available else "float32"
    if requested in {"bf16", "bfloat16", "fp16", "float16"} and not cuda_available:
        print(
            f"[dots-tts worker] {requested} requires CUDA for stable Dot.TTS inference; using float32 on CPU.",
            file=sys.stderr,
        )
        return "float32"
    return requested


def _write_audio(output_path: str, audio, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    arr = audio
    if hasattr(arr, "detach"):
        arr = arr.detach().float().cpu().squeeze().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.squeeze()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, arr, int(sample_rate))


def _generate_chunks(job: dict) -> list[str]:
    runtime = _load_runtime(job)
    generate_params = set(inspect.signature(runtime.generate).parameters)
    num_steps = int(job.get("num_steps") or 10)
    guidance_scale = float(job.get("guidance_scale") or 1.2)
    speaker_scale = float(job.get("speaker_scale") or 1.5)
    seed = job.get("seed", 42)
    normalize_text = bool(job.get("normalize_text", False))
    language = (job.get("language") or "none").strip() or "none"
    allow_xvector_only = bool(job.get("allow_xvector_only", False))

    files: list[str] = []
    for chunk_index, chunk in enumerate(job.get("chunks") or [], start=1):
        output_path = chunk["output_path"]
        prompt_audio = chunk.get("prompt_audio_path") or None
        prompt_text = chunk.get("prompt_text") or None
        if prompt_audio and not prompt_text and not allow_xvector_only:
            raise ValueError(
                f"Dot.TTS requires prompt_text with prompt_audio for {Path(prompt_audio).name}."
            )

        kwargs = {
            "text": chunk["text"],
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
        }
        if "speaker_scale" in generate_params:
            kwargs["speaker_scale"] = speaker_scale
        if prompt_audio:
            kwargs["prompt_audio_path"] = prompt_audio
        if prompt_text:
            kwargs["prompt_text"] = prompt_text
        if seed not in (None, "") and "seed" in generate_params:
            kwargs["seed"] = int(seed)
        if normalize_text and "normalize_text" in generate_params:
            kwargs["normalize_text"] = True
        if language and language.lower() != "none" and "language" in generate_params:
            kwargs["language"] = language

        text_len = len(chunk.get("text") or "")
        print(
            f"[CHUNK_START] {chunk_index}/{len(job.get('chunks') or [])} chars={text_len} output={output_path}",
            file=sys.stderr,
            flush=True,
        )
        result = runtime.generate(**kwargs)
        sample_rate = int(result.get("sample_rate") or getattr(runtime, "sample_rate", 48000))
        _write_audio(output_path, result["audio"], sample_rate)
        files.append(output_path)
        print(f"[CHUNK_DONE] {output_path}", file=sys.stderr, flush=True)

    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file")
    parser.add_argument("--prefetch-model", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    if args.check_env:
        missing = _check_required_modules()
        if missing:
            print(
                json.dumps(
                    {
                        "success": False,
                        "runtime": "dots_tts",
                        "missing": missing,
                        "error": f"Missing required modules: {', '.join(missing)}",
                    }
                )
            )
            sys.exit(1)
        print(json.dumps({"success": True, "runtime": "dots_tts"}))
        return

    if args.prefetch_model:
        _load_runtime(
            {
                "model_id": args.model_id,
                "precision": args.precision,
                "optimize": args.optimize,
            }
        )
        print("[dots-tts worker] Model ready", file=sys.stderr)
        return

    if not args.job_file:
        parser.error("--job-file is required unless --prefetch-model is used")

    with open(args.job_file, "r", encoding="utf-8-sig") as handle:
        job = json.load(handle)

    try:
        files = _generate_chunks(job)
        print(json.dumps({"success": True, "files": files}))
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"}))
        raise


if __name__ == "__main__":
    main()
