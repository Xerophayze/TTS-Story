"""
IndexTTS worker script — runs inside the IndexTTS isolated venv.

Called by the TTS-Story engine adapter via subprocess. Reads a JSON job
from stdin (or --job-file), synthesises all chunks, writes WAV files, and
prints a JSON result to stdout.

Usage:
    python tts_worker.py --job-file /path/to/job.json

Job JSON schema:
{
    "model_dir":   "checkpoints",          # path to IndexTTS checkpoints dir
    "cfg_path":    "checkpoints/config.yaml",
    "use_fp16":    false,
    "use_deepspeed": false,
    "device":      null,                   # null = auto
    "chunks": [
        {
            "text":             "Hello world.",
            "spk_audio_prompt": "/abs/path/to/voice.wav",
            "output_path":      "/abs/path/to/chunk_0000.wav"
        },
        ...
    ]
}

Result JSON (written to stdout):
{
    "success": true,
    "files": ["/abs/path/to/chunk_0000.wav", ...]
}
or on error:
{
    "success": false,
    "error": "traceback string"
}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback

# Redirect stdout to stderr so that all print() calls from IndexTTS internals
# (infer_v2.py, etc.) go to stderr. The final JSON result is written directly
# to the real stdout via _stdout below.
_stdout = sys.stdout
sys.stdout = sys.stderr

_MODEL_REPO_MAP = {
    "IndexTTS-2":   "IndexTeam/IndexTTS-2",
    "IndexTTS-1.5": "IndexTeam/IndexTTS-1.5",
    "IndexTTS":     "IndexTeam/IndexTTS",
}


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _flash_attn_error(error: str) -> bool:
    lowered = error.lower()
    return "flash_attn" in lowered or "flash-attention" in lowered


def _ensure_model(model_dir: str, model_version: str) -> None:
    """Download model weights from HuggingFace if not already present."""
    gpt_path = os.path.join(model_dir, "gpt.pth")
    if os.path.exists(gpt_path):
        return

    repo_id = _MODEL_REPO_MAP.get(model_version, "IndexTeam/IndexTTS-2")
    print(
        f"[worker] Model weights not found. Downloading {repo_id} to {model_dir} ...",
        file=sys.stderr, flush=True,
    )
    print(
        f"[worker] This is a one-time download (~2-4 GB). Please wait.",
        file=sys.stderr, flush=True,
    )

    try:
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            ignore_patterns=["*.md", "*.txt", "examples/*"],
        )
        print(f"[worker] Model download complete.", file=sys.stderr, flush=True)
    except Exception:
        raise RuntimeError(
            f"Failed to download IndexTTS model '{repo_id}'.\n"
            f"Check your internet connection or download manually:\n"
            f"  huggingface-cli download {repo_id} --local-dir {model_dir}\n\n"
            + traceback.format_exc()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="IndexTTS batch worker")
    parser.add_argument("--job-file", required=True, help="Path to JSON job file")
    args = parser.parse_args()

    with open(args.job_file, "r", encoding="utf-8") as fh:
        job = json.load(fh)

    model_dir = job.get("model_dir", "checkpoints")
    model_version = job.get("model_version", "IndexTTS-2")
    cfg_path = job.get("cfg_path", os.path.join(model_dir, "config.yaml"))
    use_fp16 = bool(job.get("use_fp16", False))
    use_deepspeed = bool(job.get("use_deepspeed", False))
    use_torch_compile = bool(job.get("use_torch_compile", False))
    use_accel = bool(job.get("use_accel", False))
    num_beams = int(job.get("num_beams", 3))
    diffusion_steps = int(job.get("diffusion_steps", 25))
    temperature = float(job.get("temperature", 0.8))
    top_p = float(job.get("top_p", 0.8))
    top_k = int(job.get("top_k", 30))
    repetition_penalty = float(job.get("repetition_penalty", 10.0))
    max_mel_tokens = int(job.get("max_mel_tokens", 1500))
    max_text_tokens_per_segment = int(job.get("max_text_tokens_per_segment", 120))
    device = job.get("device") or None
    chunks = job.get("chunks", [])

    if not _flash_attn_available() and (use_fp16 or use_deepspeed or use_accel):
        print(
            "[worker] flash_attn is not installed; disabling IndexTTS fp16, "
            "DeepSpeed, and acceleration for compatibility.",
            file=sys.stderr,
            flush=True,
        )
        use_fp16 = False
        use_deepspeed = False
        use_accel = False

    try:
        _ensure_model(model_dir, model_version)
    except Exception:
        _fail(traceback.format_exc())
        return

    try:
        from indextts.infer_v2 import IndexTTS2  # type: ignore

        try:
            tts = IndexTTS2(
                cfg_path=cfg_path,
                model_dir=model_dir,
                use_fp16=use_fp16,
                use_deepspeed=use_deepspeed,
                use_torch_compile=use_torch_compile,
                use_accel=use_accel,
                device=device,
                use_cuda_kernel=None,
            )
        except Exception:
            error = traceback.format_exc()
            if _flash_attn_error(error) and (use_fp16 or use_deepspeed or use_accel):
                print(
                    "[worker] IndexTTS requested flash_attn. Retrying with "
                    "fp16, DeepSpeed, and acceleration disabled.",
                    file=sys.stderr,
                    flush=True,
                )
                tts = IndexTTS2(
                    cfg_path=cfg_path,
                    model_dir=model_dir,
                    use_fp16=False,
                    use_deepspeed=False,
                    use_torch_compile=use_torch_compile,
                    use_accel=False,
                    device=device,
                    use_cuda_kernel=False,
                )
            else:
                raise
    except Exception:
        _fail(traceback.format_exc())
        return

    files: list[str] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        spk_audio_prompt = chunk.get("spk_audio_prompt", "")
        output_path = chunk.get("output_path", "")

        if not text.strip():
            print(f"[worker] Skipping empty chunk -> {output_path}", file=sys.stderr)
            continue

        try:
            tts.infer(
                spk_audio_prompt=spk_audio_prompt,
                text=text,
                output_path=output_path,
                verbose=False,
                num_beams=num_beams,
                diffusion_steps=diffusion_steps,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_mel_tokens=max_mel_tokens,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
            )
            files.append(output_path)
            print(f"[CHUNK_DONE] {output_path}", file=sys.stderr, flush=True)
        except Exception:
            _fail(traceback.format_exc())
            return

    result = {"success": True, "files": files}
    print(json.dumps(result), file=_stdout, flush=True)


def _fail(error: str) -> None:
    result = {"success": False, "error": error}
    print(json.dumps(result), file=_stdout, flush=True)
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _fail(traceback.format_exc())
