"""OmniVoice isolated worker — called via subprocess from the main app.

Accepts a JSON job file via --job-file.  Writes output WAV files and prints
progress markers to stderr so the parent process can track completion.

Job file schema
---------------
{
  "mode": "clone" | "design",

  // clone mode:
  "chunks": [
    {
      "text": "...",
      "ref_audio": "/path/to/prompt.wav",
      "ref_text": "optional transcript",
      "output_path": "/path/to/chunk_0000.wav",
      "_order_index": 0
    }
  ],

  // design mode (single preview):
  "text": "...",
  "instruct": "female, low pitch, british accent",
  "output_path": "/path/to/preview.wav",

  // shared:
  "model_id": "k2-fsa/OmniVoice",
  "device": "auto",   // auto | cuda | cpu
  "dtype": "float16", // float16 | bfloat16 | float32
  "num_step": 32,
  "batch_size": 1, // 1 is safest; 2-4 can improve throughput with enough VRAM
  "speed": 1.0,
  "duration_safety_margin": 0.25, // extra generation time; 0 disables
  "post_process": true   // set to false to disable silence-trimming post-processing
}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice  # type: ignore
from omnivoice.models import omnivoice as omnivoice_model_module  # type: ignore

DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
VOICE_PROMPT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "voice_prompts"


def _voice_prompt_cache_key(
    ref_audio: str,
    ref_text: str | None,
    model_id: str,
) -> str:
    """Hash everything that can change OmniVoice's encoded clone prompt."""
    digest = hashlib.sha256()
    digest.update(str(model_id).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(ref_text or "").strip().encode("utf-8"))
    digest.update(b"\0")
    with open(ref_audio, "rb") as audio_file:
        for block in iter(lambda: audio_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _create_or_load_voice_prompt(
    model: OmniVoice,
    ref_audio: str,
    ref_text: str | None,
    model_id: str,
):
    """Reuse encoded reference audio within and across generation jobs."""
    cache_key = _voice_prompt_cache_key(ref_audio, ref_text, model_id)
    cache_path = VOICE_PROMPT_CACHE_DIR / f"{cache_key}.pt"
    started = time.perf_counter()

    if cache_path.is_file():
        try:
            prompt = omnivoice_model_module.VoiceClonePrompt.load(str(cache_path))
            print(
                "[omnivoice_worker] Loaded cached voice prompt "
                f"{cache_key[:12]} in {time.perf_counter() - started:.2f}s",
                file=sys.stderr,
            )
            return prompt, cache_key
        except Exception as exc:
            print(
                "[omnivoice_worker] Cached voice prompt could not be loaded; "
                f"rebuilding it ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            cache_path.unlink(missing_ok=True)

    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    VOICE_PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    try:
        prompt.save(str(temporary_path))
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(
        "[omnivoice_worker] Encoded and cached voice prompt "
        f"{cache_key[:12]} in {time.perf_counter() - started:.2f}s",
        file=sys.stderr,
    )
    return prompt, cache_key


def _fade_in_and_pad_audio(
    audio,
    pad_duration: float = 0.1,
    fade_duration: float = 0.1,
    sample_rate: int = 24000,
):
    """Keep edge padding without fading final speech, for tensors or arrays."""
    if audio.shape[-1] == 0:
        return audio

    is_tensor = torch.is_tensor(audio)
    processed = audio.clone() if is_tensor else np.array(audio, copy=True)
    fade_samples = int(fade_duration * sample_rate)
    if fade_samples > 0:
        fade_length = min(fade_samples, processed.shape[-1] // 2)
        if fade_length > 0:
            fade_shape = [1] * (processed.ndim - 1) + [fade_length]
            if is_tensor:
                fade_in = torch.linspace(
                    0,
                    1,
                    fade_length,
                    device=processed.device,
                    dtype=processed.dtype,
                ).reshape(fade_shape)
            else:
                fade_in = np.linspace(
                    0,
                    1,
                    fade_length,
                    dtype=processed.dtype,
                ).reshape(fade_shape)
            processed[..., :fade_length] *= fade_in

    pad_samples = int(pad_duration * sample_rate)
    if pad_samples > 0:
        pad_shape = list(processed.shape)
        pad_shape[-1] = pad_samples
        if is_tensor:
            silence = torch.zeros(
                tuple(pad_shape),
                dtype=processed.dtype,
                device=processed.device,
            )
            processed = torch.cat([silence, processed, silence], dim=-1)
        else:
            silence = np.zeros(tuple(pad_shape), dtype=processed.dtype)
            processed = np.concatenate([silence, processed, silence], axis=-1)

    return processed


# OmniVoice's bundled helper applies a 100 ms fade-out even when the generated
# waveform ends in active speech. Override that helper in this isolated worker
# so the final phoneme remains untouched while the normal edge padding remains.
omnivoice_model_module.fade_and_pad_audio = _fade_in_and_pad_audio


def _apply_duration_safety_margin(model: OmniVoice, seconds: float) -> int:
    """Add fixed output-token headroom without changing the synthesis text."""
    try:
        margin_seconds = max(0.0, min(float(seconds), 2.0))
    except (TypeError, ValueError):
        margin_seconds = 0.25

    frame_rate = float(model.audio_tokenizer.config.frame_rate)
    margin_tokens = max(0, int(round(margin_seconds * frame_rate)))
    if margin_tokens <= 0:
        return 0

    original_estimate = model.duration_estimator.estimate_duration

    def estimate_with_margin(*args, **kwargs):
        return original_estimate(*args, **kwargs) + margin_tokens

    model.duration_estimator.estimate_duration = estimate_with_margin
    return margin_tokens


def _resolve_device(device: str) -> str:
    d = (device or "auto").strip().lower()
    if d == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return d


def _resolve_dtype(dtype: str) -> torch.dtype:
    d = (dtype or "float16").strip().lower()
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if d in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _ensure_model(model_id: str) -> str:
    """Download model if not cached locally; return local path string."""
    local_dir = Path(__file__).resolve().parent.parent.parent / "models" / "omnivoice"
    local_dir.mkdir(parents=True, exist_ok=True)
    model_path = local_dir / model_id.replace("/", "_")
    if not model_path.exists() or not any(model_path.iterdir()):
        print(f"[omnivoice_worker] Downloading model {model_id} ...", file=sys.stderr)
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )
    return str(model_path)


def _friendly_download_error(model_id: str, exc: Exception) -> str:
    return (
        f"OmniVoice model '{model_id}' is not cached locally and could not be "
        "downloaded from Hugging Face. Check internet/DNS access from Pinokio, "
        "or download the model during setup while online. "
        f"Original error: {type(exc).__name__}: {exc}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file")
    parser.add_argument("--prefetch-model", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    if args.prefetch_model:
        try:
            model_path = _ensure_model(args.model_id)
        except Exception as exc:
            print(_friendly_download_error(args.model_id, exc), file=sys.stderr)
            sys.exit(2)
        print(f"[omnivoice_worker] Model cached at {model_path}", file=sys.stderr)
        return

    if not args.job_file:
        parser.error("--job-file is required unless --prefetch-model is used")

    with open(args.job_file, "r", encoding="utf-8") as f:
        job = json.load(f)

    model_id = job.get("model_id") or DEFAULT_MODEL_ID
    device = _resolve_device(job.get("device") or "auto")
    dtype = _resolve_dtype(job.get("dtype") or "float16")
    num_step = int(job.get("num_step") or 32)
    batch_size = max(1, min(int(job.get("batch_size") or 1), 8))
    speed = float(job.get("speed") or 1.0)
    try:
        duration_safety_margin = max(
            0.0,
            min(float(job.get("duration_safety_margin", 0.25)), 2.0),
        )
    except (TypeError, ValueError):
        duration_safety_margin = 0.25
    post_process = job.get("post_process", True)
    mode = job.get("mode") or "clone"
    print(
        "[omnivoice_worker] Configuration: "
        f"mode={mode} steps={num_step} batch_size={batch_size} "
        f"speed={speed:.2f} dtype={str(dtype).replace('torch.', '')} device={device}",
        file=sys.stderr,
    )

    try:
        model_path = _ensure_model(model_id)
    except Exception as exc:
        print(_friendly_download_error(model_id, exc), file=sys.stderr)
        sys.exit(2)
    worker_started = time.perf_counter()
    model_load_started = time.perf_counter()
    print(f"[omnivoice_worker] Loading model from {model_path} (device={device})", file=sys.stderr)

    model = OmniVoice.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
    )
    print(
        f"[omnivoice_worker] Model loaded in {time.perf_counter() - model_load_started:.2f}s",
        file=sys.stderr,
    )
    margin_tokens = _apply_duration_safety_margin(model, duration_safety_margin)
    if margin_tokens:
        print(
            "[omnivoice_worker] Added "
            f"{float(duration_safety_margin):.2f}s ending-duration buffer "
            f"({margin_tokens} audio tokens)",
            file=sys.stderr,
        )

    sample_rate = 24000

    if mode == "clone":
        chunks = job.get("chunks") or []
        prompt_cache = {}
        generated_audio_seconds = 0.0
        generation_seconds = 0.0
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start + batch_size]
            batch_texts = []
            batch_prompts = []
            batch_prompt_keys = []
            for chunk in batch_chunks:
                ref_audio = chunk["ref_audio"]
                ref_text = chunk.get("ref_text") or None
                in_job_key = (
                    str(Path(ref_audio).resolve()),
                    str(ref_text or "").strip(),
                )
                prompt_entry = prompt_cache.get(in_job_key)
                if prompt_entry is None:
                    prompt_entry = _create_or_load_voice_prompt(
                        model,
                        ref_audio,
                        ref_text,
                        model_id,
                    )
                    prompt_cache[in_job_key] = prompt_entry
                voice_clone_prompt, prompt_key = prompt_entry
                batch_texts.append(chunk["text"])
                batch_prompts.append(voice_clone_prompt)
                batch_prompt_keys.append(prompt_key)

            use_batch = len(batch_chunks) > 1
            kwargs = dict(
                text=batch_texts if use_batch else batch_texts[0],
                voice_clone_prompt=batch_prompts if use_batch else batch_prompts[0],
                num_step=num_step,
                speed=speed,
                postprocess_output=bool(post_process),
            )

            generation_started = time.perf_counter()
            audio_list = model.generate(**kwargs)
            batch_generation_seconds = time.perf_counter() - generation_started
            generation_seconds += batch_generation_seconds
            if len(audio_list) != len(batch_chunks):
                raise RuntimeError(
                    "OmniVoice returned "
                    f"{len(audio_list)} audio result(s) for {len(batch_chunks)} input chunk(s)."
                )
            batch_audio_seconds = 0.0
            prepared_audio = []
            for chunk, generated_audio in zip(batch_chunks, audio_list):
                audio = np.asarray(generated_audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.squeeze()
                audio_seconds = float(audio.shape[-1]) / sample_rate
                batch_audio_seconds += audio_seconds
                prepared_audio.append((chunk, audio, audio_seconds))
            generated_audio_seconds += batch_audio_seconds
            batch_rtf = (
                batch_generation_seconds / batch_audio_seconds
                if batch_audio_seconds else 0.0
            )
            print(
                "[omnivoice_worker] Generated batch "
                f"{batch_start // batch_size + 1}/"
                f"{(len(chunks) + batch_size - 1) // batch_size} "
                f"size={len(batch_chunks)} audio={batch_audio_seconds:.2f}s "
                f"inference={batch_generation_seconds:.2f}s rtf={batch_rtf:.3f}",
                file=sys.stderr,
            )
            for offset, (chunk, audio, audio_seconds) in enumerate(prepared_audio):
                output_path = chunk["output_path"]
                chunk_number = batch_start + offset + 1
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(output_path, audio, sample_rate)
                print(
                    "[omnivoice_worker] Completed chunk "
                    f"{chunk_number}/{len(chunks)} "
                    f"prompt={batch_prompt_keys[offset][:12]} audio={audio_seconds:.2f}s",
                    file=sys.stderr,
                )
                print(f"[CHUNK_DONE] {output_path}", file=sys.stderr)

        overall_rtf = generation_seconds / generated_audio_seconds if generated_audio_seconds else 0.0
        print(
            "[omnivoice_worker] Clone job complete: "
            f"chunks={len(chunks)} unique_prompts={len(prompt_cache)} "
            f"batch_size={batch_size} "
            f"audio={generated_audio_seconds:.2f}s inference={generation_seconds:.2f}s "
            f"rtf={overall_rtf:.3f} total={time.perf_counter() - worker_started:.2f}s",
            file=sys.stderr,
        )

    elif mode == "design":
        text = job["text"]
        instruct = job["instruct"]
        output_path = job["output_path"]

        design_kwargs = dict(
            text=text,
            instruct=instruct,
            num_step=num_step,
            speed=speed,
            postprocess_output=bool(post_process),
        )
        generation_started = time.perf_counter()
        audio_list = model.generate(**design_kwargs)
        generation_seconds = time.perf_counter() - generation_started
        audio = np.asarray(audio_list[0], dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sample_rate)
        audio_seconds = float(audio.shape[-1]) / sample_rate
        rtf = generation_seconds / audio_seconds if audio_seconds else 0.0
        print(
            "[omnivoice_worker] Voice design complete: "
            f"audio={audio_seconds:.2f}s inference={generation_seconds:.2f}s "
            f"rtf={rtf:.3f} total={time.perf_counter() - worker_started:.2f}s",
            file=sys.stderr,
        )
        # Print the output path as the result
        print(output_path)
        print(f"[DESIGN_DONE] {output_path}", file=sys.stderr)

    else:
        print(f"[omnivoice_worker] Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
