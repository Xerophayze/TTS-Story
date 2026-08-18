"""Isolated Chatterbox Turbo worker used by the main TTS-Story process."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from huggingface_hub import snapshot_download


MODEL_ID = "ResembleAI/chatterbox-turbo"
SAMPLE_RATE = 24000


def emit(payload: dict) -> None:
    print("TTS_STORY_EVENT " + json.dumps(payload, ensure_ascii=False), flush=True)


def resolve_device(value: str) -> str:
    candidate = (value or "auto").strip().lower()
    if candidate == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this Chatterbox environment cannot access it.")
    return candidate


def patch_chatterbox_audio_types() -> None:
    """Keep reference tensors float32 across the upstream conditioning path."""
    try:
        from chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer

        def prepare_audio(self, wavs):
            prepared = []
            for wav in wavs:
                if isinstance(wav, np.ndarray):
                    wav = torch.from_numpy(wav.astype(np.float32))
                elif torch.is_tensor(wav):
                    wav = wav.float()
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                prepared.append(wav)
            return prepared

        S3Tokenizer._prepare_audio = prepare_audio
    except Exception:
        pass

    from chatterbox.tts_turbo import Conditionals, S3GEN_SR, S3_SR, T3Cond

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
        reference, sample_rate = librosa.load(wav_fpath, sr=S3GEN_SR)
        if len(reference) / float(sample_rate) <= 5.0:
            raise ValueError("Chatterbox reference audio must be longer than five seconds.")
        if norm_loudness:
            reference = self.norm_loudness(reference, sample_rate)
        reference_16k = librosa.resample(
            reference,
            orig_sr=S3GEN_SR,
            target_sr=S3_SR,
        ).astype(np.float32)
        reference = reference.astype(np.float32)
        reference = reference[: self.DEC_COND_LEN]
        s3gen_ref = self.s3gen.embed_ref(reference, S3GEN_SR, device=self.device)
        prompt_tokens = None
        if prompt_length := self.t3.hp.speech_cond_prompt_len:
            prompt_tokens, _ = self.s3gen.tokenizer.forward(
                [reference_16k[: self.ENC_COND_LEN]],
                max_len=prompt_length,
            )
            prompt_tokens = torch.atleast_2d(prompt_tokens).to(self.device)
        speaker_embedding = torch.from_numpy(
            self.ve.embeds_from_wavs([reference_16k], sample_rate=S3_SR)
        )
        speaker_embedding = speaker_embedding.mean(axis=0, keepdim=True).to(self.device)
        t3_condition = T3Cond(
            speaker_emb=speaker_embedding,
            cond_prompt_speech_tokens=prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_condition, s3gen_ref)

    ChatterboxTurboTTS.prepare_conditionals = prepare_conditionals


def coerce_tokenizer_buffers(model) -> None:
    tokenizer = getattr(getattr(model, "s3gen", None), "tokenizer", None)
    if tokenizer is None:
        return
    try:
        tokenizer.float()
    except Exception:
        pass
    for name in ("_mel_filters", "window"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, torch.Tensor) and value.dtype != torch.float32:
            setattr(tokenizer, name, value.float())


def number(extra: dict, key: str, fallback: float) -> float:
    try:
        return float(extra.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)


def run_job(job: dict) -> None:
    patch_chatterbox_audio_types()
    device = resolve_device(job.get("device") or "auto")
    token = os.environ.get("HF_TOKEN") or False
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        token=token,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
    )
    model = ChatterboxTurboTTS.from_local(model_path, device)
    coerce_tokenizer_buffers(model)
    defaults = job.get("defaults") or {}
    last_prompt = None
    for item in job.get("items") or []:
        prompt = str(item.get("audio_prompt_path") or "").strip()
        if not prompt or not Path(prompt).is_file():
            raise FileNotFoundError(f"Chatterbox reference audio was not found: {prompt or '(empty)'}")
        extra = item.get("extra") or {}
        exaggeration = number(extra, "exaggeration", number(defaults, "exaggeration", 0.0))
        if prompt != last_prompt or model.conds is None:
            model.prepare_conditionals(
                prompt,
                exaggeration=exaggeration,
                norm_loudness=bool(defaults.get("prompt_norm_loudness", True)),
            )
            last_prompt = prompt
        coerce_tokenizer_buffers(model)
        wav = model.generate(
            text=str(item.get("text") or ""),
            temperature=number(extra, "temperature", number(defaults, "temperature", 0.8)),
            top_p=number(extra, "top_p", number(defaults, "top_p", 0.95)),
            top_k=int(number(extra, "top_k", number(defaults, "top_k", 1000))),
            repetition_penalty=number(
                extra,
                "repetition_penalty",
                number(defaults, "repetition_penalty", 1.2),
            ),
            cfg_weight=number(extra, "cfg_weight", number(defaults, "cfg_weight", 0.0)),
            exaggeration=exaggeration,
            audio_prompt_path=None,
            norm_loudness=bool(extra.get("norm_loudness", defaults.get("norm_loudness", True))),
        )
        output_path = Path(item["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, wav.squeeze(0).detach().cpu().numpy().astype("float32"), SAMPLE_RATE)
        emit({"event": "chunk", "index": int(item["order_index"]), "path": str(output_path)})
    emit({"event": "complete", "count": len(job.get("items") or [])})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()
    if args.check_env:
        print(f"Chatterbox isolated environment ready; torch={torch.__version__}")
        return 0
    if not args.job_file:
        parser.error("--job-file is required")
    try:
        run_job(json.loads(Path(args.job_file).read_text(encoding="utf-8")))
        return 0
    except Exception as exc:
        print(f"CHATTERBOX_WORKER_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
