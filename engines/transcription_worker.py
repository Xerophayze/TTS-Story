"""Short-lived CPU-only SenseVoice worker. No server or persistent model process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.voice_transcription import (MODEL_REPO, MODEL_REVISION, TRANSCRIPT_EVENT,
                                     VoiceTranscriptGenerator, transcription_runtime_root)


def configure_environment():
    runtime = transcription_runtime_root()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HOME"] = str(runtime / "cache" / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(runtime / "models")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["MODELSCOPE_CACHE"] = str(runtime / "cache" / "modelscope")
    os.environ["OMP_NUM_THREADS"] = "4"
    ffmpeg_dir = ROOT / "tools" / "ffmpeg"
    if ffmpeg_dir.is_dir():
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")


def model_directory(*, download=False):
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(
        MODEL_REPO, revision=MODEL_REVISION, local_files_only=not download,
        allow_patterns=["config.yaml", "configuration.json", "model.pt", "am.mvn", "*.bpe.model"],
    ))
    for name in ("config.yaml", "configuration.json", "model.pt", "am.mvn", "chn_jpn_yue_eng_ko_spectok.bpe.model"):
        if not (directory / name).is_file():
            raise RuntimeError("Transcription model files are incomplete. Repair Voice transcription in Settings → Engines → Utilities.")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--audio")
    args = parser.parse_args()
    configure_environment()
    try:
        if args.download_model:
            print(f"Downloading {MODEL_REPO} at {MODEL_REVISION} for local CPU transcription.", flush=True)
            model_directory(download=True)
            return 0
        # Import inside this process only. The pinned FunASR distribution
        # includes SenseVoice; downloaded model Python code is never enabled.
        from funasr import AutoModel
        from funasr.register import tables
        if "SenseVoiceSmall" not in tables.model_classes:
            raise RuntimeError("This FunASR runtime does not include SenseVoiceSmall; repair the transcription tool.")
        if args.check_env:
            print("CPU transcription runtime ready; SenseVoiceSmall is registered.", flush=True)
            return 0
        if not args.audio:
            parser.error("--audio is required")
        try:
            directory = model_directory()
        except Exception as exc:
            raise RuntimeError("Transcription model files are missing. Repair Voice transcription in Settings → Engines → Utilities to download them.") from exc
        def factory(**kwargs):
            return AutoModel(**{**kwargs, "model": str(directory), "ncpu": 4})
        transcript = VoiceTranscriptGenerator(device="cpu", model_factory=factory).transcribe(args.audio)
        print(TRANSCRIPT_EVENT + json.dumps({"success": True, "transcript": transcript}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(TRANSCRIPT_EVENT + json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
