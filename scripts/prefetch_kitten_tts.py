"""Prefetch KittenTTS model files into the runtime cache."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_ID = "KittenML/kitten-tts-mini-0.8"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "models" / "kitten_tts"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch KittenTTS model files.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=os.environ.get("KITTEN_TTS_CACHE_DIR") or str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ["KITTEN_TTS_CACHE_DIR"] = args.cache_dir

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prefetching KittenTTS model: {args.model_id}")
    print(f"KittenTTS cache: {cache_dir}")

    from kittentts import KittenTTS

    KittenTTS(args.model_id, cache_dir=str(cache_dir))
    print("KittenTTS model cache ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
