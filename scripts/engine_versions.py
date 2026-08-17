#!/usr/bin/env python3
"""Report the installed TTS-Story runtime versions without importing engines."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata
from typing import Dict, Optional


MAIN_ENV_PACKAGES = {
    "Kokoro": "kokoro",
    "Chatterbox Turbo": "chatterbox-tts",
    "Pocket TTS": "pocket-tts",
    "VoxCPM": "voxcpm",
    "Qwen3 TTS": "qwen-tts",
    "FlashAttention (optional)": "flash-attn",
    "KittenTTS": "kittentts",
    "Replicate client": "replicate",
    "Edge TTS client": "edge-tts",
    "PyTorch": "torch",
    "Transformers": "transformers",
    "NumPy": "numpy",
}

MODEL_DEFAULTS = {
    "Chatterbox Turbo": "ResembleAI/chatterbox-turbo",
    "Pocket TTS": "variant b6369a24",
    "VoxCPM": "openbmb/VoxCPM1.5",
    "Qwen3 Custom Voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen3 Voice Clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "KittenTTS": "KittenML/kitten-tts-mini-0.8",
    "OmniVoice (isolated)": "k2-fsa/OmniVoice",
    "IndexTTS (isolated)": "IndexTTS-2",
    "Dot.TTS (isolated)": "rednote-hilab/dots.tts-soar",
}


def installed_version(distribution_name: str) -> Optional[str]:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def collect_versions() -> Dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            label: installed_version(distribution)
            for label, distribution in MAIN_ENV_PACKAGES.items()
        },
        "default_models": MODEL_DEFAULTS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    report = collect_versions()

    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    print(f"Python: {report['python']}")
    print(f"Platform: {report['platform']}")
    print("Main environment packages:")
    for label, version in report["packages"].items():
        print(f"  {label}: {version or 'not installed'}")
    print("Default model identifiers:")
    for label, model_id in report["default_models"].items():
        print(f"  {label}: {model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
