"""Run a short, explicit local-engine smoke test without starting a web backend.

Downloads models on first use. Outputs are kept under the engine's ignored
cache/smoke-test directory, never in an existing audiobook production.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from src.engines.isolated_proxy import IsolatedEngineProxy, engine_root


def worker_environment(engine_name):
    """Use the same saved Pocket credential as the app, without logging it."""
    if not engine_name.startswith("pocket_tts"):
        return {}
    config_path = ROOT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    token = str(config.get("huggingface_token") or os.environ.get("HF_TOKEN") or "").strip()
    return {"HF_TOKEN": token}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["kokoro", "pocket_tts_preset", "pocket_tts"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--voice")
    parser.add_argument("--pause-after-first", action="store_true")
    parser.add_argument("--transcribe", action="store_true")
    args = parser.parse_args()
    voice = args.voice or ("am_echo" if args.engine == "kokoro" else "alba")
    settings = {"device": args.device}
    if args.engine.startswith("pocket_tts"):
        settings.update(device="cpu", num_threads=4)
    engine = IsolatedEngineProxy(args.engine, environment=worker_environment(args.engine), **settings)
    target = engine_root(args.engine) / "cache" / "smoke-test" / ("pause" if args.pause_after_first else "batch")
    target.mkdir(parents=True, exist_ok=True)
    events, progress = [], []
    class SmokePause(Exception):
        pass
    def chunk(index, metadata, path):
        events.append({"index": index, "metadata": metadata, "path": path})
    def tick():
        progress.append(1)
        if args.pause_after_first:
            raise SmokePause()
    started = monotonic()
    paused = False
    try:
        files = engine.generate_batch([
            {"speaker": "first", "chunks": ["This is a short local engine test."]},
            {"speaker": "second", "chunks": ["Every speaker should have an assigned voice."]},
        ], {"first": {"voice": voice, "lang_code": "a"}, "second": {"voice": voice, "lang_code": "a"}},
            target, progress_cb=tick, chunk_cb=chunk)
    except SmokePause:
        paused = True
        files = [event["path"] for event in events]
    elapsed = monotonic() - started
    expected = 1 if paused else 2
    assert len(events) == len(files) == expected, (events, files)
    assert len(progress) == expected
    for index, event in enumerate(events):
        assert event["index"] == event["metadata"]["order_index"] == index
        samples, rate = sf.read(event["path"])
        assert rate == 24000 and len(samples) > 2400
        assert np.isfinite(samples).all() and np.max(np.abs(samples)) > 0.001
    report = {"engine": args.engine, "device": engine.device, "elapsed_seconds": round(elapsed, 2),
              "paused": paused, "chunks": events}
    if args.transcribe:
        from src.voice_transcription import VoiceTranscriptGenerator
        report["transcript"] = VoiceTranscriptGenerator().transcribe(files[0])
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
