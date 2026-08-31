"""Main-process proxy for engines installed in dedicated virtual environments."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from .base import EngineCapabilities, TtsEngineBase


logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "engines" / "isolated_engine_worker.py"
ENGINE_DIRS = {
    "kokoro": "kokoro",
    "voxcpm_local": "voxcpm_local",
    "pocket_tts": "pocket_tts",
    "pocket_tts_preset": "pocket_tts",
    "qwen3_custom": "qwen3",
    "qwen3_clone": "qwen3",
    "kitten_tts": "kitten_tts",
    "edge_tts": "edge_tts",
    "audio8_tts": "audio8_tts",
}
SAMPLE_RATES = {
    "kokoro": 24000,
    "voxcpm_local": 44100,
    "pocket_tts": 24000,
    "pocket_tts_preset": 24000,
    "qwen3_custom": 24000,
    "qwen3_clone": 24000,
    "kitten_tts": 24000,
    "edge_tts": 24000,
    "audio8_tts": 44100,
}


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def engine_root(engine: str) -> Path:
    return ROOT / "engines" / ENGINE_DIRS[engine]


def engine_python(engine: str) -> Path:
    return engine_root(engine) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def engine_marker(engine: str) -> Path:
    return engine_root(engine) / ".ready"


def isolated_engine_available(engine: str) -> bool:
    return WORKER.is_file() and engine_python(engine).is_file() and engine_marker(engine).is_file()


class IsolatedEngineProxy(TtsEngineBase):
    capabilities = EngineCapabilities(supports_voice_cloning=True, supported_languages=None)

    def __init__(
        self,
        engine_name: str,
        *,
        environment: Optional[Dict[str, str]] = None,
        **constructor,
    ) -> None:
        if not isolated_engine_available(engine_name):
            raise ImportError(f"{engine_name} isolated environment is not installed")
        self.name = engine_name
        self.constructor = constructor
        self.environment = {
            str(key): str(value)
            for key, value in (environment or {}).items()
            if value not in (None, "")
        }
        self._sample_rate = SAMPLE_RATES.get(engine_name, 24000)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _execute(self, payload: dict, progress_cb=None, chunk_cb=None) -> dict:
        payload.update({"engine": self.name, "constructor": self.constructor})
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False)
        job_file = Path(handle.name)
        output: List[str] = []
        process = None
        complete = {}
        try:
            with handle:
                json.dump(payload, handle, ensure_ascii=False, default=_json_default)
            process_environment = os.environ.copy()
            process_environment.update(self.environment)
            process = subprocess.Popen(
                [str(engine_python(self.name)), str(WORKER), "--engine", self.name, "--job-file", str(job_file)],
                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=process_environment,
            )
            for raw in process.stdout or []:
                line = raw.rstrip()
                output.append(line)
                if not line.startswith("TTS_STORY_EVENT "):
                    if line:
                        logger.info("[%s worker] %s", self.name, line)
                    continue
                event = json.loads(line[len("TTS_STORY_EVENT "):])
                if event.get("event") == "progress" and callable(progress_cb):
                    progress_cb()
                elif event.get("event") == "chunk" and callable(chunk_cb):
                    chunk_cb(event.get("chunk_index"), event.get("metadata") or {}, event.get("path"))
                elif event.get("event") == "complete":
                    complete = event
                elif event.get("event") == "engine_ready":
                    logger.info(
                        "[%s worker] engine ready device=%s dtype=%s",
                        self.name, event.get("device", "unknown"), event.get("dtype", "unknown"),
                    )
            code = process.wait()
            if code != 0:
                raise RuntimeError("Isolated engine failed:\n" + "\n".join(output[-40:]))
            return complete
        except BaseException:
            if process and process.poll() is None:
                process.terminate()
            raise
        finally:
            job_file.unlink(missing_ok=True)

    def generate_batch(self, segments: List[Dict], voice_config: Dict[str, Dict], output_dir: Path,
                       speed: float = 1.0, sample_rate: Optional[int] = None, progress_cb=None,
                       chunk_cb=None, parallel_workers: int = 1, group_by_speaker: bool = False) -> List[str]:
        event = self._execute({
            "operation": "batch", "segments": segments, "voice_config": voice_config,
            "output_dir": str(output_dir), "speed": speed, "sample_rate": sample_rate,
            "parallel_workers": parallel_workers, "group_by_speaker": group_by_speaker,
        }, progress_cb=progress_cb, chunk_cb=chunk_cb)
        return event.get("files") or []

    def generate_audio(self, text: str, **kwargs) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix=f"tts_story_{self.name}_") as directory:
            output = Path(directory) / "preview.wav"
            event = self._execute({
                "operation": "audio", "arguments": {"text": text, **kwargs},
                "output_path": str(output),
            })
            audio, _ = sf.read(event.get("path") or output, dtype="float32")
            return np.asarray(audio, dtype=np.float32)

    def list_voices(self) -> List[dict]:
        return self._execute({"operation": "voices"}).get("voices") or []

    def cleanup(self) -> None:
        pass
