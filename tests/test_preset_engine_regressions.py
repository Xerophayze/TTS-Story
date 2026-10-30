"""Adapter checks without model downloads, GPU allocation, or a web server."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.engines import kokoro_engine, pocket_tts_engine, isolated_proxy


class PauseRequested(Exception):
    pass


def stop_after_chunk():
    raise PauseRequested()


@pytest.mark.parametrize("requested,cuda,expected", [
    ("cpu", True, "cpu"), ("auto", True, "cuda"), ("auto", False, "cpu"),
])
def test_kokoro_honors_device_and_reuses_pipeline(monkeypatch, requested, cuda, expected):
    calls = []
    monkeypatch.setattr(kokoro_engine, "KOKORO_AVAILABLE", True)
    monkeypatch.setattr(kokoro_engine, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda)))
    monkeypatch.setattr(kokoro_engine, "KPipeline", lambda **kw: calls.append(kw) or object(), raising=False)
    engine = kokoro_engine.KokoroEngine(requested)
    assert engine._get_pipeline("a") is engine._get_pipeline("a")
    assert calls == [{"lang_code": "a", "device": expected}]
    assert not engine.capabilities.supports_voice_cloning
    monkeypatch.setattr(isolated_proxy, "isolated_engine_available", lambda _: True)
    assert not isolated_proxy.IsolatedEngineProxy("kokoro").capabilities.supports_voice_cloning


def test_kokoro_records_finished_clip_before_pause(monkeypatch, tmp_path):
    class Pipeline:
        def __call__(self, text, **kwargs):
            return iter([(text, "phonemes", np.zeros(240, dtype=np.float32))])

    monkeypatch.setattr(kokoro_engine, "KOKORO_AVAILABLE", True)
    engine = kokoro_engine.KokoroEngine("cpu")
    monkeypatch.setattr(engine, "_get_pipeline", lambda _: Pipeline())
    recorded = []
    with pytest.raises(PauseRequested):
        engine.generate_batch([{"speaker": "actor", "chunks": ["First", "Second"]}],
                              {"actor": {"voice": "af_heart"}}, tmp_path,
                              sample_rate=None,
                              progress_cb=stop_after_chunk,
                              chunk_cb=lambda *args: recorded.append(args))
    assert len(recorded) == 1
    assert recorded[0][0] == recorded[0][1]["order_index"] == 0
    assert (tmp_path / "chunk_0000.wav").is_file()
    assert kokoro_engine.sf.info(tmp_path / "chunk_0000.wav").samplerate == 24000
    assert not (tmp_path / "chunk_0001.wav").exists()


def pocket_stub(monkeypatch):
    class Tensor:
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.zeros(240, dtype=np.float32)

    calls = []
    engine = object.__new__(pocket_tts_engine.PocketTTSEngine)
    engine.model = SimpleNamespace(sample_rate=24000,
        generate_audio=lambda state, text, **kwargs: calls.append((state, text, kwargs)) or Tensor(),
        get_state_for_audio_prompt=lambda path, **kwargs: {"prompt": path})
    engine._voice_cache = {}
    engine._prompt_wav_cache = {}
    engine.default_prompt = None
    engine.prompt_truncate = False
    engine._explicit_num_threads = False
    engine.post_processor = SimpleNamespace(apply_post_pipeline=lambda audio, *args: audio)
    monkeypatch.setattr(engine, "_ensure_wav_prompt", lambda path: path)
    return engine, calls


def test_pocket_grouping_keeps_narrative_order_and_cached_voice_state(monkeypatch, tmp_path):
    engine, calls = pocket_stub(monkeypatch)
    recorded = []
    segments = [{"speaker": speaker, "chunks": [text]} for speaker, text in
                [("a", "First"), ("b", "Second"), ("a", "Third")]]
    paths = engine.generate_batch(segments, {"a": {"voice": "alba"}, "b": {"voice": "marius"}},
                                  tmp_path, group_by_speaker=True, chunk_cb=lambda *args: recorded.append(args))
    assert [entry[1] for entry in calls] == ["First", "Third", "Second"]
    assert calls[0][0] is calls[1][0]
    assert all(entry[2]["copy_state"] is True for entry in calls)
    assert [entry[0] for entry in recorded] == [0, 2, 1]
    assert [entry[1]["segment_index"] for entry in recorded] == [0, 2, 1]
    assert [entry[1]["order_index"] for entry in recorded] == [0, 2, 1]
    assert [Path(path).name for path in paths] == [
        "chunk_0000.wav", "chunk_0001.wav", "chunk_0002.wav"]


def test_pocket_records_finished_clip_before_pause(monkeypatch, tmp_path):
    engine, calls = pocket_stub(monkeypatch)
    recorded = []
    with pytest.raises(PauseRequested):
        engine.generate_batch([{"speaker": "a", "chunks": ["First", "Second"]}],
                              {"a": {"voice": "alba"}}, tmp_path,
                              progress_cb=stop_after_chunk, chunk_cb=lambda *args: recorded.append(args))
    assert len(recorded) == len(calls) == 1
    assert recorded[0][0] == recorded[0][1]["order_index"] == 0
    assert (tmp_path / "chunk_0000.wav").is_file()
    assert not (tmp_path / "chunk_0001.wav").exists()


def test_isolated_proxy_stops_owned_worker_after_pause_and_reports_actual_device(monkeypatch):
    import json
    monkeypatch.setattr(isolated_proxy, "isolated_engine_available", lambda _: True)
    stopped, job_paths = [], []
    class Process:
        def __init__(self, command, **kwargs):
            job_paths.append(Path(command[-1]))
            assert job_paths[-1].is_file()
            self.stdout = iter([
                'TTS_STORY_EVENT ' + json.dumps({"event": "engine_ready", "device": "cuda"}),
                'TTS_STORY_EVENT ' + json.dumps({"event": "progress"}),
            ])
    monkeypatch.setattr(isolated_proxy.subprocess, "Popen", Process)
    monkeypatch.setattr(isolated_proxy, "stop_owned_worker", lambda process, **kwargs: stopped.append((process, kwargs)))
    engine = isolated_proxy.IsolatedEngineProxy("kokoro", device="auto")
    with pytest.raises(PauseRequested):
        engine._execute({"operation": "batch"}, progress_cb=stop_after_chunk)
    assert engine.device == "cuda"
    assert len(stopped) == 1
    assert stopped[0][1] == {"graceful": False}
    assert not job_paths[0].exists()


def test_local_smoke_tool_uses_saved_pocket_auth_without_passing_it_to_other_engines(tmp_path, monkeypatch):
    import json
    from scripts import smoke_test_local_engine as smoke
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "environment-test-token")
    assert smoke.worker_environment("pocket_tts") == {"HF_TOKEN": "environment-test-token"}
    (tmp_path / "config.json").write_text(json.dumps({"huggingface_token": "saved-test-token"}), encoding="utf-8")
    assert smoke.worker_environment("pocket_tts") == {"HF_TOKEN": "saved-test-token"}
    assert smoke.worker_environment("pocket_tts_preset") == {"HF_TOKEN": "saved-test-token"}
    assert smoke.worker_environment("kokoro") == {}
