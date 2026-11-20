from pathlib import Path
from types import SimpleNamespace

import pytest

import app as app_module


@pytest.mark.parametrize("with_pause", [False, True])
def test_callback_order_maps_to_original_speech_and_preserves_pauses(monkeypatch, tmp_path, with_pause):
    job_id = "grouped-order"
    segments = [
        {"speaker": "alice", "chunks": ["First."]},
        {"speaker": "bob", "chunks": ["Second."]},
        {"speaker": "alice", "chunks": ["Third."]},
    ]
    if with_pause:
        segments[0]["chunks"].append("***")

    class Engine:
        sample_rate = 24000
        device = "test"

        def generate_batch(self, segments, voice_config, output_dir, speed=1,
                           progress_cb=None, chunk_cb=None, group_by_speaker=False):
            assert group_by_speaker is not with_pause
            chunks = [(si, ci, text) for si, segment in enumerate(segments)
                      for ci, text in enumerate(segment["chunks"])]
            indices = [0, 1, 2] if with_pause else [0, 2, 1]
            files = []
            for grouped_index, original_index in enumerate(indices):
                si, ci, text = chunks[original_index]
                path = Path(output_dir) / f"chunk_{original_index:04d}.wav"
                path.write_bytes(text.encode())
                progress_cb()
                # Grouped engines may report segment indices in grouped order.
                chunk_cb(ci, {"segment_index": grouped_index if group_by_speaker else si,
                              "chunk_index": ci, "order_index": original_index}, str(path))
                files.append(str(path))
            return files

    monkeypatch.setattr(app_module, "OUTPUT_DIR", tmp_path)
    for name in ("jobs", "pause_flags", "cancel_flags", "cancel_events"):
        monkeypatch.setattr(app_module, name, {})
    monkeypatch.setattr(app_module, "_persist_job_state", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_create_text_processor_for_engine", lambda *a: SimpleNamespace(
        process_text=lambda text: segments))
    monkeypatch.setattr(app_module, "split_text_into_book_sections", lambda *a: {
        "sections": [{"title": "Chapter 1", "content": "test"}]})
    monkeypatch.setattr(app_module, "get_tts_engine", lambda *a, **k: Engine())
    monkeypatch.setattr(app_module, "_merge_review_job", lambda _id, entry, manifest: entry.update(status="completed"))
    config = {**app_module.DEFAULT_CONFIG, "tts_engine": "breeze_tts_2", "group_chunks_by_speaker": True}
    count = 4 if with_pause else 3
    data = {"job_id": job_id, "text": "test", "config": config, "voice_assignments": {},
            "total_chunks": count, "review_mode": True, "split_by_chapter": True,
            "generate_full_story": True}
    entry = {"status": "processing", "total_chunks": count, "config_snapshot": config}
    app_module.jobs[job_id] = entry
    app_module._process_audio_job(data)

    assert entry["status"] == "completed", entry.get("error")
    spoken = [chunk for chunk in entry["chunks"] if chunk["text"] != "***"]
    assert [(chunk["speaker"], chunk["text"]) for chunk in spoken] == [
        ("alice", "First."), ("bob", "Second."), ("alice", "Third.")]
    assert [Path(chunk["file_path"]).read_bytes() for chunk in spoken] == [
        b"First.", b"Second.", b"Third."]
    if with_pause:
        assert [chunk["text"] for chunk in entry["chunks"]] == ["First.", "***", "Second.", "Third."]
