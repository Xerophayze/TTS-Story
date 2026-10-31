"""Library listing exposes cached audio durations (production, full story, chapters)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import app as app_module


SAMPLE_RATE = 8000


def _write_wav(path: Path, seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(seconds * SAMPLE_RATE))
    sf.write(str(path), np.zeros(frames, dtype="float32"), SAMPLE_RATE)
    return path


def _write_metadata(job_dir: Path, metadata: dict) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / app_module.JOB_METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")


@pytest.fixture
def library_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(app_module, "_LIBRARY_DURATION_CACHE", {})
    app_module.invalidate_library_cache()
    yield tmp_path
    app_module.invalidate_library_cache()


def _items_by_id(items):
    return {item["job_id"]: item for item in items}


def test_listing_cache_refresh_and_application_invalidation(monkeypatch, library_dir):
    calls = []
    monkeypatch.setattr(app_module, '_build_library_listing', lambda: calls.append(1) or [])
    client = app_module.app.test_client()
    assert client.get('/api/library').get_json()['cached'] is False
    assert client.get('/api/library').get_json()['cached'] is True
    assert len(calls) == 1
    assert client.get('/api/library?refresh=1').get_json()['cached'] is False
    assert len(calls) == 2
    app_module.invalidate_library_cache()
    assert client.get('/api/library').get_json()['cached'] is False
    assert len(calls) == 3


def test_chapter_production_reports_full_story_and_chapter_durations(library_dir):
    job_dir = library_dir / "book-job"
    _write_wav(job_dir / "chapter_01" / "one.wav", 1.5)
    _write_wav(job_dir / "chapter_02" / "two.wav", 2.25)
    _write_wav(job_dir / "full_story.wav", 3.75)
    _write_metadata(job_dir, {
        "chapter_mode": True,
        "output_format": "wav",
        "chapters": [
            {"index": 1, "title": "Chapter 1", "relative_path": "chapter_01/one.wav"},
            {"index": 2, "title": "Chapter 2", "relative_path": "chapter_02/two.wav"},
        ],
        "full_story": {"title": "Full Story", "relative_path": "full_story.wav"},
    })

    response = app_module.app.test_client().get("/api/library")
    payload = response.get_json()
    assert payload["success"] is True
    item = _items_by_id(payload["items"])["book-job"]

    assert item["duration_seconds"] == pytest.approx(3.75, abs=0.01)
    assert item["full_story"]["duration_seconds"] == pytest.approx(3.75, abs=0.01)
    durations = [chapter["duration_seconds"] for chapter in item["chapters"]]
    assert durations == [pytest.approx(1.5, abs=0.01), pytest.approx(2.25, abs=0.01)]
    # Rename addresses chapters by their position in metadata["chapters"].
    assert [chapter["position"] for chapter in item["chapters"]] == [0, 1]


def test_production_without_full_story_sums_chapters_and_single_files_report_their_length(library_dir):
    chapters_job = library_dir / "chapters-only"
    _write_wav(chapters_job / "chapter_01" / "a.wav", 1.0)
    _write_wav(chapters_job / "chapter_02" / "b.wav", 0.5)
    _write_metadata(chapters_job, {
        "chapter_mode": True,
        "output_format": "wav",
        "chapters": [
            {"index": 1, "title": "A", "relative_path": "chapter_01/a.wav"},
            {"index": 2, "title": "B", "relative_path": "chapter_02/b.wav"},
        ],
    })
    single_job = library_dir / "single-file"
    _write_wav(single_job / "output.wav", 2.0)

    items = _items_by_id(app_module._build_library_listing())

    assert items["chapters-only"]["duration_seconds"] == pytest.approx(1.5, abs=0.01)
    assert items["single-file"]["duration_seconds"] == pytest.approx(2.0, abs=0.01)
    assert items["single-file"]["chapters"][0]["duration_seconds"] == pytest.approx(2.0, abs=0.01)


def test_unreadable_or_missing_audio_has_no_duration_and_does_not_break_listing(library_dir):
    job_dir = library_dir / "broken-job"
    job_dir.mkdir()
    (job_dir / "chapter_01").mkdir()
    (job_dir / "chapter_01" / "bad.mp3").write_bytes(b"this is not audio")
    _write_metadata(job_dir, {
        "chapter_mode": True,
        "output_format": "mp3",
        "chapters": [
            {"index": 1, "title": "Bad", "relative_path": "chapter_01/bad.mp3"},
            {"index": 2, "title": "Gone", "relative_path": "chapter_02/missing.mp3"},
        ],
    })

    items = _items_by_id(app_module._build_library_listing())

    item = items["broken-job"]
    assert [chapter["duration_seconds"] for chapter in item["chapters"]] == [None, None]
    assert item["duration_seconds"] is None


def test_durations_are_cached_until_the_file_changes(monkeypatch, library_dir):
    job_dir = library_dir / "cached-job"
    audio = _write_wav(job_dir / "output.wav", 1.0)
    calls = []
    real_probe = app_module._probe_audio_duration_seconds

    def counting_probe(path):
        calls.append(Path(path).name)
        return real_probe(path)

    monkeypatch.setattr(app_module, "_probe_audio_duration_seconds", counting_probe)

    first = _items_by_id(app_module._build_library_listing())["cached-job"]["duration_seconds"]
    second = _items_by_id(app_module._build_library_listing())["cached-job"]["duration_seconds"]
    assert first == second == pytest.approx(1.0, abs=0.01)
    assert calls == ["output.wav"]

    _write_wav(audio, 3.0)
    stat_result = audio.stat()
    os.utime(audio, ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 5_000_000_000))
    third = _items_by_id(app_module._build_library_listing())["cached-job"]["duration_seconds"]
    assert third == pytest.approx(3.0, abs=0.01)
    assert calls == ["output.wav", "output.wav"]


def test_book_rows_report_durations_but_no_chapter_position(library_dir):
    # Book rows come from metadata["books"]; the chapter rename route addresses
    # metadata["chapters"], so book rows must not advertise a rename position.
    job_dir = library_dir / "books-job"
    _write_wav(job_dir / "book_01" / "b1.wav", 0.5)
    _write_wav(job_dir / "book_02" / "b2.wav", 0.75)
    _write_metadata(job_dir, {
        "chapter_mode": True,
        "book_mode": True,
        "output_format": "wav",
        "books": [
            {"index": 1, "title": "Book One", "relative_path": "book_01/b1.wav"},
            {"index": 2, "title": "Book Two", "relative_path": "book_02/b2.wav"},
        ],
        "chapters": [
            {"index": 0, "title": "Ch A", "book_index": 0},
            {"index": 1, "title": "Ch B", "book_index": 0},
            {"index": 2, "title": "Ch C", "book_index": 1},
        ],
    })

    item = _items_by_id(app_module._build_library_listing())["books-job"]

    assert [row["title"] for row in item["books"]] == ["Book One", "Book Two"]
    assert [row["position"] for row in item["books"]] == [None, None]
    assert [row["duration_seconds"] for row in item["books"]] == [
        pytest.approx(0.5, abs=0.01), pytest.approx(0.75, abs=0.01),
    ]
    assert item["duration_seconds"] == pytest.approx(1.25, abs=0.01)


def test_library_larger_than_the_cache_cap_keeps_every_duration(monkeypatch, library_dir):
    # Before: the cache cleared itself completely at the cap, so a big Library re-measured
    # every file on every listing rebuild.
    monkeypatch.setattr(app_module, "_LIBRARY_DURATION_CACHE_MAX_ENTRIES", 4)
    for index in range(10):
        _write_wav(library_dir / f"job-{index:02d}" / "output.wav", 0.25)
    calls = []
    real_probe = app_module._probe_audio_duration_seconds

    def counting_probe(path):
        calls.append(str(path))
        return real_probe(path)

    monkeypatch.setattr(app_module, "_probe_audio_duration_seconds", counting_probe)

    first = app_module._build_library_listing()
    assert len(first) == 10
    assert len(calls) == 10
    second = app_module._build_library_listing()
    assert len(second) == 10
    assert len(calls) == 10, "the second build must reuse every cached duration"
    assert all(item["duration_seconds"] == pytest.approx(0.25, abs=0.01) for item in second)


def test_listing_build_prunes_durations_of_deleted_files(library_dir):
    keep = _write_wav(library_dir / "keep-job" / "output.wav", 0.5)
    gone = _write_wav(library_dir / "gone-job" / "output.wav", 0.5)
    app_module._build_library_listing()
    assert {str(keep), str(gone)} <= set(app_module._LIBRARY_DURATION_CACHE)

    gone.unlink()
    gone.parent.rmdir()
    app_module._build_library_listing()

    assert str(keep) in app_module._LIBRARY_DURATION_CACHE
    assert str(gone) not in app_module._LIBRARY_DURATION_CACHE
