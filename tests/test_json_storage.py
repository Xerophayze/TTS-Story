from __future__ import annotations

import errno
import json

from src import json_storage


def test_atomic_json_write_retries_a_transient_windows_file_error(tmp_path, monkeypatch):
    destination = tmp_path / "chunks_metadata.json"
    real_replace = json_storage.os.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError(errno.EINVAL, "temporary filesystem contention")
        real_replace(source, target)

    monkeypatch.setattr(json_storage.os, "replace", flaky_replace)

    json_storage.write_json_atomic(
        destination,
        {"chunks": [{"id": "chunk-1"}]},
        retries=4,
        retry_delay=0,
    )

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "chunks": [{"id": "chunk-1"}]
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_json_write_preserves_existing_file_when_all_retries_fail(tmp_path, monkeypatch):
    destination = tmp_path / "chunks_metadata.json"
    destination.write_text('{"state": "previous"}', encoding="utf-8")

    def failed_replace(source, target):
        raise OSError(errno.EINVAL, "temporary filesystem contention")

    monkeypatch.setattr(json_storage.os, "replace", failed_replace)

    try:
        json_storage.write_json_atomic(
            destination,
            {"state": "new"},
            retries=2,
            retry_delay=0,
        )
    except OSError as exc:
        assert exc.errno == errno.EINVAL
    else:
        raise AssertionError("Expected the final filesystem error")

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "state": "previous"
    }
    assert list(tmp_path.glob("*.tmp")) == []
