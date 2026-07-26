from __future__ import annotations

import shutil

import pytest

from src import library_cleanup


def test_directory_removal_retries_transient_file_locks(tmp_path, monkeypatch):
    target = tmp_path / "locked-item"
    target.mkdir()
    (target / "audio.mp3").write_bytes(b"audio")
    real_rmtree = shutil.rmtree
    attempts = []

    def flaky_rmtree(path, onerror=None):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError("file is in use")
        real_rmtree(path, onerror=onerror)

    monkeypatch.setattr(library_cleanup.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(library_cleanup.time, "sleep", lambda _delay: None)

    library_cleanup.remove_directory_with_retries(target)

    assert len(attempts) == 3
    assert not target.exists()


def test_directory_removal_reports_failure_if_target_remains(tmp_path, monkeypatch):
    target = tmp_path / "locked-item"
    target.mkdir()
    (target / "audio.mp3").write_bytes(b"audio")

    monkeypatch.setattr(
        library_cleanup.shutil,
        "rmtree",
        lambda _path, onerror=None: (_ for _ in ()).throw(PermissionError("file is in use")),
    )
    monkeypatch.setattr(library_cleanup.time, "sleep", lambda _delay: None)

    with pytest.raises(PermissionError, match="files are still in use"):
        library_cleanup.remove_directory_with_retries(target, attempts=3)

    assert target.exists()
