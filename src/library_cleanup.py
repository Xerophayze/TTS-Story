"""Filesystem cleanup helpers for Audio Library items."""

from __future__ import annotations

import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any


def _retry_readonly_removal(func: Any, path: str, _exc_info: Any) -> None:
    """Retry a failed removal after making the target writable."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_directory_with_retries(
    directory: Path,
    *,
    attempts: int = 5,
    base_delay_seconds: float = 0.15,
) -> None:
    """Remove a directory, retrying transient Windows file locks."""
    target = Path(directory)
    if not target.exists():
        return

    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            shutil.rmtree(target, onerror=_retry_readonly_removal)
            if not target.exists():
                return
            last_error = OSError(f"Directory still exists after deletion: {target}")
        except OSError as exc:
            last_error = exc

        if attempt < attempts - 1:
            time.sleep(base_delay_seconds * (attempt + 1))

    detail = f": {last_error}" if last_error else ""
    raise PermissionError(
        "Could not completely delete the Library item because one or more "
        f"files are still in use. Stop audio playback and try again{detail}"
    )
