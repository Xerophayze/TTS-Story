"""Small, resilient helpers for JSON files that are updated while the app runs."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = True,
    retries: int = 4,
    retry_delay: float = 0.05,
) -> None:
    """Write JSON without exposing a partially-written destination file.

    Some Windows filesystems, antivirus tools, and sync utilities can briefly
    reject a file operation.  Each attempt writes a unique temporary sibling
    and atomically replaces the destination, with a short increasing delay
    between transient failures.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii)
    attempts = max(int(retries), 1)
    last_error: OSError | None = None

    for attempt in range(attempts):
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(serialized, encoding="utf-8")
            os.replace(temporary, destination)
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 < attempts:
                time.sleep(max(float(retry_delay), 0.0) * (attempt + 1))

    if last_error is not None:
        raise last_error

