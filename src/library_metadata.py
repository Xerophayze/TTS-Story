"""Small, dependency-free helpers for persisted Audio Library metadata."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping, Optional


_TITLE_PREFIX_RE = re.compile(
    r"^\s*(?:[#*_]+\s*)?title\s*(?::|[.\-–—])\s*(.+?)\s*$",
    re.IGNORECASE,
)


def infer_collection_title_from_chunks(chunks: Any) -> Optional[str]:
    """Recover a collection title from an explicit title chunk.

    This is intentionally conservative: ordinary chapter prose is ignored,
    and only short chunks beginning with an explicit ``Title:``/``Title.``
    marker qualify.
    """
    if not isinstance(chunks, (list, tuple)):
        return None
    ordered = sorted(
        (chunk for chunk in chunks if isinstance(chunk, Mapping)),
        key=lambda chunk: chunk.get("order_index")
        if isinstance(chunk.get("order_index"), (int, float))
        else float("inf"),
    )
    for chunk in ordered[:10]:
        text = chunk.get("text")
        if not isinstance(text, str):
            continue
        compact = " ".join(text.strip().split())
        if not compact or len(compact) > 240:
            continue
        match = _TITLE_PREFIX_RE.match(compact)
        if not match:
            continue
        title = match.group(1).strip(" #*_\"'“”‘’.,:;")
        if title and len(title) <= 160:
            return title
    return None


def merge_generated_library_metadata(
    existing: Any,
    generated: Any,
    chunks: Any = None,
) -> dict[str, Any]:
    """Overlay rebuilt output records without losing user-authored metadata."""
    merged = dict(existing) if isinstance(existing, Mapping) else {}
    if isinstance(generated, Mapping):
        merged.update(generated)
    if not merged.get("collection_title"):
        recovered_title = infer_collection_title_from_chunks(chunks)
        if recovered_title:
            merged["collection_title"] = recovered_title
    return merged


def get_custom_chapter_title(
    metadata: Mapping[str, Any],
    chapter_index: Any = None,
    position: Optional[int] = None,
) -> Optional[str]:
    """Return a saved custom title by list position, then manifest chapter index."""
    custom_titles = metadata.get("custom_chapter_titles") or {}
    if not isinstance(custom_titles, Mapping):
        return None

    keys = []
    if position is not None:
        keys.append(str(position))
    if chapter_index is not None:
        keys.append(str(chapter_index))

    for key in keys:
        value = custom_titles.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def can_reuse_active_review_job(
    job_entry: Optional[Mapping[str, Any]],
    expected_job_dir: Any,
) -> bool:
    """Return whether an active in-memory Library review job must be reused.

    Completed/idle entries can be refreshed from disk. Replacing an active
    entry would discard its queued/running chunk regeneration state.
    """
    if not isinstance(job_entry, Mapping):
        return False
    if not job_entry.get("review_mode") or not job_entry.get("chunks"):
        return False

    stored_job_dir = job_entry.get("job_dir")
    if not stored_job_dir or not expected_job_dir:
        return False
    regen_tasks = job_entry.get("regen_tasks") or {}
    if not any(
        isinstance(task, Mapping) and task.get("status") in {"queued", "running"}
        for task in regen_tasks.values()
    ):
        return False

    try:
        stored_path = os.path.normcase(os.path.abspath(os.fspath(stored_job_dir)))
        expected_path = os.path.normcase(os.path.abspath(os.fspath(expected_job_dir)))
    except (TypeError, ValueError, OSError):
        return False
    return stored_path == expected_path
