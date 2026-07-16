"""Small, dependency-free helpers for persisted Audio Library metadata."""

from __future__ import annotations

from typing import Any, Mapping, Optional


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
