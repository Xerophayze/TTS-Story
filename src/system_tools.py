"""Cross-platform discovery for bundled and system command-line tools."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PathLike = Union[str, os.PathLike]


def _usable_executable(path: Path, platform_name: str) -> bool:
    """Return whether *path* is a runnable file on the target platform."""

    if not path.is_file():
        return False
    return platform_name == "nt" or os.access(path, os.X_OK)


def find_system_tool(
    tool_name: str,
    *,
    project_root: PathLike = PROJECT_ROOT,
    platform_name: Optional[str] = None,
    which: Optional[Callable[[str], Optional[str]]] = None,
) -> Optional[Path]:
    """Resolve a bundled Windows executable or a native executable on PATH.

    Windows installations prefer the copy under ``tools/<name>/<name>.exe``.
    Linux and macOS prefer the native executable on PATH and never select a
    bundled Windows ``.exe``.
    """

    normalized_name = tool_name.strip()
    if not normalized_name or Path(normalized_name).name != normalized_name:
        raise ValueError("tool_name must be a simple executable name")

    platform_name = platform_name or os.name
    which = which or shutil.which
    root = Path(project_root)
    bundled_name = f"{normalized_name}.exe" if platform_name == "nt" else normalized_name
    bundled = root / "tools" / normalized_name / bundled_name

    candidates = []
    if platform_name == "nt":
        candidates.append(bundled)

    discovered = which(normalized_name)
    if discovered:
        candidates.append(Path(discovered))

    if platform_name != "nt":
        candidates.append(bundled)

    for candidate in candidates:
        if _usable_executable(candidate, platform_name):
            return candidate.resolve()
    return None


__all__ = ["find_system_tool", "PROJECT_ROOT"]
