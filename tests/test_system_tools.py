from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.system_tools import find_system_tool


class SystemToolDiscoveryTests(unittest.TestCase):
    def test_windows_prefers_bundled_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundled = root / "tools" / "sox" / "sox.exe"
            bundled.parent.mkdir(parents=True)
            bundled.write_bytes(b"test")

            result = find_system_tool(
                "sox",
                project_root=root,
                platform_name="nt",
                which=lambda _: None,
            )

            self.assertEqual(bundled.resolve(), result)

    def test_posix_never_selects_bundled_windows_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundled = root / "tools" / "sox" / "sox.exe"
            bundled.parent.mkdir(parents=True)
            bundled.write_bytes(b"test")

            result = find_system_tool(
                "sox",
                project_root=root,
                platform_name="posix",
                which=lambda _: None,
            )

            self.assertIsNone(result)

    def test_posix_uses_native_path_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / "sox"
            executable.write_bytes(b"#!/bin/sh\n")
            executable.chmod(executable.stat().st_mode | 0o111)

            result = find_system_tool(
                "sox",
                project_root=temp_dir,
                platform_name="posix",
                which=lambda _: os.fspath(executable),
            )

            self.assertEqual(executable.resolve(), result)

    def test_rejects_path_instead_of_simple_tool_name(self) -> None:
        with self.assertRaises(ValueError):
            find_system_tool("../sox")


if __name__ == "__main__":
    unittest.main()
