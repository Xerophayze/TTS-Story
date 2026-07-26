#!/usr/bin/env python3
"""Run a command with a portable timeout and preserve its exit status."""

from __future__ import annotations

import argparse
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    try:
        return subprocess.run(command, check=False, timeout=args.timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {args.timeout:g} seconds.")
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
