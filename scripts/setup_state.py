#!/usr/bin/env python3
"""Track whether an existing installation needs dependency reconciliation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as platform_module
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / ".setup_state.json"
STATE_SCHEMA = 1


def dependency_files(platform_id: str) -> tuple[str, ...]:
    platform_key = platform_id.lower()
    common = ("requirements.txt", "scripts/setup_state.py")
    if platform_key.startswith("windows"):
        return common + ("setup.bat",)
    return common + ("setup.sh", "scripts/unix_torch.sh")


def calculate_fingerprint(root: Path, platform_id: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"tts-story-setup-state-v{STATE_SCHEMA}\0{platform_id.lower()}\0".encode())
    for relative_name in dependency_files(platform_id):
        path = root / relative_name
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def read_state(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def state_matches(root: Path, state_path: Path, platform_id: str) -> bool:
    state = read_state(state_path)
    return (
        state.get("schema") == STATE_SCHEMA
        and state.get("platform") == platform_id.lower()
        and state.get("fingerprint") == calculate_fingerprint(root, platform_id)
    )


def write_state(root: Path, state_path: Path, platform_id: str) -> None:
    payload = {
        "schema": STATE_SCHEMA,
        "platform": platform_id.lower(),
        "fingerprint": calculate_fingerprint(root, platform_id),
        "python": platform_module.python_version(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, state_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fingerprint", "matches", "write"))
    parser.add_argument("--platform", required=True, dest="platform_id")
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    args = parser.parse_args()

    if args.action == "fingerprint":
        print(calculate_fingerprint(ROOT, args.platform_id))
        return 0
    if args.action == "matches":
        return 0 if state_matches(ROOT, args.state_file, args.platform_id) else 1

    write_state(ROOT, args.state_file, args.platform_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
