"""Reject sensitive or generated content before it is committed.

This intentionally uses only the Python standard library so it can run before
the project's optional dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


SECRET_CONFIG_KEYS = {
    "gemini_api_key",
    "replicate_api_key",
    "llm_local_api_key",
    "chatterbox_turbo_replicate_api_token",
}

FORBIDDEN_EXACT_PATHS = {
    ".sync_state.json",
    "config.json.syncbak",
    "data/chatterbox_voices.json",
    "data/custom_voices.json",
    "data/external_voice_archives.json",
    "data/external_voices_cache.json",
    "jobs.db",
}

FORBIDDEN_PREFIXES = (
    ".cache/",
    "data/external_voices/",
    "data/jobs/",
    "data/prep/",
    "data/voice_prompts/",
    "engines/dots-tts/.venv/",
    "engines/dots-tts/hf-cache/",
    "engines/dots-tts/models/",
    "engines/dots-tts/repo/",
    "engines/index-tts/",
    "engines/omnivoice/.venv/",
    "models/",
    "static/audio/",
    "static/samples/",
    "tools/ffmpeg/",
    "tools/rubberband/",
    "venv.old/",
    "venv/",
)

ALLOWED_GENERATED_PLACEHOLDERS = {
    "data/voice_prompts/.gitkeep",
    "static/audio/.gitkeep",
    "static/samples/.gitkeep",
}

ALLOWED_PREFIX_EXCEPTIONS = {
    "engines/index-tts/tts_worker.py",
}

SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{40,255})"),
    "Google API key": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    "AWS access key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "OpenAI-style key": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "Replicate token": re.compile(rb"r8_[A-Za-z0-9]{30,}"),
}


def normalize_path(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix().lstrip("./")


def path_problem(path: str) -> str | None:
    normalized = normalize_path(path)
    lowered = normalized.lower()
    name = PurePosixPath(lowered).name

    if lowered in FORBIDDEN_EXACT_PATHS:
        return "runtime or backup file"
    if lowered in ALLOWED_GENERATED_PLACEHOLDERS or lowered in ALLOWED_PREFIX_EXCEPTIONS:
        return None
    if any(lowered.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return "generated, downloaded, or user-data directory"
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "environment file"
    if PurePosixPath(lowered).suffix in {".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx"}:
        return "database or credential file"
    return None


def content_problems(path: str, content: bytes) -> list[str]:
    problems = [label for label, pattern in SECRET_PATTERNS.items() if pattern.search(content)]
    if normalize_path(path).lower() == "config.json":
        try:
            config = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            problems.append("config.json is not valid UTF-8 JSON")
        else:
            populated = sorted(key for key in SECRET_CONFIG_KEYS if config.get(key))
            if populated:
                problems.append("populated secret setting(s): " + ", ".join(populated))
    return problems


def run_git(repo: Path, *args: str, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def staged_paths(repo: Path) -> list[str]:
    output = run_git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    assert isinstance(output, bytes)
    return [entry.decode("utf-8", "surrogateescape") for entry in output.split(b"\0") if entry]


def staged_content(repo: Path, path: str) -> bytes:
    output = run_git(repo, "show", f":{path}")
    assert isinstance(output, bytes)
    return output


def find_repo_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise OSError("no Git repository found in this directory or its parents")


def audit_staged(repo: Path) -> list[str]:
    failures: list[str] = []
    for path in staged_paths(repo):
        if problem := path_problem(path):
            failures.append(f"{path}: {problem}")
            continue
        content = staged_content(repo, path)
        for problem in content_problems(path, content):
            failures.append(f"{path}: {problem}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="audit staged additions and modifications")
    args = parser.parse_args()
    if not args.staged:
        parser.error("--staged is required")

    try:
        repo = find_repo_root()
        failures = audit_staged(repo)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: repository safety check could not run: {exc}", file=sys.stderr)
        return 2

    if failures:
        print("ERROR: repository safety check found content that must not be committed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("[safety] Staged content passed the repository safety check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
