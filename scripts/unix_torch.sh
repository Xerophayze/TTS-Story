#!/usr/bin/env bash

# Shared PyTorch helpers for setup.sh and run.sh.
# macOS wheels are published on PyPI without the Linux/Windows "+cpu" suffix.

tts_story_python_supported() {
    "$1" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 13) else 1)' >/dev/null 2>&1
}

tts_story_find_compatible_python() {
    # Print one compatible interpreter path. Do not rely solely on `python3`:
    # macOS may expose a newer unsupported Python while a compatible Homebrew
    # or pyenv installation is present but not linked into the current shell.
    local candidate=""
    local resolved=""
    local brew_prefix=""
    local pyenv_root=""
    local minor=""

    if [ -n "${TTS_STORY_PYTHON:-}" ]; then
        if tts_story_python_supported "$TTS_STORY_PYTHON"; then
            printf '%s\n' "$TTS_STORY_PYTHON"
            return 0
        fi
        return 1
    fi

    # Keep a supported active interpreter when one exists, then prefer the
    # project's primary 3.11 target among explicitly versioned commands.
    for candidate in python3 python3.11 python3.12 python3.10 python3.9 python; do
        resolved="$(command -v "$candidate" 2>/dev/null || true)"
        if [ -n "$resolved" ] && tts_story_python_supported "$resolved"; then
            printf '%s\n' "$resolved"
            return 0
        fi
    done

    # Homebrew versioned formulae are frequently keg-only, especially when a
    # different Python is linked as `python3`.
    if command -v brew >/dev/null 2>&1; then
        for minor in 11 12 10 9; do
            brew_prefix="$(brew --prefix "python@3.$minor" 2>/dev/null || true)"
            candidate="$brew_prefix/bin/python3.$minor"
            if [ -n "$brew_prefix" ] && [ -x "$candidate" ] && tts_story_python_supported "$candidate"; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    fi

    # Cover unlinked Apple Silicon/Intel Homebrew and python.org framework
    # installations even when their shell initialization is missing.
    for minor in 11 12 10 9; do
        for candidate in \
            "/opt/homebrew/opt/python@3.$minor/bin/python3.$minor" \
            "/usr/local/opt/python@3.$minor/bin/python3.$minor" \
            "/Library/Frameworks/Python.framework/Versions/3.$minor/bin/python3"; do
            if [ -x "$candidate" ] && tts_story_python_supported "$candidate"; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    done

    # pyenv versions may exist even when `pyenv init` was not added to the
    # non-interactive shell used by an installer or launcher.
    pyenv_root="${PYENV_ROOT:-$HOME/.pyenv}"
    if command -v pyenv >/dev/null 2>&1; then
        pyenv_root="$(pyenv root 2>/dev/null || printf '%s' "$pyenv_root")"
    fi
    for minor in 11 12 10 9; do
        for candidate in "$pyenv_root"/versions/3."$minor"*/bin/python3; do
            if [ -x "$candidate" ] && tts_story_python_supported "$candidate"; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    done

    return 1
}

tts_story_install_cpu_torch() {
    local python_executable="$1"
    shift

    if [ "$(uname -s)" = "Darwin" ]; then
        "$python_executable" -m pip install --upgrade --force-reinstall "$@"
    else
        "$python_executable" -m pip install --upgrade --force-reinstall "$@" \
            --index-url https://download.pytorch.org/whl/cpu
    fi
}
