#!/usr/bin/env bash

# Shared PyTorch helpers for setup.sh and run.sh.
# macOS wheels are published on PyPI without the Linux/Windows "+cpu" suffix.

tts_story_python_supported() {
    "$1" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 13) else 1)' >/dev/null 2>&1
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
