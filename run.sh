#!/usr/bin/env bash
set -e

echo "========================================"
echo "Starting TTS-Story"
echo "========================================"
echo
echo "NOTE: First startup can pause while models initialize and caches build."
echo "Subsequent runs should be faster."
echo
echo "Quick Troubleshooting:"
echo "  - If startup fails, delete the 'venv' folder and re-run setup.sh"
echo "  - GPU users: update to the latest NVIDIA drivers"
echo "  - Run 'git pull' to pull the latest updates"
echo

OS_NAME="$(uname -s)"
IS_MACOS=0
if [ "$OS_NAME" = "Darwin" ]; then
    IS_MACOS=1
fi

is_port_available() {
    python - "$1" <<'EOF'
import socket
import sys

port = int(sys.argv[1])
addresses = [
    (socket.AF_INET, "0.0.0.0"),
]

if socket.has_ipv6:
    addresses.append((socket.AF_INET6, "::"))

for family, host in addresses:
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            raise SystemExit(1)
raise SystemExit(0)
EOF
}

find_available_port() {
    python - "$1" <<'EOF'
import socket
import sys

start_port = int(sys.argv[1])
for port in range(start_port, min(start_port + 100, 65536)):
    addresses = [
        (socket.AF_INET, "0.0.0.0"),
    ]
    if socket.has_ipv6:
        addresses.append((socket.AF_INET6, "::"))

    ok = True
    for family, host in addresses:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                ok = False
                break
    if ok:
        print(port)
        raise SystemExit(0)
raise SystemExit(1)
EOF
}

read_configured_port() {
    python - <<'EOF'
import json
from pathlib import Path

default_port = 5000
config_path = Path("config.json")

try:
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        candidate = data.get("server_port", default_port)
    else:
        candidate = default_port
    port = int(candidate)
except Exception:
    port = default_port

if not (1 <= port <= 65535):
    port = default_port

print(port)
EOF
}

# Check that virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
# shellcheck disable=SC1091
source "venv/bin/activate"

PORT_SOURCE="default"
STRICT_PORT_REQUEST=0

if [ -n "${TTS_STORY_PORT:-}" ]; then
    REQUESTED_PORT="$TTS_STORY_PORT"
    PORT_SOURCE="environment"
    STRICT_PORT_REQUEST=1
elif [ -n "${PORT:-}" ]; then
    REQUESTED_PORT="$PORT"
    PORT_SOURCE="environment"
    STRICT_PORT_REQUEST=1
else
    REQUESTED_PORT="$(read_configured_port)"
    PORT_SOURCE="config"
fi

APP_PORT="$REQUESTED_PORT"

if ! [[ "$APP_PORT" =~ ^[0-9]+$ ]] || [ "$APP_PORT" -lt 1 ] || [ "$APP_PORT" -gt 65535 ]; then
    echo "WARNING: Invalid port '$APP_PORT'. Falling back to 5000."
    APP_PORT="5000"
fi

if ! is_port_available "$APP_PORT"; then
    if [ "$STRICT_PORT_REQUEST" -eq 1 ]; then
        echo "ERROR: Requested port $APP_PORT is already in use."
        echo "Set TTS_STORY_PORT to another port and re-run, for example: TTS_STORY_PORT=5001 ./run.sh"
        exit 1
    fi

    ALT_PORT="$(find_available_port $((APP_PORT + 1)) || true)"
    if [ -z "$ALT_PORT" ]; then
        echo "ERROR: Preferred port $APP_PORT is in use and no fallback port was found in the next 100 ports."
        exit 1
    fi

    if [ "$PORT_SOURCE" = "config" ]; then
        echo "Saved port $APP_PORT is already in use. Falling back to port $ALT_PORT for this run."
    else
        echo "Port $APP_PORT is already in use. Falling back to port $ALT_PORT."
    fi
    if [ "$IS_MACOS" -eq 1 ]; then
        echo "Tip: this is commonly caused by AirPlay Receiver on macOS."
    fi
    APP_PORT="$ALT_PORT"
fi

export TTS_STORY_PORT="$APP_PORT"

# Detect NVIDIA GPU
HAS_NVIDIA=0
GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [ -n "$GPU_NAME" ]; then
        HAS_NVIDIA=1
        echo "NVIDIA GPU detected: $GPU_NAME"
    fi
fi

# Ensure CPU-only torch on systems without NVIDIA GPUs
if [ "$HAS_NVIDIA" -eq 0 ]; then
    if [ "$IS_MACOS" -eq 1 ]; then
        echo "macOS system detected. Ensuring non-CUDA PyTorch is installed..."
    else
        echo "CPU-only system detected. Ensuring CPU PyTorch is installed..."
    fi
    
    TORCH_PIN="2.6.0"
    TORCHVISION_PIN="0.21.0"
    TORCHAUDIO_PIN="2.6.0"
    
    # Check if torch is installed
    TORCH_INSTALLED=""
    if python -c "import torch" 2>/dev/null; then
        TORCH_INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    fi
    
    # Reinstall CPU-only torch if:
    # 1. FORCE_TORCH_REINSTALL=1 is set
    # 2. No torch installed
    # 3. CUDA build detected on CPU-only system
    if [ "${FORCE_TORCH_REINSTALL:-0}" = "1" ] || [ -z "$TORCH_INSTALLED" ]; then
        if [ "$IS_MACOS" -eq 1 ]; then
            echo "Installing macOS-compatible PyTorch..."
        else
            echo "Installing CPU-only PyTorch..."
        fi
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        if [ "$IS_MACOS" -eq 1 ]; then
            pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        else
            pip install --upgrade --force-reinstall \
                torch==${TORCH_PIN}+cpu \
                torchvision==${TORCHVISION_PIN}+cpu \
                torchaudio==${TORCHAUDIO_PIN}+cpu \
                --index-url https://download.pytorch.org/whl/cpu
            pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
        fi
    elif echo "$TORCH_INSTALLED" | grep -q "+cu"; then
        echo "CUDA build detected on CPU-only system. Reinstalling CPU-only torch..."
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        if [ "$IS_MACOS" -eq 1 ]; then
            pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        else
            pip install --upgrade --force-reinstall \
                torch==${TORCH_PIN}+cpu \
                torchvision==${TORCHVISION_PIN}+cpu \
                torchaudio==${TORCHAUDIO_PIN}+cpu \
                --index-url https://download.pytorch.org/whl/cpu
            pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
        fi
    else
        echo "Detected PyTorch: $TORCH_INSTALLED"
    fi
fi

# Check system tools
echo
echo "Checking system tools..."

# Check FFmpeg
if command -v ffmpeg >/dev/null 2>&1; then
    echo "✓ FFmpeg ready"
else
    echo "⚠ WARNING: FFmpeg not found. Audio processing may fail."
    echo "  Install with: sudo apt-get install ffmpeg (Ubuntu/Debian)"
    echo "             brew install ffmpeg (macOS)"
fi

# Check Rubber Band
if command -v rubberband >/dev/null 2>&1; then
    echo "✓ Rubber Band CLI ready"
else
    echo "⚠ WARNING: Rubber Band CLI not found. Audio processing may fail."
    echo "  Install with: sudo apt-get install rubberband-cli (Ubuntu/Debian)"
    echo "             brew install rubberband (macOS)"
fi

# Check SoX
if command -v sox >/dev/null 2>&1; then
    echo "✓ SoX ready"
else
    echo "⚠ WARNING: SoX not found. Audio processing may fail."
    echo "  Install with: sudo apt-get install sox (Ubuntu/Debian)"
    echo "             brew install sox (macOS)"
fi

# Check espeak-ng
if command -v espeak-ng >/dev/null 2>&1; then
    echo "✓ espeak-ng ready"
else
    echo "⚠ WARNING: espeak-ng not found. Some TTS features may fail."
    echo "  Install with: sudo apt-get install espeak-ng (Ubuntu/Debian)"
    echo "             brew install espeak-ng (macOS)"
fi

# Check CUDA availability
echo
python - << 'EOF'
try:
    import torch
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("WARNING: Could not check CUDA status:", e)
EOF

echo
echo "Starting Flask server..."
echo "Open your browser to: http://localhost:${APP_PORT}"
echo "Press Ctrl+C to stop the server"
echo

# Start the application
python app.py
