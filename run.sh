#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/unix_torch.sh"

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

# Check that setup completed and the virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run ./setup.sh first."
    exit 1
fi
if [ ! -f ".setup_complete" ]; then
    echo "ERROR: Setup has not completed successfully for this installation."
    echo "Please rerun ./setup.sh before starting TTS-Story."
    exit 1
fi

# Activate virtual environment
# shellcheck disable=SC1091
source "venv/bin/activate"

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
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "macOS system detected. Ensuring native PyTorch is installed..."
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
        echo "Installing PyTorch for this platform..."
        python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        tts_story_install_cpu_torch python \
            "torch==${TORCH_PIN}" \
            "torchvision==${TORCHVISION_PIN}" \
            "torchaudio==${TORCHAUDIO_PIN}"
        pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
    elif echo "$TORCH_INSTALLED" | grep -q "+cu"; then
        echo "CUDA build detected on a non-NVIDIA system. Reinstalling PyTorch for this platform..."
        python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        tts_story_install_cpu_torch python \
            "torch==${TORCH_PIN}" \
            "torchvision==${TORCHVISION_PIN}" \
            "torchaudio==${TORCHAUDIO_PIN}"
        pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
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
echo "Open your browser to: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo

# Start the application
python app.py
