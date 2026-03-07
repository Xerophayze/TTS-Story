#!/usr/bin/env bash
set -e

echo "========================================"
echo "TTS-Story Setup (Linux/macOS)"
echo "========================================"
echo
echo "IMPORTANT: Initial setup can take several minutes (large downloads + builds)."
echo "Please be patient and report any errors you encounter."
echo
echo "Quick Troubleshooting:"
echo "  - If setup fails, delete the 'venv' folder and re-run setup.sh"
echo "  - GPU users: update to the latest NVIDIA drivers"

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
IS_MACOS=0
if [ "$OS_NAME" = "Darwin" ]; then
    IS_MACOS=1
fi

PYTHON_BIN=""

find_compatible_python() {
    local candidate=""

    if command -v python3 >/dev/null 2>&1; then
        candidate="$(command -v python3)"
    fi

    if [ -n "$candidate" ]; then
        local candidate_minor
        candidate_minor=$("$candidate" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo "")
        if [ -n "$candidate_minor" ] && [ "$candidate_minor" -le 12 ]; then
            PYTHON_BIN="$candidate"
            return 0
        fi
    fi

    for alt in python3.12 python3.11; do
        if command -v "$alt" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "$alt")"
            return 0
        fi
    done

    if [ "$IS_MACOS" -eq 1 ] && command -v brew >/dev/null 2>&1; then
        echo "Compatible Python not found in PATH. Installing Homebrew python@3.12..."
        brew install python@3.12

        if command -v python3.12 >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v python3.12)"
            return 0
        fi

        local brew_python_prefix
        brew_python_prefix="$(brew --prefix python@3.12 2>/dev/null || true)"
        if [ -n "$brew_python_prefix" ] && [ -x "$brew_python_prefix/bin/python3.12" ]; then
            PYTHON_BIN="$brew_python_prefix/bin/python3.12"
            return 0
        fi
    fi

    return 1
}

# PyTorch versions (matching setup.bat)
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
TORCHAUDIO_VERSION="2.6.0"

# 1/12 Check Python installation
echo
echo "[1/12] Checking Python installation..."
if ! find_compatible_python; then
    echo "ERROR: Could not find a compatible Python interpreter."
    echo "This project currently needs Python 3.9-3.12 for full local install support."
    echo "Please install Python 3.12 (recommended) and re-run setup.sh."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1)
echo "Found $PYTHON_VERSION via $PYTHON_BIN"

# Check Python version (3.9+ required)
PYTHON_MAJOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.minor)')
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "ERROR: Python 3.9 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -gt 12 ]; then
    echo "ERROR: Python $PYTHON_VERSION is too new for one or more required packages (notably kokoro)."
    echo "Please install Python 3.12 or 3.11 and re-run setup.sh."
    exit 1
fi

# 1b/12 Check and install git if not present
echo
echo "[1b/12] Checking Git installation..."
if ! command -v git >/dev/null 2>&1; then
    echo "Git not found. Installing Git..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq git
    elif command -v brew >/dev/null 2>&1; then
        brew install git
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm git
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y git
    else
        echo "WARNING: Could not detect package manager to install git."
        echo "Please install git manually and re-run setup.sh."
    fi
else
    echo "Git is installed: $(git --version)"
fi

# Fix for git dubious ownership warning
git config --global --add safe.directory "*" 2>/dev/null || true

# Check and install python3-venv if not present
echo
echo "[1c/12] Checking python3-venv installation..."
if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
    echo "python3-venv not found. Installing..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3-venv python3-pip
    elif command -v brew >/dev/null 2>&1; then
        brew install python
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm python-pythonz
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3.10-venv
    else
        echo "WARNING: Could not detect package manager to install python3-venv."
    fi
fi
echo "python3-venv is available."

# 2/12 Create virtual environment
echo
echo "[2/12] Creating virtual environment..."
if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
    VENV_PYTHON_MINOR=$(venv/bin/python -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo "")
    if [ -n "$VENV_PYTHON_MINOR" ] && [ "$VENV_PYTHON_MINOR" != "$PYTHON_MINOR" ]; then
        echo "Existing virtual environment uses a different Python version. Recreating venv..."
        PROJECT_VENV_PATH="$(cd venv 2>/dev/null && pwd || true)"
        if [ -n "${VIRTUAL_ENV:-}" ] && [ -n "$PROJECT_VENV_PATH" ] && [ "$VIRTUAL_ENV" = "$PROJECT_VENV_PATH" ]; then
            echo "Deactivating current virtual environment before recreating it..."
            deactivate 2>/dev/null || true
            hash -r 2>/dev/null || true
        fi

        if ! rm -rf venv; then
            echo "rm -rf venv failed, retrying with Python cleanup..."
            "$PYTHON_BIN" - <<'EOF'
import shutil

shutil.rmtree("venv")
EOF
        fi
    fi
fi

if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "Virtual environment already exists, skipping..."
else
    "$PYTHON_BIN" -m venv venv
fi

# 3/12 Activate virtual environment
echo
echo "[3/12] Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

# 4/12 Upgrade pip
echo
echo "[4/12] Upgrading pip..."
python -m pip install --upgrade pip --quiet

# 5/12 Install PyTorch
echo
echo "[5/12] Installing PyTorch..."
echo "This may take several minutes..."
echo

# Detect NVIDIA GPU
HAS_NVIDIA=0
GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [ -n "$GPU_NAME" ]; then
        HAS_NVIDIA=1
    fi
fi

if [ "$HAS_NVIDIA" -eq 0 ]; then
    if [ "$IS_MACOS" -eq 1 ]; then
        echo "No NVIDIA GPU detected. Using macOS-compatible non-CUDA installs (CPU/MPS where supported)."
    else
        echo "No NVIDIA GPU detected. Using CPU-only installs."
    fi
fi

# Check if PyTorch is already installed
TORCH_INSTALLED=""
TORCH_CUDA=""
NEED_TORCH_INSTALL=1

if python -c "import torch" 2>/dev/null; then
    TORCH_INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    if python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null | grep -q "cuda"; then
        TORCH_CUDA="cuda"
    else
        TORCH_CUDA="cpu"
    fi
    
    if [ "$HAS_NVIDIA" -eq 1 ] && [ "$TORCH_CUDA" = "cuda" ]; then
        echo "Detected existing CUDA torch: $TORCH_INSTALLED"
        NEED_TORCH_INSTALL=0
    elif [ "$HAS_NVIDIA" -eq 0 ] && [ "$TORCH_CUDA" = "cpu" ]; then
        echo "Detected existing CPU torch: $TORCH_INSTALLED"
        NEED_TORCH_INSTALL=0
    fi
fi

if [ "$NEED_TORCH_INSTALL" -eq 0 ]; then
    echo "Skipping torch install - compatible build already present."
else
    if [ "$HAS_NVIDIA" -eq 1 ]; then
        # Try CUDA 12.4 first (most stable), then fall back
        echo "Installing PyTorch with CUDA 12.4 support..."
        if pip install torch==${TORCH_VERSION}+cu124 torchvision==${TORCHVISION_VERSION}+cu124 torchaudio==${TORCHAUDIO_VERSION}+cu124 --index-url https://download.pytorch.org/whl/cu124 2>/dev/null; then
            echo "PyTorch CUDA 12.4 installed successfully!"
        else
            # Try CUDA 12.1
            echo "CUDA 12.4 failed, trying CUDA 12.1..."
            if pip install torch==${TORCH_VERSION}+cu121 torchvision==${TORCHVISION_VERSION}+cu121 torchaudio==${TORCHAUDIO_VERSION}+cu121 --index-url https://download.pytorch.org/whl/cu121 2>/dev/null; then
                echo "PyTorch CUDA 12.1 installed successfully!"
            else
                # Try CUDA 12.6
                echo "CUDA 12.1 failed, trying CUDA 12.6..."
                if pip install torch==${TORCH_VERSION}+cu126 torchvision==${TORCHVISION_VERSION}+cu126 torchaudio==${TORCHAUDIO_VERSION}+cu126 --index-url https://download.pytorch.org/whl/cu126 2>/dev/null; then
                    echo "PyTorch CUDA 12.6 installed successfully!"
                else
                    echo "WARNING: CUDA PyTorch install failed, trying CPU version..."
                    pip install torch torchvision torchaudio
                fi
            fi
        fi
    else
        if [ "$IS_MACOS" -eq 1 ]; then
            echo "Installing macOS-compatible PyTorch (no Linux-style +cpu suffix)..."
        else
            echo "Installing CPU-only PyTorch..."
        fi
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        if [ "$IS_MACOS" -eq 1 ]; then
            pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        else
            if ! pip install --upgrade --force-reinstall torch==${TORCH_VERSION}+cpu torchvision==${TORCHVISION_VERSION}+cpu torchaudio==${TORCHAUDIO_VERSION}+cpu --index-url https://download.pytorch.org/whl/cpu; then
                echo "Pinned CPU PyTorch install failed, trying latest compatible wheels from the PyTorch CPU index..."
                pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            fi
        fi

        if [ "$IS_MACOS" -eq 1 ]; then
            echo "Skipping legacy Linux CPU compatibility pins on macOS."
        else
            pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
        fi
    fi
fi

# 6/12 Install other dependencies
echo
echo "[6/12] Installing other Python dependencies..."
# Add scipy if not in requirements (needed for pocket-tts)
if ! grep -qi "^scipy" requirements.txt 2>/dev/null; then
    echo "Adding scipy to requirements..."
    echo "scipy>=1.11.0" >> requirements.txt
fi
# Filter out torch packages (already installed) and pyopenjtalk (needs special handling)
grep -vi "^torch" requirements.txt > temp_requirements.txt 2>/dev/null || true
grep -vi "^pyopenjtalk" temp_requirements.txt > temp_requirements_filtered.txt 2>/dev/null || true
pip install -r temp_requirements_filtered.txt
rm -f temp_requirements.txt temp_requirements_filtered.txt

# Install pyopenjtalk if possible (requires compile tools)
echo
echo "Checking for pyopenjtalk (Japanese text support)..."
if command -v make >/dev/null 2>&1 && (command -v g++ >/dev/null 2>&1 || command -v c++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1); then
    echo "Build tools found. Installing pyopenjtalk..."
    pip install pyopenjtalk || echo "WARNING: pyopenjtalk failed to install. Japanese TTS features will be unavailable."
else
    echo "WARNING: Build tools not found. Skipping pyopenjtalk."
    echo "To enable Japanese TTS, install build tools and rerun setup.sh."
fi

# 7/12 Install Chatterbox Turbo runtime
echo
echo "[7/12] Installing Chatterbox Turbo runtime..."
# On macOS, chatterbox-tts currently tries to drag in an older numpy build path
# that is noisy and unnecessary for this project setup, so install the package
# itself without re-resolving torch/numpy.
if [ "$IS_MACOS" -eq 1 ]; then
    if pip install chatterbox-tts --no-deps; then
        echo "Chatterbox Turbo installed without dependency re-resolution on macOS."
    else
        echo "WARNING: chatterbox-tts install failed - Chatterbox local engine may be unavailable"
    fi
# First try with deps to get torchaudio and other required packages
elif pip install chatterbox-tts; then
    echo "Chatterbox Turbo installed with dependencies!"
else
    # If that fails, try without deps but install torchaudio manually
    echo "Installing chatterbox-tts without auto-deps, installing key dependencies manually..."
    pip install chatterbox-tts --no-deps || true
fi

# Ensure torchaudio is installed (required for Chatterbox)
echo
echo "Ensuring torchaudio is installed (required for Chatterbox)..."
pip install torchaudio --quiet || echo "WARNING: torchaudio install failed"

# Install scipy (needed by pocket-tts and other TTS engines)
echo
echo "Installing scipy (required for Pocket TTS and audio processing)..."
pip install scipy --quiet || echo "WARNING: scipy install failed"

# 8/12 Install Pocket TTS runtime
echo
echo "[8/12] Installing Pocket TTS runtime..."
# Install pocket-tts with deps to get all required packages
if pip install pocket-tts; then
    echo "Pocket TTS installed with dependencies!"
else
    echo "WARNING: pocket-tts install failed - Pocket TTS engine will not be available"
fi

# 8b/12 Install VoxCPM runtime (after pocket-tts so numpy version is set)
echo
echo "[8b/12] Installing VoxCPM 1.5 runtime..."
pip install voxcpm --no-deps || echo "WARNING: Failed to install voxcpm - VoxCPM engine will not be available"

# Ensure numpy is at a compatible version for all TTS engines
echo
echo "Ensuring numpy version compatibility..."
if [ "$IS_MACOS" -eq 1 ]; then
    echo "Skipping legacy numpy compatibility adjustment on macOS."
elif [ "$HAS_NVIDIA" -eq 0 ]; then
    # CPU-only systems need numpy<1.26.0 for older PyTorch compatibility
    pip install "numpy<1.26.0" --quiet || echo "WARNING: numpy version adjustment failed"
else
    # GPU systems can use newer numpy
    pip install "numpy>=2.0.0" --quiet || echo "WARNING: numpy version adjustment failed"
fi

# 9/12 Install optional performance extras
echo
echo "[9/12] Installing optional performance extras..."
echo "- hf_xet (faster Hugging Face downloads)"

pip install hf_xet || echo "WARNING: hf_xet install failed. Hugging Face downloads may be slower."

# Ensure voice prompts directory exists
echo
echo "[10/12] Creating data directories..."
mkdir -p data/voice_prompts

# 11/12 Install system tools (espeak-ng, sox, ffmpeg)
echo
echo "[11/12] Checking system dependencies..."

# Check for apt (Debian/Ubuntu)
install_apt() {
    echo "Installing system packages via apt..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq espeak-ng sox ffmpeg libsox-dev rubberband-cli || echo "WARNING: Some system packages failed to install"
}

# Check for brew (macOS)
install_brew() {
    echo "Installing system packages via Homebrew..."
    brew install espeak-ng sox ffmpeg rubberband || echo "WARNING: Some system packages failed to install"
}

# Check for pacman (Arch Linux)
install_pacman() {
    echo "Installing system packages via pacman..."
    sudo pacman -Sy --noconfirm espeak-ng sox ffmpeg rubberband || echo "WARNING: Some system packages failed to install"
}

# Check for dnf (Fedora)
install_dnf() {
    echo "Installing system packages via dnf..."
    sudo dnf install -y espeak-ng sox ffmpeg rubberband || echo "WARNING: Some system packages failed to install"
}

if command -v apt-get >/dev/null 2>&1; then
    install_apt
elif command -v brew >/dev/null 2>&1; then
    install_brew
elif command -v pacman >/dev/null 2>&1; then
    install_pacman
elif command -v dnf >/dev/null 2>&1; then
    install_dnf
else
    echo "WARNING: Could not detect package manager. Please install manually:"
    echo "  - espeak-ng"
    echo "  - sox"
    echo "  - ffmpeg"
    echo "  - rubberband-cli"
fi

# Verify espeak-ng
echo
echo "========================================"
echo "Checking espeak-ng..."
echo "========================================"
if command -v espeak-ng >/dev/null 2>&1; then
    echo "espeak-ng is installed!"
else
    echo "WARNING: espeak-ng not found!"
    echo "Please install espeak-ng using your package manager:"
    echo "  Ubuntu/Debian: sudo apt-get install espeak-ng"
    echo "  macOS: brew install espeak-ng"
    echo "  Arch: sudo pacman -S espeak-ng"
fi

# Verify rubberband
if command -v rubberband >/dev/null 2>&1; then
    echo "rubberband is installed!"
else
    echo "WARNING: rubberband-cli not found!"
fi

# Verify ffmpeg
if command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is installed!"
else
    echo "WARNING: ffmpeg not found!"
fi

# 12/12 Verify installation
echo
echo "[12/12] Verifying Installation..."
echo "========================================"
echo

python - << 'EOF'
import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA Device: CPU-only")
EOF

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
if [ "$IS_MACOS" -eq 1 ]; then
    echo "macOS note: Local GPU engines in this project target NVIDIA/CUDA."
    echo "On Apple Silicon, prefer Pocket TTS, KittenTTS, or Replicate-based engines."
    echo
fi
echo "Next steps:"
echo "  1. If espeak-ng is not installed, install it now"
echo "  2. Run: ./run.sh"
echo "  3. Open browser to: http://localhost:5000"
echo
