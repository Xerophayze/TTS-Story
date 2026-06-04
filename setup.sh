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

# PyTorch versions (matching setup.bat)
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
TORCHAUDIO_VERSION="2.6.0"
BLACKWELL_TORCH_VERSION="2.8.0"
BLACKWELL_TORCHVISION_VERSION="0.23.0"
BLACKWELL_TORCHAUDIO_VERSION="2.8.0"

# 1/12 Check Python installation
echo
echo "[1/12] Checking Python installation..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is not installed or not in PATH"
    echo "Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "Found $PYTHON_VERSION"

# Check Python version (3.9+ required)
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "ERROR: Python 3.9 or higher is required. Found: $PYTHON_VERSION"
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
if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "python3-venv not found. Installing..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3-venv python3-pip
    elif command -v brew >/dev/null 2>&1; then
        brew install python@3.10
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
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
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
GPU_COMPUTE_CAP=""
NEEDS_BLACKWELL_TORCH=0
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [ -n "$GPU_NAME" ]; then
        HAS_NVIDIA=1
    fi
fi

if [ "$HAS_NVIDIA" -eq 0 ]; then
    echo "No NVIDIA GPU detected. Using CPU-only installs."
else
    echo "NVIDIA GPU detected: $GPU_NAME"
    if [ -n "$GPU_COMPUTE_CAP" ]; then
        echo "NVIDIA compute capability: $GPU_COMPUTE_CAP"
    fi
    if [[ "$GPU_COMPUTE_CAP" == 12.* ]] || echo "$GPU_NAME" | grep -Eiq "RTX 50|Blackwell"; then
        NEEDS_BLACKWELL_TORCH=1
        echo "Blackwell GPU detected. PyTorch CUDA 12.8 build with sm_120 support is required."
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
        if [ "$NEEDS_BLACKWELL_TORCH" -eq 1 ]; then
            echo "Checking for Blackwell sm_120 support..."
            if python scripts/torch_cuda_probe.py --require-arch sm_120 --test-cuda >/dev/null 2>&1; then
                NEED_TORCH_INSTALL=0
            else
                echo "Existing CUDA torch does not support this GPU. Reinstalling PyTorch CUDA 12.8 build."
            fi
        elif python scripts/torch_cuda_probe.py --test-cuda >/dev/null 2>&1; then
            NEED_TORCH_INSTALL=0
        else
            echo "Existing CUDA torch failed a runtime test. Reinstalling PyTorch."
        fi
    elif [ "$HAS_NVIDIA" -eq 0 ] && [ "$TORCH_CUDA" = "cpu" ]; then
        echo "Detected existing CPU torch: $TORCH_INSTALLED"
        NEED_TORCH_INSTALL=0
    fi
fi

if [ "$NEED_TORCH_INSTALL" -eq 0 ]; then
    echo "Skipping torch install - compatible build already present."
else
    if [ "$HAS_NVIDIA" -eq 1 ]; then
        if [ "$NEEDS_BLACKWELL_TORCH" -eq 1 ]; then
            if [ "${USE_TORCH_NIGHTLY:-0}" = "1" ]; then
                echo "Installing PyTorch nightly with CUDA 12.8 support for Blackwell..."
                if ! pip install --pre --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128; then
                    echo "ERROR: PyTorch nightly CUDA 12.8 installation failed."
                    exit 1
                fi
            else
                echo "Installing PyTorch CUDA 12.8 support for Blackwell..."
                if ! pip install --upgrade --force-reinstall \
                    torch==${BLACKWELL_TORCH_VERSION} \
                    torchvision==${BLACKWELL_TORCHVISION_VERSION} \
                    torchaudio==${BLACKWELL_TORCHAUDIO_VERSION} \
                    --index-url https://download.pytorch.org/whl/cu128; then
                    echo "ERROR: PyTorch CUDA 12.8 installation failed."
                    echo "Set USE_TORCH_NIGHTLY=1 and rerun setup.sh if stable wheels do not support your GPU yet."
                    exit 1
                fi
            fi
            if ! python scripts/torch_cuda_probe.py --require-arch sm_120 --test-cuda; then
                echo "ERROR: Installed PyTorch still cannot run on this Blackwell GPU."
                echo "Try updating NVIDIA drivers, then rerun setup.sh. As a fallback, set USE_TORCH_NIGHTLY=1."
                exit 1
            fi
        else
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
        fi
    else
        echo "Installing CPU-only PyTorch..."
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        pip install --upgrade --force-reinstall torch==${TORCH_VERSION}+cpu torchvision==${TORCHVISION_VERSION}+cpu torchaudio==${TORCHAUDIO_VERSION}+cpu --index-url https://download.pytorch.org/whl/cpu
        pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
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
if command -v make >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    echo "Build tools found. Installing pyopenjtalk..."
    pip install pyopenjtalk || echo "WARNING: pyopenjtalk failed to install. Japanese TTS features will be unavailable."
else
    echo "WARNING: Build tools not found. Skipping pyopenjtalk."
    echo "To enable Japanese TTS, install build tools and rerun setup.sh."
fi

# 7/12 Install Chatterbox Turbo runtime
echo
echo "[7/12] Installing Chatterbox Turbo runtime..."
# First try with deps to get torchaudio and other required packages
if pip install chatterbox-tts; then
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
if [ "$HAS_NVIDIA" -eq 0 ]; then
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

# 9a/12 Setup OmniVoice isolated environment
echo
echo "[9a/12] Setting up OmniVoice isolated environment..."
echo "OmniVoice requires torch 2.8, so it runs in its own isolated venv."
OMNIVOICE_DIR="$(pwd)/engines/omnivoice"
OMNIVOICE_PYTHON="$OMNIVOICE_DIR/.venv/bin/python"
if [ ! -x "$OMNIVOICE_PYTHON" ] && [ -x "$OMNIVOICE_DIR/.venv/bin/python3" ]; then
    OMNIVOICE_PYTHON="$OMNIVOICE_DIR/.venv/bin/python3"
fi
OMNIVOICE_READY=0
if [ -f "$OMNIVOICE_DIR/.omnivoice_ready" ]; then
    if [ -f "$OMNIVOICE_DIR/omnivoice_worker.py" ] && { [ -x "$OMNIVOICE_DIR/.venv/bin/python" ] || [ -x "$OMNIVOICE_DIR/.venv/bin/python3" ]; }; then
        OMNIVOICE_READY=1
    else
        echo "OmniVoice setup marker is stale or incomplete. Repairing OmniVoice setup..."
        rm -f "$OMNIVOICE_DIR/.omnivoice_ready"
    fi
fi

if [ "$OMNIVOICE_READY" -eq 1 ]; then
    echo "OmniVoice isolated environment already set up. Skipping."
else
    mkdir -p "$OMNIVOICE_DIR"
    echo "Creating OmniVoice isolated virtual environment..."
    if python -m venv "$OMNIVOICE_DIR/.venv"; then
        OMNIVOICE_PYTHON="$OMNIVOICE_DIR/.venv/bin/python"
        if [ ! -x "$OMNIVOICE_PYTHON" ]; then
            OMNIVOICE_PYTHON="$OMNIVOICE_DIR/.venv/bin/python3"
        fi
        echo "Installing omnivoice package..."
        if "$OMNIVOICE_PYTHON" -m pip install omnivoice; then
            if [ "$HAS_NVIDIA" -eq 1 ]; then
                echo "Upgrading OmniVoice torch to CUDA 12.8 build for GPU acceleration..."
                "$OMNIVOICE_PYTHON" -m pip install "torch==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128 || \
                    echo "WARNING: CUDA torch install failed, OmniVoice will run on CPU."
            fi
            echo "Installing OmniVoice helper packages..."
            if "$OMNIVOICE_PYTHON" -m pip install soundfile huggingface-hub; then
                echo "Verifying OmniVoice isolated environment..."
                if "$OMNIVOICE_PYTHON" -c "import omnivoice, soundfile, huggingface_hub"; then
                    touch "$OMNIVOICE_DIR/.omnivoice_ready"
                    echo "OmniVoice isolated environment ready."
                else
                    echo "WARNING: OmniVoice verification failed. OmniVoice will not be available."
                fi
            else
                echo "WARNING: Failed to install OmniVoice helper packages. OmniVoice will not be available."
            fi
        else
            echo "WARNING: Failed to install omnivoice in isolated venv. OmniVoice will not be available."
        fi
    else
        echo "WARNING: Failed to create OmniVoice venv. OmniVoice will not be available."
    fi
fi

# 9b/12 Setup IndexTTS isolated environment (optional)
echo
echo "[9b/12] Setting up IndexTTS isolated environment (optional)..."
echo "IndexTTS uses its own isolated venv to avoid dependency conflicts."
INDEX_TTS_DIR="$(pwd)/engines/index-tts"
INDEX_TTS_READY=0
if [ -f "$INDEX_TTS_DIR/.indextts_ready" ]; then
    if [ -f "$INDEX_TTS_DIR/tts_worker.py" ] && [ -f "$INDEX_TTS_DIR/pyproject.toml" ] && { [ -x "$INDEX_TTS_DIR/.venv/bin/python" ] || [ -x "$INDEX_TTS_DIR/.venv/bin/python3" ]; }; then
        INDEX_TTS_READY=1
    else
        echo "IndexTTS setup marker is stale or incomplete. Repairing IndexTTS setup..."
        rm -f "$INDEX_TTS_DIR/.indextts_ready"
    fi
fi

if [ "$INDEX_TTS_READY" -eq 1 ]; then
    echo "IndexTTS already set up. Skipping clone and sync."
else
    if ! command -v git >/dev/null 2>&1; then
        echo "WARNING: git not found. Skipping IndexTTS setup."
        echo "To install IndexTTS manually:"
        echo "  git clone https://github.com/index-tts/index-tts.git engines/index-tts"
        echo "  cd engines/index-tts && uv sync"
    else
        if command -v uv >/dev/null 2>&1; then
            UV_CMD=(uv)
        elif python -m uv --version >/dev/null 2>&1; then
            UV_CMD=(python -m uv)
        else
            echo "uv not found. Installing uv package manager..."
            if pip install -U uv --quiet; then
                UV_CMD=(python -m uv)
            else
                echo "WARNING: Failed to install uv. Skipping IndexTTS setup."
                UV_CMD=()
            fi
        fi

        if [ "${#UV_CMD[@]}" -gt 0 ]; then
            if [ ! -f "$INDEX_TTS_DIR/pyproject.toml" ]; then
                echo "Cloning IndexTTS repository (skipping LFS audio examples)..."
                INDEX_TTS_CLONE_TMP="$(mktemp -d)"
                GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/index-tts/index-tts.git "$INDEX_TTS_CLONE_TMP"
                if [ ! -f "$INDEX_TTS_CLONE_TMP/pyproject.toml" ]; then
                    echo "WARNING: Failed to clone IndexTTS (pyproject.toml missing). Skipping IndexTTS setup."
                else
                    mkdir -p "$INDEX_TTS_DIR"
                    cp -a "$INDEX_TTS_CLONE_TMP"/. "$INDEX_TTS_DIR"/
                    echo "IndexTTS cloned successfully."
                fi
                rm -rf "$INDEX_TTS_CLONE_TMP"
            else
                echo "IndexTTS already cloned. Pulling latest changes..."
                GIT_LFS_SKIP_SMUDGE=1 git -C "$INDEX_TTS_DIR" pull --ff-only >/dev/null 2>&1 || true
            fi

            if [ ! -f "$INDEX_TTS_DIR/tts_worker.py" ]; then
                echo "WARNING: IndexTTS worker file is missing: $INDEX_TTS_DIR/tts_worker.py"
                echo "Run git pull from the TTS-Story repository, then rerun setup.sh."
            elif [ -f "$INDEX_TTS_DIR/pyproject.toml" ]; then
                echo "Installing IndexTTS dependencies (this may take several minutes)..."
                if (cd "$INDEX_TTS_DIR" && "${UV_CMD[@]}" sync); then
                    touch "$INDEX_TTS_DIR/.indextts_ready"
                    echo "IndexTTS environment ready."
                    echo "Model weights will be downloaded automatically on first use (~2-4 GB)."
                else
                    echo "WARNING: IndexTTS dependency install failed."
                    echo "Try manually: cd engines/index-tts && uv sync"
                fi
            fi
        fi
    fi
fi

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

if [ "$NEEDS_BLACKWELL_TORCH" -eq 1 ]; then
    python scripts/torch_cuda_probe.py --require-arch sm_120 --test-cuda
elif [ "$HAS_NVIDIA" -eq 1 ]; then
    python scripts/torch_cuda_probe.py --test-cuda
else
    python scripts/torch_cuda_probe.py
fi

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "  1. If espeak-ng is not installed, install it now"
echo "  2. Run: ./run.sh"
echo "  3. Open browser to: http://localhost:5000"
echo
