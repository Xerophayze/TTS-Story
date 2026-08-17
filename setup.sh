#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/unix_torch.sh"

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

UPDATE_MODE=0
REPAIR_MODE=0
PINOKIO_MODE="${TTS_STORY_PINOKIO:-0}"
if [ "$PINOKIO_MODE" = "1" ]; then
    echo "Pinokio managed-environment mode enabled (no sudo prompts)."
fi
if [ "${1:-}" = "--update" ] || { [ -z "${1:-}" ] && [ -f ".setup_complete" ]; }; then
    UPDATE_MODE=1
    echo "Fast update mode enabled. Existing environments will be reused when valid."
elif [ "${1:-}" = "--repair" ]; then
    REPAIR_MODE=1
    echo "Repair mode enabled. Setup will perform full dependency reconciliation."
elif [ -n "${1:-}" ]; then
    echo "ERROR: Unknown setup option: ${1}"
    echo "Use --update for a fast update or --repair for full reconciliation."
    exit 1
fi

# PyTorch versions (matching setup.bat)
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
TORCHAUDIO_VERSION="2.6.0"
CHATTERBOX_TTS_VERSION="0.1.6"
POCKET_TTS_VERSION="1.0.3"
BLACKWELL_TORCH_VERSION="2.8.0"
BLACKWELL_TORCHVISION_VERSION="0.23.0"
BLACKWELL_TORCHAUDIO_VERSION="2.8.0"
PLATFORM="$(uname -s)"
ARCHITECTURE="$(uname -m)"
SETUP_PLATFORM_ID="$(printf '%s-%s' "$PLATFORM" "$ARCHITECTURE" | tr '[:upper:]' '[:lower:]')"

# 1/12 Check Python installation
echo
echo "[1/12] Checking Python installation..."
PYTHON_BIN="$(tts_story_find_compatible_python || true)"
if [ -z "$PYTHON_BIN" ]; then
    DEFAULT_PYTHON_VERSION="not found"
    if command -v python3 >/dev/null 2>&1; then
        DEFAULT_PYTHON_VERSION="$(python3 --version 2>&1 || true)"
    fi
    echo "ERROR: Python 3.9 through 3.12 is required. Default python3: $DEFAULT_PYTHON_VERSION"
    echo "Install Python 3.11 with Homebrew (brew install python@3.11), or set:"
    echo "  TTS_STORY_PYTHON=/full/path/to/python3.11 ./setup.sh"
    exit 1
fi
PYTHON_VERSION="$("$PYTHON_BIN" --version 2>&1)"
echo "Using $PYTHON_VERSION at $PYTHON_BIN"

# 1b/12 Check and install git if not present
echo
echo "[1b/12] Checking Git installation..."
if ! command -v git >/dev/null 2>&1; then
    if [ "$PINOKIO_MODE" = "1" ]; then
        echo "ERROR: Git is missing from the Pinokio managed environment."
        echo "Run the Pinokio Install action again so its Conda prerequisites can be repaired."
        exit 1
    else
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
    if [ "$PINOKIO_MODE" = "1" ]; then
        echo "ERROR: Python venv support is missing from the Pinokio managed environment."
        echo "Run the Pinokio Install action again so its Conda environment can be repaired."
        exit 1
    else
        echo "python3-venv not found. Installing..."
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq python3-venv python3-pip
        elif command -v brew >/dev/null 2>&1; then
            brew install python@3.11
            PYTHON_BIN="$(tts_story_find_compatible_python || true)"
        elif command -v pacman >/dev/null 2>&1; then
            sudo pacman -Sy --noconfirm python-pythonz
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y python3.10-venv
        else
            echo "WARNING: Could not detect package manager to install python3-venv."
        fi
    fi
fi
if [ -z "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
    echo "ERROR: venv support is unavailable for a compatible Python interpreter."
    exit 1
fi
echo "Python venv support is available through $PYTHON_BIN."

# 2/12 Create virtual environment
echo
echo "[2/12] Creating virtual environment..."
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    VENV_MM="$(venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [ -z "$VENV_MM" ]; then
        echo "Existing venv Python cannot start. Recreating venv."
        rm -rf venv
        "$PYTHON_BIN" -m venv venv
    elif tts_story_python_supported venv/bin/python; then
        echo "Virtual environment already exists, skipping..."
    else
        echo "Existing venv Python is outside the supported 3.9-3.12 range (detected: ${VENV_MM:-unknown}). Recreating venv."
        rm -rf venv
        "$PYTHON_BIN" -m venv venv
    fi
else
    "$PYTHON_BIN" -m venv venv
fi

# 3/12 Activate virtual environment
echo
echo "[3/12] Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate
if ! tts_story_python_supported python; then
    echo "ERROR: The virtual environment must use Python 3.9 through 3.12."
    echo "Delete the 'venv' folder, install a supported Python version, and rerun setup.sh."
    exit 1
fi

if [ "$UPDATE_MODE" -eq 1 ] && [ "$REPAIR_MODE" -eq 0 ] && [ -f ".setup_complete" ]; then
    if python scripts/setup_state.py matches --platform "$SETUP_PLATFORM_ID" >/dev/null 2>&1; then
        if python -c "import flask, soundfile, torch" >/dev/null 2>&1; then
            echo
            echo "Dependency definitions are unchanged and the existing environment passed its health check."
            echo "No setup work is required for this update."
            echo
            echo "========================================"
            echo "Setup Complete!"
            echo "========================================"
            exit 0
        fi
        echo "Existing environment failed the fast health check. Running dependency reconciliation."
    else
        echo "Dependency definitions changed or setup state is unavailable. Running dependency reconciliation."
    fi
fi
rm -f .setup_complete

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
    if [ "$PLATFORM" = "Darwin" ]; then
        echo "macOS ${ARCHITECTURE} detected. Installing native macOS PyTorch wheels."
    else
        echo "No NVIDIA GPU detected. Using CPU-only installs."
    fi
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
        if [ "$PLATFORM" = "Darwin" ]; then
            echo "Installing native macOS PyTorch..."
        else
            echo "Installing CPU-only PyTorch..."
        fi
        python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        tts_story_install_cpu_torch python \
            "torch==${TORCH_VERSION}" \
            "torchvision==${TORCHVISION_VERSION}" \
            "torchaudio==${TORCHAUDIO_VERSION}"
        pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
    fi
fi

# 6/12 Install other dependencies
echo
echo "[6/12] Installing other Python dependencies..."
MAIN_DEPS_STAMP="venv/.main_deps_stamp"
MAIN_DEPS_SIGNATURE="$(python - <<'PY'
import hashlib
from pathlib import Path
payload = Path("requirements.txt").read_bytes() + b"|main-deps-v1"
print(hashlib.sha256(payload).hexdigest())
PY
)"
SKIP_MAIN_DEPS=0
if [ "$UPDATE_MODE" -eq 1 ] && [ -f "$MAIN_DEPS_STAMP" ] && [ "$(cat "$MAIN_DEPS_STAMP")" = "$MAIN_DEPS_SIGNATURE" ]; then
    SKIP_MAIN_DEPS=1
fi
if [ "$SKIP_MAIN_DEPS" -eq 1 ]; then
    echo "Main Python dependencies are unchanged. Skipping dependency reinstall for update."
else
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
echo "$MAIN_DEPS_SIGNATURE" > "$MAIN_DEPS_STAMP"
fi

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
# Core dependencies are managed by requirements.txt/PyTorch setup. Avoid
# allowing chatterbox-tts to silently replace shared engine dependencies.
if pip install "chatterbox-tts==${CHATTERBOX_TTS_VERSION}" --no-deps; then
    if python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; import importlib.metadata as m; print('Chatterbox Turbo import OK; package version:', m.version('chatterbox-tts'))"; then
        echo "Chatterbox Turbo runtime validated!"
    else
        echo "WARNING: chatterbox-tts installed but its runtime import failed."
        echo "The application health endpoint will report the underlying dependency error."
    fi
else
    echo "WARNING: Failed to install chatterbox-tts==${CHATTERBOX_TTS_VERSION}."
    echo "Chatterbox Turbo will be unavailable, but setup will continue for other engines."
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
if pip install "pocket-tts==${POCKET_TTS_VERSION}"; then
    if python -c "from pocket_tts import TTSModel; import importlib.metadata as m; print('Pocket TTS import OK; package version:', m.version('pocket-tts'))"; then
        echo "Pocket TTS runtime validated!"
    else
        echo "WARNING: Pocket TTS installed but its runtime import failed"
    fi
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

echo
echo "Checking optional FlashAttention 2 acceleration for Qwen3-TTS..."
if [ "${INSTALL_FLASH_ATTN:-1}" = "0" ]; then
    echo "INSTALL_FLASH_ATTN=0 set. Skipping FlashAttention installation."
else
    python scripts/flash_attention_setup.py install || \
        echo "WARNING: FlashAttention setup failed. Qwen3 will use PyTorch SDPA or eager attention."
fi

# 9a/12 Install KittenTTS runtime (optional, CPU-only)
echo
echo "[9a/12] Installing KittenTTS runtime (optional, CPU-only)..."
if pip install https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl; then
    if [ "${PREFETCH_KITTEN_TTS_MODEL:-1}" != "0" ]; then
        echo "Prefetching KittenTTS model cache (set PREFETCH_KITTEN_TTS_MODEL=0 to skip)..."
        if command -v timeout >/dev/null 2>&1; then
            timeout 300 python scripts/prefetch_kitten_tts.py --model-id "KittenML/kitten-tts-mini-0.8" || echo "WARNING: KittenTTS model prefetch failed. First generation will retry the download."
        else
            python scripts/prefetch_kitten_tts.py --model-id "KittenML/kitten-tts-mini-0.8" || echo "WARNING: KittenTTS model prefetch failed. First generation will retry the download."
        fi
    else
        echo "PREFETCH_KITTEN_TTS_MODEL=0 set. Skipping KittenTTS model prefetch."
    fi
else
    echo "WARNING: Failed to install kittentts - KittenTTS engine will not be available"
fi

# 9b/12 Setup OmniVoice isolated environment
echo
echo "[9b/12] Setting up OmniVoice isolated environment..."
echo "OmniVoice requires torch 2.8, so it runs in its own isolated venv."
OMNIVOICE_DIR="$(pwd)/engines/omnivoice"
OMNIVOICE_PYTHON="$OMNIVOICE_DIR/.venv/bin/python"
if [ ! -x "$OMNIVOICE_PYTHON" ] && [ -x "$OMNIVOICE_DIR/.venv/bin/python3" ]; then
    OMNIVOICE_PYTHON="$OMNIVOICE_DIR/.venv/bin/python3"
fi
OMNIVOICE_READY=0
if [ -f "$OMNIVOICE_DIR/.omnivoice_ready" ]; then
    if [ -f "$OMNIVOICE_DIR/omnivoice_worker.py" ] && { [ -x "$OMNIVOICE_DIR/.venv/bin/python" ] || [ -x "$OMNIVOICE_DIR/.venv/bin/python3" ]; }; then
        if "$OMNIVOICE_PYTHON" -c "import omnivoice, torch, torchaudio, soundfile, huggingface_hub" >/dev/null 2>&1; then
            OMNIVOICE_READY=1
        else
            echo "OmniVoice setup marker is stale or incomplete. Repairing OmniVoice setup..."
            rm -f "$OMNIVOICE_DIR/.omnivoice_ready"
        fi
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
                echo "Installing matched OmniVoice torch/torchaudio CUDA 12.8 builds for GPU acceleration..."
                if ! "$OMNIVOICE_PYTHON" -m pip install --upgrade --force-reinstall --no-deps "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128; then
                    echo "WARNING: CUDA torch/torchaudio install failed. Trying matched CPU builds."
                    tts_story_install_cpu_torch "$OMNIVOICE_PYTHON" --no-deps "torch==2.8.0" "torchaudio==2.8.0" || \
                        echo "WARNING: CPU torch/torchaudio install failed. OmniVoice will not be available."
                fi
            else
                echo "Installing matched OmniVoice torch/torchaudio builds for this platform..."
                tts_story_install_cpu_torch "$OMNIVOICE_PYTHON" --no-deps "torch==2.8.0" "torchaudio==2.8.0" || \
                    echo "WARNING: CPU torch/torchaudio install failed. OmniVoice will not be available."
            fi
            echo "Installing OmniVoice helper packages..."
            if "$OMNIVOICE_PYTHON" -m pip install soundfile huggingface-hub; then
                echo "Verifying OmniVoice isolated environment..."
                if "$OMNIVOICE_PYTHON" -c "import omnivoice, torch, torchaudio, soundfile, huggingface_hub; print('OmniVoice torch:', torch.__version__, 'torchaudio:', torchaudio.__version__)"; then
                    if [ "${PREFETCH_OMNIVOICE_MODEL:-1}" != "0" ]; then
                        echo "Prefetching OmniVoice model cache (set PREFETCH_OMNIVOICE_MODEL=0 to skip)..."
                        "$OMNIVOICE_PYTHON" "$OMNIVOICE_DIR/omnivoice_worker.py" --prefetch-model --model-id "k2-fsa/OmniVoice" || \
                            echo "WARNING: OmniVoice model prefetch failed. First generation will retry the download."
                    else
                        echo "PREFETCH_OMNIVOICE_MODEL=0 set. Skipping OmniVoice model prefetch."
                    fi
                    touch "$OMNIVOICE_DIR/.omnivoice_ready"
                    echo "OmniVoice isolated environment ready."
                else
                    echo "WARNING: OmniVoice verification failed. OmniVoice will not be available."
                    rm -f "$OMNIVOICE_DIR/.omnivoice_ready"
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

# 9c/12 Setup IndexTTS isolated environment (optional)
echo
echo "[9c/12] Setting up IndexTTS isolated environment (optional)..."
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

echo
echo "[9d/12] Setting up Dot.TTS isolated environment (optional)..."
echo "Dot.TTS pins newer torch/transformers versions, so it runs in its own venv."
DOTS_TTS_DIR="$(pwd)/engines/dots-tts"
DOTS_TTS_REPO="$DOTS_TTS_DIR/repo"
DOTS_TTS_PYTHON="$DOTS_TTS_DIR/.venv/bin/python"
if [ ! -x "$DOTS_TTS_PYTHON" ] && [ -x "$DOTS_TTS_DIR/.venv/bin/python3" ]; then
    DOTS_TTS_PYTHON="$DOTS_TTS_DIR/.venv/bin/python3"
fi
DOTS_TTS_READY=0
DOTS_TTS_REPO_UPDATED=0
if [ -d "$DOTS_TTS_REPO/.git" ] && command -v git >/dev/null 2>&1; then
    DOTS_TTS_REPO_BEFORE="$(git -C "$DOTS_TTS_REPO" rev-parse HEAD 2>/dev/null || true)"
    echo "Checking Dot.TTS upstream updates..."
    git -C "$DOTS_TTS_REPO" pull --ff-only >/dev/null 2>&1 || true
    DOTS_TTS_REPO_AFTER="$(git -C "$DOTS_TTS_REPO" rev-parse HEAD 2>/dev/null || true)"
    if [ -n "$DOTS_TTS_REPO_BEFORE" ] && [ -n "$DOTS_TTS_REPO_AFTER" ] && [ "$DOTS_TTS_REPO_BEFORE" != "$DOTS_TTS_REPO_AFTER" ]; then
        DOTS_TTS_REPO_UPDATED=1
        echo "Dot.TTS upstream updated. Refreshing editable install."
        rm -f "$DOTS_TTS_DIR/.dots_tts_ready"
    fi
fi
if [ -f "$DOTS_TTS_DIR/.dots_tts_ready" ]; then
    if [ -f "$DOTS_TTS_DIR/dots_tts_worker.py" ] && [ -f "$DOTS_TTS_REPO/pyproject.toml" ] && [ -x "$DOTS_TTS_PYTHON" ]; then
        if "$DOTS_TTS_PYTHON" "$DOTS_TTS_DIR/dots_tts_worker.py" --check-env >/dev/null 2>&1; then
            DOTS_TTS_READY=1
        else
            echo "Dot.TTS setup marker is stale or incomplete. Repairing Dot.TTS setup..."
            rm -f "$DOTS_TTS_DIR/.dots_tts_ready"
        fi
    else
        echo "Dot.TTS setup marker is stale or incomplete. Repairing Dot.TTS setup..."
        rm -f "$DOTS_TTS_DIR/.dots_tts_ready"
    fi
fi

if [ "$DOTS_TTS_READY" -eq 1 ]; then
    echo "Dot.TTS isolated environment already set up. Skipping."
elif ! command -v git >/dev/null 2>&1; then
    echo "WARNING: git not found. Skipping Dot.TTS setup."
    echo "To install Dot.TTS manually:"
    echo "  git clone https://github.com/rednote-hilab/dots.tts.git engines/dots-tts/repo"
    echo "  python -m venv engines/dots-tts/.venv"
    echo "  engines/dots-tts/.venv/bin/python -m pip install -c engines/dots-tts/repo/constraints/recommended.txt torch torchaudio transformers huggingface-hub loguru 'langcodes[data]' einops librosa soundfile numpy pydantic PyYAML safetensors torchdiffeq tqdm lingua-language-detector"
    echo "  engines/dots-tts/.venv/bin/python -m pip install -e engines/dots-tts/repo --no-deps"
else
    mkdir -p "$DOTS_TTS_DIR"
    if [ ! -f "$DOTS_TTS_REPO/pyproject.toml" ]; then
        echo "Cloning Dot.TTS repository..."
        git clone https://github.com/rednote-hilab/dots.tts.git "$DOTS_TTS_REPO" || echo "WARNING: Failed to clone Dot.TTS."
    else
        echo "Dot.TTS already cloned. Pulling latest changes..."
        git -C "$DOTS_TTS_REPO" pull --ff-only >/dev/null 2>&1 || true
    fi

    if [ -f "$DOTS_TTS_REPO/pyproject.toml" ]; then
        if [ ! -x "$DOTS_TTS_PYTHON" ]; then
            echo "Creating Dot.TTS isolated virtual environment..."
            python -m venv "$DOTS_TTS_DIR/.venv" || echo "WARNING: Failed to create Dot.TTS venv."
            DOTS_TTS_PYTHON="$DOTS_TTS_DIR/.venv/bin/python"
            if [ ! -x "$DOTS_TTS_PYTHON" ] && [ -x "$DOTS_TTS_DIR/.venv/bin/python3" ]; then
                DOTS_TTS_PYTHON="$DOTS_TTS_DIR/.venv/bin/python3"
            fi
        fi

        if [ -x "$DOTS_TTS_PYTHON" ]; then
            echo "Installing Dot.TTS dependencies (this may take several minutes)..."
            "$DOTS_TTS_PYTHON" -m pip install --upgrade pip
            DOTS_TTS_TORCH_READY=0
            if [ "${DOTS_TTS_CUDA:-1}" != "0" ] && command -v nvidia-smi >/dev/null 2>&1; then
                echo "Installing Dot.TTS CUDA torch/torchaudio builds..."
                if "$DOTS_TTS_PYTHON" -m pip install --upgrade --force-reinstall --no-deps \
                    "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" \
                    --index-url https://download.pytorch.org/whl/cu128; then
                    DOTS_TTS_TORCH_READY=1
                fi
            fi
            if [ "$DOTS_TTS_TORCH_READY" -eq 0 ]; then
                echo "Installing Dot.TTS torch/torchaudio builds for this platform..."
                tts_story_install_cpu_torch "$DOTS_TTS_PYTHON" --no-deps \
                    "torch==2.8.0" "torchaudio==2.8.0" || true
            fi
            DOTS_TTS_RUNTIME_DEPS=(
                transformers huggingface-hub loguru "langcodes[data]"
                einops librosa soundfile numpy pydantic PyYAML safetensors
                torchdiffeq tqdm lingua-language-detector
            )
            if "$DOTS_TTS_PYTHON" -m pip install -c "$DOTS_TTS_REPO/constraints/recommended.txt" "${DOTS_TTS_RUNTIME_DEPS[@]}" \
                && "$DOTS_TTS_PYTHON" -m pip install -e "$DOTS_TTS_REPO" --no-deps; then
                echo "Verifying Dot.TTS isolated environment..."
                if "$DOTS_TTS_PYTHON" "$DOTS_TTS_DIR/dots_tts_worker.py" --check-env; then
                    if [ "${PREFETCH_DOTS_TTS_MODEL:-0}" = "1" ]; then
                        echo "Prefetching Dot.TTS model cache (set PREFETCH_DOTS_TTS_MODEL=0 to skip)..."
                        "$DOTS_TTS_PYTHON" "$DOTS_TTS_DIR/dots_tts_worker.py" --prefetch-model --model-id "rednote-hilab/dots.tts-soar" || \
                            echo "WARNING: Dot.TTS model prefetch failed. First generation will retry the download."
                    else
                        echo "Dot.TTS model prefetch is opt-in. Set PREFETCH_DOTS_TTS_MODEL=1 to download during setup."
                    fi
                    touch "$DOTS_TTS_DIR/.dots_tts_ready"
                    echo "Dot.TTS isolated environment ready."
                else
                    echo "WARNING: Dot.TTS verification failed. Dot.TTS will not be available."
                fi
            else
                echo "WARNING: Dot.TTS dependency install failed. Dot.TTS will not be available."
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

if [ "$PINOKIO_MODE" = "1" ]; then
    echo "Using system tools from Pinokio's managed Conda environment."
elif command -v apt-get >/dev/null 2>&1; then
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

python scripts/flash_attention_setup.py diagnose

python scripts/setup_state.py write --platform "$SETUP_PLATFORM_ID"
touch .setup_complete
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
