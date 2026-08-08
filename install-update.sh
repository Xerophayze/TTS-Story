#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/Xerophayze/TTS-Story.git"
REPO_DIR="TTS-Story"

pull_updates() {
    local config_backup=""
    if [ -f "config.json" ] && [ -n "$(git status --porcelain -- config.json 2>/dev/null)" ]; then
        config_backup="$(mktemp "${TMPDIR:-/tmp}/tts-story-config.XXXXXX.json")"
        cp config.json "$config_backup"
        echo "Preserving local config.json settings during repository update..."
        if ! git restore --source=HEAD --staged --worktree -- config.json; then
            cp "$config_backup" config.json
            rm -f "$config_backup"
            echo "ERROR: Could not prepare tracked config.json for update."
            return 1
        fi
    fi

    local pull_result=0
    git pull --ff-only || pull_result=$?

    if [ -n "$config_backup" ]; then
        cp "$config_backup" config.json
        rm -f "$config_backup"
        echo "Restored local config.json settings."
    fi
    return "$pull_result"
}

echo "========================================"
echo "TTS-Story Linux/macOS Install/Update"
echo "========================================"
echo

# Fix git safe.directory warning
git config --global --add safe.directory "*" 2>/dev/null || true

# Check if already in the repository (running from within the project folder)
if [ -d ".git" ] && [ -f "setup.sh" ]; then
    echo "Running from within TTS-Story repository."
    echo "Updating in-place..."
    pull_updates
    echo
    
    # Fix incomplete venv if exists
    if [ -d "venv" ] && [ ! -f "venv/bin/activate" ]; then
        echo "Removing incomplete virtual environment..."
        rm -rf venv
    fi
    
    echo "Running setup.sh..."
    chmod +x setup.sh
    ./setup.sh
    
    echo
    echo "✅ Update complete."
    exit 0
fi

# Not in repo - clone/update from outside

# Check if git is installed
echo "Checking Git installation..."
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
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y git
    elif command -v apk >/dev/null 2>&1; then
        sudo apk add git
    else
        echo "ERROR: Could not detect package manager to install git."
        exit 1
    fi
    
    if ! command -v git >/dev/null 2>&1; then
        echo "ERROR: Git installation failed."
        exit 1
    fi
fi

echo "Git is installed: $(git --version)"
echo

# setup.sh selects a compatible interpreter before checking venv support.
# Do not probe the default `python3` here: on macOS it may be Python 3.13+
# even when a supported Homebrew or pyenv Python is already installed.
echo "Python selection and venv support will be verified by setup.sh."

# Clone or update repository
echo "Cloning or updating repository..."
if [ -d "$REPO_DIR" ]; then
    if [ -d "$REPO_DIR/.git" ]; then
        echo "Repository found. Pulling latest updates..."
        if [ -O "$REPO_DIR" ]; then
            echo "Repository owned by current user."
        else
            echo "Fixing repository ownership..."
            sudo chown -R $(whoami):$(whoami) "$REPO_DIR" 2>/dev/null || true
        fi
        cd "$REPO_DIR"
        pull_updates
        cd ..
    else
        echo "ERROR: $REPO_DIR exists but is not a Git repository."
        exit 1
    fi
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo
echo "========================================"
echo "Running setup.sh to install dependencies..."
echo "========================================"
echo

if [ -f "$REPO_DIR/setup.sh" ]; then
    cd "$REPO_DIR"
    chmod +x setup.sh
    
    # Remove incomplete venv if exists
    if [ -d "venv" ] && [ ! -f "venv/bin/activate" ]; then
        echo "Removing incomplete virtual environment..."
        rm -rf venv
    fi
    
    ./setup.sh
    cd ..
else
    echo "ERROR: setup.sh not found in $REPO_DIR."
    exit 1
fi

echo
echo "✅ Install/update complete."
