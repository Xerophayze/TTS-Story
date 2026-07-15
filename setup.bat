@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
echo ========================================
echo TTS-Story Setup
echo ========================================
echo.
echo IMPORTANT: Initial setup can take several minutes (large downloads + builds).
echo Please be patient and report any errors you encounter.
echo.
echo Quick Troubleshooting:
echo  - If setup fails, delete the "venv" folder and re-run install-update.bat
echo  - GPU users: update to the latest NVIDIA drivers
echo.

set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
set "PYTHON_INSTALLER=%TEMP%\python-installer.exe"
set "TORCH_VERSION=2.6.0"
set "TORCHVISION_VERSION=0.21.0"
set "TORCHAUDIO_VERSION=2.6.0"
set "CHATTERBOX_TTS_VERSION=0.1.6"
set "POCKET_TTS_VERSION=1.0.3"
set "BLACKWELL_TORCH_VERSION=2.8.0"
set "BLACKWELL_TORCHVISION_VERSION=0.23.0"
set "BLACKWELL_TORCHAUDIO_VERSION=2.8.0"
set "UPDATE_MODE=0"
if /i "%~1"=="--update" set "UPDATE_MODE=1"
if "%UPDATE_MODE%"=="1" echo Pinokio update mode enabled. Existing environments will be reused when valid.

REM Check Python installation. Prefer an existing project venv during updates,
REM because Pinokio may invoke setup.bat from a Conda base Python.
echo [1/12] Checking Python installation...
set "PYTHON_CMD=python"
set "USE_EXISTING_VENV=0"
set "VENV_PYTHON=venv\Scripts\python.exe"
set "VENV_PY_MM="
set "VENV_CFG_MM="
set "VENV_CFG_VERSION="
if exist "%VENV_PYTHON%" (
    for /f "delims=" %%V in ('"%VENV_PYTHON%" -c "import sys; print(str(sys.version_info[0])+chr(46)+str(sys.version_info[1]))" 2^>nul') do set "VENV_PY_MM=%%V"
    if exist "venv\pyvenv.cfg" (
        for /f "tokens=2 delims==" %%V in ('findstr /B /C:"version" "venv\pyvenv.cfg" 2^>nul') do set "VENV_CFG_VERSION=%%V"
        set "VENV_CFG_VERSION=!VENV_CFG_VERSION: =!"
        set "VENV_CFG_MM=!VENV_CFG_VERSION:~0,4!"
    )
    if "!VENV_PY_MM!"=="3.11" (
        set "USE_EXISTING_VENV=1"
        echo Found existing project venv Python 3.11. Reusing it for update.
        goto :PythonReady
    ) else if defined VENV_PY_MM (
        echo Existing venv Python is not 3.11 ^(detected: !VENV_PY_MM!^). Recreating venv.
        rmdir /s /q venv >nul 2>&1
    ) else (
        echo Existing venv Python cannot start ^(pyvenv.cfg: !VENV_CFG_MM!^). Recreating venv.
        rmdir /s /q venv >nul 2>&1
    )
)

set "PYTHON_VERSION="
set "PYTHON_MM="
python --version >nul 2>&1
if errorlevel 1 (
    set "PYTHON_VERSION="
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
    for /f "delims=" %%i in ('python -c "import sys; print(str(sys.version_info[0])+chr(46)+str(sys.version_info[1]))" 2^>nul') do set "PYTHON_MM=%%i"
)

if "%PYTHON_MM%"=="3.11" goto :FoundPythonCommand

py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    for /f "tokens=2" %%i in ('py -3.11 --version 2^>^&1') do set "PYTHON_VERSION=%%i"
    goto :FoundPythonCommand
)

call :FindPython311Dir
if defined PY_DIR (
    set "PATH=%PY_DIR%;%PY_DIR%\Scripts;%PATH%"
    goto :FoundPythonCommand
)

if "%PYTHON_VERSION%"=="" (
    echo Python not found. Installing Python 3.11.9...
) else (
    echo Detected Python %PYTHON_VERSION%. Installing required Python 3.11.9...
)
powershell -NoLogo -NoProfile -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $url='%PYTHON_INSTALLER_URL%'; try { Invoke-WebRequest -Uri $url -OutFile '%PYTHON_INSTALLER%' -UseBasicParsing -ErrorAction Stop } catch { try { Start-BitsTransfer -Source $url -Destination '%PYTHON_INSTALLER%' -ErrorAction Stop } catch { Write-Error $_.Exception.Message; exit 1 } }"
if errorlevel 1 (
    echo ERROR: Failed to download Python installer.
    echo If this is a Pinokio update and venv already exists, delete only an incomplete venv and rerun Update.
    pause
    exit /b 1
)
"%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=0 Include_test=0
if errorlevel 1 (
    echo ERROR: Python installer failed.
    pause
    exit /b 1
)
call :FindPython311Dir
if not defined PY_DIR (
    echo ERROR: Python 3.11 installed but install path not found.
    echo Please restart your terminal or install Python 3.11 manually.
    pause
    exit /b 1
)
set "PATH=%PY_DIR%;%PY_DIR%\Scripts;%PATH%"

:FoundPythonCommand
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11 installed but still not found.
    echo Please restart your terminal and rerun setup.bat.
    pause
    exit /b 1
)

set "PYTHON_MM="
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set "PYTHON_VERSION=%%i"
for /f "delims=" %%i in ('%PYTHON_CMD% -c "import sys; print(str(sys.version_info[0])+chr(46)+str(sys.version_info[1]))" 2^>nul') do set "PYTHON_MM=%%i"
if not "%PYTHON_MM%"=="3.11" (
    echo ERROR: Python 3.11 is required. Current: %PYTHON_VERSION%
    echo Please ensure Python 3.11 is installed and rerun setup.bat.
    pause
    exit /b 1
)
echo Found Python %PYTHON_VERSION%
:PythonReady

set "HAS_NVIDIA=0"
set "GPU_NAME="
set "GPU_COMPUTE_CAP="
set "NEEDS_BLACKWELL_TORCH=0"
for /f "delims=" %%G in ('powershell -NoLogo -NoProfile -Command "$g=(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1).Name; if ($g) { $g }"') do set "GPU_NAME=%%G"
if defined GPU_NAME (
    set "HAS_NVIDIA=1"
    echo NVIDIA GPU detected: !GPU_NAME!
    for /f "tokens=1 delims=," %%C in ('nvidia-smi --query-gpu^=compute_cap --format^=csv 2^>nul ^| findstr /R "^[0-9]"') do (
        if not defined GPU_COMPUTE_CAP set "GPU_COMPUTE_CAP=%%C"
    )
    if defined GPU_COMPUTE_CAP echo NVIDIA compute capability: !GPU_COMPUTE_CAP!
    if "!GPU_COMPUTE_CAP:~0,3!"=="12." set "NEEDS_BLACKWELL_TORCH=1"
    powershell -NoLogo -NoProfile -Command "if ('!GPU_NAME!' -match 'RTX 50|Blackwell') { exit 0 } else { exit 1 }" >nul 2>&1
    if not errorlevel 1 set "NEEDS_BLACKWELL_TORCH=1"
    if "!NEEDS_BLACKWELL_TORCH!"=="1" (
        echo Blackwell GPU detected. PyTorch CUDA 12.8 build with sm_120 support is required.
    )
) else (
    echo No NVIDIA GPU detected. Using CPU-only installs.
)

REM Create virtual environment
echo.
echo [2/12] Creating virtual environment...
if "%USE_EXISTING_VENV%"=="1" (
    echo Virtual environment already exists, skipping...
) else if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo.
echo [3/12] Activating virtual environment...
call venv\Scripts\activate.bat
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Virtual environment did not activate with Python 3.11.
    echo Delete the "venv" folder and rerun setup.bat if this environment is corrupted.
    exit /b 1
)

REM Upgrade pip
echo.
echo [4/12] Upgrading pip...
python -m pip install --upgrade pip --quiet

set "SKIP_MAIN_DEPS=0"
set "MAIN_DEPS_STAMP=venv\.main_deps_stamp"
set "MAIN_DEPS_SIGNATURE="
for /f "delims=" %%H in ('powershell -NoLogo -NoProfile -Command "$h=(Get-FileHash 'requirements.txt' -Algorithm SHA256).Hash; Write-Output ($h + ';main-deps-v2;torch=%TORCH_VERSION%;torchvision=%TORCHVISION_VERSION%;torchaudio=%TORCHAUDIO_VERSION%;blackwell=%BLACKWELL_TORCH_VERSION%')" 2^>nul') do set "MAIN_DEPS_SIGNATURE=%%H"
if "%UPDATE_MODE%"=="1" (
    if exist "%MAIN_DEPS_STAMP%" (
        set /p EXISTING_MAIN_DEPS_SIGNATURE=<"%MAIN_DEPS_STAMP%"
        if defined MAIN_DEPS_SIGNATURE (
            if "!EXISTING_MAIN_DEPS_SIGNATURE!"=="!MAIN_DEPS_SIGNATURE!" (
                set "SKIP_MAIN_DEPS=1"
            )
        )
    )
)

REM Install PyTorch (let pip/PyTorch auto-detect CUDA)
echo.
echo [5/12] Installing PyTorch...
echo This may take several minutes...
echo.

REM Check if CUDA is available using Python
python -c "import sys; sys.exit(0)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not working in virtual environment
    pause
    exit /b 1
)

set "TORCH_INSTALLED="
set "TORCH_CUDA="
set "NEED_TORCH_INSTALL=1"
set "TORCH_INFO_FILE=%TEMP%\tts_torch_info.txt"
python -c "import torch,sys; print(torch.__version__); print('cuda' if torch.cuda.is_available() else 'cpu')" > "%TORCH_INFO_FILE%" 2>nul
if exist "%TORCH_INFO_FILE%" (
    for /f "usebackq delims=" %%V in ("%TORCH_INFO_FILE%") do (
        if not defined TORCH_INSTALLED (
            set "TORCH_INSTALLED=%%V"
        ) else (
            set "TORCH_CUDA=%%V"
        )
    )
    del "%TORCH_INFO_FILE%" >nul 2>&1
)

if "%SKIP_TORCH_INSTALL%"=="1" (
    echo SKIP_TORCH_INSTALL=1 set. Skipping torch install/update.
    set "NEED_TORCH_INSTALL=0"
) else if "%FORCE_TORCH_REINSTALL%"=="1" (
    echo FORCE_TORCH_REINSTALL=1 set. Reinstalling torch.
) else if defined TORCH_INSTALLED (
    if "%HAS_NVIDIA%"=="1" (
        if /i "%TORCH_CUDA%"=="cuda" (
            if "%NEEDS_BLACKWELL_TORCH%"=="1" (
                echo Detected existing CUDA torch: %TORCH_INSTALLED%
                echo Checking for Blackwell sm_120 support...
                python scripts\torch_cuda_probe.py --require-arch sm_120 --test-cuda >nul 2>&1
                if errorlevel 1 (
                    echo Existing CUDA torch does not support this GPU. Reinstalling PyTorch CUDA 12.8 build.
                ) else (
                    set "NEED_TORCH_INSTALL=0"
                )
            ) else (
                echo Detected existing CUDA torch: %TORCH_INSTALLED%
                python scripts\torch_cuda_probe.py --test-cuda >nul 2>&1
                if errorlevel 1 (
                    echo Existing CUDA torch failed a runtime test. Reinstalling PyTorch.
                ) else (
                    set "NEED_TORCH_INSTALL=0"
                )
            )
        )
    ) else (
        if /i "%TORCH_CUDA%"=="cpu" (
            echo Detected existing CPU torch: %TORCH_INSTALLED%
            set "NEED_TORCH_INSTALL=0"
        )
    )
)

if "%NEED_TORCH_INSTALL%"=="0" (
    echo Skipping torch install - compatible build already present.
) else (

    if "%HAS_NVIDIA%"=="1" (
        if "%NEEDS_BLACKWELL_TORCH%"=="1" (
            if "%USE_TORCH_NIGHTLY%"=="1" (
                echo Installing PyTorch nightly with CUDA 12.8 support for Blackwell...
                pip install --pre --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
            ) else (
                echo Installing PyTorch CUDA 12.8 support for Blackwell...
                pip install --upgrade --force-reinstall torch==%BLACKWELL_TORCH_VERSION% torchvision==%BLACKWELL_TORCHVISION_VERSION% torchaudio==%BLACKWELL_TORCHAUDIO_VERSION% --index-url https://download.pytorch.org/whl/cu128
            )
            if errorlevel 1 (
                echo.
                echo PyTorch CUDA 12.8 installation failed. Set USE_TORCH_NIGHTLY=1 and rerun setup.bat if stable wheels do not support your GPU yet.
                exit /b 1
            )
            python scripts\torch_cuda_probe.py --require-arch sm_120 --test-cuda
            if errorlevel 1 (
                echo.
                echo ERROR: Installed PyTorch still cannot run on this Blackwell GPU.
                echo Try updating NVIDIA drivers, then rerun setup.bat. As a fallback, set USE_TORCH_NIGHTLY=1.
                exit /b 1
            )
        ) else (
            REM Install PyTorch with CUDA 12.4 (available for torch 2.6.0)
            echo Installing PyTorch with CUDA 12.4 support...
            pip install torch==%TORCH_VERSION%+cu124 torchvision==%TORCHVISION_VERSION%+cu124 torchaudio==%TORCHAUDIO_VERSION%+cu124 --index-url https://download.pytorch.org/whl/cu124

            if errorlevel 1 (
                echo.
                echo PyTorch installation failed, trying CPU version...
                pip install torch torchvision torchaudio
            )
        )
    ) else (
        echo Installing CPU-only PyTorch...
        pip uninstall -y torch torchvision torchaudio >nul 2>&1
        pip install --upgrade --force-reinstall torch==%TORCH_VERSION%+cpu torchvision==%TORCHVISION_VERSION%+cpu torchaudio==%TORCHAUDIO_VERSION%+cpu --index-url https://download.pytorch.org/whl/cpu
        if errorlevel 1 (
            echo.
            echo CPU-only PyTorch install failed, trying default index...
            pip install --upgrade --force-reinstall torch torchvision torchaudio
        )
        pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
    )
)

REM Install other dependencies (excluding torch + chatterbox runtime handled separately)
echo.
echo [6/12] Installing other Python dependencies...
if "%SKIP_MAIN_DEPS%"=="1" (
    echo Main Python dependencies are unchanged. Skipping dependency reinstall for update.
    goto :AfterMainDependencyInstall
)
findstr /v /i "torch" requirements.txt > temp_requirements.txt
findstr /v /i "pyopenjtalk" temp_requirements.txt > temp_requirements_filtered.txt
del temp_requirements.txt
pip install -r temp_requirements_filtered.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
del temp_requirements_filtered.txt

echo.
echo Installing optional pyopenjtalk (Japanese text support)...
where cl >nul 2>&1
if errorlevel 1 (
    echo WARNING: MSVC build tools not found. Skipping pyopenjtalk.
    echo To enable Japanese TTS, install MSVC Build Tools:
    echo https://aka.ms/vs/17/release/vs_buildtools.exe
    echo Then select "Desktop development with C++" and rerun setup.bat.
) else (
    pip install pyopenjtalk
    if errorlevel 1 (
        echo WARNING: pyopenjtalk failed to install. Japanese TTS features will be unavailable.
    )
)

REM Install local Chatterbox runtime
echo.
echo [7/12] Installing Chatterbox Turbo runtime...
pip install chatterbox-tts==%CHATTERBOX_TTS_VERSION% --no-deps
if errorlevel 1 (
    echo WARNING: Failed to install chatterbox-tts==%CHATTERBOX_TTS_VERSION%.
    echo Chatterbox Turbo will be unavailable, but setup will continue for other engines.
) else (
    python -c "from chatterbox.tts_turbo import ChatterboxTurboTTS; import importlib.metadata as m; print('Chatterbox Turbo import OK; package version:', m.version('chatterbox-tts'))"
    if errorlevel 1 (
        echo WARNING: chatterbox-tts installed but its runtime import failed.
        echo The application health endpoint will report the underlying dependency error.
    )
)

echo.
echo Validating Pocket TTS runtime...
python -c "from pocket_tts import TTSModel; import importlib.metadata as m; print('Pocket TTS import OK; package version:', m.version('pocket-tts'))"
if errorlevel 1 (
    echo WARNING: Pocket TTS is installed incorrectly or unavailable. Rerun setup to repair it.
)

REM Install VoxCPM runtime
echo.
echo [8/12] Installing VoxCPM 1.5 runtime...
pip install voxcpm --no-deps
if errorlevel 1 (
    echo WARNING: Failed to install voxcpm - VoxCPM engine will not be available
)
if defined MAIN_DEPS_SIGNATURE (
    >"%MAIN_DEPS_STAMP%" echo(!MAIN_DEPS_SIGNATURE!
)
:AfterMainDependencyInstall

REM Setup OmniVoice isolated environment
echo.
echo [9/12] Setting up OmniVoice isolated environment...
echo OmniVoice requires torch==2.8 which conflicts with main deps, so it runs in its own venv.
set "OMNIVOICE_DIR=%~dp0engines\omnivoice"
if "%OMNIVOICE_DIR:~-1%"=="\" set "OMNIVOICE_DIR=%OMNIVOICE_DIR:~0,-1%"
if exist "%OMNIVOICE_DIR%\.omnivoice_ready" (
    if exist "%OMNIVOICE_DIR%\omnivoice_worker.py" (
        if exist "%OMNIVOICE_DIR%\.venv\Scripts\python.exe" (
            "%OMNIVOICE_DIR%\.venv\Scripts\python.exe" -c "import omnivoice, torch, torchaudio, soundfile, huggingface_hub; print('OmniVoice torch:', torch.__version__, 'torchaudio:', torchaudio.__version__)" >nul 2>&1
            if not errorlevel 1 (
                echo OmniVoice isolated environment already set up. Skipping.
                goto :AfterOmniVoice
            )
        )
    )
    echo OmniVoice setup marker is stale or incomplete. Repairing OmniVoice setup...
    del "%OMNIVOICE_DIR%\.omnivoice_ready" >nul 2>&1
)
if not exist "%OMNIVOICE_DIR%" mkdir "%OMNIVOICE_DIR%"
echo Creating OmniVoice isolated virtual environment...
python -m venv "%OMNIVOICE_DIR%\.venv"
if errorlevel 1 (
    echo WARNING: Failed to create OmniVoice venv. OmniVoice will not be available.
    goto :AfterOmniVoice
)
echo Installing OmniVoice and its dependencies (torch 2.8 + transformers 5.3)...
echo This may take several minutes...
echo Installing omnivoice package...
"%OMNIVOICE_DIR%\.venv\Scripts\pip.exe" install omnivoice
if errorlevel 1 (
    echo WARNING: Failed to install omnivoice in isolated venv. OmniVoice will not be available.
    goto :AfterOmniVoice
)
if "%HAS_NVIDIA%"=="1" (
    echo Installing matched torch/torchaudio CUDA 12.8 builds for GPU acceleration...
    "%OMNIVOICE_DIR%\.venv\Scripts\pip.exe" install --upgrade --force-reinstall --no-deps "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo WARNING: CUDA torch/torchaudio install failed. Trying matched CPU builds.
        "%OMNIVOICE_DIR%\.venv\Scripts\pip.exe" install --upgrade --force-reinstall --no-deps "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cpu
        if errorlevel 1 (
            echo WARNING: CPU torch/torchaudio install failed. OmniVoice will not be available.
            goto :AfterOmniVoice
        )
    )
)
if not "%HAS_NVIDIA%"=="1" (
    echo Installing matched torch/torchaudio CPU builds...
    "%OMNIVOICE_DIR%\.venv\Scripts\pip.exe" install --upgrade --force-reinstall --no-deps "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo WARNING: CPU torch/torchaudio install failed. OmniVoice will not be available.
        goto :AfterOmniVoice
    )
)
echo Installing soundfile and huggingface-hub in OmniVoice venv...
"%OMNIVOICE_DIR%\.venv\Scripts\pip.exe" install soundfile huggingface-hub
if errorlevel 1 (
    echo WARNING: Failed to install OmniVoice helper packages. OmniVoice will not be available.
    goto :AfterOmniVoice
)
echo Verifying OmniVoice isolated environment...
"%OMNIVOICE_DIR%\.venv\Scripts\python.exe" -c "import omnivoice, torch, torchaudio, soundfile, huggingface_hub; print('OmniVoice torch:', torch.__version__, 'torchaudio:', torchaudio.__version__)"
if errorlevel 1 (
    echo WARNING: OmniVoice verification failed. OmniVoice will not be available.
    goto :AfterOmniVoice
)
if not "%PREFETCH_OMNIVOICE_MODEL%"=="0" (
    echo Prefetching OmniVoice model cache ^(set PREFETCH_OMNIVOICE_MODEL=0 to skip^)...
    "%OMNIVOICE_DIR%\.venv\Scripts\python.exe" "%OMNIVOICE_DIR%\omnivoice_worker.py" --prefetch-model --model-id "k2-fsa/OmniVoice"
    if errorlevel 1 (
        echo WARNING: OmniVoice model prefetch failed. First generation will retry the download.
    )
) else (
    echo PREFETCH_OMNIVOICE_MODEL=0 set. Skipping OmniVoice model prefetch.
)
type nul > "%OMNIVOICE_DIR%\.omnivoice_ready"
echo OmniVoice isolated environment ready.
:AfterOmniVoice

REM Install KittenTTS runtime (optional, CPU-only)
echo.
echo [10/12] Installing KittenTTS runtime (optional, CPU-only)...
pip install https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl
if errorlevel 1 (
    echo WARNING: Failed to install kittentts - KittenTTS engine will not be available
    goto :AfterKittenTTS
)
if not "%PREFETCH_KITTEN_TTS_MODEL%"=="0" (
    echo Prefetching KittenTTS model cache ^(set PREFETCH_KITTEN_TTS_MODEL=0 to skip^)...
    powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$p=Start-Process -FilePath '%~dp0venv\Scripts\python.exe' -ArgumentList @('%~dp0scripts\prefetch_kitten_tts.py','--model-id','KittenML/kitten-tts-mini-0.8') -NoNewWindow -PassThru; if (-not $p.WaitForExit(300000)) { Write-Host 'KittenTTS prefetch timed out after 300 seconds.'; try { $p.Kill() } catch {}; exit 124 }; exit $p.ExitCode"
    if errorlevel 1 (
        echo WARNING: KittenTTS model prefetch failed. First generation will retry the download.
    )
) else (
    echo PREFETCH_KITTEN_TTS_MODEL=0 set. Skipping KittenTTS model prefetch.
)
:AfterKittenTTS

REM Setup IndexTTS isolated environment (optional)
echo.
echo [10/12] Setting up IndexTTS isolated environment (optional)...
echo IndexTTS uses its own isolated venv to avoid dependency conflicts.
set "INDEX_TTS_DIR=%~dp0engines\index-tts"
REM Strip trailing backslash if present
if "%INDEX_TTS_DIR:~-1%"=="\" set "INDEX_TTS_DIR=%INDEX_TTS_DIR:~0,-1%"
if exist "%INDEX_TTS_DIR%\.indextts_ready" (
    if exist "%INDEX_TTS_DIR%\tts_worker.py" (
        if exist "%INDEX_TTS_DIR%\pyproject.toml" (
            if exist "%INDEX_TTS_DIR%\.venv\Scripts\python.exe" (
                echo IndexTTS already set up. Skipping clone and sync.
                goto :AfterIndexTTS
            )
        )
    )
    echo IndexTTS setup marker is stale or incomplete. Repairing IndexTTS setup...
    del "%INDEX_TTS_DIR%\.indextts_ready" >nul 2>&1
)
where git >nul 2>&1
if errorlevel 1 (
    echo WARNING: git not found. Skipping IndexTTS setup.
    echo To install IndexTTS manually:
    echo   1. git clone https://github.com/index-tts/index-tts.git engines\index-tts
    echo   2. cd engines\index-tts ^&^& uv sync
    echo   3. Download model: uv run huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=checkpoints
    goto :AfterIndexTTS
)
where uv >nul 2>&1
if not errorlevel 1 (
    set "UV_EXE=uv"
    set "UV_ARGS="
) else (
    python -m uv --version >nul 2>&1
    if not errorlevel 1 (
        set "UV_EXE=python"
        set "UV_ARGS=-m uv"
    ) else (
        echo uv not found. Installing uv package manager...
        pip install -U uv --quiet
        if errorlevel 1 (
            echo WARNING: Failed to install uv. Skipping IndexTTS setup.
            echo Install uv manually from https://docs.astral.sh/uv/ then run:
            echo   cd engines\index-tts ^&^& uv sync
            goto :AfterIndexTTS
        )
        set "UV_EXE=python"
        set "UV_ARGS=-m uv"
    )
)
if not exist "%INDEX_TTS_DIR%\pyproject.toml" (
    echo Cloning IndexTTS repository ^(skipping LFS audio examples^)...
    set "INDEX_TTS_CLONE_TMP=%TEMP%\indextts_clone_%RANDOM%"
    set "GIT_LFS_SKIP_SMUDGE=1"
    git clone https://github.com/index-tts/index-tts.git "!INDEX_TTS_CLONE_TMP!"
    set "GIT_LFS_SKIP_SMUDGE=0"
    if not exist "!INDEX_TTS_CLONE_TMP!\pyproject.toml" (
        echo WARNING: Failed to clone IndexTTS ^(pyproject.toml missing^). Skipping IndexTTS setup.
        goto :AfterIndexTTS
    )
    if not exist "%INDEX_TTS_DIR%" mkdir "%INDEX_TTS_DIR%"
    xcopy /E /I /Y "!INDEX_TTS_CLONE_TMP!\*" "%INDEX_TTS_DIR%\" >nul
    rmdir /s /q "!INDEX_TTS_CLONE_TMP!" >nul 2>&1
    echo IndexTTS cloned successfully.
) else (
    echo IndexTTS already cloned. Pulling latest changes...
    set "GIT_LFS_SKIP_SMUDGE=1"
    git -C "%INDEX_TTS_DIR%" pull --ff-only >nul 2>&1
    set "GIT_LFS_SKIP_SMUDGE=0"
)
if not exist "%INDEX_TTS_DIR%\tts_worker.py" (
    echo WARNING: IndexTTS worker file is missing: %INDEX_TTS_DIR%\tts_worker.py
    echo Run git pull from the TTS-Story repository, then rerun setup.bat.
    goto :AfterIndexTTS
)
echo Installing IndexTTS dependencies (this may take several minutes)...
echo Note: Skipping deepspeed extra - it cannot build on Windows without special CUDA tooling.
pushd "%INDEX_TTS_DIR%"
%UV_EXE% %UV_ARGS% sync
if errorlevel 1 (
    popd
    goto :IndexTTSFailed
)
popd
echo IndexTTS environment ready.
echo Model weights will be downloaded automatically on first use (~2-4 GB).
echo Note: deepspeed was skipped. IndexTTS will run in standard mode.
type nul > "%INDEX_TTS_DIR%\.indextts_ready"
goto :AfterIndexTTS
:IndexTTSFailed
echo WARNING: IndexTTS dependency install failed.
echo Try manually: cd engines\index-tts ^&^& uv sync
:AfterIndexTTS

REM Setup Dot.TTS isolated environment (optional)
echo.
echo [10b/12] Setting up Dot.TTS isolated environment (optional)...
echo Dot.TTS pins newer torch/transformers versions, so it runs in its own venv.
set "DOTS_TTS_DIR=%~dp0engines\dots-tts"
set "DOTS_TTS_REPO=%DOTS_TTS_DIR%\repo"
if "%DOTS_TTS_DIR:~-1%"=="\" set "DOTS_TTS_DIR=%DOTS_TTS_DIR:~0,-1%"
set "DOTS_TTS_REPO_UPDATED=0"
if exist "%DOTS_TTS_REPO%\.git" (
    where git >nul 2>&1
    if not errorlevel 1 (
        set "DOTS_TTS_REPO_BEFORE="
        set "DOTS_TTS_REPO_AFTER="
        for /f "delims=" %%H in ('git -C "%DOTS_TTS_REPO%" rev-parse HEAD 2^>nul') do set "DOTS_TTS_REPO_BEFORE=%%H"
        echo Checking Dot.TTS upstream updates...
        git -C "%DOTS_TTS_REPO%" pull --ff-only >nul 2>&1
        for /f "delims=" %%H in ('git -C "%DOTS_TTS_REPO%" rev-parse HEAD 2^>nul') do set "DOTS_TTS_REPO_AFTER=%%H"
        if defined DOTS_TTS_REPO_BEFORE if defined DOTS_TTS_REPO_AFTER if not "!DOTS_TTS_REPO_BEFORE!"=="!DOTS_TTS_REPO_AFTER!" (
            set "DOTS_TTS_REPO_UPDATED=1"
            echo Dot.TTS upstream updated. Refreshing editable install.
            del "%DOTS_TTS_DIR%\.dots_tts_ready" >nul 2>&1
        )
    )
)
if exist "%DOTS_TTS_DIR%\.dots_tts_ready" (
    if exist "%DOTS_TTS_DIR%\dots_tts_worker.py" (
        if exist "%DOTS_TTS_REPO%\pyproject.toml" (
            if exist "%DOTS_TTS_DIR%\.venv\Scripts\python.exe" (
                "%DOTS_TTS_DIR%\.venv\Scripts\python.exe" "%DOTS_TTS_DIR%\dots_tts_worker.py" --check-env >nul 2>&1
                if not errorlevel 1 (
                    echo Dot.TTS isolated environment already set up. Skipping.
                    goto :AfterDotsTTS
                )
            )
        )
    )
    echo Dot.TTS setup marker is stale or incomplete. Repairing Dot.TTS setup...
    del "%DOTS_TTS_DIR%\.dots_tts_ready" >nul 2>&1
)
where git >nul 2>&1
if errorlevel 1 (
    echo WARNING: git not found. Skipping Dot.TTS setup.
    echo To install Dot.TTS manually:
    echo   1. git clone https://github.com/rednote-hilab/dots.tts.git engines\dots-tts\repo
    echo   2. python -m venv engines\dots-tts\.venv
    echo   3. engines\dots-tts\.venv\Scripts\python.exe -m pip install -c engines\dots-tts\repo\constraints\recommended.txt torch torchaudio transformers huggingface-hub loguru "langcodes[data]" einops librosa soundfile numpy pydantic PyYAML safetensors torchdiffeq tqdm lingua-language-detector
    echo   4. engines\dots-tts\.venv\Scripts\python.exe -m pip install -e engines\dots-tts\repo --no-deps
    goto :AfterDotsTTS
)
if not exist "%DOTS_TTS_DIR%" mkdir "%DOTS_TTS_DIR%"
if not exist "%DOTS_TTS_REPO%\pyproject.toml" (
    echo Cloning Dot.TTS repository...
    git clone https://github.com/rednote-hilab/dots.tts.git "%DOTS_TTS_REPO%"
    if errorlevel 1 (
        echo WARNING: Failed to clone Dot.TTS. Skipping Dot.TTS setup.
        goto :AfterDotsTTS
    )
) else (
    echo Dot.TTS already cloned. Pulling latest changes...
    git -C "%DOTS_TTS_REPO%" pull --ff-only >nul 2>&1
)
if not exist "%DOTS_TTS_DIR%\.venv\Scripts\python.exe" (
    echo Creating Dot.TTS isolated virtual environment...
    python -m venv "%DOTS_TTS_DIR%\.venv"
    if errorlevel 1 (
        echo WARNING: Failed to create Dot.TTS venv. Dot.TTS will not be available.
        goto :AfterDotsTTS
    )
)
echo Installing Dot.TTS dependencies (this may take several minutes)...
"%DOTS_TTS_DIR%\.venv\Scripts\python.exe" -m pip install --upgrade pip
set "DOTS_TTS_TORCH_INSTALLED=0"
if not "%DOTS_TTS_CUDA%"=="0" (
    where nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        echo Installing Dot.TTS CUDA torch/torchaudio builds...
        "%DOTS_TTS_DIR%\.venv\Scripts\python.exe" -m pip install --upgrade --force-reinstall --no-deps "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128
        if not errorlevel 1 set "DOTS_TTS_TORCH_INSTALLED=1"
    )
)
if "%DOTS_TTS_TORCH_INSTALLED%"=="0" (
    echo Installing Dot.TTS CPU torch/torchaudio builds...
    "%DOTS_TTS_DIR%\.venv\Scripts\python.exe" -m pip install --upgrade --force-reinstall --no-deps "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cpu
)
"%DOTS_TTS_DIR%\.venv\Scripts\python.exe" -m pip install -c "%DOTS_TTS_REPO%\constraints\recommended.txt" transformers huggingface-hub loguru "langcodes[data]" einops librosa soundfile numpy pydantic PyYAML safetensors torchdiffeq tqdm lingua-language-detector
if errorlevel 1 (
    echo WARNING: Dot.TTS runtime dependency install failed. Dot.TTS will not be available.
    goto :AfterDotsTTS
)
"%DOTS_TTS_DIR%\.venv\Scripts\python.exe" -m pip install -e "%DOTS_TTS_REPO%" --no-deps
if errorlevel 1 (
    echo WARNING: Dot.TTS dependency install failed. Dot.TTS will not be available.
    goto :AfterDotsTTS
)
echo Verifying Dot.TTS isolated environment...
"%DOTS_TTS_DIR%\.venv\Scripts\python.exe" "%DOTS_TTS_DIR%\dots_tts_worker.py" --check-env
if errorlevel 1 (
    echo WARNING: Dot.TTS verification failed. Dot.TTS will not be available.
    goto :AfterDotsTTS
)
if "%PREFETCH_DOTS_TTS_MODEL%"=="1" (
    echo Prefetching Dot.TTS model cache ^(set PREFETCH_DOTS_TTS_MODEL=0 to skip^)...
    "%DOTS_TTS_DIR%\.venv\Scripts\python.exe" "%DOTS_TTS_DIR%\dots_tts_worker.py" --prefetch-model --model-id "rednote-hilab/dots.tts-soar"
    if errorlevel 1 echo WARNING: Dot.TTS model prefetch failed. First generation will retry the download.
) else (
    echo Dot.TTS model prefetch is opt-in. Set PREFETCH_DOTS_TTS_MODEL=1 to download during setup.
)
type nul > "%DOTS_TTS_DIR%\.dots_tts_ready"
echo Dot.TTS isolated environment ready.
:AfterDotsTTS

REM Optional performance extras (best-effort)
echo.
echo [11/12] Installing optional performance extras...
echo - hf_xet (faster Hugging Face downloads)
pip install hf_xet
if errorlevel 1 (
    echo WARNING: hf_xet install failed. Hugging Face downloads may be slower.
)

call :EnsureVoicePromptFolder
call :InstallSox
call :InstallRubberBand
call :InstallFFmpeg

call :InstallVCRedist

REM Check for espeak-ng
echo.
echo ========================================
echo Checking espeak-ng...
echo ========================================
where espeak-ng >nul 2>&1
if errorlevel 1 (
    call :InstallEspeakNg
) else (
    echo espeak-ng is installed!
)

REM Verify installation
echo.
echo ========================================
echo Verifying Installation
echo ========================================
echo.
if "%NEEDS_BLACKWELL_TORCH%"=="1" (
    python scripts\torch_cuda_probe.py --require-arch sm_120 --test-cuda >nul 2>&1
) else if "%HAS_NVIDIA%"=="1" (
    python scripts\torch_cuda_probe.py --test-cuda >nul 2>&1
) else (
    python scripts\torch_cuda_probe.py >nul 2>&1
)
if errorlevel 1 (
    echo WARNING: PyTorch verification failed.
    echo If this is a CPU-only system, rerun setup.bat to reinstall the CPU-only torch build.
    echo If you have an NVIDIA GPU, ensure the NVIDIA driver is installed and rerun setup.bat.
) else (
    if "%NEEDS_BLACKWELL_TORCH%"=="1" (
        python scripts\torch_cuda_probe.py --require-arch sm_120 --test-cuda
    ) else if "%HAS_NVIDIA%"=="1" (
        python scripts\torch_cuda_probe.py --test-cuda
    ) else (
        python scripts\torch_cuda_probe.py
    )
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. If espeak-ng is not installed, install it now
echo 2. Run: run.bat
echo 3. Open browser to: http://localhost:5000
echo.
exit /b 0

:FindPython311Dir
set "PY_DIR="
if exist "%LocalAppData%\Programs\Python\Python311" set "PY_DIR=%LocalAppData%\Programs\Python\Python311"
if not defined PY_DIR if exist "%ProgramFiles%\Python311" set "PY_DIR=%ProgramFiles%\Python311"
if not defined PY_DIR (
    for /f "delims=" %%D in ('dir /b /ad /o-n "%LocalAppData%\Programs\Python\Python311*" 2^>nul') do (
        set "PY_DIR=%LocalAppData%\Programs\Python\%%D"
        exit /b 0
    )
)
if not defined PY_DIR (
    for /f "delims=" %%D in ('dir /b /ad /o-n "%ProgramFiles%\Python311*" 2^>nul') do (
        set "PY_DIR=%ProgramFiles%\%%D"
        exit /b 0
    )
)
exit /b 0

:RunUvSync
REM Subroutine: cd into %1 and run uv sync --all-extras, return errorlevel
cd /d "%~1"
%UV_EXE% %UV_ARGS% sync --all-extras
exit /b %errorlevel%

:InstallVCRedist
echo.
echo Installing Microsoft Visual C++ Redistributable...
set "VC_REDIST_URL=https://aka.ms/vs/17/release/vc_redist.x64.exe"
set "VC_REDIST_EXE=%TEMP%\vc_redist.x64.exe"
REM Check if VC++ 2015-2022 x64 is already installed (any version >= 14.0)
powershell -NoLogo -NoProfile -Command "exit $(if (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue) { 0 } elseif (Get-ChildItem 'HKLM:\SOFTWARE\Classes\Installer\Dependencies' -ErrorAction SilentlyContinue | Where-Object { (Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue).DisplayName -match 'Visual C\+\+ 201[5-9]|Visual C\+\+ 202[0-9]' }) { 0 } else { 1 })" >nul 2>&1
if not errorlevel 1 (
    echo Visual C++ Redistributable already installed. Skipping.
    goto :EOF
)
powershell -NoLogo -NoProfile -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $url='%VC_REDIST_URL%'; try { Invoke-WebRequest -Uri $url -OutFile '%VC_REDIST_EXE%' -UseBasicParsing -ErrorAction Stop } catch { try { Start-BitsTransfer -Source $url -Destination '%VC_REDIST_EXE%' -ErrorAction Stop } catch { Write-Error $_.Exception.Message; exit 1 } }" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to download Visual C++ Redistributable.
    echo You may need to install manually from: %VC_REDIST_URL%
    goto :EOF
)
"%VC_REDIST_EXE%" /install /quiet /norestart
set "VC_ERR=%errorlevel%"
if "%VC_ERR%"=="1638" (
    echo Visual C++ Redistributable already installed (newer version present^).
) else if "%VC_ERR%"=="0" (
    echo Visual C++ Redistributable installed successfully.
) else (
    echo WARNING: Visual C++ Redistributable install failed (code %VC_ERR%^).
    echo You may need to install manually from: %VC_REDIST_URL%
)
goto :EOF

:InstallEspeakNg
echo.
echo Installing espeak-ng...
set "ESPEAK_URL="
set "ESPEAK_FALLBACK_URL=https://github.com/espeak-ng/espeak-ng/releases/latest/download/espeak-ng-X64.msi"
set "ESPEAK_FALLBACK_URL_2=https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-1.51-X64.msi"
set "ESPEAK_MSI=%TEMP%\espeak-ng-X64.msi"
powershell -NoLogo -NoProfile -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $api='https://api.github.com/repos/espeak-ng/espeak-ng/releases/latest'; $headers=@{ 'User-Agent'='TTS-Story-Installer' }; $urls=@(); try { $release=Invoke-RestMethod -Uri $api -Headers $headers -ErrorAction Stop; $asset=$release.assets | Where-Object { $_.name -match 'x64\.msi$' -or $_.name -match 'X64\.msi$' } | Select-Object -First 1; if ($asset) { $urls += $asset.browser_download_url } } catch { } $urls += '%ESPEAK_FALLBACK_URL%'; $urls += '%ESPEAK_FALLBACK_URL_2%'; $ok=$false; foreach ($u in $urls) { try { Invoke-WebRequest -Uri $u -OutFile '%ESPEAK_MSI%' -UseBasicParsing -ErrorAction Stop; $ok=$true; break } catch { try { Start-BitsTransfer -Source $u -Destination '%ESPEAK_MSI%' -ErrorAction Stop; $ok=$true; break } catch { } } } if (-not $ok) { Write-Error 'Failed to download espeak-ng.'; exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to download espeak-ng.
    echo Please install manually from: https://github.com/espeak-ng/espeak-ng/releases
    goto :EOF
)
msiexec /i "%ESPEAK_MSI%" /qn /norestart
if errorlevel 1 (
    echo WARNING: espeak-ng install failed. Please install manually.
    goto :EOF
)
where espeak-ng >nul 2>&1
if errorlevel 1 (
    echo WARNING: espeak-ng installed but not found in PATH. A restart may be required.
) else (
    echo espeak-ng installed successfully.
)
goto :EOF

:EnsureVoicePromptFolder
set "VOICE_PROMPTS_DIR=%~dp0data\voice_prompts"
if not exist "%VOICE_PROMPTS_DIR%" (
    mkdir "%VOICE_PROMPTS_DIR%" >nul 2>&1
)
goto :EOF

:InstallSox
echo.
echo [10/12] Installing SoX...
set "SOX_DIR=%~dp0tools\sox"
set "SOX_URL_1=https://gigenet.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2-win64.zip"
set "SOX_URL_2=https://gigenet.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2-win32.zip"
set "SOX_EXE_URL=https://gigenet.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2-win64.exe"
set "SOX_ZIP=%TEMP%\sox_download.zip"
set "SOX_EXTRACT=%TEMP%\sox_extract"

if exist "%SOX_DIR%\sox.exe" (
    echo SoX already present.
    goto :EOF
)

if not exist "%~dp0tools" (
    mkdir "%~dp0tools"
) >nul 2>&1

set "SOX_INSTALLED="
call :TrySoxDownload "%SOX_URL_1%"
if not defined SOX_INSTALLED call :TrySoxDownload "%SOX_URL_2%"

:AfterSoxDownload
if not defined SOX_INSTALLED (
    echo WARNING: SoX install failed. Attempting to download the official installer...
    if not exist "%SOX_DIR%" mkdir "%SOX_DIR%" >nul 2>&1
    set "SOX_SETUP_EXE=%SOX_DIR%\sox-14.4.2-win64.exe"
    powershell -NoLogo -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13; Invoke-WebRequest -Uri '%SOX_EXE_URL%' -OutFile '%SOX_SETUP_EXE%' -UseBasicParsing -ErrorAction Stop; } catch { Write-Error $_.Exception.Message; exit 1 }" >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Failed to download SoX installer. Install manually from https://sourceforge.net/projects/sox/files/sox/
    ) else (
        echo SoX installer downloaded: %SOX_SETUP_EXE%
        echo Run it manually, then re-run setup.bat or run.bat.
    )
)

if exist "%SOX_ZIP%" del "%SOX_ZIP%" >nul 2>&1
if exist "%SOX_EXTRACT%" rmdir /s /q "%SOX_EXTRACT%" >nul 2>&1
goto :EOF

:TrySoxDownload
set "SOX_URL=%~1"
if exist "%SOX_ZIP%" del "%SOX_ZIP%" >nul 2>&1
if exist "%SOX_EXTRACT%" rmdir /s /q "%SOX_EXTRACT%" >nul 2>&1

echo Downloading SoX from %SOX_URL% ...
powershell -NoLogo -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13; Invoke-WebRequest -Uri '%SOX_URL%' -OutFile '%SOX_ZIP%' -UseBasicParsing -ErrorAction Stop; } catch { Write-Error $_.Exception.Message; exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to download SoX from %SOX_URL%.
    goto :EOF
)

powershell -NoLogo -NoProfile -Command "try { if ((Get-Item '%SOX_ZIP%').Length -lt 102400) { exit 2 } } catch { exit 2 }" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Downloaded SoX archive looks invalid.
    goto :EOF
)

powershell -Command "Expand-Archive -Path '%SOX_ZIP%' -DestinationPath '%SOX_EXTRACT%' -Force" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to extract SoX archive.
    goto :EOF
)

set "SOX_SOURCE="
for /f "delims=" %%F in ('dir /b /s "%SOX_EXTRACT%\sox.exe" 2^>nul') do (
    set "SOX_SOURCE=%%F"
    goto :FoundSox
)
:FoundSox
if not defined SOX_SOURCE (
    echo WARNING: sox.exe not found in downloaded archive.
    goto :EOF
)

for %%F in ("%SOX_SOURCE%") do set "SOX_SOURCE_DIR=%%~dpF"
if not defined SOX_SOURCE_DIR (
    echo WARNING: Unable to determine SoX source directory.
    goto :EOF
)

if exist "%SOX_DIR%" rmdir /s /q "%SOX_DIR%" >nul 2>&1
mkdir "%SOX_DIR%" >nul 2>&1
xcopy /E /I /Y "%SOX_SOURCE_DIR%*.*" "%SOX_DIR%\" >nul
if errorlevel 1 (
    echo WARNING: Failed to copy SoX files to tools directory.
    goto :EOF
)

echo SoX installed to %SOX_DIR%.
set "SOX_INSTALLED=1"
goto :EOF

:InstallRubberBand
echo.
echo [11/12] Installing Rubber Band CLI...
set "RB_DIR=%~dp0tools\rubberband"
set "RB_URL=https://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip"
set "RB_ZIP=%TEMP%\rubberband_cli.zip"
set "RB_EXTRACT=%TEMP%\rubberband_cli_extract"

if exist "%RB_DIR%\rubberband.exe" (
    echo Rubber Band CLI already present.
    goto :EOF
)

if not exist "%~dp0tools" (
    mkdir "%~dp0tools"
) >nul 2>&1

if exist "%RB_ZIP%" del "%RB_ZIP%" >nul 2>&1
if exist "%RB_EXTRACT%" rmdir /s /q "%RB_EXTRACT%" >nul 2>&1

echo Downloading Rubber Band CLI from: %RB_URL%
powershell -NoLogo -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13; Invoke-WebRequest -Uri '%RB_URL%' -OutFile '%RB_ZIP%' -UseBasicParsing -ErrorAction Stop; } catch { Write-Error $_.Exception.Message; exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to download Rubber Band CLI. FX will fall back to basic processing.
    goto :EOF
)

powershell -Command "Expand-Archive -Path '%RB_ZIP%' -DestinationPath '%RB_EXTRACT%' -Force" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to extract Rubber Band CLI archive.
    goto :EOF
)

set "RB_SOURCE="
for /f "delims=" %%F in ('dir /b /s "%RB_EXTRACT%\rubberband.exe" 2^>nul') do (
    set "RB_SOURCE=%%F"
    goto :FoundRB
)
:FoundRB
if not defined RB_SOURCE (
    echo WARNING: rubberband.exe not found in downloaded archive.
    goto :EOF
)

for %%F in ("%RB_SOURCE%") do set "RB_SOURCE_DIR=%%~dpF"
if not defined RB_SOURCE_DIR (
    echo WARNING: Unable to determine Rubber Band source directory.
    goto :EOF
)

if exist "%RB_DIR%" rmdir /s /q "%RB_DIR%" >nul 2>&1
mkdir "%RB_DIR%" >nul 2>&1
xcopy /E /I /Y "%RB_SOURCE_DIR%*.*" "%RB_DIR%\" >nul
if errorlevel 1 (
    echo WARNING: Failed to copy Rubber Band CLI files to tools directory.
    goto :EOF
)

echo Rubber Band CLI installed to %RB_DIR%.

if exist "%RB_ZIP%" del "%RB_ZIP%" >nul 2>&1
if exist "%RB_EXTRACT%" rmdir /s /q "%RB_EXTRACT%" >nul 2>&1
goto :EOF

:InstallFFmpeg
echo.
echo [12/12] Installing FFmpeg...
set "FF_DIR=%~dp0tools\ffmpeg"
set "FF_URL_1=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
set "FF_URL_2=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
set "FF_ZIP=%TEMP%\ffmpeg_download.zip"
set "FF_EXTRACT=%TEMP%\ffmpeg_extract"

if exist "%FF_DIR%\ffmpeg.exe" (
    echo FFmpeg already present.
    goto :EOF
)

if not exist "%~dp0tools" (
    mkdir "%~dp0tools"
) >nul 2>&1

if exist "%FF_ZIP%" del "%FF_ZIP%" >nul 2>&1
if exist "%FF_EXTRACT%" rmdir /s /q "%FF_EXTRACT%" >nul 2>&1

echo Downloading FFmpeg...
powershell -NoLogo -NoProfile -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $urls=@('%FF_URL_1%','%FF_URL_2%'); $ok=$false; foreach ($u in $urls) { try { Invoke-WebRequest -Uri $u -OutFile '%FF_ZIP%' -UseBasicParsing -ErrorAction Stop; $ok=$true; break } catch { try { Start-BitsTransfer -Source $u -Destination '%FF_ZIP%' -ErrorAction Stop; $ok=$true; break } catch { } } } if (-not $ok) { Write-Error 'Failed to download FFmpeg from all sources.'; exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to download FFmpeg. Audio merging may use system FFmpeg.
    goto :EOF
)

echo Extracting FFmpeg...
powershell -Command "Expand-Archive -Path '%FF_ZIP%' -DestinationPath '%FF_EXTRACT%' -Force" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to extract FFmpeg archive.
    goto :EOF
)

set "FF_SOURCE="
for /f "delims=" %%F in ('dir /b /s "%FF_EXTRACT%\ffmpeg.exe" 2^>nul') do (
    set "FF_SOURCE=%%F"
    goto :FoundFF
)
:FoundFF
if not defined FF_SOURCE (
    echo WARNING: ffmpeg.exe not found in downloaded archive.
    goto :EOF
)

for %%F in ("%FF_SOURCE%") do set "FF_SOURCE_DIR=%%~dpF"
if not defined FF_SOURCE_DIR (
    echo WARNING: Unable to determine FFmpeg source directory.
    goto :EOF
)

if exist "%FF_DIR%" rmdir /s /q "%FF_DIR%" >nul 2>&1
mkdir "%FF_DIR%" >nul 2>&1
REM Copy just the executables (ffmpeg.exe, ffprobe.exe, ffplay.exe)
copy /Y "%FF_SOURCE_DIR%ffmpeg.exe" "%FF_DIR%\" >nul
copy /Y "%FF_SOURCE_DIR%ffprobe.exe" "%FF_DIR%\" >nul 2>&1
copy /Y "%FF_SOURCE_DIR%ffplay.exe" "%FF_DIR%\" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to copy FFmpeg files to tools directory.
    goto :EOF
)

echo FFmpeg installed to %FF_DIR%.

if exist "%FF_ZIP%" del "%FF_ZIP%" >nul 2>&1
if exist "%FF_EXTRACT%" rmdir /s /q "%FF_EXTRACT%" >nul 2>&1
goto :EOF
