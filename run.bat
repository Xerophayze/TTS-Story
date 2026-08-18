@echo off
setlocal EnableExtensions EnableDelayedExpansion
echo ========================================
echo Starting TTS-Story
echo ========================================
echo.
echo NOTE: First startup can pause while models initialize and caches build.
echo Subsequent runs should be faster.
echo.
echo Quick Troubleshooting:
echo  - If startup fails, delete the "venv" folder and re-run install-update.bat
echo  - GPU users: update to the latest NVIDIA drivers
echo  - Run install-update.bat to pull the latest updates
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Ensure FFmpeg is on PATH if bundled
set "FFMPEG_EXE=%~dp0tools\ffmpeg\ffmpeg.exe"
if exist "%FFMPEG_EXE%" (
    set "PATH=%~dp0tools\ffmpeg;%PATH%"
    echo FFmpeg ready: %FFMPEG_EXE%
) else (
    echo WARNING: FFmpeg not found (expected %FFMPEG_EXE%)
)

REM Detect NVIDIA GPU (for CPU-only torch fallback)
set "HAS_NVIDIA=0"
set "GPU_NAME="
for /f "delims=" %%G in ('powershell -NoLogo -NoProfile -Command "$g=(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1).Name; if ($g) { $g }"') do set "GPU_NAME=%%G"
if defined GPU_NAME (
    set "HAS_NVIDIA=1"
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Ensure CPU-only torch on systems without NVIDIA GPUs
if "%SKIP_TORCH_INSTALL%"=="1" (
    echo SKIP_TORCH_INSTALL=1 set. Skipping torch checks.
) else if "%HAS_NVIDIA%"=="0" (
    set "TORCH_PIN=2.6.0"
    set "TORCHVISION_PIN=0.21.0"
    set "TORCHAUDIO_PIN=2.6.0"
    set "TORCH_INSTALLED="
    for /f "delims=" %%V in ('python -c "import torch, sys; sys.stdout.write(torch.__version__)" 2^>nul') do set "TORCH_INSTALLED=%%V"
    if "%FORCE_TORCH_REINSTALL%"=="1" (
        echo FORCE_TORCH_REINSTALL=1 set. Reinstalling CPU-only torch...
        pip uninstall -y torch torchvision torchaudio >nul 2>&1
        pip install --upgrade --force-reinstall torch==!TORCH_PIN!+cpu torchvision==!TORCHVISION_PIN!+cpu torchaudio==!TORCHAUDIO_PIN!+cpu --index-url https://download.pytorch.org/whl/cpu
        pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
    ) else if not defined TORCH_INSTALLED (
        echo WARNING: PyTorch failed to import. Reinstalling CPU-only torch...
        pip uninstall -y torch torchvision torchaudio >nul 2>&1
        pip install --upgrade --force-reinstall torch==!TORCH_PIN!+cpu torchvision==!TORCHVISION_PIN!+cpu torchaudio==!TORCHAUDIO_PIN!+cpu --index-url https://download.pytorch.org/whl/cpu
        pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
    ) else (
        echo Detected PyTorch: !TORCH_INSTALLED!
        echo !TORCH_INSTALLED! | findstr /i "+cu" >nul 2>&1
        if not errorlevel 1 (
            echo CUDA build detected on CPU-only system. Reinstalling CPU-only torch...
            pip uninstall -y torch torchvision torchaudio >nul 2>&1
            pip install --upgrade --force-reinstall torch==!TORCH_PIN!+cpu torchvision==!TORCHVISION_PIN!+cpu torchaudio==!TORCHAUDIO_PIN!+cpu --index-url https://download.pytorch.org/whl/cpu
            pip install --upgrade "numpy<1.26.0" "pillow<12.0" "fsspec<=2025.3.0" "filelock>=3.20.1,<4"
        )
    )
)

REM Ensure Rubber Band CLI is on PATH if bundled
set "RB_EXE=%~dp0tools\rubberband\rubberband.exe"
if exist "%RB_EXE%" (
    set "PATH=%~dp0tools\rubberband;%PATH%"
    echo Rubber Band CLI ready: %RB_EXE%
) else (
    echo WARNING: Rubber Band CLI not found (expected %RB_EXE%)
)

REM Ensure SoX is on PATH if bundled
set "SOX_EXE=%~dp0tools\sox\sox.exe"
if exist "%SOX_EXE%" (
    set "PATH=%~dp0tools\sox;%PATH%"
    echo SoX ready: %SOX_EXE%
) else (
    echo WARNING: SoX not found (expected %SOX_EXE%)
)

REM Skip CUDA check at startup (can hang on some systems)
REM CUDA availability will be detected when the app starts

echo.
echo Starting Flask server...
echo Open your browser to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

REM Start the application
set "TTS_STORY_RESTARTABLE=1"
:run_server
python app.py
set "SERVER_EXIT_CODE=!errorlevel!"
if "!SERVER_EXIT_CODE!"=="75" (
    echo.
    echo Restart requested. Relaunching TTS-Story backend...
    set "TTS_STORY_SKIP_BROWSER=1"
    goto run_server
)
exit /b !SERVER_EXIT_CODE!
