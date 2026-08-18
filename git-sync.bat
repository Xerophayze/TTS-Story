@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Prefer the fully guided local sync script when it is present.
if exist "%SCRIPT_DIR%\sync-to-git.bat" (
    call "%SCRIPT_DIR%\sync-to-git.bat"
    exit /b %errorlevel%
)

set "PYTHON=python"
if exist "%SCRIPT_DIR%\venv\Scripts\python.exe" set "PYTHON=%SCRIPT_DIR%\venv\Scripts\python.exe"

git -C "%SCRIPT_DIR%" check-ignore -q config.json
if errorlevel 1 (
    echo ERROR: config.json is not ignored. Refusing to synchronize.
    pause
    exit /b 1
)

"%PYTHON%" "%SCRIPT_DIR%\scripts\check_repo_safety.py" --working-tree
if errorlevel 1 (
    echo ERROR: Repository safety audit failed. Nothing was staged or pushed.
    pause
    exit /b 1
)

REM Usage: git-sync.bat [optional commit message]
REM If you omit a commit message, a timestamp-based message is used.

for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set TODAY=%%a-%%b-%%c
for /f "tokens=1-2 delims=:." %%a in ("%time%") do set NOW=%%a%%b

if "%~1"=="" (
    set "COMMIT_MSG=Auto-sync !TODAY! !NOW!"
) else (
    set "COMMIT_MSG=%*"
)

echo === Checking repository status ===
git status
if errorlevel 1 goto :git_error

echo === Staging all changes ===
git -C "%SCRIPT_DIR%" add -A
if errorlevel 1 goto :git_error

"%PYTHON%" "%SCRIPT_DIR%\scripts\check_repo_safety.py" --staged
if errorlevel 1 (
    echo ERROR: Staged repository safety audit failed. Nothing was committed or pushed.
    pause
    exit /b 1
)

echo === Committing with message: !COMMIT_MSG! ===
git -C "%SCRIPT_DIR%" commit -m "!COMMIT_MSG!"
if errorlevel 1 goto :commit_error

echo === Pulling latest changes (rebase) ===
git -C "%SCRIPT_DIR%" pull --rebase
if errorlevel 1 goto :git_error

echo === Pushing to origin ===
git -C "%SCRIPT_DIR%" push
if errorlevel 1 goto :git_error

echo.
echo ✅ Repository is up to date with origin.
goto :eof

:commit_error
echo.
echo ⚠️ Nothing to commit (working tree clean) or commit failed.
echo Skipping pull/push.
goto :eof

:git_error
echo.
echo ❌ Git command failed. Please review the messages above.
pause
goto :eof
