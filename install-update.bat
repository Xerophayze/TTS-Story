@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO_URL=https://github.com/Xerophayze/TTS-Story.git"
set "REPO_DIR=TTS-Story"
set "GIT_INSTALLER=%TEMP%\git-installer.exe"

echo ========================================
echo TTS-Story Windows Install/Update
echo ========================================
echo.

REM Check if already in the repository (running from within the project folder)
if exist ".git" (
    if exist "setup.bat" (
        echo Running from within TTS-Story repository.
        echo Updating in-place...
        call :pull_updates
        if errorlevel 1 (
            echo ERROR: Git pull failed.
            pause
            exit /b 1
        )
        echo.
        echo Running fast dependency-aware update...
        call setup.bat --update
        if errorlevel 1 (
            echo ERROR: Update setup failed.
            pause
            exit /b 1
        )
        echo.
        echo ✅ Update complete.
        pause
        exit /b 0
    )
)

REM Not in repo - clone/update from outside

echo Checking Git installation...
where git >nul 2>&1
if errorlevel 1 (
    echo Git not found. Downloading and installing Git for Windows...
    powershell -NoLogo -NoProfile -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $api='https://api.github.com/repos/git-for-windows/git/releases/latest'; $headers=@{ 'User-Agent'='TTS-Story-Installer' }; try { $release=Invoke-RestMethod -Uri $api -Headers $headers -ErrorAction Stop; $asset=$release.assets | Where-Object { $_.name -match '64-bit\.exe$' -and $_.name -notmatch 'portable' -and $_.name -notmatch 'mingit' } | Select-Object -First 1; if (-not $asset) { throw 'Unable to find Git 64-bit installer asset.' } $url=$asset.browser_download_url; try { Invoke-WebRequest -Uri $url -OutFile '%GIT_INSTALLER%' -UseBasicParsing -ErrorAction Stop } catch { Start-BitsTransfer -Source $url -Destination '%GIT_INSTALLER%' -ErrorAction Stop }; if ((Get-Item '%GIT_INSTALLER%').Length -lt 1048576) { throw 'Downloaded Git installer is unexpectedly small.' } } catch { Write-Error $_.Exception.Message; exit 1 }"
    if errorlevel 1 (
        echo GitHub download failed. Trying Windows Package Manager...
        where winget >nul 2>&1
        if not errorlevel 1 (
            winget install --id Git.Git -e --source winget --silent --accept-package-agreements --accept-source-agreements
        )
    ) else (
        "%GIT_INSTALLER%" /VERYSILENT /NORESTART /NOCANCEL /SP-
        if errorlevel 1 echo WARNING: Downloaded Git installer returned an error.
    )

    set "PATH=%ProgramFiles%\Git\cmd;%LocalAppData%\Programs\Git\cmd;%PATH%"
    where git >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Git could not be installed automatically.
        echo Install it manually from https://git-scm.com/download/win and rerun this file.
        pause
        exit /b 1
    )
) else (
    echo Git is installed.
)

echo.
echo Cloning or updating repository...
set "EXISTING_INSTALL=0"
if exist "%REPO_DIR%" (
    if exist "%REPO_DIR%\.git" (
        set "EXISTING_INSTALL=1"
        echo Repository found. Pulling latest updates...
        pushd "%REPO_DIR%"
        call :pull_updates
        if errorlevel 1 (
            echo ERROR: Git pull failed.
            popd
            pause
            exit /b 1
        )
        popd
    ) else (
        echo ERROR: "%REPO_DIR%" exists but is not a Git repository.
        echo Please rename or remove the folder and re-run this script.
        pause
        exit /b 1
    )
) else (
    git clone "%REPO_URL%" "%REPO_DIR%"
    if errorlevel 1 (
        echo ERROR: Git clone failed.
        pause
        exit /b 1
    )
)

echo.
echo Running setup.bat...
if exist "%REPO_DIR%\setup.bat" (
    pushd "%REPO_DIR%"
    if "%EXISTING_INSTALL%"=="1" (
        call setup.bat --update
    ) else (
        call setup.bat
    )
    if errorlevel 1 (
        echo ERROR: Setup failed.
        popd
        pause
        exit /b 1
    )
    popd
) else (
    echo ERROR: setup.bat not found in %REPO_DIR%.
    pause
    exit /b 1
)

echo.
echo ✅ Install/update complete.
pause
exit /b 0

:pull_updates
REM Preserve settings from older installations where config.json was tracked.
REM After the migration is pulled, the restored local file is ignored by Git.
set "CONFIG_UPDATE_BACKUP="
set "CONFIG_UPDATE_STATUS="
for /f "delims=" %%I in ('git status --porcelain -- config.json 2^>nul') do set "CONFIG_UPDATE_STATUS=%%I"
if defined CONFIG_UPDATE_STATUS if exist "config.json" (
    set "CONFIG_UPDATE_BACKUP=%TEMP%\tts-story-config-!RANDOM!-!RANDOM!.json"
    copy /Y "config.json" "!CONFIG_UPDATE_BACKUP!" >nul
    if errorlevel 1 (
        echo ERROR: Could not preserve local config.json before updating.
        exit /b 1
    )
    echo Preserving local config.json settings during repository update...
    git restore --source=HEAD --staged --worktree -- config.json >nul 2>&1
    if errorlevel 1 (
        copy /Y "!CONFIG_UPDATE_BACKUP!" "config.json" >nul
        del /Q "!CONFIG_UPDATE_BACKUP!" >nul 2>&1
        echo ERROR: Could not prepare tracked config.json for update.
        exit /b 1
    )
)

git pull --ff-only
set "PULL_RESULT=!ERRORLEVEL!"
if defined CONFIG_UPDATE_BACKUP (
    copy /Y "!CONFIG_UPDATE_BACKUP!" "config.json" >nul
    del /Q "!CONFIG_UPDATE_BACKUP!" >nul 2>&1
    echo Restored local config.json settings.
)
exit /b !PULL_RESULT!
