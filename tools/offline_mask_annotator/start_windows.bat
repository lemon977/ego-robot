@echo off
setlocal
cd /d "%~dp0"
set "CONFIG_PATH=%~1"
if "%CONFIG_PATH%"=="" set "CONFIG_PATH=%CD%\config.json"

where ffmpeg >nul 2>nul || (
  echo ERROR: ffmpeg.exe not found on PATH.
  pause
  exit /b 2
)
where ffprobe >nul 2>nul || (
  echo ERROR: ffprobe.exe not found on PATH.
  pause
  exit /b 2
)

if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv || goto :error
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-lock.txt || goto :error
".venv\Scripts\python.exe" annotator.py run --config "%CONFIG_PATH%"
exit /b %ERRORLEVEL%

:error
echo.
echo Startup failed. See README_ZH.md.
pause
exit /b 2
