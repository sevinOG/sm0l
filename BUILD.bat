@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python is required on PATH.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating venv...
  python -m venv .venv
)

echo Installing deps...
".venv\Scripts\python.exe" -m pip install -U pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m pip install pyinstaller

echo Generating icon...
".venv\Scripts\python.exe" scripts\make_icon.py

echo Compiling...
".venv\Scripts\python.exe" -m compileall -q src sm0l.py

echo Building EXE...
".venv\Scripts\python.exe" -m PyInstaller build.spec --noconfirm --clean

if exist "dist\sm0l\sm0l.exe" (
  echo.
  echo Built: dist\sm0l\sm0l.exe
) else (
  echo BUILD FAILED
  exit /b 1
)
