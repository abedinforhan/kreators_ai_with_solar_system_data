@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo            KREATORS AI - SETUP
echo ==================================================
echo.
echo This will install everything needed to train your
echo own AI. This can take 10-15 minutes. Please wait.
echo.

py -3.12 --version >nul 2>nul
if %errorlevel%==0 (
    echo Python 3.12 found.
    goto :setup_venv
)

echo Python 3.12 not found. Downloading and installing it now...
echo (You may see a Windows permission prompt - click Yes.)
echo.

set PYTHON_URL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
set PYTHON_INSTALLER=%TEMP%\kreators_python_installer.exe

powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'"

if not exist "%PYTHON_INSTALLER%" (
    echo ERROR: Could not download Python.
    echo Install Python 3.12 manually from https://www.python.org/downloads/
    pause
    exit /b 1
)

"%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1
timeout /t 15 /nobreak >nul
del "%PYTHON_INSTALLER%"

:setup_venv
echo.
echo Creating your AI workspace...

if exist .venv rmdir /s /q .venv

py -3.12 -m venv .venv
if not exist .venv\Scripts\python.exe (
    echo ERROR: Could not create Python environment.
    pause
    exit /b 1
)

echo.
echo Installing AI tools...
.venv\Scripts\python.exe -m pip install --upgrade pip
if %errorlevel% neq 0 ( echo ERROR: pip upgrade failed. & pause & exit /b 1 )

.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 ( echo ERROR: torch install failed. & pause & exit /b 1 )

.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 ( echo ERROR: package install failed. & pause & exit /b 1 )

echo.
echo Downloading Qwen3-0.6B model (about 1 GB, takes a few minutes)...
.venv\Scripts\python.exe -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B'); print('Qwen ready.')"
if %errorlevel% neq 0 ( echo ERROR: Qwen model download failed. & pause & exit /b 1 )

echo.
echo ==================================================
echo   SETUP COMPLETE!
echo   Run start.bat to begin.
echo ==================================================
pause
