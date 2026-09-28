@echo off

if not exist .venv\Scripts\python.exe (
    echo Setup has not been completed yet.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

:menu
cls
echo ==================================================
echo                  KREATORS AI
echo ==================================================
echo.
echo   [1] Train My AI
echo   [2] Test My Trained AI (after training)
echo   [3] Test Untrained Brain (base model - compare!)
echo   [4] Exit
echo.
set /p choice="Choose an option: "

if "%choice%"=="1" (
    echo.
    echo Starting training...
    echo.
    .venv\Scripts\python.exe train.py
    echo.
    pause
    goto menu
)
if "%choice%"=="2" (
    .venv\Scripts\python.exe test.py
    goto menu
)
if "%choice%"=="3" (
    .venv\Scripts\python.exe test_base.py
    goto menu
)
if "%choice%"=="4" (
    exit
)
goto menu