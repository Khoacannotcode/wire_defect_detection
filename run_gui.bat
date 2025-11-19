@echo off
REM Wire Defect Detection - GUI Launcher for Windows
REM Double-click this file to run the GUI application
REM Script automatically activates virtual environment and runs GUI

echo ========================================
echo Wire Defect Detection - GUI Launcher
echo ========================================
echo.

REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run setup_environment.sh first to create the virtual environment.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in virtual environment!
    echo.
    echo Please check your virtual environment setup.
    echo.
    pause
    exit /b 1
)

REM Check if GUI script exists
if not exist "gui_detection_runner.py" (
    echo [ERROR] gui_detection_runner.py not found!
    echo.
    echo Please ensure you are in the shipping directory.
    echo.
    pause
    exit /b 1
)

REM Run GUI
echo [INFO] Starting GUI application...
echo.
python gui_detection_runner.py

REM If GUI exits, keep window open to show any errors
if errorlevel 1 (
    echo.
    echo [ERROR] GUI application exited with an error.
    echo.
    pause
)

