@echo off
setlocal enabledelayedexpansion

REM Root directory of your project
set ROOT_DIR=C:\Users\Administrator\Desktop\Agentic_AI_framework\backend

REM Python from virtual environment
set PYTHON_EXE=%ROOT_DIR%\venv\Scripts\python.exe

echo ===============================================
echo   Compiling all .py files to .pyd using venv
echo ===============================================

for /R "%ROOT_DIR%" %%f in (*.py) do (
    REM Skip __init__.py and anything inside venv folder
    echo %%f | findstr /I "\\venv\\" >nul
    if errorlevel 1 (
        if /I not "%%~nxf"=="__init__.py" (
            echo Compiling %%f ...
            "%PYTHON_EXE%" -m nuitka --module "%%f" --remove-output --no-pyi-file --output-dir="%%~dpf"
        )
    )
)

echo ===============================================
echo   Compilation finished!
echo ===============================================
pause
