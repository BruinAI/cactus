@echo off
setlocal enabledelayedexpansion

:: Default model configuration
set "DEFAULT_MODEL_ID=LiquidAI/LFM2-1.2B"

:: Get script directory and project root
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%i in ("%SCRIPT_DIR%") do set "PROJECT_ROOT=%%~dpi"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "BUILD_DIR=%SCRIPT_DIR%\build"

:: Check for command argument
if "%~1"=="" (
    call :print_usage
    exit /b 1
)

:: Set model ID (use default if not provided)
set "COMMAND=%~1"
if "%~2"=="" (
    set "MODEL_ID=%DEFAULT_MODEL_ID%"
) else (
    set "MODEL_ID=%~2"
)

:: Process command
if /i "%COMMAND%"=="build" (
    call :build_chat
    exit /b !ERRORLEVEL!
)
if /i "%COMMAND%"=="download" (
    call :download_model "%MODEL_ID%"
    exit /b !ERRORLEVEL!
)
if /i "%COMMAND%"=="run" (
    call :run_chat "%MODEL_ID%"
    exit /b !ERRORLEVEL!
)
if /i "%COMMAND%"=="help" goto show_help
if /i "%COMMAND%"=="-h" goto show_help
if /i "%COMMAND%"=="--help" goto show_help

echo [31mError: Unknown command '%COMMAND%'[0m
echo.
call :print_usage
exit /b 1

:show_help
call :print_usage
exit /b 0

:print_usage
echo Usage: %~nx0 ^<command^> [model_id]
echo.
echo Commands:
echo   build         - Build the chat application only
echo   download      - Download model weights only
echo   run           - Build, download (if needed), and run the chat app
echo.
echo Arguments:
echo   model_id      - HuggingFace model ID (e.g., LiquidAI/LFM2-1.2B)
echo.
echo Examples:
echo   %~nx0 build
echo   %~nx0 download LiquidAI/LFM2-1.2B
echo   %~nx0 run
echo   %~nx0 run Qwen/Qwen3-0.6B
echo   %~nx0 run google/gemma-3-270m-it
echo.
echo Common models:
echo   LiquidAI/LFM2-1.2B (default)
echo   Qwen/Qwen3-0.6B
echo   google/gemma-3-270m-it
echo   google/gemma-3-1b-it
exit /b 0

:get_model_dir_name
:: Convert model ID to directory name
:: e.g., "LiquidAI/LFM2-1.2B" -> "lfm2-1.2b"
:: e.g., "Qwen/Qwen3-0.6B" -> "qwen3-0.6b"
:: e.g., "google/gemma-3-270m-it" -> "gemma3-270m"
set "model_id_input=%~1"
for /f "tokens=2 delims=/" %%a in ("%model_id_input%") do set "model_name=%%a"
:: Convert to lowercase
for %%L in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
    set "model_name=!model_name:%%L=%%L!"
)
call :to_lower model_name
:: Remove -it suffix
set "model_name=!model_name:-it=!"
set "MODEL_DIR_NAME=%model_name%"
exit /b 0

:to_lower
:: Helper function to convert string to lowercase
set "str=!%~1!"
for %%a in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
    set "str=!str:%%a=%%a!"
)
set "str=!str:A=a!"
set "str=!str:B=b!"
set "str=!str:C=c!"
set "str=!str:D=d!"
set "str=!str:E=e!"
set "str=!str:F=f!"
set "str=!str:G=g!"
set "str=!str:H=h!"
set "str=!str:I=i!"
set "str=!str:J=j!"
set "str=!str:K=k!"
set "str=!str:L=l!"
set "str=!str:M=m!"
set "str=!str:N=n!"
set "str=!str:O=o!"
set "str=!str:P=p!"
set "str=!str:Q=q!"
set "str=!str:R=r!"
set "str=!str:S=s!"
set "str=!str:T=t!"
set "str=!str:U=u!"
set "str=!str:V=v!"
set "str=!str:W=w!"
set "str=!str:X=x!"
set "str=!str:Y=y!"
set "str=!str:Z=z!"
set "%~1=!str!"
exit /b 0

:build_chat
echo [34mBuilding Cactus chat...[0m
echo =======================
echo.

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

:: Build Cactus library if needed
cd /d "%PROJECT_ROOT%\cactus"
if not exist "build\libcactus.a" (
    echo [33mCactus library not found. Building...[0m
    call build.bat
    if errorlevel 1 (
        echo [31mError: Failed to build Cactus library[0m
        exit /b 1
    )
) else (
    echo [32mCactus library found.[0m
)

cd /d "%BUILD_DIR%"

echo Compiling chat.cpp...

:: Use g++ on Windows (MinGW or similar)
g++ -std=c++17 -O3 ^
    -I"%PROJECT_ROOT%" ^
    "%SCRIPT_DIR%\chat.cpp" ^
    "%PROJECT_ROOT%\cactus\build\libcactus.a" ^
    -o chat.exe ^
    -pthread

if errorlevel 1 (
    echo [31mError: Failed to compile chat.cpp[0m
    exit /b 1
)

echo [32mBuild complete: %BUILD_DIR%\chat.exe[0m
echo.
exit /b 0

:download_model
set "model_id=%~1"
call :get_model_dir_name "%model_id%"
set "model_dir=%MODEL_DIR_NAME%"
set "weights_dir=%PROJECT_ROOT%\weights\%model_dir%"

if exist "%weights_dir%\config.txt" (
    echo [32mModel weights found at %weights_dir%[0m
    exit /b 0
)

echo.
echo [33mModel weights not found. Downloading %model_id%...[0m
echo =============================================

cd /d "%PROJECT_ROOT%"

:: Check for Python3
where python3 >nul 2>nul
if errorlevel 1 (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [31mError: Python not found. Cannot download weights automatically.[0m
        echo Please run manually: python tools/convert_hf.py %model_id% weights/%model_dir%/ --precision INT8
        exit /b 1
    )
    set "PYTHON_CMD=python"
) else (
    set "PYTHON_CMD=python3"
)

:: Check for required packages
%PYTHON_CMD% -c "import numpy, torch, transformers" 2>nul
if errorlevel 1 (
    echo [31mError: Required Python packages not found.[0m
    echo Please check the README for setup instructions.
    exit /b 1
)

echo Running: %PYTHON_CMD% tools/convert_hf.py %model_id% weights/%model_dir%/ --precision INT8
%PYTHON_CMD% tools/convert_hf.py "%model_id%" "weights/%model_dir%/" --precision INT8

if errorlevel 1 (
    echo [31mError: Failed to download weights.[0m
    echo Please run manually: %PYTHON_CMD% tools/convert_hf.py %model_id% weights/%model_dir%/ --precision INT8
    exit /b 1
)

echo [32mSuccessfully downloaded and converted weights[0m
exit /b 0

:run_chat
set "model_id=%~1"
call :get_model_dir_name "%model_id%"
set "model_dir=%MODEL_DIR_NAME%"
set "weights_dir=%PROJECT_ROOT%\weights\%model_dir%"

:: Build the chat app
call :build_chat
if errorlevel 1 exit /b 1

:: Download model if needed
call :download_model "%model_id%"
if errorlevel 1 exit /b 1

:: Run the chat app
cls
echo [32mStarting Cactus Chat with model: %model_id%[0m
echo.

"%BUILD_DIR%\chat.exe" "%weights_dir%"
exit /b 0
