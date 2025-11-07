@echo off
setlocal

echo Building Cactus library...

cd /d "%~dp0"

if exist build rmdir /s /q build

mkdir build
cd build

cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF >nul 2>&1
if %errorlevel% neq 0 exit /b %errorlevel%

cmake --build . --config Release
if %errorlevel% neq 0 exit /b %errorlevel%

echo Cactus library built successfully!
echo Library location: %cd%\lib\libcactus.a
