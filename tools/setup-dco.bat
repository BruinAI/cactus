@echo off
setlocal enabledelayedexpansion

echo Setting up DCO for the Cactus project...

git config core.hooksPath .githooks

echo [32m✓[0m Git hooks configured to use .githooks directory

for /f "tokens=*" %%a in ('git config user.name') do set "name=%%a"
for /f "tokens=*" %%b in ('git config user.email') do set "email=%%b"

if "!name!"=="" (
    set missing=1
) else if "!email!"=="" (
    set missing=1
) else (
    set missing=0
)

if !missing!==1 (
    echo.
    echo [33m⚠️  Warning: Git user configuration is incomplete[0m
    echo.
    echo Please configure your git identity:
    echo   git config --global user.name "Your Name"
    echo   git config --global user.email "your.email@example.com"
    echo.
) else (
    echo [32m✓[0m Git user configured as: !name! ^<!email!^>
)

echo.
echo DCO setup complete!
echo.
echo From now on, your commits will automatically be signed-off.
echo You can also manually sign commits with: git commit -s
echo.
echo To learn more about contributing, see CONTRIBUTING.md
