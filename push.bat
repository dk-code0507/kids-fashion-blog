@echo off
cd /d "%~dp0"
echo.
echo ===== kosodatezukan auto push =====
echo.

git add -A

git diff --cached --quiet
if %errorlevel% == 0 (
    echo No changes to push.
    pause
    exit /b 0
)

set /p msg="Commit message (Enter for default): "
if "%msg%"=="" set msg=update: content update

git commit -m "%msg%"
git push

echo.
echo Done! Vercel will deploy automatically.
echo.
pause
