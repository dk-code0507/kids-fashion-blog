@echo off
chcp 65001 > nul
cd /d C:\Users\dkoba\projects\kids-fashion-blog
git add -A
git status
echo.
set /p msg="Commit message: "
git commit -m "%msg%"
git push origin main
echo.
echo Done! Vercel will auto-deploy now.
pause
