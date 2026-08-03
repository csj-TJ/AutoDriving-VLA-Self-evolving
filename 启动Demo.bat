@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动 Driving Reflection Demo...
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8000'"
python scripts\run_demo.py
if errorlevel 1 (
  echo.
  echo 启动失败：请确认已安装 Python 和 requirements.txt 中的依赖。
  pause
)
