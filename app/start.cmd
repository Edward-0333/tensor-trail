@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
where npm >nul 2>nul
if errorlevel 1 (
  echo 请先安装 Node.js 20 或更新版本，再运行本文件。
  pause
  exit /b 1
)
if not exist node_modules (
  call npm.cmd ci --registry https://registry.npmjs.org
  if errorlevel 1 (
    echo 依赖安装失败，请查看上方信息。
    pause
    exit /b 1
  )
)
call npm.cmd run dev -- --port 5178 --open
pause
