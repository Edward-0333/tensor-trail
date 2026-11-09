@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Please install Node.js 24 LTS. See README.md.
  pause
  exit /b 1
)
if not exist node_modules\.package-lock.json (
  node scripts/npm.cjs ci --ignore-scripts --no-audit --no-fund
  if errorlevel 1 goto failed
)
node scripts/npm.cjs run setup:desktop
if errorlevel 1 goto failed
node scripts/npm.cjs run build:desktop
if errorlevel 1 goto failed
node scripts/desktop.cjs --detach
if errorlevel 1 goto failed
exit /b 0
:failed
echo Desktop startup failed. Please check the output above.
pause
exit /b 1
