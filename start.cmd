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
node scripts/npm.cjs --version >nul
if errorlevel 1 (
  pause
  exit /b 1
)
if not exist node_modules\.package-lock.json (
  node scripts/npm.cjs ci --no-audit --no-fund
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
node scripts/npm.cjs run dev:web -- --port 5178 --open
pause
