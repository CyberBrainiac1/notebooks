@echo off
setlocal
cd /d "%~dp0"

if exist ".venv311\Scripts\python.exe" (
  .\.venv311\Scripts\python.exe scripts\ollama_autopilot.py --auto-start-ollama
) else (
  py -3.11 scripts\ollama_autopilot.py --auto-start-ollama
)

endlocal
