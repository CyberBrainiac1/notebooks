@echo off
setlocal

set "PYEXE=.venv311\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

echo [1/2] Bootstrapping vector memory...
"%PYEXE%" scripts\bootstrap_pranav_memory.py ^
  --auto-start-ollama ^
  --auto-pull-embed-model ^
  --embed-model nomic-embed-text

if errorlevel 1 (
  echo Bootstrap failed.
  exit /b 1
)

echo [2/2] Starting 8B vector-memory assistant...
"%PYEXE%" scripts\ollama_vector_memory_assistant.py ^
  --auto-start-ollama ^
  --chat-model llama3.1:8b ^
  --auto-pull-chat-model ^
  --embed-model nomic-embed-text ^
  --auto-pull-embed-model ^
  --base-dataset data/pranav_profile_qa_v4.jsonl

endlocal
