@echo off
rem Starts Agentic RAG on http://localhost:8501 (creates the virtual environment on first run).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
start "" http://localhost:8501
.venv\Scripts\python.exe -m agentic_rag --port 8501
