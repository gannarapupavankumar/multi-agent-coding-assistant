# Multi-Agent Coding Assistant

A FastAPI-based local multi-agent coding assistant that can inspect a repo, retrieve relevant code context, generate code changes, review them, apply approved changes, run tests, and use test feedback for repair.

## What This Project Does

Send the backend a coding task and a `repo_path`. The assistant can:

- Retrieve relevant files from the target repository.
- Use hybrid BM25 + ChromaDB RAG context retrieval.
- Rank, prune, and pack context before LLM calls.
- Call planner, developer, reviewer, and improver agents.
- Review generated changes before writing them.
- Apply approved changes safely inside the target repo.
- Run tests with safe command discovery.
- Repair code using test feedback when tests fail.

## Features Implemented

- FastAPI backend
- Modular backend structure
- LangGraph orchestration
- Adaptive execution mode
- `fast_path` for small tasks
- Stepwise execution for larger tasks
- Groq LLM support
- Optional Ollama fallback config
- ChromaDB RAG
- sentence-transformers embeddings
- BM25-style retrieval
- Context ranking, pruning, and prompt packing
- Static validation
- Apply changes service
- Test runner service
- `auto_apply`
- `run_tests`
- Goal completion checks
- Step satisfaction checks
- Token usage estimate
- Rate-limit guard
- Backend defaults so users do not need to pass many thresholds

## Current Working Proof

Calculator workflow:

```text
Task: Add sextuple function to calculator.js and tests. sextuple(3) should return 18.
Result:
- status: approved
- stop_reason: tests_passed
- execution_mode: fast_path
- task_complexity: small
- tests passed: 15
```
## API Usage

Minimal `POST /run-workflow` payload:

```json
{
  "task": "Add sextuple function to calculator.js and tests. sextuple(3) should return 18.",
  "repo_path": "C:\\Users\\ganna\\OneDrive\\Documents\\sample_test_agent",
  "auto_apply": true,
  "run_tests": true
}
```

## Other Endpoints

- `GET /health`
- `POST /index-repo`
- `POST /retrieve-context`
- `POST /run-tests`
- `POST /apply-changes`

## How To Run Locally

PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

The API will usually be available at:

```text
http://127.0.0.1:8000
```

## Environment Variables

Create a local `.env` file in `backend/`. Do not commit it.

Safe example:

```env
GROQ_API_KEY=your_groq_api_key_here
DEFAULT_LLM_PROVIDER=groq
PLANNER_LLM_PROVIDER=groq
CODING_LLM_PROVIDER=groq
REVIEW_LLM_PROVIDER=groq
IMPROVE_LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-120b
ENABLE_OLLAMA_FALLBACK=false
```

## Tech Stack

- Python
- FastAPI
- LangGraph
- Groq
- ChromaDB
- sentence-transformers
- Local repo analysis
- Jest test runner support

## Project Purpose

This is a learning and portfolio project for understanding coding agents, RAG, LangGraph workflows, token-aware execution, and test-feedback repair.

## GitHub Setup

After creating an empty GitHub repository named `multi-agent-coding-assistant`, connect and push with:

```powershell
git branch -M main
git remote add origin https://github.com/<my-username>/multi-agent-coding-assistant.git
git push -u origin main
```

If GitHub CLI is installed, you can alternatively run:

```powershell
gh repo create multi-agent-coding-assistant --public --source=. --remote=origin --push
```
