# Multi-Agent Coding Assistant

A FastAPI-based local multi-agent coding assistant that can inspect a repo, retrieve relevant code context, generate code changes, review them, apply approved changes, run tests, and use test feedback for repair.

## What This Project Does

Send the backend a coding task and a `repo_path`. The assistant can:

- Retrieve relevant files from the target repository.
- Use hybrid BM25 + ChromaDB RAG context retrieval.
- Run context ranking to prioritize useful source, test, config, and package files.
- Run context pruning to remove lock files, dependency folders, generated files, huge files, and low-value context.
- Run prompt packing to send a compact, token-budgeted context bundle to each LLM agent.
- Call planner, developer, reviewer, and improver agents.
- Review generated changes before writing them.
- Apply approved changes safely inside the target repo.
- Run tests with safe command discovery.
- Repair code using test feedback when tests fail.

## Implementation Overview

The backend is split into small modules so each part has one job:

```text
backend/app/main.py                         FastAPI app creation
backend/app/api/routes.py                   API route handlers
backend/app/models/schemas.py               Pydantic request models
backend/app/config.py                       environment and runtime defaults
backend/app/agents/                         planner, developer, reviewer, improver agents
backend/app/services/llm_service.py         Groq/Ollama provider calls
backend/app/services/langgraph_workflow_service.py
                                             LangGraph workflow orchestration
backend/app/services/repo_service.py        repo scanning and matched file search
backend/app/services/apply_service.py       safe file writing
backend/app/services/test_runner_service.py safe test command discovery and execution
backend/app/context/                        retrieval, RAG, ranking, pruning, prompt packing
backend/app/validators/                     sanitization and deterministic validation
```

## Context Pipeline

The workflow does not send an entire repository to the LLM. It uses a token-aware context pipeline:

1. **Context retrieval**
   - Starts from repo search results.
   - Skips unsafe or noisy paths such as `node_modules`, lock files, build outputs, coverage, `.git`, and virtual environments.
   - Uses ChromaDB RAG retrieval to pull semantically relevant code chunks.

2. **Repo indexing and code chunking**
   - Indexes useful source and text files into local ChromaDB storage at `backend/data/chroma`.
   - Uses local sentence-transformers embeddings.
   - Splits code into smaller chunks before vector search.
   - Summarizes `package.json` to scripts and dependencies instead of sending the full file.

3. **Hybrid retrieval**
   - Combines keyword/BM25-style matching with ChromaDB semantic retrieval.
   - Deduplicates by file path.
   - Keeps exact filename, symbol, source-file, and test-file matches high priority.

4. **Context ranking**
   - Scores files using BM25 plus deterministic priority signals.
   - Prioritizes likely source files, related test files, task keyword matches, and target symbols.

5. **Context pruning**
   - Enforces per-file and total context token budgets.
   - Truncates large files when needed.
   - Skips low-priority files when the context budget is full.

6. **Prompt packing**
   - Builds compact prompts for Developer, Reviewer, and Improve agents.
   - Includes only selected packed context, original task, current subtask when stepwise, review output, and test diagnostics when available.

## Agent Workflow

The primary workflow endpoint uses LangGraph:

- Classifies task complexity deterministically as `small`, `medium`, or `large`.
- Uses `fast_path` for small focused tasks.
- Uses stepwise execution for larger tasks.
- Treats planner steps as suggestions, not commands.
- Prefers existing repo conventions and existing files.
- Checks whether a step is already satisfied before spending another LLM call.
- Reviews generated changes before applying them.
- Applies changes only when `auto_apply=true` and reviewer approval succeeds.
- Runs tests only when `run_tests=true`.
- Uses test failure diagnostics to drive Improve Agent repair.
- Stops when tests pass, the goal is complete, or safety limits are reached.

## Features Implemented

- FastAPI backend
- Modular backend structure
- LangGraph orchestration
- Adaptive execution mode
- `fast_path` for small tasks
- Stepwise execution for larger tasks
- Task complexity classification
- Planner step filtering and merging
- Groq LLM support
- Optional Ollama fallback config
- ChromaDB RAG
- sentence-transformers embeddings
- BM25-style retrieval
- Context retrieval
- Context ranking
- Context pruning
- Prompt packing
- Token counting with `tiktoken`
- Static validation
- Output sanitization for generated file content
- Full updated file content output instead of fragile unified diffs
- Apply changes service
- Test runner service
- Safe test command allowlist
- `.agentconfig.json` test command support
- `auto_apply`
- `run_tests`
- Goal completion checks
- Step satisfaction checks
- Token usage estimate
- Rate-limit guard
- Test-feedback repair loop
- Workflow trace with node-level details
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

## Known Limitations / In Progress

- Larger multi-file workflows are still being tuned.
- Test-feedback repair quality depends on how much useful output the test framework emits.
- The assistant currently suggests and applies full updated file contents, not partial diffs.
- The frontend is still a simple scaffold; most functionality is currently backend-first.

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
- `POST /ask`
- `POST /plan`
- `POST /repo-search`
- `POST /index-repo`
- `POST /retrieve-context`
- `POST /develop`
- `POST /review`
- `POST /improve`
- `POST /run-workflow`
- `POST /run-tests`
- `POST /apply-changes`

## Run Workflow Options

Minimal requests can use only `task`, `repo_path`, `auto_apply`, and `run_tests`. The backend provides defaults for safety and rate-limit controls.

Common optional fields:

```json
{
  "max_time_seconds": 180,
  "max_agent_steps": 8,
  "test_timeout_seconds": 60,
  "repeated_failure_threshold": 3,
  "no_improvement_threshold": 2,
  "empty_changed_files_threshold": 2,
  "repeated_test_failure_threshold": 2,
  "max_llm_calls_per_workflow": 12,
  "max_estimated_tokens_per_workflow": 30000
}
```

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
OLLAMA_MODEL=qwen2.5-coder:1.5b
ENABLE_OLLAMA_FALLBACK=false
USE_LANGGRAPH_WORKFLOW=true
MAX_ESTIMATED_TOKENS_PER_MINUTE=9000
MIN_SECONDS_BETWEEN_LLM_CALLS=2
```

Note: environment variable names are safe to commit in examples, but real `.env` files and real API keys must never be committed.

## Tech Stack

- Python
- FastAPI
- LangGraph
- Groq
- Optional local Ollama
- ChromaDB
- sentence-transformers
- rank-bm25
- tiktoken
- Local repo analysis
- Jest, pytest, Maven, Gradle, SBT, dotnet, Go, Cargo, and Make test runner support

## Safety Notes

- `.env` and `*.env` files are ignored.
- Virtual environments are ignored.
- `node_modules` is ignored.
- Generated ChromaDB data at `backend/data/chroma` is ignored.
- Apply Changes prevents path traversal and writes only inside the requested `repo_path`.
- Test execution uses an allowlist and does not run arbitrary user-provided shell commands.

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
