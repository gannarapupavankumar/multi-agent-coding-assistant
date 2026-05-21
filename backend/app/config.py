import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "groq")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")

PLANNER_LLM_PROVIDER = os.getenv("PLANNER_LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
CODING_LLM_PROVIDER = os.getenv("CODING_LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
REVIEW_LLM_PROVIDER = os.getenv("REVIEW_LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
IMPROVE_LLM_PROVIDER = os.getenv("IMPROVE_LLM_PROVIDER", DEFAULT_LLM_PROVIDER)

JSON_SYSTEM_PROMPT = (
    "Return only valid JSON. Do not include markdown. Do not include explanations. "
    "Do not include <think> blocks."
)

USE_LANGGRAPH_WORKFLOW = os.getenv("USE_LANGGRAPH_WORKFLOW", "true").lower() == "true"
MAX_ESTIMATED_TOKENS_PER_MINUTE = int(os.getenv("MAX_ESTIMATED_TOKENS_PER_MINUTE", "9000"))
MIN_SECONDS_BETWEEN_LLM_CALLS = float(os.getenv("MIN_SECONDS_BETWEEN_LLM_CALLS", "2"))
