import json

from app.config import PLANNER_LLM_PROVIDER
from app.services.llm_service import call_llm
from app.validators.sanitizer import clean_llm_response


def run_planner(task: str) -> dict[str, object]:
    planner_prompt = f"""
You are a Planner Agent for a coding assistant.

Create a practical plan for the requested coding task. Identify likely files to inspect, implementation steps, and risks.
Return raw JSON only. Do not use markdown, code fences, explanations, comments, or extra text.

The JSON must exactly follow this format:
{{
  "agent": "Planner Agent",
  "goal": "string",
  "steps": ["string"],
  "files_to_check": ["string"],
  "risks": ["string"]
}}

Task: {task}
"""

    data = call_llm(planner_prompt, provider=PLANNER_LLM_PROVIDER)
    response_text = clean_llm_response(data.get("response", ""))
    try:
        planner_data = json.loads(response_text)
    except json.JSONDecodeError:
        planner_data = {
            "agent": "Planner Agent",
            "goal": task,
            "steps": [response_text],
            "files_to_check": [],
            "risks": ["Planner response was not valid JSON."],
        }

    return {
        "agent": planner_data.get("agent", "Planner Agent"),
        "goal": planner_data.get("goal", task),
        "steps": planner_data.get("steps", []),
        "files_to_check": planner_data.get("files_to_check", []),
        "risks": planner_data.get("risks", []),
    }
