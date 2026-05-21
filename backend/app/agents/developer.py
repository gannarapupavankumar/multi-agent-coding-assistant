import json

from app.config import CODING_LLM_PROVIDER
from app.context.prompt_packer import build_context_pack, build_developer_prompt
from app.services.llm_service import call_llm
from app.validators.sanitizer import clean_llm_response, sanitize_changed_files


def run_developer(
    task: str,
    repo_path: str,
    matched_files: list[str],
    context_pack: dict[str, object] | None = None,
) -> dict[str, object]:
    context_pack = context_pack or build_context_pack(repo_path, matched_files, task)
    developer_prompt = build_developer_prompt(task, context_pack)

    data = call_llm(developer_prompt, provider=CODING_LLM_PROVIDER)
    response_text = clean_llm_response(data.get("response", ""))
    try:
        developer_data = json.loads(response_text)
    except json.JSONDecodeError:
        developer_data = {
            "agent": "Developer Agent",
            "summary": "Developer response was not valid JSON.",
            "changed_files": [],
        }
    developer_data = sanitize_changed_files(developer_data)

    return {
        "agent": developer_data.get("agent", "Developer Agent"),
        "summary": developer_data.get("summary", ""),
        "changed_files": developer_data.get("changed_files", []),
    }
