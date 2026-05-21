import json

from app.config import IMPROVE_LLM_PROVIDER
from app.context.prompt_packer import build_context_pack, build_improver_prompt
from app.services.llm_service import call_llm
from app.validators.sanitizer import clean_llm_response, sanitize_changed_files


def run_improver(
    task: str,
    repo_path: str,
    matched_files: list[str],
    developer_output: dict,
    review_output: dict,
    context_pack: dict[str, object] | None = None,
    test_feedback: dict[str, object] | None = None,
) -> dict[str, object]:
    context_pack = context_pack or build_context_pack(repo_path, matched_files, task)
    improve_prompt = build_improver_prompt(
        task,
        context_pack,
        developer_output,
        review_output,
        test_feedback,
    )

    data = call_llm(improve_prompt, provider=IMPROVE_LLM_PROVIDER)
    response_text = clean_llm_response(data.get("response", ""))
    try:
        improve_data = json.loads(response_text)
    except json.JSONDecodeError:
        improve_data = {
            "agent": "Improve Patch Agent",
            "summary": "Improve Patch Agent response was not valid JSON.",
            "changed_files": [],
        }
    improve_data = sanitize_changed_files(improve_data)

    return {
        "agent": improve_data.get("agent", "Improve Patch Agent"),
        "summary": improve_data.get("summary", ""),
        "changed_files": improve_data.get("changed_files", []),
    }
