import json

from app.config import REVIEW_LLM_PROVIDER
from app.context.context_pruner import prune_context
from app.context.context_ranker import rank_context
from app.context.prompt_packer import build_reviewer_prompt
from app.context.token_counter import count_tokens
from app.models.schemas import DeveloperOutput
from app.services.llm_service import call_llm
from app.validators.sanitizer import clean_llm_response
from app.validators.static_validator import validate_changed_files_for_review


def _context_pack_from_original_files(
    task: str,
    developer_output: DeveloperOutput,
) -> dict[str, object]:
    candidates = [
        {
            "path": file.file,
            "content": file.content,
            "kind": "file",
            "tokens": count_tokens(file.content),
        }
        for file in developer_output.original_files
    ]
    ranked = rank_context(task, candidates)
    pruned = prune_context(ranked)
    return {
        **pruned,
        "retrieved_files": [candidate["path"] for candidate in candidates],
        "ranked_files": [file_info["path"] for file_info in ranked],
    }


def run_reviewer(
    task: str,
    developer_output: DeveloperOutput,
    context_pack: dict[str, object] | None = None,
) -> dict[str, object]:
    validation_issues = validate_changed_files_for_review(developer_output)
    context_pack = context_pack or _context_pack_from_original_files(task, developer_output)
    reviewer_prompt = build_reviewer_prompt(
        task,
        context_pack,
        developer_output.model_dump_json(indent=2),
        validation_issues,
    )

    data = call_llm(reviewer_prompt, provider=REVIEW_LLM_PROVIDER)
    response_text = clean_llm_response(data.get("response", ""))
    try:
        reviewer_data = json.loads(response_text)
    except json.JSONDecodeError:
        reviewer_data = {
            "agent": "Reviewer Agent",
            "approved": False,
            "issues": ["Reviewer response was not valid JSON."],
            "improvements": [response_text],
            "risk_level": "medium",
        }

    if validation_issues:
        model_issues = reviewer_data.get("issues", [])
        reviewer_data["approved"] = False
        reviewer_data["issues"] = model_issues + validation_issues
        reviewer_data["risk_level"] = "high"

    return {
        "agent": reviewer_data.get("agent", "Reviewer Agent"),
        "approved": reviewer_data.get("approved", False),
        "issues": reviewer_data.get("issues", []),
        "improvements": reviewer_data.get("improvements", []),
        "risk_level": reviewer_data.get("risk_level", "medium"),
    }
