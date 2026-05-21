from app.context.token_counter import count_tokens, truncate_to_token_budget

DEFAULT_MAX_PER_FILE_TOKENS = 1200
DEFAULT_MAX_TOTAL_CONTEXT_TOKENS = 6000


def prune_context(
    ranked_files: list[dict[str, object]],
    existing_skipped_files: list[dict[str, str]] | None = None,
    max_per_file_tokens: int = DEFAULT_MAX_PER_FILE_TOKENS,
    max_total_context_tokens: int = DEFAULT_MAX_TOTAL_CONTEXT_TOKENS,
) -> dict[str, object]:
    included = []
    skipped = list(existing_skipped_files or [])
    truncated = []
    total_tokens = 0

    for file_info in ranked_files:
        path = str(file_info["path"])
        content = str(file_info["content"])
        original_tokens = count_tokens(content)
        file_token_budget = min(max_per_file_tokens, max_total_context_tokens - total_tokens)

        if file_token_budget <= 0:
            skipped.append({"path": path, "reason": "total context token budget exhausted"})
            continue

        if original_tokens > file_token_budget:
            content = truncate_to_token_budget(content, file_token_budget)
            truncated.append(path)

        current_tokens = count_tokens(content)
        if total_tokens + current_tokens > max_total_context_tokens:
            skipped.append({"path": path, "reason": "total context token budget exceeded"})
            continue

        included.append(
            {
                **file_info,
                "content": content,
                "tokens": current_tokens,
                "truncated": path in truncated,
            }
        )
        total_tokens += current_tokens

    return {
        "files": included,
        "included_files": [file_info["path"] for file_info in included],
        "skipped_files": skipped,
        "truncated_files": truncated,
        "total_context_tokens": total_tokens,
    }
