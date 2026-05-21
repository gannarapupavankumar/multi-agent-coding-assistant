try:
    import tiktoken
except Exception:  # pragma: no cover - fallback covers missing optional runtime
    tiktoken = None


def _get_encoding():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return None


def count_tokens(text: str) -> int:
    encoding = _get_encoding()
    if encoding is None:
        return max(1, len(text) // 4)
    try:
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""

    encoding = _get_encoding()
    if encoding is None:
        return text[: max_tokens * 4]

    try:
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])
    except Exception:
        return text[: max_tokens * 4]
