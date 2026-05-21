import json
from pathlib import Path

from app.context.token_counter import count_tokens
from app.services.repo_service import resolve_repo_path

EXCLUDED_FILE_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "venv",
}


def _is_excluded(path: str) -> bool:
    parts = Path(path).parts
    return Path(path).name in EXCLUDED_FILE_NAMES or any(
        part in EXCLUDED_DIR_NAMES for part in parts
    )


def _summarize_package_json(content: str) -> str:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content

    summary = {
        key: data.get(key, {})
        for key in ("scripts", "dependencies", "devDependencies")
        if data.get(key)
    }
    return json.dumps(summary, indent=2)


def retrieve_context(repo_path: str, matched_files: list[str]) -> dict[str, object]:
    path = resolve_repo_path(repo_path)
    candidates = []
    skipped = []

    for matched_file in matched_files:
        if _is_excluded(matched_file):
            skipped.append({"path": matched_file, "reason": "excluded generated/dependency/lock file"})
            continue

        file_path = path / matched_file
        if not file_path.exists() or not file_path.is_file():
            skipped.append({"path": matched_file, "reason": "file missing"})
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            skipped.append({"path": matched_file, "reason": "file unreadable"})
            continue

        kind = "package" if Path(matched_file).name == "package.json" else "file"
        if kind == "package":
            content = _summarize_package_json(content)

        candidates.append(
            {
                "path": matched_file,
                "content": content,
                "kind": kind,
                "tokens": count_tokens(content),
            }
        )

    return {
        "candidates": candidates,
        "skipped_files": skipped,
        "retrieved_files": [candidate["path"] for candidate in candidates],
    }


def is_context_excluded(path: str) -> bool:
    return _is_excluded(path)


def summarize_package_json(content: str) -> str:
    return _summarize_package_json(content)
