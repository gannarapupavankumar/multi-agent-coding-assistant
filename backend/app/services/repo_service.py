from pathlib import Path

from fastapi import HTTPException

IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
SEARCH_KEYWORDS = {
    "auth",
    "controller",
    "email",
    "register",
    "registration",
    "route",
    "schema",
    "test",
    "user",
    "validation",
}


def resolve_repo_path(repo_path: str) -> Path:
    path = Path(repo_path).expanduser()
    if not path.exists() or not path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"repo_path does not exist or is not a directory: {repo_path}",
        )
    return path


def scan_repo(task: str, repo_path: str) -> dict[str, object]:
    path = resolve_repo_path(repo_path)
    task_keywords = {
        word.lower()
        for word in task.replace("-", " ").replace("_", " ").split()
        if len(word) >= 4
    }
    keywords = SEARCH_KEYWORDS | task_keywords
    matched_files = []

    for file_path in path.rglob("*"):
        if any(part in IGNORED_DIRS for part in file_path.parts):
            continue
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(path).as_posix()
        path_text = relative_path.lower()
        filename_matches = sorted(keyword for keyword in keywords if keyword in path_text)

        content_matches = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore").lower()
            content_matches = sorted(keyword for keyword in keywords if keyword in content)
        except OSError:
            pass

        if filename_matches or content_matches:
            reasons = []
            if filename_matches:
                reasons.append(f"filename matches: {', '.join(filename_matches[:5])}")
            if content_matches:
                reasons.append(f"content matches: {', '.join(content_matches[:5])}")

            matched_files.append(
                {
                    "path": relative_path,
                    "reason": "; ".join(reasons),
                }
            )

    return {
        "agent": "Repo Search Agent",
        "task": task,
        "repo_path": repo_path,
        "matched_files": matched_files,
    }


def read_file_sections(repo_path: str, matched_files: list[str]) -> list[str]:
    path = resolve_repo_path(repo_path)
    file_sections = []
    for matched_file in matched_files:
        file_path = path / matched_file
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"matched file does not exist: {matched_file}",
            )

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"could not read matched file: {matched_file}",
            ) from exc

        file_sections.append(
            f"File: {matched_file}\n"
            "```text\n"
            f"{content}\n"
            "```"
        )

    return file_sections


def read_original_files(repo_path: str, matched_files: list[str]) -> list[dict[str, str]]:
    path = resolve_repo_path(repo_path)
    original_files = []
    for matched_file in matched_files:
        try:
            content = (path / matched_file).read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"could not read {matched_file}: {exc}",
            ) from exc

        original_files.append({"file": matched_file, "content": content})

    return original_files

