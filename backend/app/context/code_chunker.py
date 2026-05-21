from pathlib import Path

LANGUAGE_BY_SUFFIX = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".py": "python",
    ".java": "java",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".scala": "scala",
    ".kt": "kotlin",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
}
INDEXABLE_SUFFIXES = set(LANGUAGE_BY_SUFFIX)


def is_indexable_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in INDEXABLE_SUFFIXES


def language_for_path(file_path: str) -> str:
    return LANGUAGE_BY_SUFFIX.get(Path(file_path).suffix.lower(), "text")


def file_type_for_path(file_path: str) -> str:
    lower_path = file_path.lower()
    if "test" in lower_path or "spec" in lower_path:
        return "test"
    if Path(file_path).name == "package.json":
        return "package"
    if lower_path.endswith((".md", ".yaml", ".yml", ".json")):
        return "config"
    return "source"


def chunk_code(content: str, max_chunk_chars: int = 1500, overlap_chars: int = 180) -> list[str]:
    if len(content) <= max_chunk_chars:
        return [content]

    chunks = []
    start = 0
    while start < len(content):
        end = min(start + max_chunk_chars, len(content))
        chunks.append(content[start:end])
        if end == len(content):
            break
        start = max(0, end - overlap_chars)

    return chunks
