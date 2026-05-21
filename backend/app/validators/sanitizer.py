import json
import re

LANGUAGE_LABELS = {"javascript", "js", "text", "typescript"}


def clean_llm_response(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            _, end_index = decoder.raw_decode(cleaned[index:])
            return cleaned[index : index + end_index].strip()
        except json.JSONDecodeError:
            continue

    return cleaned


def clean_changed_file_content(content: str) -> str:
    lines = content.splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if lines and lines[0].strip().lower() in LANGUAGE_LABELS:
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    if lines and lines[0].strip().startswith("```"):
        fence_language = lines[0].strip()[3:].strip().lower()
        if not fence_language or fence_language in LANGUAGE_LABELS:
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
            if lines and lines[-1].strip() == "```":
                lines.pop()
                while lines and not lines[-1].strip():
                    lines.pop()

    return "\n".join(lines)


def sanitize_changed_files(output: dict) -> dict:
    sanitized_output = {**output}
    changed_files = []

    for changed_file in output.get("changed_files", []):
        if not isinstance(changed_file, dict):
            continue

        sanitized_file = {**changed_file}
        content = sanitized_file.get("content", "")
        if isinstance(content, str):
            sanitized_file["content"] = clean_changed_file_content(content)
        changed_files.append(sanitized_file)

    sanitized_output["changed_files"] = changed_files
    return sanitized_output
