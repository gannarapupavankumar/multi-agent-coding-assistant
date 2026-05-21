import json

from app.context.context_pruner import prune_context
from app.context.context_ranker import rank_context
from app.context.context_retriever import retrieve_context


def build_context_pack(repo_path: str, matched_files: list[str], task: str) -> dict[str, object]:
    retrieved = retrieve_context(repo_path, matched_files)
    candidates = list(retrieved["candidates"])
    rag_details = {
        "status": "not_run",
        "retrieved_chunks": [],
        "retrieved_files": [],
        "warning": "",
        "indexing_result": {},
    }

    try:
        from app.context.rag_retriever import retrieve_rag_context

        rag_chunks = retrieve_rag_context(task, repo_path)
        existing_paths = {candidate["path"] for candidate in candidates}
        for chunk in rag_chunks:
            file_path = chunk.get("file_path", "")
            if not file_path:
                continue
            if file_path in existing_paths:
                continue

            candidates.append(
                {
                    "path": file_path,
                    "content": chunk.get("content", ""),
                    "kind": chunk.get("file_type", "file"),
                    "tokens": 0,
                    "source": "rag",
                    "rag_score": chunk.get("score"),
                }
            )
            existing_paths.add(file_path)

        rag_details = {
            "status": "ok",
            "retrieved_chunks": [
                {
                    "file_path": chunk.get("file_path", ""),
                    "score": chunk.get("score"),
                    "language": chunk.get("language", ""),
                    "file_type": chunk.get("file_type", ""),
                }
                for chunk in rag_chunks
            ],
            "retrieved_files": sorted({chunk.get("file_path", "") for chunk in rag_chunks if chunk.get("file_path")}),
            "warning": "",
            "indexing_result": rag_chunks[0].get("indexing_result", {}) if rag_chunks else {},
        }
    except Exception as exc:
        rag_details = {
            "status": "warning",
            "retrieved_chunks": [],
            "retrieved_files": [],
            "warning": f"RAG retrieval unavailable: {exc}",
            "indexing_result": {},
        }

    ranked = rank_context(task, candidates)
    pruned = prune_context(ranked, retrieved["skipped_files"])

    return {
        **pruned,
        "retrieved_files": retrieved["retrieved_files"],
        "bm25_retrieved_files": retrieved["retrieved_files"],
        "rag": rag_details,
        "ranked_files": [file_info["path"] for file_info in ranked],
    }


def format_context_files(context_pack: dict[str, object]) -> str:
    sections = []
    for file_info in context_pack.get("files", []):
        sections.append(
            f"File: {file_info['path']}\n"
            f"Tokens: {file_info['tokens']}\n"
            "```text\n"
            f"{file_info['content']}\n"
            "```"
        )
    return "\n\n".join(sections)


def context_trace_details(context_pack: dict[str, object]) -> dict[str, object]:
    return {
        "retrieved_files": context_pack.get("retrieved_files", []),
        "bm25_retrieved_files": context_pack.get("bm25_retrieved_files", []),
        "rag": context_pack.get("rag", {}),
        "ranked_files": context_pack.get("ranked_files", []),
        "included_files": context_pack.get("included_files", []),
        "skipped_files": context_pack.get("skipped_files", []),
        "truncated_files": context_pack.get("truncated_files", []),
        "total_context_tokens": context_pack.get("total_context_tokens", 0),
    }


def build_developer_prompt(task: str, context_pack: dict[str, object]) -> str:
    return f"""
You are a Developer Agent for a coding assistant.

Return raw JSON only. Do not use markdown, code fences, explanations, comments, or extra text.
Do not write files. Only suggest changes.
Return actual code-level changes for the requested task, not vague descriptions.
Use only the provided packed context.
Do not invent file paths.
Return full updated file contents for each changed file. Do not return a unified diff.
Only include files from the packed context in changed_files.
Do not reference functions, classes, imports, variables, or modules that are not defined unless the updated file content also defines or imports them.
If the task changes API behavior, include test file updates if a test file is in the packed context.
If there is not enough context to create reliable updated file contents, say what file is missing in "summary" and return an empty changed_files list instead of guessing.

The JSON must exactly follow this format:
{{
  "agent": "Developer Agent",
  "summary": "string",
  "changed_files": [
    {{
      "file": "string",
      "reason": "string",
      "content": "full updated file content as string"
    }}
  ]
}}

Task: {task}

Packed context:
{format_context_files(context_pack)}
"""


def build_improver_prompt(
    task: str,
    context_pack: dict[str, object],
    developer_output: dict,
    review_output: dict,
    test_feedback: dict[str, object] | None = None,
) -> str:
    test_feedback_section = ""
    if test_feedback:
        test_feedback_section = f"""
Test failure feedback:
{json.dumps(test_feedback, indent=2)}
"""

    return f"""
You are an Improve Patch Agent for a coding assistant.

Fix the developer output using the reviewer issues, improvements, and test failure feedback when provided.
When test failure feedback is provided, repair the previous changed files using those diagnostics.
Do not blindly regenerate a different solution. Focus on the failure_summary, failure_type, exit_code, stdout_tail, and stderr_tail.
Return raw JSON only. Do not use markdown, code fences, explanations, comments, or extra text.
Do not write files. Only suggest changes.
Do not return vague descriptions. Return concrete code-level changes.
Use only the provided packed context.

The improved output must:
- directly address the original task
- preserve existing working logic unless the task requires changing it
- avoid missing helper functions or undefined references
- include test changes when behavior changes
- return full updated file contents for every changed file
- not return a unified diff
- modify only files included in the packed context

The JSON must exactly follow this format:
{{
  "agent": "Improve Patch Agent",
  "summary": "string",
  "changed_files": [
    {{
      "file": "string",
      "reason": "string",
      "content": "full updated file content as string"
    }}
  ]
}}

Task: {task}

Previous developer output:
{json.dumps(developer_output, indent=2)}

Reviewer output:
{json.dumps(review_output, indent=2)}

{test_feedback_section}

Packed context:
{format_context_files(context_pack)}
"""


def build_reviewer_prompt(
    task: str,
    context_pack: dict[str, object],
    developer_output_json: str,
    validation_issues: list[str],
) -> str:
    return f"""
You are a Reviewer Agent for a coding assistant.

Review the developer output against the original task and packed original context.
Be strict. Approve only if:
- changed_files contains real full-file implementation details
- changed file contents directly address the task
- every changed file path is relevant to the task and present in context
- every changed file content is directly writable to disk as source code
- referenced functions, classes, imports, variables, and modules are defined or added
- imported dependencies are already present in context or are added to package.json
- referenced local files exist in context
- likely syntax issues are avoided
- edge cases are considered
- tests are added or suggested when behavior changes
- no obvious security or maintainability issue is introduced

Reject changed_files if:
- content includes markdown code fences such as ```javascript
- content includes non-code labels such as typeScript
- content is not directly valid source code
- it imports a dependency that is not already present or added to package.json
- it references files that do not exist in the original context
- it changes existing behavior unnecessarily
- test code does not match the current test style
- module syntax does not match the existing file style

Reject vague descriptions, empty changed_files, unified diffs, partial snippets, or invented file paths.
If important context is missing, set approved to false and explain what is missing in issues.
Return raw JSON only. Do not use markdown, code fences, explanations, comments, or extra text.

The JSON must exactly follow this format:
{{
  "agent": "Reviewer Agent",
  "approved": true,
  "issues": ["string"],
  "improvements": ["string"],
  "risk_level": "low | medium | high"
}}

Task: {task}

Local validation issues that must be treated as reviewer issues:
{json.dumps(validation_issues, indent=2)}

Developer output:
{developer_output_json}

Packed original context:
{format_context_files(context_pack)}
"""
