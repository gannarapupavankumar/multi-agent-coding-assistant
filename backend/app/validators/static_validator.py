import posixpath

from app.models.schemas import DeveloperOutput


def validate_changed_files_for_review(developer_output: DeveloperOutput) -> list[str]:
    issues = []
    original_by_file = {file.file: file.content for file in developer_output.original_files}
    matched_paths = {posixpath.normpath(path) for path in original_by_file}
    matched_stems = {posixpath.splitext(path)[0] for path in matched_paths}
    matched_index_dirs = {
        posixpath.dirname(path)
        for path in matched_paths
        if posixpath.splitext(posixpath.basename(path))[0] == "index"
    }
    package_json_content = original_by_file.get("package.json", "")

    if not developer_output.changed_files:
        issues.append("changed_files is empty.")

    for changed_file in developer_output.changed_files:
        content = changed_file.content
        stripped = content.strip()
        original_content = original_by_file.get(changed_file.file, "")
        normalized_changed_file = posixpath.normpath(changed_file.file)

        if matched_paths and normalized_changed_file not in matched_paths:
            issues.append(f"{changed_file.file} is not one of the matched files.")
        if "```" in content:
            issues.append(f"{changed_file.file} contains markdown code fences.")
        if "typeScript" in content or stripped.lower().startswith("typescript"):
            issues.append(f"{changed_file.file} contains non-code labels.")
        if stripped.startswith("Here ") or stripped.startswith("This "):
            issues.append(f"{changed_file.file} appears to contain explanation text, not source code.")
        if original_content and "require(" in original_content and (
            "\nimport " in content or stripped.startswith("import ")
        ):
            issues.append(f"{changed_file.file} changes CommonJS module style to ES module syntax.")
        if original_content and "module.exports" in original_content and "export default" in content:
            issues.append(f"{changed_file.file} changes CommonJS exports to ES module exports.")

        for quote in ('require("', "require('", 'from "', "from '"):
            start = 0
            while True:
                index = content.find(quote, start)
                if index == -1:
                    break
                path_start = index + len(quote)
                path_end = content.find(quote[-1], path_start)
                if path_end == -1:
                    break
                referenced_path = content[path_start:path_end]
                start = path_end + 1
                if not referenced_path.startswith("."):
                    if (
                        referenced_path not in original_content
                        and f'"{referenced_path}"' not in package_json_content
                    ):
                        issues.append(
                            f"{changed_file.file} imports {referenced_path}, which is not present in the original file or package.json context."
                        )
                    continue

                base_dir = posixpath.dirname(normalized_changed_file)
                normalized = posixpath.normpath(posixpath.join(base_dir, referenced_path))
                if (
                    matched_paths
                    and normalized not in matched_paths
                    and normalized not in matched_stems
                    and normalized not in matched_index_dirs
                ):
                    issues.append(
                        f"{changed_file.file} references {referenced_path}, which is not present in matched file context."
                    )

    return issues
