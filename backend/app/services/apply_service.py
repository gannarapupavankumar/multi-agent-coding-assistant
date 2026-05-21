from pathlib import Path

from fastapi import HTTPException

from app.models.schemas import ApplyChangesRequest


def apply_approved_changes(request: ApplyChangesRequest) -> dict[str, object]:
    if request.approved is not True:
        raise HTTPException(
            status_code=400,
            detail="Changes were not approved. Refusing to write files.",
        )

    repo_path = Path(request.repo_path).expanduser().resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"repo_path does not exist or is not a directory: {request.repo_path}",
        )

    files_written = []
    for changed_file in request.changed_files:
        target_path = (repo_path / changed_file.file).resolve()
        if repo_path != target_path and repo_path not in target_path.parents:
            raise HTTPException(
                status_code=400,
                detail=f"file path escapes repo_path: {changed_file.file}",
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(changed_file.content, encoding="utf-8")
        files_written.append(changed_file.file)

    return {
        "status": "applied",
        "files_written": files_written,
    }
