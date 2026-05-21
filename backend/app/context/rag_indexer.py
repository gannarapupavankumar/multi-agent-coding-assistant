import hashlib
import os
from pathlib import Path

import chromadb

from app.config import BACKEND_DIR
from app.context.code_chunker import (
    chunk_code,
    file_type_for_path,
    is_indexable_file,
    language_for_path,
)
from app.context.context_retriever import is_context_excluded, summarize_package_json
from app.context.embedding_service import embed_texts
from app.services.repo_service import resolve_repo_path

CHROMA_DIR = BACKEND_DIR / "data" / "chroma"
COLLECTION_NAME = "repo_code_chunks"


def _collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def _stable_repo_id(repo_path: Path) -> str:
    return hashlib.sha1(str(repo_path.resolve()).encode("utf-8")).hexdigest()


def _chunk_id(repo_id: str, file_path: str, chunk_index: int, content: str) -> str:
    content_hash = hashlib.sha1(content.encode("utf-8")).hexdigest()
    raw = f"{repo_id}:{file_path}:{chunk_index}:{content_hash}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def index_repo(repo_path: str) -> dict[str, object]:
    path = resolve_repo_path(repo_path).resolve()
    repo_id = _stable_repo_id(path)
    collection = _collection()
    files_indexed = 0
    chunks_indexed = 0
    files_skipped = []

    for root, dir_names, file_names in os.walk(path):
        root_path = Path(root)
        kept_dirs = []
        for dir_name in dir_names:
            dir_path = root_path / dir_name
            relative_dir = dir_path.relative_to(path).as_posix()
            if is_context_excluded(relative_dir):
                files_skipped.append({"path": relative_dir, "reason": "excluded directory"})
            else:
                kept_dirs.append(dir_name)
        dir_names[:] = kept_dirs

        for file_name in file_names:
            file_path = root_path / file_name

            relative_path = file_path.relative_to(path).as_posix()
            if is_context_excluded(relative_path):
                files_skipped.append({"path": relative_path, "reason": "excluded"})
                continue
            if not is_indexable_file(file_path):
                files_skipped.append({"path": relative_path, "reason": "not indexable"})
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                files_skipped.append({"path": relative_path, "reason": "unreadable"})
                continue

            if file_path.name == "package.json":
                content = summarize_package_json(content)

            chunks = chunk_code(content)
            ids = [_chunk_id(repo_id, relative_path, index, chunk) for index, chunk in enumerate(chunks)]
            embeddings = embed_texts(chunks)
            metadatas = [
                {
                    "repo_path": str(path),
                    "repo_id": repo_id,
                    "file_path": relative_path,
                    "language": language_for_path(relative_path),
                    "file_type": file_type_for_path(relative_path),
                    "chunk_index": index,
                }
                for index, _ in enumerate(chunks)
            ]

            collection.delete(
                where={
                    "$and": [
                        {"repo_id": repo_id},
                        {"file_path": relative_path},
                    ]
                }
            )
            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            files_indexed += 1
            chunks_indexed += len(chunks)

    return {
        "status": "indexed",
        "files_indexed": files_indexed,
        "chunks_indexed": chunks_indexed,
        "files_skipped": files_skipped,
    }
