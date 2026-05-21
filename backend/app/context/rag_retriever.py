import hashlib
from pathlib import Path

import chromadb

from app.config import BACKEND_DIR
from app.context.embedding_service import embed_texts
from app.context.rag_indexer import COLLECTION_NAME, index_repo
from app.services.repo_service import resolve_repo_path

CHROMA_DIR = BACKEND_DIR / "data" / "chroma"


def _collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def _stable_repo_id(repo_path: Path) -> str:
    return hashlib.sha1(str(repo_path.resolve()).encode("utf-8")).hexdigest()


def retrieve_rag_context(task: str, repo_path: str, top_k: int = 8) -> list[dict[str, object]]:
    path = resolve_repo_path(repo_path).resolve()
    repo_id = _stable_repo_id(path)
    indexing_result = index_repo(str(path))
    collection = _collection()
    query_embedding = embed_texts([task])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"repo_id": repo_id},
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append(
            {
                "file_path": metadata.get("file_path", ""),
                "content": document,
                "score": distance,
                "distance": distance,
                "language": metadata.get("language", "text"),
                "file_type": metadata.get("file_type", "source"),
                "source": "rag",
                "indexing_result": indexing_result,
            }
        )

    return retrieved
