import re

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

SOURCE_HINTS = {"src", "lib", "app"}
TEST_HINTS = {"test", "tests", "spec", "__tests__"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


def _priority_score(task_tokens: set[str], path: str, content: str, kind: str) -> float:
    path_lower = path.lower()
    content_lower = content.lower()
    score = 0.0

    if kind == "package":
        score += 1.5
    if any(hint in path_lower.split("/") for hint in SOURCE_HINTS):
        score += 4.0
    if any(hint in path_lower for hint in TEST_HINTS):
        score += 4.0
    if any(token in path_lower for token in task_tokens):
        score += 5.0
    if any(token in content_lower for token in task_tokens):
        score += 3.0
    if "lock" in path_lower:
        score -= 100.0

    return score


def rank_context(task: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    if not candidates:
        return []

    task_tokens = set(_tokens(task))
    corpus = [_tokens(f"{candidate['path']} {candidate['content']}") for candidate in candidates]
    bm25_scores = [0.0] * len(candidates)

    if BM25Okapi is not None:
        try:
            bm25 = BM25Okapi(corpus)
            bm25_scores = list(bm25.get_scores(_tokens(task)))
        except Exception:
            bm25_scores = [0.0] * len(candidates)

    ranked = []
    for index, candidate in enumerate(candidates):
        priority = _priority_score(
            task_tokens,
            str(candidate["path"]),
            str(candidate["content"]),
            str(candidate.get("kind", "file")),
        )
        ranked_candidate = {
            **candidate,
            "rank_score": float(bm25_scores[index]) + priority,
            "priority_score": priority,
        }
        ranked.append(ranked_candidate)

    return sorted(ranked, key=lambda item: item["rank_score"], reverse=True)
