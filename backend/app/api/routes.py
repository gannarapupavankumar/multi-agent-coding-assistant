from fastapi import APIRouter

from app.agents.developer import run_developer
from app.agents.improver import run_improver
from app.agents.planner import run_planner
from app.agents.reviewer import run_reviewer
from app.context.rag_indexer import index_repo
from app.context.rag_retriever import retrieve_rag_context
from app.config import USE_LANGGRAPH_WORKFLOW
from app.models.schemas import (
    ApplyChangesRequest,
    AskRequest,
    DevelopRequest,
    IndexRepoRequest,
    ImproveRequest,
    PlanRequest,
    RetrieveContextRequest,
    RepoSearchRequest,
    ReviewRequest,
    RunTestsRequest,
    RunWorkflowRequest,
)
from app.services.apply_service import apply_approved_changes
from app.services.langgraph_workflow_service import run_langgraph_workflow
from app.services.llm_service import call_ollama_prompt
from app.services.repo_service import scan_repo
from app.services.test_runner_service import run_tests_for_repo
from app.services.workflow_service import run_multi_agent_workflow

router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ask")
def ask(request: AskRequest) -> dict[str, str]:
    return call_ollama_prompt(request.prompt)


@router.post("/plan")
def plan(request: PlanRequest) -> dict[str, object]:
    return run_planner(request.task)


@router.post("/repo-search")
def repo_search(request: RepoSearchRequest) -> dict[str, object]:
    return scan_repo(request.task, request.repo_path)


@router.post("/index-repo")
def index_repository(request: IndexRepoRequest) -> dict[str, object]:
    return index_repo(request.repo_path)


@router.post("/retrieve-context")
def retrieve_context(request: RetrieveContextRequest) -> dict[str, object]:
    retrieved = retrieve_rag_context(request.task, request.repo_path, request.top_k)
    return {
        "task": request.task,
        "retrieved": [
            {
                "file_path": item.get("file_path", ""),
                "language": item.get("language", ""),
                "file_type": item.get("file_type", ""),
                "source": item.get("source", "rag"),
                "content_preview": str(item.get("content", ""))[:500],
            }
            for item in retrieved
        ],
    }


@router.post("/develop")
def develop(request: DevelopRequest) -> dict[str, object]:
    return run_developer(request.task, request.repo_path, request.matched_files)


@router.post("/review")
def review(request: ReviewRequest) -> dict[str, object]:
    return run_reviewer(request.task, request.developer_output)


@router.post("/improve")
def improve(request: ImproveRequest) -> dict[str, object]:
    return run_improver(
        request.task,
        request.repo_path,
        request.matched_files,
        request.developer_output,
        request.review_output,
    )


@router.post("/run-workflow")
def run_workflow(request: RunWorkflowRequest) -> dict[str, object]:
    if USE_LANGGRAPH_WORKFLOW:
        return run_langgraph_workflow(request)
    return run_multi_agent_workflow(request)


@router.post("/apply-changes")
def apply_changes(request: ApplyChangesRequest) -> dict[str, object]:
    return apply_approved_changes(request)


@router.post("/run-tests")
def run_tests(request: RunTestsRequest) -> dict[str, object]:
    return run_tests_for_repo(request.repo_path, request.timeout_seconds)
