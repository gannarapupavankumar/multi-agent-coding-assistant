from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    prompt: str


class PlanRequest(BaseModel):
    task: str


class RepoSearchRequest(BaseModel):
    task: str
    repo_path: str


class IndexRepoRequest(BaseModel):
    repo_path: str


class RetrieveContextRequest(BaseModel):
    repo_path: str
    task: str
    top_k: int = 8


class DevelopRequest(BaseModel):
    task: str
    repo_path: str
    matched_files: list[str]


class SuggestedChange(BaseModel):
    file: str
    change: str


class ChangedFile(BaseModel):
    file: str
    reason: str
    content: str


class OriginalFile(BaseModel):
    file: str
    content: str


class DeveloperOutput(BaseModel):
    summary: str
    changed_files: list[ChangedFile]
    original_files: list[OriginalFile] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    task: str
    developer_output: DeveloperOutput


class ImproveRequest(BaseModel):
    task: str
    repo_path: str
    matched_files: list[str]
    developer_output: dict
    review_output: dict


class ApplyChangedFile(BaseModel):
    file: str
    content: str


class ApplyChangesRequest(BaseModel):
    repo_path: str
    approved: bool
    changed_files: list[ApplyChangedFile]


class RunTestsRequest(BaseModel):
    repo_path: str
    timeout_seconds: int = 60


class RunWorkflowRequest(BaseModel):
    task: str
    repo_path: str
    max_time_seconds: int = 180
    max_agent_steps: int = 8
    repeated_failure_threshold: int = 3
    no_improvement_threshold: int = 2
    empty_changed_files_threshold: int = 2
    auto_apply: bool = False
    run_tests: bool = False
    test_timeout_seconds: int = 60
    repeated_test_failure_threshold: int = 2
    max_llm_calls_per_workflow: int = 12
    max_estimated_tokens_per_workflow: int = 30000
    max_estimated_tokens_per_minute: int | None = None
    min_seconds_between_llm_calls: float | None = None
