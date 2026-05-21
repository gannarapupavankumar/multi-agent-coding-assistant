import json
import re
import time
from typing import Any, TypedDict

from fastapi import HTTPException
from langgraph.graph import END, StateGraph

from app.agents.developer import run_developer
from app.agents.improver import run_improver
from app.agents.planner import run_planner
from app.agents.reviewer import run_reviewer
from app.config import MAX_ESTIMATED_TOKENS_PER_MINUTE, MIN_SECONDS_BETWEEN_LLM_CALLS
from app.context.prompt_packer import build_context_pack, context_trace_details
from app.context.token_counter import count_tokens
from app.models.schemas import (
    ApplyChangedFile,
    ApplyChangesRequest,
    DeveloperOutput,
    RunWorkflowRequest,
)
from app.services.apply_service import apply_approved_changes
from app.services.llm_service import LLMProviderError
from app.services.repo_service import scan_repo
from app.services.test_runner_service import run_tests_for_repo

DESCRIPTIVE_STEP_MARKERS = (
    "open ",
    "review ",
    "inspect ",
    "check ",
    "locate ",
    "find ",
    "run test",
    "verify ",
    "ensure ",
    "commit ",
    "document ",
)
ARCHITECTURE_TERMS = {
    "controller",
    "controllers",
    "service",
    "services",
    "model",
    "models",
    "dto",
    "dtos",
    "schema",
    "schemas",
    "repository",
    "repositories",
    "repo",
    "helper",
    "helpers",
    "config",
    "middleware",
    "route",
    "routes",
}
NEW_STRUCTURE_MARKERS = (
    "new file",
    "create file",
    "create a file",
    "add file",
    "create new",
    "add new",
    "new layer",
)


class WorkflowState(TypedDict, total=False):
    task: str
    repo_path: str
    max_time_seconds: int
    max_agent_steps: int
    repeated_failure_threshold: int
    no_improvement_threshold: int
    empty_changed_files_threshold: int
    repeated_test_failure_threshold: int
    max_llm_calls_per_workflow: int
    max_estimated_tokens_per_workflow: int
    max_estimated_tokens_per_minute: int
    min_seconds_between_llm_calls: float
    auto_apply: bool
    run_tests: bool
    test_timeout_seconds: int
    plan: dict[str, Any]
    plan_steps: list[dict[str, Any]]
    current_step_index: int
    current_subtask: str
    completed_steps: list[str]
    failed_steps: list[str]
    step_results: list[dict[str, Any]]
    step_attempts: dict[str, int]
    per_step_context: dict[str, Any]
    per_step_changed_files: dict[str, list[dict[str, Any]]]
    matched_files: list[str]
    context_pipeline_result: dict[str, Any]
    original_files: list[dict[str, str]]
    developer_output: dict[str, Any]
    review_output: dict[str, Any]
    final_changed_files: list[dict[str, Any]]
    apply_result: dict[str, Any]
    test_result: dict[str, Any]
    test_feedback: dict[str, Any]
    final_test_pending: bool
    step_satisfied: bool
    satisfaction_reason: str
    test_attempts: int
    attempts: int
    agent_steps: int
    tool_errors: int
    repeated_issue_counts: dict[str, int]
    repeated_test_failure_counts: dict[str, int]
    previous_changed_files: str | None
    no_improvement_count: int
    empty_changed_files_count: int
    total_llm_calls: int
    total_estimated_input_tokens: int
    total_estimated_output_tokens: int
    llm_token_events: list[dict[str, Any]]
    last_llm_call_at: float
    status: str
    stop_reason: str
    workflow_trace: list[dict[str, Any]]
    start_time: float
    elapsed_seconds: float
    task_complexity: str
    execution_mode: str
    goal_complete: bool
    goal_completion_reason: str


def run_langgraph_workflow(request: RunWorkflowRequest) -> dict[str, object]:
    graph = _build_graph()
    final_state = graph.invoke(_initial_state(request), {"recursion_limit": 100})
    return _build_response(final_state)


def _build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("task_complexity_node", task_complexity_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("plan_decomposition_node", plan_decomposition_node)
    graph.add_node("select_next_step_node", select_next_step_node)
    graph.add_node("repo_search_node", repo_search_node)
    graph.add_node("context_pipeline_node", context_pipeline_node)
    graph.add_node("step_satisfaction_node", step_satisfaction_node)
    graph.add_node("developer_node", developer_node)
    graph.add_node("reviewer_node", reviewer_node)
    graph.add_node("apply_changes_node", apply_changes_node)
    graph.add_node("test_runner_node", test_runner_node)
    graph.add_node("improve_node", improve_node)
    graph.add_node("step_completion_node", step_completion_node)
    graph.add_node("goal_completion_node", goal_completion_node)
    graph.add_node("final_node", final_node)

    graph.set_entry_point("repo_search_node")
    graph.add_conditional_edges("task_complexity_node", _route_after_complexity)
    graph.add_conditional_edges("planner_node", _route_after_planner)
    graph.add_conditional_edges("plan_decomposition_node", _route_after_plan_decomposition)
    graph.add_conditional_edges("select_next_step_node", _route_after_select_next_step)
    graph.add_conditional_edges("repo_search_node", _route_after_repo_search)
    graph.add_conditional_edges("context_pipeline_node", _route_after_context_node)
    graph.add_conditional_edges("step_satisfaction_node", _route_after_satisfaction_node)
    graph.add_conditional_edges("developer_node", _route_after_developer_node)
    graph.add_conditional_edges("reviewer_node", _route_after_reviewer_node)
    graph.add_conditional_edges("apply_changes_node", _route_after_apply_node)
    graph.add_conditional_edges("test_runner_node", _route_after_test_node)
    graph.add_conditional_edges("improve_node", _route_after_developer_node)
    graph.add_conditional_edges("step_completion_node", _route_after_step_completion)
    graph.add_conditional_edges("goal_completion_node", _route_after_goal_completion)
    graph.add_edge("final_node", END)
    return graph.compile()


def _initial_state(request: RunWorkflowRequest) -> WorkflowState:
    return {
        "task": request.task,
        "repo_path": request.repo_path,
        "max_time_seconds": request.max_time_seconds,
        "max_agent_steps": request.max_agent_steps,
        "repeated_failure_threshold": request.repeated_failure_threshold,
        "no_improvement_threshold": request.no_improvement_threshold,
        "empty_changed_files_threshold": request.empty_changed_files_threshold,
        "repeated_test_failure_threshold": request.repeated_test_failure_threshold,
        "max_llm_calls_per_workflow": request.max_llm_calls_per_workflow,
        "max_estimated_tokens_per_workflow": request.max_estimated_tokens_per_workflow,
        "max_estimated_tokens_per_minute": request.max_estimated_tokens_per_minute or MAX_ESTIMATED_TOKENS_PER_MINUTE,
        "min_seconds_between_llm_calls": request.min_seconds_between_llm_calls or MIN_SECONDS_BETWEEN_LLM_CALLS,
        "auto_apply": request.auto_apply,
        "run_tests": request.run_tests,
        "test_timeout_seconds": request.test_timeout_seconds,
        "plan": {},
        "plan_steps": [],
        "current_step_index": -1,
        "current_subtask": "",
        "completed_steps": [],
        "failed_steps": [],
        "step_results": [],
        "step_attempts": {},
        "per_step_context": {},
        "per_step_changed_files": {},
        "matched_files": [],
        "context_pipeline_result": {},
        "original_files": [],
        "developer_output": {},
        "review_output": {},
        "final_changed_files": [],
        "apply_result": {},
        "test_result": {},
        "test_feedback": {},
        "final_test_pending": False,
        "step_satisfied": False,
        "satisfaction_reason": "",
        "test_attempts": 0,
        "attempts": 0,
        "agent_steps": 0,
        "tool_errors": 0,
        "repeated_issue_counts": {},
        "repeated_test_failure_counts": {},
        "previous_changed_files": None,
        "no_improvement_count": 0,
        "empty_changed_files_count": 0,
        "total_llm_calls": 0,
        "total_estimated_input_tokens": 0,
        "total_estimated_output_tokens": 0,
        "llm_token_events": [],
        "last_llm_call_at": 0,
        "status": "",
        "stop_reason": "",
        "workflow_trace": [],
        "start_time": time.monotonic(),
        "elapsed_seconds": 0,
        "task_complexity": "",
        "execution_mode": "",
        "goal_complete": False,
        "goal_completion_reason": "",
    }


def planner_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    if not _prepare_llm_call(state, "planner_node", state["task"]):
        return state
    try:
        state["plan"] = run_planner(state["task"])
        _record_llm_output(state, "planner_node", state["plan"])
        _increment_agent_step(state)
        _trace(state, "planner_node", "ok", "completed")
    except Exception as exc:
        _handle_step_error(state, "planner_node", exc)
    return state


def task_complexity_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    complexity, execution_mode, reason = _classify_task_complexity(state)
    state["task_complexity"] = complexity
    state["execution_mode"] = execution_mode
    if execution_mode == "fast_path":
        state["current_subtask"] = state["task"]
    _trace(
        state,
        "task_complexity_node",
        "ok",
        json.dumps(
            {
                "task_complexity": complexity,
                "selected_execution_mode": execution_mode,
                "reason": reason,
                "token_savings_reason": "small task uses one-shot path and skips plan decomposition"
                if execution_mode == "fast_path"
                else "stepwise selected for broader task scope",
            }
        ),
    )
    return state


def plan_decomposition_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    steps, skipped_steps = _decompose_plan_steps(
        state["task"],
        state.get("plan", {}),
        state.get("matched_files", []),
    )
    state["plan_steps"] = steps
    _trace(
        state,
        "plan_decomposition_node",
        "ok",
        json.dumps(
            {
                "plan_steps": steps,
                "skipped_steps": skipped_steps,
                "execution_mode": state.get("execution_mode", "stepwise"),
            }
        ),
    )
    return state


def select_next_step_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state

    next_index = state["current_step_index"] + 1
    if next_index >= len(state["plan_steps"]):
        state["current_subtask"] = ""
        if state["run_tests"]:
            state["final_test_pending"] = True
            _trace(state, "select_next_step_node", "ok", "all steps complete; final test queued")
        else:
            _stop(state, "approved", "approved")
            _trace(state, "select_next_step_node", "ok", "all steps complete")
        return state

    state["current_step_index"] = next_index
    selected_step = state["plan_steps"][next_index]
    selected_step["status"] = "pending"
    state["current_subtask"] = selected_step["description"]
    state["matched_files"] = []
    state["context_pipeline_result"] = {}
    state["original_files"] = []
    state["developer_output"] = {}
    state["review_output"] = {}
    state["test_feedback"] = {}
    state["step_satisfied"] = False
    state["satisfaction_reason"] = ""
    _trace(
        state,
        "select_next_step_node",
        "ok",
        json.dumps(
            {
                "current_step_index": next_index,
                "current_step_id": selected_step["id"],
                "current_subtask": {
                    "title": selected_step["title"],
                    "description": selected_step["description"],
                    "type": selected_step["type"],
                },
                "completed_step_ids": [step.get("id") for step in state["plan_steps"] if step.get("status") == "completed"],
            }
        ),
    )
    return state


def repo_search_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    try:
        result = scan_repo(_retrieval_query(state), state["repo_path"])
        state["matched_files"] = [file["path"] for file in result.get("matched_files", [])]
        _increment_agent_step(state)
        _trace(
            state,
            "repo_search_node",
            "ok",
            json.dumps({"current_subtask": _active_task(state), "matched_files": state["matched_files"]}),
        )
        if not state["matched_files"]:
            _stop(state, "stopped", "no_files_found")
            _trace(state, "repo_search_node", "stopped", "no matched files found")
    except Exception as exc:
        _handle_step_error(state, "repo_search_node", exc)
    return state


def context_pipeline_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    try:
        context_pack = build_context_pack(state["repo_path"], state["matched_files"], _retrieval_query(state))
        state["context_pipeline_result"] = context_pack
        state["original_files"] = [
            {"file": file_info["path"], "content": file_info["content"]}
            for file_info in context_pack.get("files", [])
        ]
        step_key = _step_key(state)
        state["per_step_context"][step_key] = context_trace_details(context_pack)
        _trace(
            state,
            "context_pipeline_node",
            "ok",
            json.dumps({"current_subtask": _active_task(state), **context_trace_details(context_pack)}),
        )
    except Exception as exc:
        _handle_tool_error(state, "context_pipeline_node", exc)
    return state


def step_satisfaction_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    satisfied, reason = _is_step_satisfied(state)
    state["step_satisfied"] = satisfied
    state["satisfaction_reason"] = reason
    _trace(
        state,
        "step_satisfaction_node",
        "satisfied" if satisfied else "needs_work",
        json.dumps(
            {
                "current_step_id": _current_step(state).get("id", _step_key(state)),
                "current_subtask": _active_task(state),
                "reason": reason,
            }
        ),
    )
    return state


def developer_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    if not _prepare_llm_call(state, "developer_node", _developer_input_text(state)):
        return state
    try:
        result = run_developer(
            _developer_task(state),
            state["repo_path"],
            state["matched_files"],
            state["context_pipeline_result"],
        )
        _record_developer_output(state, result)
        _record_llm_output(state, "developer_node", result)
        _increment_agent_step(state)
        _trace(state, "developer_node", "ok", json.dumps({"current_subtask": _active_task(state)}))
    except Exception as exc:
        _handle_step_error(state, "developer_node", exc)
    return state


def improve_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    if not _prepare_llm_call(state, "improve_node", _improver_input_text(state)):
        return state
    try:
        result = run_improver(
            _improver_task(state),
            state["repo_path"],
            state["matched_files"],
            state["developer_output"],
            state["review_output"],
            state["context_pipeline_result"],
            state.get("test_feedback") or None,
        )
        _record_developer_output(state, result)
        _record_llm_output(state, "improve_node", result)
        _increment_agent_step(state)
        _trace(
            state,
            "improve_node",
            "ok",
            json.dumps(
                {
                    "current_subtask": _active_task(state),
                    "repair_triggered_from_test_feedback": bool(state.get("test_feedback")),
                }
            ),
        )
    except Exception as exc:
        _handle_step_error(state, "improve_node", exc)
    return state


def reviewer_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    if not _prepare_llm_call(state, "reviewer_node", _reviewer_input_text(state)):
        return state
    try:
        developer_output_model = DeveloperOutput(
            **{
                **state["developer_output"],
                "original_files": state["original_files"],
            }
        )
        state["review_output"] = run_reviewer(
            _developer_task(state),
            developer_output_model,
            state["context_pipeline_result"],
        )
        _record_llm_output(state, "reviewer_node", state["review_output"])
        _increment_agent_step(state)
        _trace(state, "reviewer_node", "ok", json.dumps({"current_subtask": _active_task(state)}))
    except Exception as exc:
        _handle_step_error(state, "reviewer_node", exc)
        return state

    if state["review_output"].get("approved") is True:
        _trace(state, "reviewer_node", "approved", "reviewer approved the changed files")
        return state

    for issue in state["review_output"].get("issues", []):
        normalized_issue = _normalize_issue(f"{_active_task(state)} {issue}")
        counts = state["repeated_issue_counts"]
        counts[normalized_issue] = counts.get(normalized_issue, 0) + 1
        _trace(
            state,
            "reviewer_node",
            "warning",
            (
                f"reviewer issue count {counts[normalized_issue]}/"
                f"{state['repeated_failure_threshold']}: {issue}"
            ),
        )
        if counts[normalized_issue] >= state["repeated_failure_threshold"]:
            _current_step(state)["status"] = "failed"
            state["failed_steps"].append(_current_step(state))
            _stop(state, "stopped", "repeated_failure")
            break
    return state


def apply_changes_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    try:
        changed_files = [
            ApplyChangedFile(file=changed_file["file"], content=changed_file["content"])
            for changed_file in state["developer_output"].get("changed_files", [])
        ]
        state["apply_result"] = apply_approved_changes(
            ApplyChangesRequest(
                repo_path=state["repo_path"],
                approved=True,
                changed_files=changed_files,
            )
        )
        if state.get("run_tests") and state.get("apply_result", {}).get("status") == "applied":
            state["test_result"] = {}
        _trace(
            state,
            "apply_changes_node",
            "ok",
            json.dumps(
                {
                    "current_subtask": _active_task(state),
                    "files_written": state["apply_result"].get("files_written", []),
                }
            ),
        )
    except Exception as exc:
        _handle_tool_error(state, "apply_changes_node", exc)
    return state


def test_runner_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    try:
        state["test_attempts"] += 1
        state["test_result"] = run_tests_for_repo(state["repo_path"], state["test_timeout_seconds"])
        _trace(
            state,
            "test_runner_node",
            str(state["test_result"].get("status", "unknown")),
            json.dumps(_test_trace_details(state["test_result"])),
        )
    except Exception as exc:
        _handle_tool_error(state, "test_runner_node", exc)
        return state

    if state["test_result"].get("status") == "passed":
        _stop(state, "approved", "tests_passed")
        return state

    if state["test_result"].get("status") == "skipped":
        reason = str(state["test_result"].get("reason", ""))
        _stop(state, "stopped", "test_runner_unavailable" if "executable" in reason.lower() else "tests_skipped")
        return state

    signature = _test_failure_signature(state["test_result"])
    counts = state["repeated_test_failure_counts"]
    counts[signature] = counts.get(signature, 0) + 1
    _trace(
        state,
        "test_runner_node",
        "warning",
        f"test failure count {counts[signature]}/{state['repeated_test_failure_threshold']}",
    )
    if counts[signature] >= state["repeated_test_failure_threshold"]:
        _stop(state, "stopped", "repeated_failure")
        return state

    state["test_feedback"] = _summarize_test_failure(state["test_result"])
    state["final_test_pending"] = False
    if not state.get("current_subtask"):
        state["current_subtask"] = state["task"]
    _trace(
        state,
        "test_runner_node",
        "queued",
        json.dumps(
            {
                "current_subtask": _active_task(state),
                "failure_summary": state["test_feedback"].get("failure_summary", ""),
                "failure_type": state["test_feedback"].get("failure_type", ""),
                "repair_triggered_from_test_feedback": True,
            }
        ),
    )
    return state


def step_completion_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    subtask = _active_task(state)
    changed_files = state["developer_output"].get("changed_files", [])
    if subtask not in state["completed_steps"]:
        state["completed_steps"].append(subtask)
    _current_step(state)["status"] = "completed"
    state["per_step_changed_files"][_step_key(state)] = changed_files
    state["step_results"].append(
        {
            "step_id": _current_step(state).get("id", _step_key(state)),
            "step": subtask,
            "type": _current_step(state).get("type", "unknown"),
            "status": "completed",
            "satisfied_without_changes": bool(state.get("step_satisfied")) and not changed_files,
            "satisfaction_reason": state.get("satisfaction_reason", ""),
            "changed_files": [changed_file.get("file", "") for changed_file in changed_files],
            "approved": state["review_output"].get("approved", False),
        }
    )
    _trace(
        state,
        "step_completion_node",
        "ok",
        json.dumps(
            {
                "completed_step": subtask,
                "completed_step_id": _current_step(state).get("id", _step_key(state)),
                "satisfied_without_changes": bool(state.get("step_satisfied")) and not changed_files,
                "changed_files": [item.get("file", "") for item in changed_files],
            }
        ),
    )
    return state


def goal_completion_node(state: WorkflowState) -> WorkflowState:
    if _hard_stop(state):
        return state
    complete, reason = _is_goal_complete(state)
    state["goal_complete"] = complete
    state["goal_completion_reason"] = reason
    _trace(
        state,
        "goal_completion_node",
        "complete" if complete else "needs_work",
        json.dumps(
            {
                "task_complexity": state.get("task_complexity", ""),
                "execution_mode": state.get("execution_mode", ""),
                "goal_complete": complete,
                "reason": reason,
                "tests_requested": state.get("run_tests", False),
                "test_status": state.get("test_result", {}).get("status", ""),
            }
        ),
    )
    if not complete:
        return state
    _stop(state, "approved", "tests_passed" if state.get("test_result", {}).get("status") == "passed" else "approved")
    return state


def final_node(state: WorkflowState) -> WorkflowState:
    state["elapsed_seconds"] = _elapsed(state)
    if not state.get("status"):
        _stop(state, "approved", "approved")
    _trace(state, "final_node", state["status"], state["stop_reason"])
    return state


def _record_developer_output(state: WorkflowState, result: dict[str, Any]) -> None:
    state["attempts"] += 1
    step_key = _step_key(state)
    state["step_attempts"][step_key] = state["step_attempts"].get(step_key, 0) + 1
    state["developer_output"] = result
    state["final_changed_files"] = result.get("changed_files", [])
    current_changed_files = json.dumps(state["final_changed_files"], sort_keys=True)

    if state["final_changed_files"]:
        state["empty_changed_files_count"] = 0
    else:
        satisfied, reason = _is_step_satisfied(state)
        if satisfied:
            state["step_satisfied"] = True
            state["satisfaction_reason"] = reason
            _trace(
                state,
                "step_satisfaction_node",
                "satisfied",
                json.dumps(
                    {
                        "current_step_id": _current_step(state).get("id", _step_key(state)),
                        "current_subtask": _active_task(state),
                        "reason": f"empty changed_files but already satisfied: {reason}",
                    }
                ),
            )
            return
        state["empty_changed_files_count"] += 1
        _trace(
            state,
            "workflow_stop_check",
            "warning",
            f"empty changed_files count {state['empty_changed_files_count']}/{state['empty_changed_files_threshold']}",
        )
        if state["empty_changed_files_count"] >= state["empty_changed_files_threshold"]:
            _current_step(state)["status"] = "failed"
            state["failed_steps"].append(_current_step(state))
            _stop(state, "stopped", "empty_changed_files")
            return

    previous_changed_files = state.get("previous_changed_files")
    if previous_changed_files is not None and current_changed_files == previous_changed_files:
        state["no_improvement_count"] += 1
        _trace(
            state,
            "workflow_stop_check",
            "warning",
            f"no improvement count {state['no_improvement_count']}/{state['no_improvement_threshold']}",
        )
        if state["no_improvement_count"] >= state["no_improvement_threshold"]:
            _current_step(state)["status"] = "failed"
            state["failed_steps"].append(_current_step(state))
            _stop(state, "stopped", "no_patch_improvement")
            return
    else:
        state["no_improvement_count"] = 0
    state["previous_changed_files"] = current_changed_files


def _route_after_planner(state: WorkflowState) -> str:
    return "final_node" if state.get("stop_reason") else "plan_decomposition_node"


def _route_after_complexity(state: WorkflowState) -> str:
    if state.get("stop_reason"):
        return "final_node"
    if state.get("execution_mode") == "fast_path":
        return "context_pipeline_node"
    return "planner_node"


def _route_after_plan_decomposition(state: WorkflowState) -> str:
    return "final_node" if state.get("stop_reason") else "select_next_step_node"


def _route_after_select_next_step(state: WorkflowState) -> str:
    if state.get("stop_reason"):
        return "final_node"
    if state.get("final_test_pending"):
        return "test_runner_node"
    return "repo_search_node"


def _route_after_repo_search(state: WorkflowState) -> str:
    if state.get("stop_reason"):
        return "final_node"
    if not state.get("execution_mode"):
        return "task_complexity_node"
    return "context_pipeline_node"


def _route_after_context_node(state: WorkflowState) -> str:
    return "final_node" if state.get("stop_reason") else "step_satisfaction_node"


def _route_after_satisfaction_node(state: WorkflowState) -> str:
    if state.get("stop_reason"):
        return "final_node"
    if state.get("step_satisfied"):
        return "goal_completion_node" if state.get("execution_mode") == "fast_path" else "step_completion_node"
    return "developer_node"


def _route_after_developer_node(state: WorkflowState) -> str:
    if state.get("stop_reason"):
        return "final_node"
    return "step_completion_node" if state.get("step_satisfied") else "reviewer_node"


def _route_after_reviewer_node(state: WorkflowState) -> str:
    if state.get("stop_reason"):
        return "final_node"
    if state.get("review_output", {}).get("approved") is True:
        if state.get("execution_mode") == "fast_path" and not state.get("auto_apply"):
            return "goal_completion_node"
        return "apply_changes_node" if state.get("auto_apply") else "step_completion_node"
    return "improve_node"


def _route_after_apply_node(state: WorkflowState) -> str:
    if state.get("stop_reason"):
        return "final_node"
    return "goal_completion_node" if state.get("execution_mode") == "fast_path" else "step_completion_node"


def _route_after_step_completion(state: WorkflowState) -> str:
    return "final_node" if state.get("stop_reason") else "goal_completion_node"


def _route_after_goal_completion(state: WorkflowState) -> str:
    if state.get("stop_reason"):
        return "final_node"
    if state.get("goal_complete"):
        return "final_node"
    if state.get("run_tests"):
        test_status = state.get("test_result", {}).get("status", "")
        implementation_complete = state.get("implementation_complete", False)
        if test_status == "failed":
            return "improve_node"
        if test_status == "" and implementation_complete:
            if not state.get("auto_apply") and state.get("developer_output"):
                _stop(state, "stopped", "tests_skipped")
                return "final_node"
            return "test_runner_node"
    if state.get("execution_mode") in {"stepwise", "stepwise_with_checkpoints"}:
        return "select_next_step_node"
    if state.get("developer_output"):
        return "improve_node"
    return "developer_node"


def _route_after_test_node(state: WorkflowState) -> str:
    if state.get("stop_reason"):
        return "final_node"
    return "improve_node"


def _normalize_plan_steps(task: str, plan: dict[str, Any]) -> list[dict[str, Any]]:
    steps, _ = _decompose_plan_steps(task, plan, [])
    return steps


def _decompose_plan_steps(
    task: str,
    plan: dict[str, Any],
    matched_files: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    raw_steps = [str(step).strip() for step in plan.get("steps", []) if str(step).strip()]
    actionable = []
    skipped_steps = []
    for step in raw_steps:
        cleaned_step = _clean_step_text(step)
        normalized = _normalize_issue(cleaned_step)
        if _is_descriptive_step(normalized, task):
            skipped_steps.append({"step": cleaned_step, "reason": "non-actionable planner step"})
            continue
        architecture_reason = _architecture_skip_reason(cleaned_step, task, matched_files)
        if architecture_reason:
            skipped_steps.append({"step": cleaned_step, "reason": architecture_reason})
            continue
        if len(normalized.split()) < 3:
            skipped_steps.append({"step": cleaned_step, "reason": "too small to execute safely"})
            continue
        actionable.append(cleaned_step)

    if not actionable:
        actionable = [task]

    merged = []
    seen = set()
    seen_targets = set()
    target_to_index = {}
    for step in actionable:
        canonical = _canonical_step(step)
        if canonical in seen:
            continue
        target = _target_key(step)
        if target and target in seen_targets and _classify_step(step) == "source":
            existing_index = target_to_index.get(target)
            if existing_index is not None:
                merged[existing_index] = _merge_related_step_text(merged[existing_index], step)
            continue
        seen.add(canonical)
        if target:
            seen_targets.add(target)
            target_to_index[target] = len(merged)
        merged.append(step)

    if len(merged) > 4:
        merged = _merge_small_steps(merged)

    return [
        _step_object(index=index, text=step)
        for index, step in enumerate(merged or [task], start=1)
    ], skipped_steps


def _classify_task_complexity(state: WorkflowState) -> tuple[str, str, str]:
    task = state["task"]
    lower_task = task.lower()
    matched_files = state.get("matched_files", [])
    words = re.findall(r"[A-Za-z0-9_]+", lower_task)
    score = 0
    reasons = []

    if len(words) > 45:
        score += 2
        reasons.append("long task")
    elif len(words) > 22:
        score += 1
        reasons.append("moderate task length")

    if len(matched_files) > 8:
        score += 2
        reasons.append("many matched files")
    elif len(matched_files) > 4:
        score += 1
        reasons.append("several matched files")

    if any(phrase in lower_task for phrase in ("new file", "create file", "new endpoint", "new api", "new page", "database", "migration")):
        score += 2
        reasons.append("likely new files or schema/API work")

    if any(word in lower_task for word in ("refactor", "migration", "multi-module", "architecture", "workflow", "integration")):
        score += 2
        reasons.append("broad refactor/integration wording")

    if any(word in lower_task for word in ("test", "tests", "spec")):
        score += 1
        reasons.append("tests requested")

    if any(word in lower_task for word in ("fix", "bug", "small", "function", "validator", "validation")) and len(matched_files) <= 5:
        score -= 1
        reasons.append("small bug/function signal")

    if score <= 1:
        return "small", "fast_path", "; ".join(reasons) or "short focused task"
    if score <= 3:
        return "medium", "stepwise", "; ".join(reasons)
    return "large", "stepwise_with_checkpoints", "; ".join(reasons)


def _is_goal_complete(state: WorkflowState) -> tuple[bool, str]:
    implementation_complete, implementation_reason = _is_implementation_complete(state)
    state["implementation_complete"] = implementation_complete

    if state.get("run_tests"):
        test_status = state.get("test_result", {}).get("status", "")
        if test_status == "passed":
            return True, "Tests passed."
        if test_status == "failed":
            failure_summary = state.get("test_result", {}).get("failure_summary", "")
            return False, f"Tests failed; goal is not complete. {failure_summary}".strip()
        if test_status == "skipped":
            return False, "Tests were requested but the test runner skipped validation."
        if implementation_complete:
            return False, f"{implementation_reason} Tests are requested and have not run yet."
        return False, implementation_reason

    return implementation_complete, implementation_reason


def _is_implementation_complete(state: WorkflowState) -> tuple[bool, str]:
    if state.get("execution_mode") == "fast_path" and state.get("review_output", {}).get("approved") is True:
        return True, "Reviewer approved current output."
    if state.get("step_satisfied"):
        return True, state.get("satisfaction_reason", "Step satisfaction check passed.")

    symbols = _symbols_from_text(state["task"])
    if not symbols:
        return bool(state.get("completed_steps")), "No concrete symbol found; relying on completed steps."

    searchable_text = _combined_context_and_changes(state)
    missing = [symbol for symbol in symbols if symbol.lower() not in searchable_text.lower()]
    if missing:
        return False, f"Missing requested symbol(s): {', '.join(missing)}."

    if _has_source_definition(searchable_text, symbols):
        return True, f"Requested symbol(s) appear implemented: {', '.join(symbols)}."
    return False, f"Requested symbol(s) exist but implementation could not be confirmed: {', '.join(symbols)}."


def _clean_step_text(step: str) -> str:
    step = re.sub(r"^\s*(step\s*)?\d+[\).\:-]\s*", "", step, flags=re.IGNORECASE).strip()
    step = re.sub(r"\s+", " ", step)
    return _truncate_words(step, 28)


def _step_object(index: int, text: str) -> dict[str, str]:
    step_type = _classify_step(text)
    return {
        "id": f"step-{index}",
        "title": _title_from_step(text),
        "description": text,
        "type": step_type,
        "status": "pending",
    }


def _title_from_step(text: str) -> str:
    title = re.sub(r"^(implement|add|create|update|modify|fix)\s+", "", text, flags=re.IGNORECASE).strip()
    return _truncate_words(title[:1].upper() + title[1:], 8) if title else "Workflow step"


def _classify_step(text: str) -> str:
    lower = text.lower()
    if "test" in lower or "spec" in lower:
        return "test"
    if any(word in lower for word in ("export", "wire", "route", "register", "hook up", "connect")):
        return "source"
    if any(word in lower for word in ("refactor", "cleanup", "rename", "restructure")):
        return "refactor"
    if any(word in lower for word in ("validate", "verify", "check", "run")):
        return "validation"
    if any(word in lower for word in ("implement", "add", "create", "update", "modify", "fix", "change")):
        return "source"
    return "unknown"


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _is_descriptive_step(normalized_step: str, task: str) -> bool:
    if any(normalized_step.startswith(marker) for marker in DESCRIPTIVE_STEP_MARKERS):
        return True
    if any(marker in normalized_step for marker in ("commit message", "commit changes", "git commit", "pull request")):
        return True
    if any(word in normalized_step for word in ("readme", "documentation", "docs")) and not any(
        word in task.lower() for word in ("readme", "documentation", "docs")
    ):
        return True
    return any(
        marker in normalized_step
        for marker in (
            "run the tests",
            "run tests",
            "verify the tests",
            "ensure tests",
            "make sure tests",
        )
    )


def _architecture_skip_reason(step: str, task: str, matched_files: list[str]) -> str:
    lower_step = step.lower()
    lower_task = task.lower()
    step_terms = {term for term in ARCHITECTURE_TERMS if term in lower_step}
    if not step_terms:
        return ""

    explicitly_requested = any(term in lower_task for term in step_terms)
    repo_has_convention = any(_path_has_architecture_term(path, step_terms) for path in matched_files)
    asks_for_new_structure = any(marker in lower_step for marker in NEW_STRUCTURE_MARKERS)
    existing_files_available = bool(matched_files)

    if asks_for_new_structure and not explicitly_requested and not repo_has_convention:
        return "planner suggested new architecture not requested and not present in repo conventions"

    if not explicitly_requested and not repo_has_convention and existing_files_available:
        return "planner architecture step does not match existing repo conventions"

    return ""


def _path_has_architecture_term(path: str, terms: set[str]) -> bool:
    normalized_path = path.lower().replace("\\", "/")
    if any(term in normalized_path for term in terms):
        return True
    path_parts = {
        part.lower()
        for piece in re.split(r"[\\/]", path)
        for part in re.split(r"[^A-Za-z0-9_]+", piece)
        if part
    }
    return bool(path_parts & terms)


def _merge_small_steps(steps: list[str]) -> list[str]:
    source_steps = [step for step in steps if _looks_like_source_step(step)]
    test_steps = [step for step in steps if _looks_like_test_step(step)]
    other_steps = [step for step in steps if step not in source_steps and step not in test_steps]
    merged = []
    if source_steps:
        merged.append("Implement source changes: " + "; ".join(source_steps))
    merged.extend(other_steps[:2])
    if test_steps:
        merged.append("Add or update tests: " + "; ".join(test_steps))
    return merged[:4]


def _looks_like_source_step(step: str) -> bool:
    lower = step.lower()
    return any(word in lower for word in ("implement", "add", "update", "modify", "fix", "create")) and "test" not in lower


def _looks_like_test_step(step: str) -> bool:
    return "test" in step.lower()


def _is_wiring_step(step: str) -> bool:
    lower = step.lower()
    return any(word in lower for word in ("export", "wire", "wiring", "register", "route", "hook up", "connect"))


def _merge_related_step_text(existing: str, new_step: str) -> str:
    if _is_wiring_step(new_step):
        if "also export or wire it where required" in existing:
            return existing
        return _truncate_words(f"{existing}; also export or wire it where required", 32)
    if "also update related existing files if required" in existing:
        return existing
    return _truncate_words(f"{existing}; also update related existing files if required", 32)


def _canonical_step(step: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", step.lower()).strip()


def _target_key(step: str) -> str:
    function_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", step)
    if function_match:
        return function_match.group(1).lower()
    named_target_match = re.search(
        r"\b(?:export|wire|register|route|connect|implement|add|create|update|modify|fix)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        step,
        flags=re.IGNORECASE,
    )
    if named_target_match:
        return named_target_match.group(1).lower()
    quoted_match = re.search(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'", step)
    if quoted_match:
        return next(value for value in quoted_match.groups() if value).lower()
    return ""


def _is_step_satisfied(state: WorkflowState) -> tuple[bool, str]:
    step = _current_step(state)
    description = str(step.get("description", _active_task(state)))
    step_type = str(step.get("type", "unknown"))
    symbols = _symbols_from_text(description)
    if not symbols:
        return False, "No concrete symbol or target found for deterministic satisfaction check."

    searchable_text = _combined_context_and_changes(state)
    missing_symbols = [symbol for symbol in symbols if symbol.lower() not in searchable_text.lower()]
    if missing_symbols:
        return False, f"Missing target symbol(s): {', '.join(missing_symbols)}."

    lower_description = description.lower()
    if step_type == "test" or "test" in lower_description:
        if not _has_test_context_for_symbols(state, symbols):
            return False, "Target symbol exists, but no relevant test context mentions it."
        return True, f"Relevant test context already covers {', '.join(symbols)}."

    if any(word in lower_description for word in ("export", "wire", "wiring", "register", "route", "hook up", "connect")):
        if _has_wiring_for_symbols(searchable_text, symbols):
            return True, f"Wiring/export appears present for {', '.join(symbols)}."
        return False, f"Target symbol exists, but wiring/export was not found for {', '.join(symbols)}."

    if _has_source_definition(searchable_text, symbols):
        return True, f"Source implementation appears present for {', '.join(symbols)}."
    return False, f"Target symbol exists, but a source definition was not found for {', '.join(symbols)}."


def _combined_context_and_changes(state: WorkflowState) -> str:
    parts = []
    for file_info in state.get("context_pipeline_result", {}).get("files", []):
        parts.append(str(file_info.get("content", "")))
    for changed_files in state.get("per_step_changed_files", {}).values():
        for changed_file in changed_files:
            parts.append(str(changed_file.get("content", "")))
    for changed_file in state.get("final_changed_files", []):
        parts.append(str(changed_file.get("content", "")))
    return "\n".join(parts)


def _has_test_context_for_symbols(state: WorkflowState, symbols: list[str]) -> bool:
    for file_info in state.get("context_pipeline_result", {}).get("files", []):
        path = str(file_info.get("path", "")).lower()
        content = str(file_info.get("content", "")).lower()
        if ("test" in path or "spec" in path) and all(symbol.lower() in content for symbol in symbols):
            return True
    for changed_files in state.get("per_step_changed_files", {}).values():
        for changed_file in changed_files:
            path = str(changed_file.get("file", "")).lower()
            content = str(changed_file.get("content", "")).lower()
            if ("test" in path or "spec" in path) and all(symbol.lower() in content for symbol in symbols):
                return True
    return False


def _has_wiring_for_symbols(text: str, symbols: list[str]) -> bool:
    for symbol in symbols:
        escaped = re.escape(symbol)
        patterns = (
            rf"\bexport\s+.*\b{escaped}\b",
            rf"\bmodule\.exports\b[\s\S]{{0,500}}\b{escaped}\b",
            rf"\bexports\.{escaped}\b",
            rf"\brouter\.\w+\b[\s\S]{{0,300}}\b{escaped}\b",
            rf"\b{escaped}\b[\s,}})\]]",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            continue
        return False
    return True


def _has_source_definition(text: str, symbols: list[str]) -> bool:
    for symbol in symbols:
        escaped = re.escape(symbol)
        patterns = (
            rf"\bfunction\s+{escaped}\s*\(",
            rf"\bconst\s+{escaped}\s*=",
            rf"\blet\s+{escaped}\s*=",
            rf"\bvar\s+{escaped}\s*=",
            rf"\bdef\s+{escaped}\s*\(",
            rf"\bclass\s+{escaped}\b",
            rf"\bfunc\s+{escaped}\s*\(",
            rf"\b{escaped}\s*[:=]\s*\(",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            continue
        return False
    return True


def _symbols_from_text(text: str) -> list[str]:
    symbols = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        symbol = match.group(1)
        if symbol.lower() not in {"if", "for", "while", "switch", "return", "function", "test", "expect"}:
            symbols.append(symbol)
    for match in re.finditer(r"`([^`]+)`|\"([A-Za-z_][A-Za-z0-9_]*)\"|'([A-Za-z_][A-Za-z0-9_]*)'", text):
        symbol = next(value for value in match.groups() if value)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol):
            symbols.append(symbol)
    unique = []
    for symbol in symbols:
        if symbol not in unique:
            unique.append(symbol)
    return unique[:3]


def _active_task(state: WorkflowState) -> str:
    return state.get("current_subtask") or state["task"]


def _current_step(state: WorkflowState) -> dict[str, Any]:
    index = state.get("current_step_index", -1)
    if 0 <= index < len(state.get("plan_steps", [])):
        return state["plan_steps"][index]
    return {
        "id": "fast-path",
        "title": "Workflow task",
        "description": _active_task(state),
        "type": "unknown",
        "status": "pending",
    }


def _retrieval_query(state: WorkflowState) -> str:
    return f"{_active_task(state)}\nBackground task: {state['task']}"


def _developer_task(state: WorkflowState) -> str:
    completed = _completed_steps_summary(state)
    return (
        f"Original task: {state['task']}\n"
        f"Current subtask: {_active_task(state)}\n"
        f"Completed steps: {completed}\n"
        "Implement only the current subtask. Preserve completed work."
    )


def _improver_task(state: WorkflowState) -> str:
    return (
        f"{_developer_task(state)}\n"
        "Repair the previous output for the current subtask using reviewer/test feedback."
    )


def _completed_steps_summary(state: WorkflowState) -> str:
    completed = [
        f"{step.get('id')}: {step.get('title')}"
        for step in state.get("plan_steps", [])
        if step.get("status") == "completed"
    ]
    return "; ".join(completed) if completed else "None"


def _step_key(state: WorkflowState) -> str:
    current_step = _current_step(state)
    return str(current_step.get("id") or max(state.get("current_step_index", 0), 0))


def _hard_stop(state: WorkflowState) -> bool:
    if state.get("stop_reason"):
        return True
    if _elapsed(state) >= state["max_time_seconds"]:
        _stop(state, "stopped", "max_time_reached")
        _trace(state, "workflow_stop_check", "stopped", "max time reached")
        return True
    if state["agent_steps"] >= state["max_agent_steps"]:
        _stop(state, "stopped", "max_steps_reached")
        _trace(state, "workflow_stop_check", "stopped", "max agent steps reached")
        return True
    return False


def _prepare_llm_call(state: WorkflowState, node_name: str, input_text: str) -> bool:
    if _hard_stop(state):
        return False
    input_tokens = count_tokens(input_text)
    if state["total_llm_calls"] + 1 > state["max_llm_calls_per_workflow"]:
        _stop(state, "stopped", "max_llm_calls_reached")
        _trace(state, "rate_limit_guard", "stopped", "max LLM calls reached")
        return False
    if state["total_estimated_input_tokens"] + state["total_estimated_output_tokens"] + input_tokens > state["max_estimated_tokens_per_workflow"]:
        _stop(state, "stopped", "max_estimated_tokens_reached")
        _trace(state, "rate_limit_guard", "stopped", "workflow token budget reached")
        return False

    now = time.monotonic()
    recent_tokens = sum(
        event["tokens"]
        for event in state["llm_token_events"]
        if now - event["timestamp"] < 60
    )
    wait_seconds = max(0.0, state["min_seconds_between_llm_calls"] - (now - state["last_llm_call_at"]))
    if recent_tokens + input_tokens > state["max_estimated_tokens_per_minute"]:
        wait_seconds = max(wait_seconds, 60 - (now - state["llm_token_events"][0]["timestamp"]) if state["llm_token_events"] else 0)

    if wait_seconds > 0:
        if _elapsed(state) + wait_seconds >= state["max_time_seconds"]:
            _stop(state, "stopped", "llm_rate_limit_guard")
            _trace(
                state,
                "rate_limit_guard",
                "stopped",
                json.dumps({"node": node_name, "estimated_input_tokens": input_tokens, "wait_seconds": round(wait_seconds, 2)}),
            )
            return False
        _trace(
            state,
            "rate_limit_guard",
            "wait",
            json.dumps({"node": node_name, "estimated_input_tokens": input_tokens, "wait_seconds": round(wait_seconds, 2)}),
        )
        time.sleep(min(wait_seconds, 5))

    state["total_llm_calls"] += 1
    state["total_estimated_input_tokens"] += input_tokens
    state["last_llm_call_at"] = time.monotonic()
    state["llm_token_events"].append({"timestamp": state["last_llm_call_at"], "tokens": input_tokens})
    _trace(
        state,
        "rate_limit_guard",
        "ok",
        json.dumps({"node": node_name, "estimated_input_tokens": input_tokens, "llm_calls": state["total_llm_calls"]}),
    )
    return True


def _record_llm_output(state: WorkflowState, node_name: str, output: object) -> None:
    output_tokens = count_tokens(json.dumps(output, default=str))
    state["total_estimated_output_tokens"] += output_tokens
    _trace(state, "token_usage", "ok", json.dumps({"node": node_name, "estimated_output_tokens": output_tokens}))


def _developer_input_text(state: WorkflowState) -> str:
    return json.dumps(
        {
            "original_task": state["task"],
            "current_subtask": _active_task(state),
            "completed_steps": _completed_steps_summary(state),
            "context": _context_for_token_estimate(state),
            "matched_files": state.get("matched_files", []),
        },
        default=str,
    )


def _reviewer_input_text(state: WorkflowState) -> str:
    return json.dumps(
        {
            "original_task": state["task"],
            "current_subtask": _active_task(state),
            "developer_output": state.get("developer_output", {}),
            "context": _context_for_token_estimate(state),
        },
        default=str,
    )


def _improver_input_text(state: WorkflowState) -> str:
    return json.dumps(
        {
            "original_task": state["task"],
            "current_subtask": _active_task(state),
            "completed_steps": _completed_steps_summary(state),
            "developer_output": state.get("developer_output", {}),
            "review_output": state.get("review_output", {}),
            "test_feedback": state.get("test_feedback", {}),
            "context": _context_for_token_estimate(state),
        },
        default=str,
    )


def _context_for_token_estimate(state: WorkflowState) -> dict[str, object]:
    context_pack = state.get("context_pipeline_result", {})
    return {
        "trace": context_trace_details(context_pack),
        "files": [
            {
                "path": file_info.get("path", ""),
                "content": file_info.get("content", ""),
            }
            for file_info in context_pack.get("files", [])
        ],
    }


def _increment_agent_step(state: WorkflowState) -> None:
    state["agent_steps"] += 1


def _handle_step_error(state: WorkflowState, node_name: str, exc: Exception) -> None:
    _increment_agent_step(state)
    if isinstance(exc, LLMProviderError):
        _trace(state, node_name, "error", str(exc.detail))
        _stop(state, "error", exc.stop_reason)
        return
    _handle_tool_error(state, node_name, exc)


def _handle_tool_error(state: WorkflowState, node_name: str, exc: Exception) -> None:
    state["tool_errors"] += 1
    detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
    _trace(state, node_name, "error", str(detail))
    _stop(state, "error", "tool_error")


def _stop(state: WorkflowState, status: str, stop_reason: str) -> None:
    state["status"] = status
    state["stop_reason"] = stop_reason


def _trace(state: WorkflowState, step: str, status: str, details: str) -> None:
    state["workflow_trace"].append(
        {
            "step": step,
            "status": status,
            "details": details,
            "elapsed_seconds": _elapsed(state),
        }
    )


def _elapsed(state: WorkflowState) -> float:
    return round(time.monotonic() - state["start_time"], 2)


def _normalize_issue(issue: str) -> str:
    return " ".join(issue.lower().split())


def _test_failure_signature(result: dict[str, object]) -> str:
    text = "\n".join(
        [
            str(result.get("command", "")),
            str(result.get("exit_code", "")),
            str(result.get("failure_type", "")),
            str(result.get("failure_summary", "")),
            _safe_text(result.get("stdout", ""))[-1200:],
            _safe_text(result.get("stderr", ""))[-1200:],
            str(result.get("reason", "")),
        ]
    )
    return _normalize_issue(text)


def _summarize_test_failure(result: dict[str, object]) -> dict[str, object]:
    return {
        "status": result.get("status", ""),
        "command": result.get("command", ""),
        "exit_code": result.get("exit_code", 0),
        "reason": result.get("reason", ""),
        "failure_summary": result.get("failure_summary", result.get("reason", "")),
        "failure_type": result.get("failure_type", ""),
        "timed_out": result.get("timed_out", False),
        "working_directory": result.get("working_directory", ""),
        "stdout_tail": _safe_text(result.get("stdout_tail", result.get("stdout", "")))[-2000:],
        "stderr_tail": _safe_text(result.get("stderr_tail", result.get("stderr", "")))[-2000:],
    }


def _test_trace_details(result: dict[str, object]) -> dict[str, object]:
    details = {
        "command": result.get("command", ""),
        "status": result.get("status", ""),
        "exit_code": result.get("exit_code", 0),
        "reason": result.get("reason", ""),
        "failure_summary": result.get("failure_summary", result.get("reason", "")),
        "failure_type": result.get("failure_type", ""),
        "repair_triggered_from_test_feedback": result.get("status") == "failed",
    }
    if result.get("status") == "failed":
        details["stdout_tail"] = _safe_text(result.get("stdout_tail", result.get("stdout", "")))[-1000:]
        details["stderr_tail"] = _safe_text(result.get("stderr_tail", result.get("stderr", "")))[-1000:]
    return details


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _build_response(state: WorkflowState) -> dict[str, object]:
    return {
        "task": state["task"],
        "status": state.get("status") or "stopped",
        "stop_reason": state.get("stop_reason") or "completed",
        "elapsed_seconds": _elapsed(state),
        "agent_steps": state.get("agent_steps", 0),
        "attempts": state.get("attempts", 0),
        "plan": state.get("plan", {}),
        "matched_files": state.get("matched_files", []),
        "final_developer_output": state.get("developer_output", {}),
        "final_changed_files": state.get("final_changed_files", []),
        "final_review_output": state.get("review_output", {}),
        "final_patch": "",
        "apply_result": state.get("apply_result", {}),
        "test_result": state.get("test_result", {}),
        "test_attempts": state.get("test_attempts", 0),
        "workflow_trace": state.get("workflow_trace", []),
        "orchestrator": "langgraph",
        "execution_mode": state.get("execution_mode", "stepwise"),
        "task_complexity": state.get("task_complexity", ""),
        "completed_steps": state.get("completed_steps", []),
        "failed_steps": state.get("failed_steps", []),
        "step_results": state.get("step_results", []),
        "plan_steps": state.get("plan_steps", []),
        "token_usage_estimate": {
            "llm_calls": state.get("total_llm_calls", 0),
            "estimated_input_tokens": state.get("total_estimated_input_tokens", 0),
            "estimated_output_tokens": state.get("total_estimated_output_tokens", 0),
            "estimated_total_tokens": state.get("total_estimated_input_tokens", 0)
            + state.get("total_estimated_output_tokens", 0),
        },
    }
