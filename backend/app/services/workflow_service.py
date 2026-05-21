import json
import time

from fastapi import HTTPException

from app.agents.developer import run_developer
from app.agents.improver import run_improver
from app.agents.planner import run_planner
from app.agents.reviewer import run_reviewer
from app.context.prompt_packer import build_context_pack, context_trace_details
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


def run_multi_agent_workflow(request: RunWorkflowRequest) -> dict[str, object]:
    start_time = time.monotonic()
    agent_steps = 0
    attempts = 0
    tool_errors = 0
    repeated_issue_counts = {}
    repeated_test_failure_counts = {}
    previous_changed_files = None
    no_improvement_count = 0
    empty_changed_files_count = 0
    plan_output = {}
    matched_files = []
    context_pack = {}
    original_files = []
    final_developer_output = {}
    final_review_output = {}
    apply_result = {}
    test_result = {}
    test_attempts = 0
    test_feedback = {}
    workflow_trace = []

    def elapsed() -> float:
        return round(time.monotonic() - start_time, 2)

    def trace(step: str, status: str, details: str) -> None:
        workflow_trace.append(
            {
                "step": step,
                "status": status,
                "details": details,
                "elapsed_seconds": elapsed(),
            }
        )

    def build_response(status: str, stop_reason: str) -> dict[str, object]:
        return {
            "task": request.task,
            "status": status,
            "stop_reason": stop_reason,
            "elapsed_seconds": elapsed(),
            "agent_steps": agent_steps,
            "attempts": attempts,
            "plan": plan_output,
            "matched_files": matched_files,
            "final_developer_output": final_developer_output,
            "final_changed_files": final_developer_output.get("changed_files", []),
            "final_review_output": final_review_output,
            "final_patch": "",
            "apply_result": apply_result,
            "test_result": test_result,
            "test_attempts": test_attempts,
            "workflow_trace": workflow_trace,
        }

    def stop_reason_if_needed() -> str | None:
        if elapsed() >= request.max_time_seconds:
            return "max_time_reached"
        if agent_steps >= request.max_agent_steps:
            return "max_steps_reached"
        return None

    def normalize_issue(issue: str) -> str:
        return " ".join(issue.lower().split())

    def test_failure_signature(result: dict[str, object]) -> str:
        text = "\n".join(
            [
                str(result.get("command", "")),
                str(result.get("exit_code", "")),
                str(result.get("failure_type", "")),
                str(result.get("failure_summary", "")),
                safe_text(result.get("stdout", ""))[-1200:],
                safe_text(result.get("stderr", ""))[-1200:],
                str(result.get("reason", "")),
            ]
        )
        return normalize_issue(text)

    def summarize_test_failure(result: dict[str, object]) -> dict[str, object]:
        return {
            "status": result.get("status", ""),
            "command": result.get("command", ""),
            "exit_code": result.get("exit_code", 0),
            "reason": result.get("reason", ""),
            "failure_summary": result.get("failure_summary", result.get("reason", "")),
            "failure_type": result.get("failure_type", ""),
            "timed_out": result.get("timed_out", False),
            "stdout_tail": safe_text(result.get("stdout", ""))[-2000:],
            "stderr_tail": safe_text(result.get("stderr", ""))[-2000:],
        }

    def safe_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def test_trace_details(result: dict[str, object]) -> dict[str, object]:
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
            details["stdout_tail"] = safe_text(result.get("stdout", ""))[-1000:]
            details["stderr_tail"] = safe_text(result.get("stderr", ""))[-1000:]
        return details

    def apply_current_changes() -> dict[str, object]:
        changed_files = [
            ApplyChangedFile(
                file=changed_file["file"],
                content=changed_file["content"],
            )
            for changed_file in final_developer_output.get("changed_files", [])
        ]
        return apply_approved_changes(
            ApplyChangesRequest(
                repo_path=request.repo_path,
                approved=True,
                changed_files=changed_files,
            )
        )

    def run_step(step_name: str, action):
        nonlocal agent_steps, tool_errors

        while True:
            stop_reason = stop_reason_if_needed()
            if stop_reason:
                return False, None, stop_reason

            try:
                result = action()
                agent_steps += 1
                trace(step_name, "ok", "completed")
                return True, result, None
            except LLMProviderError as exc:
                agent_steps += 1
                trace(step_name, "error", str(exc.detail))
                return False, None, exc.stop_reason
            except HTTPException as exc:
                agent_steps += 1
                tool_errors += 1
                trace(step_name, "error", str(exc.detail))
                if tool_errors >= 2:
                    return False, None, "tool_error"
            except Exception as exc:
                agent_steps += 1
                tool_errors += 1
                trace(step_name, "error", str(exc))
                if tool_errors >= 2:
                    return False, None, "tool_error"

    ok, result, stop_reason = run_step(
        "Planner Agent",
        lambda: run_planner(request.task),
    )
    if not ok:
        return build_response("error", stop_reason or "tool_error")
    plan_output = result

    ok, result, stop_reason = run_step(
        "Repo Search Agent",
        lambda: scan_repo(request.task, request.repo_path),
    )
    if not ok:
        return build_response("error", stop_reason or "tool_error")

    matched_files = [file["path"] for file in result.get("matched_files", [])]
    if not matched_files:
        trace("Repo Search Agent", "stopped", "no matched files found")
        return build_response("stopped", "no_files_found")

    context_pack = build_context_pack(request.repo_path, matched_files, request.task)
    context_details = context_trace_details(context_pack)
    trace("Context Pipeline", "ok", json.dumps(context_details))
    original_files = [
        {"file": file_info["path"], "content": file_info["content"]}
        for file_info in context_pack.get("files", [])
    ]

    while True:
        stop_reason = stop_reason_if_needed()
        if stop_reason:
            return build_response("stopped", stop_reason)

        if attempts == 0:
            ok, result, stop_reason = run_step(
                "Developer Agent",
                lambda: run_developer(
                    request.task,
                    request.repo_path,
                    matched_files,
                    context_pack,
                ),
            )
        else:
            ok, result, stop_reason = run_step(
                "Improve Patch Agent",
                lambda: run_improver(
                    request.task,
                    request.repo_path,
                    matched_files,
                    final_developer_output,
                    final_review_output,
                    context_pack,
                    test_feedback,
                ),
            )

        if not ok:
            return build_response("error", stop_reason or "tool_error")

        attempts += 1
        final_developer_output = result
        changed_files = final_developer_output.get("changed_files", [])
        current_changed_files = json.dumps(
            changed_files,
            sort_keys=True,
        )

        if changed_files:
            empty_changed_files_count = 0
        else:
            empty_changed_files_count += 1
            trace(
                "Workflow",
                "warning",
                f"empty changed_files count {empty_changed_files_count}/{request.empty_changed_files_threshold}",
            )
            if empty_changed_files_count >= request.empty_changed_files_threshold:
                trace("Workflow", "stopped", "changed_files stayed empty")
                return build_response("stopped", "empty_changed_files")

        if (
            previous_changed_files is not None
            and current_changed_files == previous_changed_files
        ):
            no_improvement_count += 1
            trace(
                "Workflow",
                "warning",
                f"no improvement count {no_improvement_count}/{request.no_improvement_threshold}",
            )
            if no_improvement_count >= request.no_improvement_threshold:
                trace("Workflow", "stopped", "changed files unchanged across attempts")
                return build_response("stopped", "no_patch_improvement")
        else:
            no_improvement_count = 0
        previous_changed_files = current_changed_files

        try:
            developer_output_model = DeveloperOutput(
                **{
                    **final_developer_output,
                    "original_files": original_files,
                }
            )
        except Exception as exc:
            trace("Reviewer Agent", "error", f"developer output invalid: {exc}")
            return build_response("error", "tool_error")

        ok, result, stop_reason = run_step(
            "Reviewer Agent",
            lambda: run_reviewer(request.task, developer_output_model, context_pack),
        )
        if not ok:
            return build_response("error", stop_reason or "tool_error")

        final_review_output = result
        if final_review_output.get("approved") is True:
            trace("Workflow", "approved", "reviewer approved the changed files")
            if not request.auto_apply and not request.run_tests:
                return build_response("approved", "approved")
            if request.run_tests and not request.auto_apply:
                trace(
                    "Test Runner",
                    "skipped",
                    "run_tests requires auto_apply=true so generated changes can be tested",
                )
                return build_response("stopped", "tests_skipped")

            try:
                apply_result = apply_current_changes()
                trace(
                    "Apply Changes",
                    "ok",
                    json.dumps(
                        {
                            "files_written": apply_result.get("files_written", []),
                        }
                    ),
                )
            except HTTPException as exc:
                trace("Apply Changes", "error", str(exc.detail))
                return build_response("error", "tool_error")

            if not request.run_tests:
                return build_response("approved", "approved")

            test_attempts += 1
            test_result = run_tests_for_repo(
                request.repo_path,
                request.test_timeout_seconds,
            )
            trace(
                "Test Runner",
                str(test_result.get("status", "unknown")),
                json.dumps(test_trace_details(test_result)),
            )

            if test_result.get("status") == "passed":
                return build_response("approved", "tests_passed")
            if test_result.get("status") == "skipped":
                reason = str(test_result.get("reason", ""))
                stop_reason = (
                    "test_runner_unavailable"
                    if "executable" in reason.lower()
                    else "tests_skipped"
                )
                return build_response("stopped", stop_reason)

            signature = test_failure_signature(test_result)
            repeated_test_failure_counts[signature] = (
                repeated_test_failure_counts.get(signature, 0) + 1
            )
            trace(
                "Workflow",
                "warning",
                f"test failure count {repeated_test_failure_counts[signature]}/{request.repeated_test_failure_threshold}",
            )
            if (
                repeated_test_failure_counts[signature]
                >= request.repeated_test_failure_threshold
            ):
                return build_response("stopped", "repeated_failure")

            test_feedback = summarize_test_failure(test_result)
            trace(
                "Test Feedback Improve",
                "queued",
                json.dumps(
                    {
                        "command": test_feedback.get("command", ""),
                        "status": test_feedback.get("status", ""),
                        "exit_code": test_feedback.get("exit_code", 0),
                        "reason": test_feedback.get("reason", ""),
                        "failure_summary": test_feedback.get("failure_summary", ""),
                        "failure_type": test_feedback.get("failure_type", ""),
                        "repair_triggered_from_test_feedback": True,
                    }
                ),
            )
            continue

        for issue in final_review_output.get("issues", []):
            normalized_issue = normalize_issue(issue)
            repeated_issue_counts[normalized_issue] = (
                repeated_issue_counts.get(normalized_issue, 0) + 1
            )
            trace(
                "Workflow",
                "warning",
                f"reviewer issue count {repeated_issue_counts[normalized_issue]}/{request.repeated_failure_threshold}: {issue}",
            )
            if repeated_issue_counts[normalized_issue] >= request.repeated_failure_threshold:
                trace("Workflow", "stopped", f"reviewer issue repeated: {issue}")
                return build_response("stopped", "repeated_failure")
