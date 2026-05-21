import json
import re
import shutil
import subprocess
from pathlib import Path

from app.services.repo_service import resolve_repo_path

SAFE_COMMANDS = {
    "npm test": ["npm", "test"],
    "pytest": ["pytest"],
    "mvn test": ["mvn", "test"],
    "gradle test": ["gradle", "test"],
    "./gradlew test": ["./gradlew", "test"],
    "sbt test": ["sbt", "test"],
    "dotnet test": ["dotnet", "test"],
    "go test ./...": ["go", "test", "./..."],
    "cargo test": ["cargo", "test"],
    "make test": ["make", "test"],
}

FAILURE_KEYWORDS = (
    "assertionerror",
    "fail",
    "failed",
    "error",
    "expected",
    "received",
    "assert",
    "traceback",
    "exception",
    "syntaxerror",
    "referenceerror",
    "typeerror",
    "modulenotfound",
    "module not found",
    "cannot find module",
    "importerror",
    "cannot find package",
    "panic:",
    "compilation failed",
)
MAX_OUTPUT_CHARS = 20000
MAX_TAIL_CHARS = 4000
MAX_SUMMARY_CHARS = 900

NOISE_LINE_PREFIXES = (
    "> ",
    "ran all test suites",
    "test suites:",
    "tests:",
    "snapshots:",
    "time:",
)

NOISE_LINE_CONTAINS = (
    " npm test",
    " jest",
    " run `npm audit",
)

HIGH_VALUE_MARKERS = (
    "expected:",
    "received:",
    "expect(",
    "assertionerror",
    "syntaxerror",
    "referenceerror",
    "typeerror",
    "cannot find module",
    "module not found",
    "modulenotfound",
    "importerror",
    "no module named",
    "traceback",
    "panic:",
    "error:",
    "failed",
    "fail ",
    " at ",
)


def run_tests_for_repo(repo_path: str, timeout_seconds: int = 60) -> dict[str, object]:
    path = resolve_repo_path(repo_path).resolve()
    detection = _select_test_command(path)
    command = detection.get("command", "")
    project_type = detection.get("project_type", "unknown")

    if not command:
        reason = detection.get("reason", "Test command unavailable.")
        return {
            "status": "skipped",
            "project_type": project_type,
            "command": "",
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "stdout_tail": "",
            "stderr_tail": "",
            "reason": reason,
            "failure_summary": reason,
            "failure_type": "command_unavailable",
            "timed_out": False,
            "working_directory": str(path),
        }

    args = _resolve_command_args(SAFE_COMMANDS[command])
    if args is None:
        return {
            "status": "skipped",
            "project_type": project_type,
            "command": command,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "stdout_tail": "",
            "stderr_tail": "",
            "reason": "Test executable was not found on PATH.",
            "failure_summary": f"Test executable for '{command}' was not found on PATH.",
            "failure_type": "command_unavailable",
            "timed_out": False,
            "working_directory": str(path),
        }

    try:
        result = subprocess.run(
            args,
            cwd=path,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _normalize_output(exc.stdout)
        stderr = _normalize_output(exc.stderr)
        diagnostics = _build_failure_diagnostics(
            command=command,
            exit_code=124,
            stdout=stdout,
            stderr=stderr or f"Test command timed out after {timeout_seconds} seconds.",
            timed_out=True,
            working_directory=str(path),
        )
        return {
            "status": "failed",
            "project_type": project_type,
            "command": command,
            "exit_code": 124,
            "stdout": diagnostics["stdout"],
            "stderr": diagnostics["stderr"],
            "stdout_tail": diagnostics["stdout_tail"],
            "stderr_tail": diagnostics["stderr_tail"],
            "reason": diagnostics["failure_summary"],
            "failure_summary": diagnostics["failure_summary"],
            "failure_type": diagnostics["failure_type"],
            "timed_out": True,
            "working_directory": str(path),
        }

    stdout = _normalize_output(result.stdout)
    stderr = _normalize_output(result.stderr)
    diagnostics = {
        "stdout": _limit_output(stdout),
        "stderr": _limit_output(stderr),
        "stdout_tail": _tail_output(stdout),
        "stderr_tail": _tail_output(stderr),
        "failure_summary": "",
        "failure_type": "",
    }
    if result.returncode != 0:
        diagnostics = _build_failure_diagnostics(
            command=command,
            exit_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
            working_directory=str(path),
        )

    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "project_type": project_type,
        "command": command,
        "exit_code": result.returncode,
        "stdout": diagnostics["stdout"],
        "stderr": diagnostics["stderr"],
        "stdout_tail": diagnostics["stdout_tail"],
        "stderr_tail": diagnostics["stderr_tail"],
        "reason": diagnostics["failure_summary"],
        "failure_summary": diagnostics["failure_summary"],
        "failure_type": diagnostics["failure_type"],
        "timed_out": False,
        "working_directory": str(path),
    }


def _normalize_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _build_failure_diagnostics(
    command: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    working_directory: str,
) -> dict[str, str]:
    limited_stdout = _limit_output(stdout)
    limited_stderr = _limit_output(stderr)
    failure_type = _classify_failure(limited_stdout, limited_stderr, timed_out)
    failure_summary = _summarize_failure(
        command,
        limited_stdout,
        limited_stderr,
        exit_code,
        failure_type,
        timed_out,
        working_directory,
    )
    return {
        "stdout": limited_stdout,
        "stderr": limited_stderr,
        "stdout_tail": _tail_output(limited_stdout),
        "stderr_tail": _tail_output(limited_stderr),
        "failure_summary": failure_summary,
        "failure_type": failure_type,
    }


def _limit_output(output: str) -> str:
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    return output[: MAX_OUTPUT_CHARS // 2] + "\n...[output truncated]...\n" + output[-MAX_OUTPUT_CHARS // 2 :]


def _tail_output(output: str) -> str:
    if len(output) <= MAX_TAIL_CHARS:
        return output
    return output[-MAX_TAIL_CHARS:]


def _classify_failure(stdout: str, stderr: str, timed_out: bool) -> str:
    if timed_out:
        return "timeout"

    combined = f"{stderr}\n{stdout}".lower()
    if any(marker in combined for marker in ("cannot find module", "module not found", "modulenotfound", "importerror", "cannot find package", "no module named")):
        return "module_import_error"
    if any(marker in combined for marker in ("syntaxerror", "parse error", "unexpected token", "compilation failed", "compile error")):
        return "syntax_error"
    if any(marker in combined for marker in ("assertionerror", "expected", "received", "assert", "should be", "to be", "comparison failure")):
        return "assertion_failure"
    if any(marker in combined for marker in ("timed out", "timeout", "deadline exceeded")):
        return "timeout"
    if any(marker in combined for marker in ("not recognized", "command not found", "not found on path", "enoent")):
        return "command_unavailable"
    if "fail" in combined or "error" in combined or "exception" in combined:
        return "test_error"
    return "unknown_failure"


def _summarize_failure(
    command: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    failure_type: str,
    timed_out: bool,
    working_directory: str,
) -> str:
    combined = "\n".join(part for part in (stderr, stdout) if part).strip()
    if not combined:
        timeout_detail = " The process timed out." if timed_out else ""
        return (
            f"{command} failed with exit code {exit_code} in {working_directory} "
            f"and produced no output.{timeout_detail}"
        )

    meaningful_lines = _meaningful_failure_lines(combined)

    if meaningful_lines:
        return _trim_summary(f"{failure_type}: " + " | ".join(meaningful_lines))

    for raw_line in _clean_output_lines(combined):
        line = raw_line.strip()
        if line:
            return _trim_summary(f"{failure_type}: {line}")

    return (
        f"{command} failed with exit code {exit_code} in {working_directory}. "
        "No meaningful diagnostic lines were found in test output."
    )


def _trim_summary(summary: str) -> str:
    return summary[:MAX_SUMMARY_CHARS]


def _meaningful_failure_lines(output: str) -> list[str]:
    lines = _clean_output_lines(output)
    scored: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _is_noise_line(stripped):
            continue
        lower_line = stripped.lower()
        score = 0
        if any(marker in lower_line for marker in HIGH_VALUE_MARKERS):
            score += 4
        if any(keyword in lower_line for keyword in FAILURE_KEYWORDS):
            score += 2
        if _looks_like_stack_line(stripped):
            score += 1
        if score:
            scored.append((score, index, stripped))

    if not scored:
        return []

    selected_indexes = sorted(index for _, index, _ in sorted(scored, key=lambda item: (-item[0], item[1]))[:8])
    return [lines[index].strip() for index in selected_indexes]


def _clean_output_lines(output: str) -> list[str]:
    return [_strip_ansi_codes(line) for line in output.splitlines()]


def _strip_ansi_codes(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _is_noise_line(line: str) -> bool:
    lower_line = line.lower()
    if any(lower_line.startswith(prefix) for prefix in NOISE_LINE_PREFIXES):
        return True
    return any(marker in lower_line for marker in NOISE_LINE_CONTAINS)


def _looks_like_stack_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("at ") or stripped.startswith("File ") or re.match(r"^\w.*:\d+:\d+", stripped) is not None


def _select_test_command(repo_path: Path) -> dict[str, str]:
    configured = _read_agent_config(repo_path)
    if configured is not None:
        command = configured.get("test_command", "")
        project_type = configured.get("project_type", "configured")
        if command not in SAFE_COMMANDS:
            return {
                "project_type": project_type,
                "command": "",
                "reason": "Configured test command is not allowlisted.",
            }
        return {
            "project_type": project_type,
            "command": command,
            "reason": "",
        }

    project_type = _detect_project_type(repo_path)
    command = _command_for_project(repo_path, project_type)
    if command:
        return {
            "project_type": project_type,
            "command": command,
            "reason": "",
        }

    return {
        "project_type": project_type,
        "command": "",
        "reason": "No supported test command found.",
    }


def _resolve_command_args(args: list[str]) -> list[str] | None:
    executable = args[0]
    if executable.startswith("./"):
        return args

    resolved = shutil.which(executable)
    if resolved is None:
        return None
    return [resolved, *args[1:]]


def _read_agent_config(repo_path: Path) -> dict[str, str] | None:
    config_path = repo_path / ".agentconfig.json"
    if not config_path.exists():
        return None

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "project_type": "configured",
            "test_command": "",
        }

    return {
        "project_type": str(data.get("project_type", "configured")),
        "test_command": str(data.get("test_command", "")),
    }


def _detect_project_type(repo_path: Path) -> str:
    if (repo_path / "package.json").exists():
        return "node"
    if any((repo_path / name).exists() for name in ("pytest.ini", "pyproject.toml", "requirements.txt")):
        return "python"
    if (repo_path / "pom.xml").exists():
        return "java-maven"
    if any((repo_path / name).exists() for name in ("build.gradle", "build.gradle.kts")):
        return "java-gradle"
    if (repo_path / "build.sbt").exists():
        return "scala-sbt"
    if any(repo_path.glob("*.csproj")):
        return "dotnet"
    if (repo_path / "go.mod").exists():
        return "go"
    if (repo_path / "Cargo.toml").exists():
        return "rust"
    if _makefile_has_test_target(repo_path):
        return "make"
    return "unknown"


def _command_for_project(repo_path: Path, project_type: str) -> str:
    if project_type == "node" and _package_has_test_script(repo_path):
        return "npm test"
    if project_type == "python":
        return "pytest"
    if project_type == "java-maven":
        return "mvn test"
    if project_type == "java-gradle":
        return "./gradlew test" if (repo_path / "gradlew").exists() else "gradle test"
    if project_type == "scala-sbt":
        return "sbt test"
    if project_type == "dotnet":
        return "dotnet test"
    if project_type == "go":
        return "go test ./..."
    if project_type == "rust":
        return "cargo test"
    if project_type == "make" and _makefile_has_test_target(repo_path):
        return "make test"
    return ""


def _package_has_test_script(repo_path: Path) -> bool:
    try:
        data = json.loads((repo_path / "package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("scripts", {}).get("test"))


def _makefile_has_test_target(repo_path: Path) -> bool:
    makefile = repo_path / "Makefile"
    if not makefile.exists():
        return False
    try:
        lines = makefile.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return False
    return any(line.startswith("test:") for line in lines)
