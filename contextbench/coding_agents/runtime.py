# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-03-22 from upstream ContextBench.
# Summary of changes: support unscored setup prompts that run before the scored benchmark prompt.

"""Runtime helpers for Codex and Claude CLI execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import checkout
from .files import ensure_dir, read_json, safe_path_component, usage_error, write_json
from .prompting import build_prompt
from .records import build_setup_run_record, build_task_record
from .response_parsing import (
    build_claude_raw_response,
    build_codex_raw_response,
)
from ..agents.claude.parser import ClaudeAgentParser
from ..agents.codex.parser import CodexAgentParser
from .types import ClaudeRawResponse, CodexRawResponse, CommandResult, SetupRunRecord, StructuredOutput, TaskRecord, TokenUsage, ToolCall


@dataclass(frozen=True)
class _InvocationResult:
    prompt_path: Path
    stderr_path: Path
    raw_response_path: Path
    command_result: CommandResult
    structured_output: StructuredOutput | None
    token_usage: TokenUsage | None
    tool_calls: list[ToolCall]
    started_at: float
    completed_at: float


def resolve_repo_from_task(task: dict[str, object], cache_dir: Path) -> str:
    repo_url = str(task.get("repo_url") or "").strip()
    if repo_url:
        return repo_url
    original_inst_id = str(task.get("original_inst_id") or task.get("instance_id") or "").strip()
    import re

    match = re.match(r"^([A-Za-z0-9_.-]+)__([A-Za-z0-9_.-]+)-\d+$", original_inst_id)
    if not match:
        return ""
    owner, repo = match.group(1), match.group(2)
    local = cache_dir / f"github.com__{owner}__{repo}"
    if (local / ".git").exists():
        return str(local)
    return f"https://github.com/{owner}/{repo}.git"


def reset_workspace(workspace_path: Path) -> None:
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(workspace_path), check=False, capture_output=True)
    subprocess.run(["git", "clean", "-fdx"], cwd=str(workspace_path), check=False, capture_output=True)


def git_diff(workspace_path: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary"],
        cwd=str(workspace_path),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout or ""


def codex_runtime_root(task_dir: Path) -> Path:
    return task_dir / "codex-runtime"


def build_codex_command(
    *,
    workspace_path: Path,
    schema_path: Path | None,
    final_output_path: Path | None,
    model: str | None,
    reasoning_effort: str | None,
    writable_dirs: Sequence[Path] = (),
    extra_args: Sequence[str],
) -> tuple[list[str], str]:
    # Codex does not expose a Claude-style verbose flag; --json is the richest machine-readable mode.
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--json",
        "--cd",
        str(workspace_path),
    ]
    if schema_path is not None:
        command.extend(["--output-schema", str(schema_path)])
    if final_output_path is not None:
        command.extend(["--output-last-message", str(final_output_path)])
    if model:
        command.extend(["--model", model])
    if reasoning_effort:
        command.extend(["-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"])
    seen_dirs: set[str] = set()
    for path in writable_dirs:
        resolved = str(path.resolve())
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        command.extend(["--add-dir", resolved])
    command.extend(extra_args)
    command.append("-")
    return command, "codex-events.jsonl"


def build_claude_command(
    *,
    schema_path: Path | None,
    prompt: str,
    model: str | None,
    reasoning_effort: str | None,
    extra_args: Sequence[str],
    settings_path: Path,
    mcp_config_path: Path,
) -> tuple[list[str], str]:
    # Always request verbose JSON so we preserve the richest available response envelope per run.
    command = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--verbose",
        "--no-session-persistence",
        "--permission-mode",
        "auto",
        "--setting-sources",
        "",
        "--settings",
        str(settings_path),
        "--disable-slash-commands",
        "--mcp-config",
        str(mcp_config_path),
        "--strict-mcp-config",
        "--tools",
        "default",
    ]
    if schema_path is not None:
        command.extend(["--json-schema", json.dumps(read_json(schema_path), ensure_ascii=False)])
    if model:
        command.extend(["--model", model])
    if reasoning_effort:
        command.extend(["--effort", reasoning_effort])
    command.extend(extra_args)
    command.append(prompt)
    return command, "claude-output.json"


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    stdin_text: str | None,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
    env: dict[str, str] | None = None,
) -> CommandResult:
    ensure_dir(stdout_path.parent)
    ensure_dir(stderr_path.parent)
    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd),
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        stdout_path.write_text(_coerce_output_text(result.stdout), encoding="utf-8")
        stderr_path.write_text(_coerce_output_text(result.stderr), encoding="utf-8")
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "signal": None,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_coerce_output_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_coerce_output_text(exc.stderr), encoding="utf-8")
        return {
            "ok": False,
            "exit_code": None,
            "signal": "SIGTERM",
            "timeout": True,
        }


def _merge_json_objects(base: object, override: object) -> object:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _merge_json_objects(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _coerce_output_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _resolve_runtime_path(roots: dict[str, Path], *, target_root: str, relative_path: str) -> Path:
    base = roots.get(target_root)
    if base is None:
        raise usage_error(f"Runtime target root '{target_root}' is not available for this agent")
    if Path(relative_path).is_absolute():
        raise usage_error(f"Runtime path must be relative: {relative_path}")
    target = (base / relative_path).resolve()
    base_real = base.resolve()
    try:
        if os.path.commonpath([str(base_real), str(target)]) != str(base_real):
            raise usage_error(f"Runtime path escapes target root '{target_root}': {relative_path}")
    except ValueError as exc:
        raise usage_error(f"Invalid runtime path for target root '{target_root}': {relative_path}") from exc
    return target


def _apply_materialized_files(
    specs: Sequence[dict[str, object]] | None,
    *,
    roots: dict[str, Path],
) -> None:
    for spec in specs or ():
        path_value = str(spec.get("path") or "").strip()
        if not path_value:
            raise usage_error("Materialized runtime files require a non-empty 'path'")
        target_root = str(spec.get("target_root") or "task_dir")
        format_name = str(spec.get("format") or "text").strip().lower()
        target_path = _resolve_runtime_path(roots, target_root=target_root, relative_path=path_value)
        ensure_dir(target_path.parent)
        if format_name == "json":
            write_json(target_path, spec.get("content"))
            continue
        if format_name != "text":
            raise usage_error(f"Unsupported materialized file format: {format_name}")
        content = spec.get("content")
        if isinstance(content, str):
            text = content
        else:
            text = json.dumps(content, indent=2, ensure_ascii=False)
        target_path.write_text(text, encoding="utf-8")


def _apply_copy_paths(
    specs: Sequence[dict[str, object]] | None,
    *,
    roots: dict[str, Path],
) -> None:
    for spec in specs or ():
        source_path = Path(str(spec.get("source") or "")).expanduser()
        if not source_path.exists():
            raise usage_error(f"Runtime copy source does not exist: {source_path}")
        destination = str(spec.get("destination") or "").strip()
        target_root = str(spec.get("target_root") or "task_dir")
        if source_path.is_file() and not destination:
            destination = source_path.name
        target_path = _resolve_runtime_path(roots, target_root=target_root, relative_path=destination or ".")
        if source_path.is_dir():
            ensure_dir(target_path.parent)
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            continue
        if destination.endswith("/") or destination in ("", "."):
            target_path = _resolve_runtime_path(
                roots,
                target_root=target_root,
                relative_path=str((Path(destination or ".") / source_path.name).as_posix()),
            )
        ensure_dir(target_path.parent)
        shutil.copy2(source_path, target_path)


def prepare_codex_runtime_env(
    task_dir: Path,
    source_codex_dir: Path | None = None,
    *,
    materialized_files: Sequence[dict[str, object]] | None = None,
    copy_paths: Sequence[dict[str, object]] | None = None,
) -> dict[str, str]:
    source_root = source_codex_dir or (Path.home() / ".codex")
    auth_path = source_root / "auth.json"
    if not auth_path.exists():
        raise usage_error(f"Codex auth is unavailable: expected {auth_path}")

    runtime_root = codex_runtime_root(task_dir)
    home_dir = runtime_root / "home"
    codex_home = home_dir / ".codex"
    xdg_config_home = runtime_root / "xdg-config"
    xdg_data_home = runtime_root / "xdg-data"
    xdg_cache_home = runtime_root / "xdg-cache"

    for path in (codex_home, xdg_config_home, xdg_data_home, xdg_cache_home):
        ensure_dir(path)

    shutil.copy2(auth_path, codex_home / "auth.json")
    runtime_roots = {
        "task_dir": task_dir,
        "runtime_root": runtime_root,
        "home_dir": home_dir,
        "codex_home": codex_home,
        "xdg_config_home": xdg_config_home,
        "xdg_data_home": xdg_data_home,
        "xdg_cache_home": xdg_cache_home,
    }
    _apply_copy_paths(copy_paths, roots=runtime_roots)
    _apply_materialized_files(materialized_files, roots=runtime_roots)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home_dir),
            "XDG_CONFIG_HOME": str(xdg_config_home),
            "XDG_DATA_HOME": str(xdg_data_home),
            "XDG_CACHE_HOME": str(xdg_cache_home),
            "OTEL_SDK_DISABLED": "true",
        }
    )
    return env


def validate_claude_auth() -> None:
    result = subprocess.run(
        ["claude", "auth", "status", "--json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise usage_error(f"Claude auth status failed: {result.stderr or result.stdout}".strip())
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise usage_error("Claude auth status did not return valid JSON") from exc
    if payload.get("loggedIn") is not True:
        raise usage_error("Claude Code is not logged in for non-interactive use (`claude auth status` returned loggedIn=false)")


def prepare_claude_runtime_files(
    task_dir: Path,
    *,
    settings_overrides: dict[str, object] | None = None,
    mcp_config_overrides: dict[str, object] | None = None,
    materialized_files: Sequence[dict[str, object]] | None = None,
    copy_paths: Sequence[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    settings_path = task_dir / "claude.settings.json"
    mcp_config_path = task_dir / "claude.mcp.json"
    write_json(settings_path, _merge_json_objects({}, settings_overrides or {}))
    write_json(mcp_config_path, _merge_json_objects({"mcpServers": {}}, mcp_config_overrides or {}))
    runtime_roots = {
        "task_dir": task_dir,
        "runtime_root": task_dir,
    }
    _apply_copy_paths(copy_paths, roots=runtime_roots)
    _apply_materialized_files(materialized_files, roots=runtime_roots)
    return settings_path, mcp_config_path


def validate_claude_isolation(raw_response: ClaudeRawResponse) -> None:
    response = raw_response.get("response")
    if not isinstance(response, list):
        raise usage_error("Claude raw response is missing the expected verbose event array")

    init_event = None
    for item in response:
        if isinstance(item, dict) and item.get("type") == "system" and item.get("subtype") == "init":
            init_event = item
            break
    if not isinstance(init_event, dict):
        raise usage_error("Claude verbose response is missing the init event needed for isolation validation")

    if init_event.get("plugins"):
        raise usage_error("Claude isolation failed: user plugins are still loaded")
    if init_event.get("mcp_servers"):
        raise usage_error("Claude isolation failed: MCP servers are still exposed")
    slash_commands = init_event.get("slash_commands")
    if isinstance(slash_commands, list) and len(slash_commands) > 0:
        raise usage_error("Claude isolation failed: slash commands are still enabled")


def _normalize_codex_reasoning_effort(reasoning_effort: str | None) -> str | None:
    return reasoning_effort


def _normalize_claude_reasoning_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None
    if reasoning_effort == "xhigh":
        return "max"
    if reasoning_effort in {"none", "minimal"}:
        raise usage_error(
            "Claude does not support reasoning_effort values 'none' or 'minimal'; use low, medium, high, or xhigh"
        )
    return reasoning_effort


_CODEX_RETRYABLE_ERROR_SNIPPETS = (
    "failed to connect to websocket",
    "currently experiencing high demand",
    "missing bearer or basic authentication in header",
    "falling back from websockets to https transport",
)
_CODEX_RETRY_DELAYS_SECONDS = (2, 5)


def _write_prompt_file(task_dir: Path, filename: str, prompt: str) -> Path:
    path = task_dir / filename
    path.write_text(prompt, encoding="utf-8")
    return path


def _attempt_path(path: Path, attempt_index: int) -> Path:
    return path.with_name(f"{path.stem}.attempt{attempt_index}{path.suffix}")


def _archive_retry_artifacts(paths: Sequence[Path | None], *, attempt_index: int) -> None:
    for path in paths:
        if path is None or not path.exists():
            continue
        shutil.copy2(path, _attempt_path(path, attempt_index))


def _codex_raw_response_text(raw_response: CodexRawResponse) -> str:
    fragments: list[str] = []
    for event in raw_response.get("events", []):
        if not isinstance(event, dict):
            continue
        message = str(event.get("message") or "").strip()
        if message:
            fragments.append(message)
        error = event.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            if message:
                fragments.append(message)
    final_message = raw_response.get("final_message")
    if isinstance(final_message, str) and final_message.strip():
        fragments.append(final_message.strip())
    return "\n".join(fragments)


def _should_retry_codex_failure(
    *,
    command_result: CommandResult,
    raw_response: CodexRawResponse,
    stderr_path: Path,
) -> bool:
    if command_result["ok"] or command_result["timeout"]:
        return False
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    haystack = "\n".join((stderr_text, _codex_raw_response_text(raw_response))).lower()
    return any(snippet in haystack for snippet in _CODEX_RETRYABLE_ERROR_SNIPPETS)


def _run_codex_invocation(
    *,
    task_dir: Path,
    workspace_path: Path,
    prompt: str,
    prompt_filename: str,
    stderr_filename: str,
    raw_response_filename: str,
    raw_output_filename: str,
    final_output_filename: str | None,
    timeout: int,
    model: str | None,
    reasoning_effort: str | None,
    extra_args: Sequence[str],
    env: dict[str, str] | None,
    schema_path: Path | None,
) -> _InvocationResult:
    parser = CodexAgentParser()
    prompt_path = _write_prompt_file(task_dir, prompt_filename, prompt)
    stderr_path = task_dir / stderr_filename
    raw_response_path = task_dir / raw_response_filename
    final_output_path = task_dir / final_output_filename if final_output_filename else None
    writable_dirs = [codex_runtime_root(task_dir)]
    command, _ = build_codex_command(
        workspace_path=workspace_path,
        schema_path=schema_path,
        final_output_path=final_output_path,
        model=model,
        reasoning_effort=_normalize_codex_reasoning_effort(reasoning_effort),
        writable_dirs=writable_dirs,
        extra_args=extra_args,
    )
    raw_output_path = task_dir / raw_output_filename
    started_at = time.time()
    raw_response: CodexRawResponse = {"agent": "codex", "response_format": "jsonl-events", "events": []}
    command_result: CommandResult = {"ok": False, "exit_code": None, "signal": None, "timeout": False}
    max_attempts = len(_CODEX_RETRY_DELAYS_SECONDS) + 1
    completed_at = started_at
    for attempt_index in range(1, max_attempts + 1):
        for path in (raw_output_path, stderr_path, raw_response_path, final_output_path):
            if path is not None and path.exists():
                path.unlink()
        command_result = run_command(
            command,
            cwd=workspace_path,
            stdin_text=prompt,
            stdout_path=raw_output_path,
            stderr_path=stderr_path,
            timeout=timeout,
            env=env,
        )
        completed_at = time.time()
        raw_response = build_codex_raw_response(raw_output_path, final_output_path)
        write_json(raw_response_path, raw_response)
        if attempt_index >= max_attempts or not _should_retry_codex_failure(
            command_result=command_result,
            raw_response=raw_response,
            stderr_path=stderr_path,
        ):
            break
        _archive_retry_artifacts(
            [raw_output_path, stderr_path, raw_response_path, final_output_path],
            attempt_index=attempt_index,
        )
        time.sleep(_CODEX_RETRY_DELAYS_SECONDS[attempt_index - 1])
    structured_output = parser.extract_structured_output(raw_response) if schema_path is not None else None
    return _InvocationResult(
        prompt_path=prompt_path,
        stderr_path=stderr_path,
        raw_response_path=raw_response_path,
        command_result=command_result,
        structured_output=structured_output,
        token_usage=parser.extract_token_usage(raw_response),
        tool_calls=parser.extract_tool_calls(raw_response),
        started_at=started_at,
        completed_at=completed_at,
    )


def _run_claude_invocation(
    *,
    task_dir: Path,
    workspace_path: Path,
    prompt: str,
    prompt_filename: str,
    stderr_filename: str,
    raw_response_filename: str,
    timeout: int,
    model: str | None,
    reasoning_effort: str | None,
    extra_args: Sequence[str],
    env: dict[str, str] | None,
    schema_path: Path | None,
    settings_path: Path,
    mcp_config_path: Path,
    validate_isolation: bool,
) -> _InvocationResult:
    parser = ClaudeAgentParser()
    prompt_path = _write_prompt_file(task_dir, prompt_filename, prompt)
    stderr_path = task_dir / stderr_filename
    raw_response_path = task_dir / raw_response_filename
    command, _ = build_claude_command(
        schema_path=schema_path,
        prompt=prompt,
        model=model,
        reasoning_effort=_normalize_claude_reasoning_effort(reasoning_effort),
        extra_args=extra_args,
        settings_path=settings_path,
        mcp_config_path=mcp_config_path,
    )
    started_at = time.time()
    command_result = run_command(
        command,
        cwd=workspace_path,
        stdin_text=None,
        stdout_path=raw_response_path,
        stderr_path=stderr_path,
        timeout=timeout,
        env=env,
    )
    completed_at = time.time()
    raw_response: ClaudeRawResponse = build_claude_raw_response(raw_response_path)
    write_json(raw_response_path, raw_response)
    if validate_isolation:
        validate_claude_isolation(raw_response)
    structured_output = parser.extract_structured_output(raw_response) if schema_path is not None else None
    return _InvocationResult(
        prompt_path=prompt_path,
        stderr_path=stderr_path,
        raw_response_path=raw_response_path,
        command_result=command_result,
        structured_output=structured_output,
        token_usage=parser.extract_token_usage(raw_response),
        tool_calls=parser.extract_tool_calls(raw_response),
        started_at=started_at,
        completed_at=completed_at,
    )


def run_coding_agent_task(
    *,
    task: dict[str, object],
    agent: str,
    output_dir: Path,
    cache_dir: Path,
    schema_path: Path,
    timeout: int,
    model: str | None = None,
    reasoning_effort: str | None = None,
    agent_args: Sequence[str] = (),
    env_overrides: dict[str, str] | None = None,
    prompt_preamble: str | None = None,
    setup: dict[str, object] | None = None,
    workspace_key: str | None = None,
) -> TaskRecord:
    if agent not in {"codex", "claude"}:
        raise usage_error(f"Unsupported coding agent: {agent}")

    repo_url = resolve_repo_from_task(task, cache_dir)
    if not repo_url:
        raise usage_error(f"Could not resolve repository for {task.get('instance_id') or task.get('original_inst_id')}")

    task = dict(task)
    task["repo_url"] = repo_url
    workspace = checkout(
        repo_url,
        task.get("commit") or "",
        str(cache_dir),
        verbose=True,
        workspace_key=workspace_key,
    )
    if not workspace:
        raise usage_error(f"Checkout failed for {task.get('instance_id') or task.get('original_inst_id')}")

    workspace_path = Path(workspace)
    reset_workspace(workspace_path)

    task_dir = (output_dir / safe_path_component(task.get("instance_id") or task.get("original_inst_id") or "task")).resolve()
    ensure_dir(task_dir)

    prompt = build_prompt(task, agent)
    if prompt_preamble:
        prompt = prompt_preamble.rstrip() + "\n\n" + prompt
    prompt_path = _write_prompt_file(task_dir, "prompt.txt", prompt)

    setup_dict: dict[str, Any] = dict(setup or {})
    copy_paths = setup_dict.get("copy_paths")
    materialized_files = setup_dict.get("files_to_materialize")
    setup_prompt = str(setup_dict.get("setup_prompt") or "").strip()
    setup_timeout_value = setup_dict.get("setup_prompt_timeout")
    setup_timeout = timeout
    if setup_prompt and setup_timeout_value is not None:
        if isinstance(setup_timeout_value, bool) or not isinstance(setup_timeout_value, int) or setup_timeout_value <= 0:
            raise usage_error("setup_prompt_timeout must be a positive integer when provided")
        setup_timeout = setup_timeout_value

    setup_run: SetupRunRecord | None = None
    if agent == "codex":
        codex_env = prepare_codex_runtime_env(
            task_dir,
            materialized_files=materialized_files if isinstance(materialized_files, Sequence) else None,
            copy_paths=copy_paths if isinstance(copy_paths, Sequence) else None,
        )
        if env_overrides:
            codex_env.update(env_overrides)
        if setup_prompt:
            setup_result = _run_codex_invocation(
                task_dir=task_dir,
                workspace_path=workspace_path,
                prompt=setup_prompt,
                prompt_filename="setup-prompt.txt",
                stderr_filename="setup-stderr.log",
                raw_response_filename="setup-raw-response.json",
                raw_output_filename="setup-codex-events.jsonl",
                final_output_filename="setup-last-message.txt",
                timeout=setup_timeout,
                model=model,
                reasoning_effort=reasoning_effort,
                extra_args=agent_args,
                env=codex_env,
                schema_path=None,
            )
            setup_run = build_setup_run_record(
                prompt_path=setup_result.prompt_path,
                stderr_path=setup_result.stderr_path,
                command_result=setup_result.command_result,
                raw_response_path=setup_result.raw_response_path,
                token_usage=setup_result.token_usage,
                tool_calls=setup_result.tool_calls,
                started_at=setup_result.started_at,
                completed_at=setup_result.completed_at,
            )
            if not setup_result.command_result["ok"]:
                diff_text = git_diff(workspace_path)
                diff_path: Path | None = None
                if diff_text.strip():
                    diff_path = task_dir / "workspace.diff"
                    diff_path.write_text(diff_text, encoding="utf-8")
                record = build_task_record(
                    task=task,
                    agent=agent,
                    workspace_path=workspace_path,
                    task_dir=task_dir,
                    prompt_path=prompt_path,
                    command_result=setup_result.command_result,
                    structured_output=None,
                    token_usage=None,
                    tool_calls=[],
                    raw_response_path=None,
                    diff_path=diff_path,
                    model_patch=diff_text,
                    started_at=setup_result.started_at,
                    completed_at=setup_result.completed_at,
                    setup_run=setup_run,
                )
                suffix = "codex"
                record_path = (
                    task_dir / f"{safe_path_component(task.get('instance_id') or task.get('original_inst_id') or 'task')}.{suffix}-record.json"
                )
                write_json(record_path, record)
                return record

        main_result = _run_codex_invocation(
            task_dir=task_dir,
            workspace_path=workspace_path,
            prompt=prompt,
            prompt_filename="prompt.txt",
            stderr_filename="stderr.log",
            raw_response_filename="raw-response.json",
            raw_output_filename="codex-events.jsonl",
            final_output_filename="final-output.json",
            timeout=timeout,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=agent_args,
            env=codex_env,
            schema_path=schema_path,
        )
    else:
        validate_claude_auth()
        command_env = os.environ.copy()
        if env_overrides:
            command_env.update(env_overrides)
        settings_overrides = setup_dict.get("claude_settings_overrides")
        mcp_config_overrides = setup_dict.get("claude_mcp_config")
        settings_path, mcp_config_path = prepare_claude_runtime_files(
            task_dir,
            settings_overrides=settings_overrides if isinstance(settings_overrides, dict) else None,
            mcp_config_overrides=mcp_config_overrides if isinstance(mcp_config_overrides, dict) else None,
            materialized_files=materialized_files if isinstance(materialized_files, Sequence) else None,
            copy_paths=copy_paths if isinstance(copy_paths, Sequence) else None,
        )
        if setup_prompt:
            setup_result = _run_claude_invocation(
                task_dir=task_dir,
                workspace_path=workspace_path,
                prompt=setup_prompt,
                prompt_filename="setup-prompt.txt",
                stderr_filename="setup-stderr.log",
                raw_response_filename="setup-raw-response.json",
                timeout=setup_timeout,
                model=model,
                reasoning_effort=reasoning_effort,
                extra_args=agent_args,
                env=command_env,
                schema_path=None,
                settings_path=settings_path,
                mcp_config_path=mcp_config_path,
                validate_isolation=True,
            )
            setup_run = build_setup_run_record(
                prompt_path=setup_result.prompt_path,
                stderr_path=setup_result.stderr_path,
                command_result=setup_result.command_result,
                raw_response_path=setup_result.raw_response_path,
                token_usage=setup_result.token_usage,
                tool_calls=setup_result.tool_calls,
                started_at=setup_result.started_at,
                completed_at=setup_result.completed_at,
            )
            if not setup_result.command_result["ok"]:
                diff_text = git_diff(workspace_path)
                diff_path: Path | None = None
                if diff_text.strip():
                    diff_path = task_dir / "workspace.diff"
                    diff_path.write_text(diff_text, encoding="utf-8")
                record = build_task_record(
                    task=task,
                    agent=agent,
                    workspace_path=workspace_path,
                    task_dir=task_dir,
                    prompt_path=prompt_path,
                    command_result=setup_result.command_result,
                    structured_output=None,
                    token_usage=None,
                    tool_calls=[],
                    raw_response_path=None,
                    diff_path=diff_path,
                    model_patch=diff_text,
                    started_at=setup_result.started_at,
                    completed_at=setup_result.completed_at,
                    setup_run=setup_run,
                )
                suffix = "claude"
                record_path = (
                    task_dir / f"{safe_path_component(task.get('instance_id') or task.get('original_inst_id') or 'task')}.{suffix}-record.json"
                )
                write_json(record_path, record)
                return record

        main_result = _run_claude_invocation(
            task_dir=task_dir,
            workspace_path=workspace_path,
            prompt=prompt,
            prompt_filename="prompt.txt",
            stderr_filename="stderr.log",
            raw_response_filename="raw-response.json",
            timeout=timeout,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=agent_args,
            env=command_env,
            schema_path=schema_path,
            settings_path=settings_path,
            mcp_config_path=mcp_config_path,
            validate_isolation=True,
        )

    diff_text = git_diff(workspace_path)
    diff_path: Path | None = None
    if diff_text.strip():
        diff_path = task_dir / "workspace.diff"
        diff_path.write_text(diff_text, encoding="utf-8")

    record = build_task_record(
        task=task,
        agent=agent,
        workspace_path=workspace_path,
        task_dir=task_dir,
        prompt_path=main_result.prompt_path,
        command_result=main_result.command_result,
        structured_output=main_result.structured_output,
        token_usage=main_result.token_usage,
        tool_calls=main_result.tool_calls,
        raw_response_path=main_result.raw_response_path,
        diff_path=diff_path,
        model_patch=diff_text,
        started_at=main_result.started_at,
        completed_at=main_result.completed_at,
        setup_run=setup_run,
    )

    suffix = "codex" if agent == "codex" else "claude"
    record_path = task_dir / f"{safe_path_component(task.get('instance_id') or task.get('original_inst_id') or 'task')}.{suffix}-record.json"
    write_json(record_path, record)
    return record
