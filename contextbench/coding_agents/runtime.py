# SPDX-License-Identifier: Apache-2.0
# Fork note: Modified by Norbert Laszlo on 2026-03-22 from upstream ContextBench.
# Summary of changes: support unscored setup prompts that run before the scored benchmark prompt.

"""Runtime helpers for Codex and Claude CLI execution."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..agents.claude.runtime import (
    build_command as build_claude_command,
    prepare_runtime_files as prepare_claude_runtime_files,
    run_invocation as _run_claude_invocation,
    validate_auth as validate_claude_auth,
    validate_isolation as validate_claude_isolation,
)
from ..agents.codex.runtime import (
    build_command as build_codex_command,
    prepare_runtime_env as prepare_codex_runtime_env,
    run_invocation as _run_codex_invocation,
    runtime_root as codex_runtime_root,
)
from ..agents.registry import get_coding_agent_adapter
from ..core import checkout
from .files import ensure_dir, safe_path_component, usage_error, write_json
from .prompting import build_prompt
from .records import build_setup_run_record, build_task_record
from .runtime_common import run_command, write_prompt_file
from .types import SetupRunRecord, TaskRecord


def _record_path_for_task(*, task_dir: Path, task: dict[str, object], suffix: str) -> Path:
    task_key = safe_path_component(task.get("instance_id") or task.get("original_inst_id") or "task")
    return task_dir / f"{task_key}.{suffix}-record.json"


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
    try:
        adapter = get_coding_agent_adapter(agent)
    except ValueError as exc:
        raise usage_error(str(exc)) from exc
    agent = adapter.name

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
    prompt_path = write_prompt_file(task_dir, "prompt.txt", prompt)

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

    prepared_runtime = adapter.prepare_runtime(
        task_dir=task_dir,
        setup=setup_dict,
        env_overrides=env_overrides,
    )
    setup_run: SetupRunRecord | None = None
    if setup_prompt:
        setup_result = adapter.run_setup_invocation(
            task_dir=task_dir,
            workspace_path=workspace_path,
            prompt=setup_prompt,
            timeout=setup_timeout,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=tuple(agent_args),
            prepared_runtime=prepared_runtime,
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
            record_path = _record_path_for_task(task_dir=task_dir, task=task, suffix=adapter.record_suffix)
            write_json(record_path, record)
            return record

    main_result = adapter.run_main_invocation(
        task_dir=task_dir,
        workspace_path=workspace_path,
        prompt=prompt,
        timeout=timeout,
        model=model,
        reasoning_effort=reasoning_effort,
        extra_args=tuple(agent_args),
        schema_path=schema_path,
        prepared_runtime=prepared_runtime,
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

    record_path = _record_path_for_task(task_dir=task_dir, task=task, suffix=adapter.record_suffix)
    write_json(record_path, record)
    return record
