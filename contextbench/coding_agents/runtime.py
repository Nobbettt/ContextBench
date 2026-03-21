# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers for Codex and Claude CLI execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from ..core import checkout
from .files import ensure_dir, read_json, safe_path_component, usage_error, write_json
from .prompting import build_prompt
from .records import build_task_record
from .response_parsing import (
    build_claude_raw_response,
    build_codex_raw_response,
)
from ..agents.claude.parser import ClaudeAgentParser
from ..agents.codex.parser import CodexAgentParser
from .types import ClaudeRawResponse, CodexRawResponse, CommandResult, TaskRecord, TokenUsage


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


def build_codex_command(
    *,
    workspace_path: Path,
    schema_path: Path,
    final_output_path: Path,
    model: str | None,
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
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(final_output_path),
        "--json",
        "--cd",
        str(workspace_path),
    ]
    if model:
        command.extend(["--model", model])
    command.extend(extra_args)
    command.append("-")
    return command, "codex-events.jsonl"


def build_claude_command(
    *,
    schema_path: Path,
    prompt: str,
    model: str | None,
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
        "--json-schema",
        json.dumps(read_json(schema_path), ensure_ascii=False),
    ]
    if model:
        command.extend(["--model", model])
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
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "signal": None,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        return {
            "ok": False,
            "exit_code": None,
            "signal": "SIGTERM",
            "timeout": True,
        }


def prepare_codex_runtime_env(task_dir: Path, source_codex_dir: Path | None = None) -> dict[str, str]:
    source_root = source_codex_dir or (Path.home() / ".codex")
    auth_path = source_root / "auth.json"
    if not auth_path.exists():
        raise usage_error(f"Codex auth is unavailable: expected {auth_path}")

    runtime_root = task_dir / "codex-runtime"
    home_dir = runtime_root / "home"
    codex_home = home_dir / ".codex"
    xdg_config_home = runtime_root / "xdg-config"
    xdg_data_home = runtime_root / "xdg-data"
    xdg_cache_home = runtime_root / "xdg-cache"

    for path in (codex_home, xdg_config_home, xdg_data_home, xdg_cache_home):
        ensure_dir(path)

    shutil.copy2(auth_path, codex_home / "auth.json")
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


def prepare_claude_runtime_files(task_dir: Path) -> tuple[Path, Path]:
    settings_path = task_dir / "claude.settings.json"
    mcp_config_path = task_dir / "claude.mcp.json"
    write_json(settings_path, {})
    write_json(mcp_config_path, {"mcpServers": {}})
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


def run_coding_agent_task(
    *,
    task: dict[str, object],
    agent: str,
    output_dir: Path,
    cache_dir: Path,
    schema_path: Path,
    timeout: int,
    model: str | None = None,
    agent_args: Sequence[str] = (),
) -> TaskRecord:
    if agent not in {"codex", "claude"}:
        raise usage_error(f"Unsupported coding agent: {agent}")

    repo_url = resolve_repo_from_task(task, cache_dir)
    if not repo_url:
        raise usage_error(f"Could not resolve repository for {task.get('instance_id') or task.get('original_inst_id')}")

    task = dict(task)
    task["repo_url"] = repo_url
    workspace = checkout(repo_url, task.get("commit") or "", str(cache_dir), verbose=True)
    if not workspace:
        raise usage_error(f"Checkout failed for {task.get('instance_id') or task.get('original_inst_id')}")

    workspace_path = Path(workspace)
    reset_workspace(workspace_path)

    task_dir = output_dir / safe_path_component(task.get("instance_id") or task.get("original_inst_id") or "task")
    ensure_dir(task_dir)

    prompt = build_prompt(task, agent)
    prompt_path = task_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    stderr_path = task_dir / "stderr.log"
    started_at = time.time()
    raw_response_path = task_dir / "raw-response.json"
    if agent == "codex":
        parser = CodexAgentParser()
        codex_env = prepare_codex_runtime_env(task_dir)
        final_output_path = task_dir / "final-output.json"
        command, raw_output_name = build_codex_command(
            workspace_path=workspace_path,
            schema_path=schema_path,
            final_output_path=final_output_path,
            model=model,
            extra_args=agent_args,
        )
        raw_output_path = task_dir / raw_output_name
        command_result = run_command(
            command,
            cwd=workspace_path,
            stdin_text=prompt,
            stdout_path=raw_output_path,
            stderr_path=stderr_path,
            timeout=timeout,
            env=codex_env,
        )
        raw_response: CodexRawResponse = build_codex_raw_response(raw_output_path, final_output_path)
        write_json(raw_response_path, raw_response)
        structured_output = parser.extract_structured_output(raw_response)
        token_usage: TokenUsage | None = parser.extract_token_usage(raw_response)
        tool_calls = parser.extract_tool_calls(raw_response)
    else:
        validate_claude_auth()
        parser = ClaudeAgentParser()
        settings_path, mcp_config_path = prepare_claude_runtime_files(task_dir)
        raw_output_path = raw_response_path
        command, _ = build_claude_command(
            schema_path=schema_path,
            prompt=prompt,
            model=model,
            extra_args=agent_args,
            settings_path=settings_path,
            mcp_config_path=mcp_config_path,
        )
        command_result = run_command(
            command,
            cwd=workspace_path,
            stdin_text=None,
            stdout_path=raw_output_path,
            stderr_path=stderr_path,
            timeout=timeout,
        )
        raw_response: ClaudeRawResponse = build_claude_raw_response(raw_output_path)
        write_json(raw_response_path, raw_response)
        validate_claude_isolation(raw_response)
        structured_output = parser.extract_structured_output(raw_response)
        token_usage = parser.extract_token_usage(raw_response)
        tool_calls = parser.extract_tool_calls(raw_response)

    completed_at = time.time()
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
        command_result=command_result,
        structured_output=structured_output,
        token_usage=token_usage,
        tool_calls=tool_calls,
        raw_response_path=raw_response_path,
        diff_path=diff_path,
        model_patch=diff_text,
        started_at=started_at,
        completed_at=completed_at,
    )

    suffix = "codex" if agent == "codex" else "claude"
    record_path = task_dir / f"{safe_path_component(task.get('instance_id') or task.get('original_inst_id') or 'task')}.{suffix}-record.json"
    write_json(record_path, record)
    return record
