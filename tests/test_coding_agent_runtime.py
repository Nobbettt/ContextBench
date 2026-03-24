# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from contextbench.coding_agents import build_prompt
from contextbench.coding_agents.constants import CODEX_OUTPUT_SCHEMA_PATH
from contextbench.coding_agents.runtime import (
    build_claude_command,
    build_codex_command,
    prepare_claude_runtime_files,
    prepare_codex_runtime_env,
    run_command,
    run_coding_agent_task,
    validate_claude_auth,
    validate_claude_isolation,
)


def test_build_prompt_uses_pr_description_and_workflow_scaffold() -> None:
    prompt = build_prompt(
        {
            "bench": "Verified",
            "repo": "psf/requests",
            "instance_id": "psf__requests-1142",
            "prompt": "Fix the bug.",
        },
        "codex",
    )

    assert "Consider the following PR description:" in prompt
    assert "Work inside the checked-out repository workspace for this task." in prompt
    assert "Populate these fields carefully: status, final_answer, notes, and the final repository context you relied on most." in prompt
    assert "You may leave retrieval_steps empty if you do not have a concise chronological summary." in prompt
    assert "Fix the bug." in prompt


def test_build_prompt_dispatches_agent_specific_identity() -> None:
    codex_prompt = build_prompt(
        {"bench": "Verified", "instance_id": "task-1", "prompt": "Fix it."},
        "codex",
    )
    claude_prompt = build_prompt(
        {"bench": "Verified", "instance_id": "task-1", "prompt": "Fix it."},
        "claude",
    )

    assert "Task ID:" not in codex_prompt
    assert "Task ID:" not in claude_prompt
    assert "Agent:" not in codex_prompt
    assert "Agent:" not in claude_prompt


def test_build_prompt_accepts_agent_aliases() -> None:
    prompt = build_prompt(
        {"bench": "Verified", "instance_id": "task-1", "prompt": "Fix it."},
        "claude-code",
    )

    assert "Consider the following PR description:" in prompt


def test_run_module_supports_codex_dry_run_with_task_data(tmp_path) -> None:
    task_data = tmp_path / "tasks.json"
    task_csv = tmp_path / "tasks.csv"
    task_data.write_text(
        json.dumps(
            [
                {
                    "instance_id": "psf__requests-1142",
                    "original_inst_id": "psf__requests-1142",
                    "repo_url": "https://github.com/psf/requests.git",
                    "base_commit": "abc123",
                    "problem_statement": "Fix a bug in requests.",
                    "language": "python",
                }
            ]
        ),
        encoding="utf-8",
    )
    task_csv.write_text(
        "bench,instance_id,original_inst_id,language,status,patch_files,patch_blocks,patch_span,gold_context_length,num_agents,repo,commit\n"
        "Verified,psf__requests-1142,psf__requests-1142,python,pass,1,1,1,10,1,,abc123\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "contextbench.run",
            "--agent",
            "codex",
            "--task-data",
            str(task_data),
            "--task-csv",
            str(task_csv),
            "--dry-run",
        ],
        cwd=".",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Loaded 1 tasks" in result.stderr


def test_claude_command_uses_verbose_json_mode(tmp_path, schema_path) -> None:
    settings_path = tmp_path / "claude.settings.json"
    mcp_config_path = tmp_path / "claude.mcp.json"
    settings_path.write_text("{}", encoding="utf-8")
    mcp_config_path.write_text('{"mcpServers": {}}', encoding="utf-8")

    command, _ = build_claude_command(
        schema_path=schema_path,
        prompt="test prompt",
        model=None,
        reasoning_effort=None,
        extra_args=[],
        settings_path=settings_path,
        mcp_config_path=mcp_config_path,
    )

    assert "--verbose" in command
    assert command[:4] == ["claude", "--print", "--output-format", "json"]
    assert "--settings" in command
    assert "--mcp-config" in command
    assert "--setting-sources" in command
    assert "--disable-slash-commands" in command
    assert "--strict-mcp-config" in command


def test_claude_command_omits_schema_when_not_requested(tmp_path) -> None:
    settings_path = tmp_path / "claude.settings.json"
    mcp_config_path = tmp_path / "claude.mcp.json"
    settings_path.write_text("{}", encoding="utf-8")
    mcp_config_path.write_text('{"mcpServers": {}}', encoding="utf-8")

    command, _ = build_claude_command(
        schema_path=None,
        prompt="bootstrap prompt",
        model=None,
        reasoning_effort=None,
        extra_args=[],
        settings_path=settings_path,
        mcp_config_path=mcp_config_path,
    )

    assert "--json-schema" not in command


def test_codex_command_uses_json_event_mode(tmp_path, schema_path) -> None:
    command, _ = build_codex_command(
        workspace_path=tmp_path,
        schema_path=schema_path,
        final_output_path=tmp_path / "final-output.json",
        model=None,
        reasoning_effort=None,
        extra_args=[],
    )

    assert "--json" in command
    assert "--verbose" not in command


def test_codex_command_omits_schema_when_not_requested(tmp_path) -> None:
    command, _ = build_codex_command(
        workspace_path=tmp_path,
        schema_path=None,
        final_output_path=tmp_path / "setup-last-message.txt",
        model=None,
        reasoning_effort=None,
        extra_args=[],
    )

    assert "--output-schema" not in command
    assert "--output-last-message" in command


def test_codex_command_includes_add_dir_for_runtime_root(tmp_path, schema_path) -> None:
    runtime_root = tmp_path / "runtime-root"
    runtime_root.mkdir()

    command, _ = build_codex_command(
        workspace_path=tmp_path,
        schema_path=schema_path,
        final_output_path=tmp_path / "final-output.json",
        model=None,
        reasoning_effort=None,
        writable_dirs=[runtime_root],
        extra_args=[],
    )

    assert "--add-dir" in command
    assert str(runtime_root.resolve()) in command


def test_claude_command_maps_xhigh_reasoning_effort_to_max(tmp_path, schema_path) -> None:
    settings_path = tmp_path / "claude.settings.json"
    mcp_config_path = tmp_path / "claude.mcp.json"
    settings_path.write_text("{}", encoding="utf-8")
    mcp_config_path.write_text('{"mcpServers": {}}', encoding="utf-8")

    command, _ = build_claude_command(
        schema_path=schema_path,
        prompt="test prompt",
        model=None,
        reasoning_effort="max",
        extra_args=[],
        settings_path=settings_path,
        mcp_config_path=mcp_config_path,
    )

    assert "--effort" in command
    assert "max" in command


def test_codex_command_includes_reasoning_effort_override(tmp_path, schema_path) -> None:
    command, _ = build_codex_command(
        workspace_path=tmp_path,
        schema_path=schema_path,
        final_output_path=tmp_path / "final-output.json",
        model=None,
        reasoning_effort="xhigh",
        extra_args=[],
    )

    assert "-c" in command
    assert 'model_reasoning_effort="xhigh"' in command


def test_validate_claude_isolation_accepts_clean_verbose_response() -> None:
    raw_response = {
        "agent": "claude",
        "response_format": "json",
        "response": [
            {
                "type": "system",
                "subtype": "init",
                "plugins": [],
                "mcp_servers": {},
                "slash_commands": [],
            }
        ],
    }

    validate_claude_isolation(raw_response)


def test_validate_claude_isolation_rejects_loaded_plugins() -> None:
    raw_response = {
        "agent": "claude",
        "response_format": "json",
        "response": [
            {
                "type": "system",
                "subtype": "init",
                "plugins": ["skill"],
                "mcp_servers": {},
                "slash_commands": [],
            }
        ],
    }

    with pytest.raises(RuntimeError, match="plugins are still loaded"):
        validate_claude_isolation(raw_response)


def test_validate_claude_auth_rejects_logged_out(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout='{"loggedIn": false, "authMethod": "none"}', stderr="")

    monkeypatch.setattr("contextbench.coding_agents.runtime.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="not logged in"):
        validate_claude_auth()


def test_run_command_timeout_decodes_byte_output(tmp_path, monkeypatch) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs.get("timeout", 30),
            output=b"partial stdout\n",
            stderr=b"partial stderr\n",
        )

    monkeypatch.setattr("contextbench.coding_agents.runtime.subprocess.run", fake_run)

    result = run_command(
        ["codex", "exec", "-"],
        cwd=tmp_path,
        stdin_text="prompt",
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=30,
    )

    assert result == {"ok": False, "exit_code": None, "signal": "SIGTERM", "timeout": True}
    assert stdout_path.read_text(encoding="utf-8") == "partial stdout\n"
    assert stderr_path.read_text(encoding="utf-8") == "partial stderr\n"


def test_prepare_codex_runtime_env_copies_auth_only(tmp_path) -> None:
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text('{"token":"abc"}', encoding="utf-8")
    (source_codex_dir / "config.toml").write_text('profile = "should-not-copy"\n', encoding="utf-8")

    env = prepare_codex_runtime_env(tmp_path / "task", source_codex_dir=source_codex_dir)

    isolated_home = Path(env["HOME"]) / ".codex"
    assert (isolated_home / "auth.json").exists()
    assert not (isolated_home / "config.toml").exists()
    assert env["OTEL_SDK_DISABLED"] == "true"
    assert env["HOME"] != str(Path.home())


def test_prepare_codex_runtime_env_applies_runtime_files(tmp_path) -> None:
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text('{"token":"abc"}', encoding="utf-8")

    extra_dir = tmp_path / "variant-files"
    extra_dir.mkdir()
    (extra_dir / "plugin.json").write_text('{"enabled":true}', encoding="utf-8")

    env = prepare_codex_runtime_env(
        tmp_path / "task",
        source_codex_dir=source_codex_dir,
        copy_paths=[
            {
                "source": str(extra_dir),
                "destination": "plugins",
                "target_root": "codex_home",
            }
        ],
        materialized_files=[
            {
                "path": "settings/variant.json",
                "content": {"mode": "compare"},
                "format": "json",
                "target_root": "xdg_config_home",
            }
        ],
    )

    isolated_home = Path(env["HOME"]) / ".codex"
    assert (isolated_home / "plugins" / "plugin.json").exists()
    assert json.loads((Path(env["XDG_CONFIG_HOME"]) / "settings" / "variant.json").read_text(encoding="utf-8")) == {
        "mode": "compare"
    }


def test_prepare_codex_runtime_env_copies_directory_to_nested_home_path(tmp_path) -> None:
    source_codex_dir = tmp_path / "source-codex"
    source_codex_dir.mkdir()
    (source_codex_dir / "auth.json").write_text('{"token":"abc"}', encoding="utf-8")

    skills_dir = tmp_path / "superpowers-skills"
    (skills_dir / "using-superpowers").mkdir(parents=True)
    (skills_dir / "using-superpowers" / "SKILL.md").write_text("name: using-superpowers\n", encoding="utf-8")

    env = prepare_codex_runtime_env(
        tmp_path / "task",
        source_codex_dir=source_codex_dir,
        copy_paths=[
            {
                "source": str(skills_dir),
                "destination": ".agents/skills/superpowers",
                "target_root": "home_dir",
            }
        ],
    )

    skill_path = Path(env["HOME"]) / ".agents" / "skills" / "superpowers" / "using-superpowers" / "SKILL.md"
    assert skill_path.exists()


def test_prepare_claude_runtime_files_applies_overrides_and_materialized_files(tmp_path) -> None:
    settings_path, mcp_config_path = prepare_claude_runtime_files(
        tmp_path,
        settings_overrides={"permissions": {"allow": ["Read"]}},
        mcp_config_overrides={"mcpServers": {"demo": {"command": "demo-mcp"}}},
        materialized_files=[
            {
                "path": "notes/setup.txt",
                "content": "variant setup",
                "format": "text",
                "target_root": "task_dir",
            }
        ],
    )

    assert json.loads(settings_path.read_text(encoding="utf-8")) == {"permissions": {"allow": ["Read"]}}
    assert json.loads(mcp_config_path.read_text(encoding="utf-8")) == {
        "mcpServers": {"demo": {"command": "demo-mcp"}}
    }
    assert (tmp_path / "notes" / "setup.txt").read_text(encoding="utf-8") == "variant setup"


def test_run_coding_agent_task_codex_writes_record_and_diff(tmp_path, monkeypatch, make_final_output) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = Path("results")
    cache_dir = Path("cache")
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-1",
        "original_inst_id": "task-1",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir)},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.build_command",
        lambda **kwargs: (
            captured.setdefault("final_output_path", kwargs["final_output_path"])
            and captured.setdefault("reasoning_effort", kwargs["reasoning_effort"])
            and captured.setdefault("writable_dirs", kwargs["writable_dirs"])
            and ["codex", "exec", "-"],
            "codex-events.jsonl",
        ),
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "diff --git a/a.py b/a.py\n")

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        captured["command"] = list(command)
        captured["cwd"] = cwd
        captured["stdin_text"] = stdin_text
        captured["env"] = env
        stdout_path.write_text(
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4, "output_tokens": 2}}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_path"]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text(
            json.dumps(make_final_output(task_id="task-1", touched_files=["a.py"], retrieved_context_files=["a.py"])),
            encoding="utf-8",
        )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        reasoning_effort="high",
        env_overrides={"EXPERIMENT": "1"},
        prompt_preamble="Variant instructions",
    )

    task_dir = (tmp_path / "results" / "task-1").resolve()
    record_path = task_dir / "task-1.codex-record.json"

    assert record["agent"] == "codex"
    assert record["final_output"]["task_id"] == "task-1"
    assert record["tool_calls"] == []
    assert record["model_patch"].startswith("diff --git")
    assert Path(record["raw_response_path"]).exists()
    assert Path(record["diff_path"]).exists()
    assert record_path.exists()
    prompt_text = (task_dir / "prompt.txt").read_text(encoding="utf-8")
    assert prompt_text.startswith("Variant instructions")
    assert "Consider the following PR description:" in prompt_text
    assert "Work inside the checked-out repository workspace for this task." in prompt_text
    assert "You may leave touched_files empty if uncertain; actual file changes can be inferred from the repository diff." in prompt_text
    assert captured["reasoning_effort"] == "high"
    assert isinstance(captured["final_output_path"], Path)
    assert captured["final_output_path"].is_absolute()
    assert [str(path) for path in captured["writable_dirs"]] == [str((task_dir / "codex-runtime").resolve())]
    assert Path(record["task_dir"]).is_absolute()
    assert captured["cwd"] == workspace_path
    assert captured["env"] == {"HOME": str(task_dir), "EXPERIMENT": "1"}


def test_run_coding_agent_task_codex_retries_transient_failure(tmp_path, monkeypatch, make_final_output) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-retry",
        "original_inst_id": "task-retry",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {"attempt": 0, "final_output_path": None}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir)},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.build_command",
        lambda **kwargs: (
            captured.__setitem__("final_output_path", kwargs["final_output_path"]) or ["codex", "exec", "-"],
            "codex-events.jsonl",
        ),
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")
    monkeypatch.setattr("contextbench.agents.codex.runtime.time.sleep", lambda seconds: None)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        captured["attempt"] = int(captured["attempt"]) + 1
        attempt = int(captured["attempt"])
        if attempt == 1:
            stdout_path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "thread.started", "thread_id": "t-1"}),
                        json.dumps({"type": "turn.started"}),
                        json.dumps({"type": "error", "message": "Reconnecting... 2/5 (We're currently experiencing high demand, which may cause temporary errors.)"}),
                        json.dumps({"type": "turn.failed", "error": {"message": "unexpected status 401 Unauthorized: Missing bearer or basic authentication in header"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            stderr_path.write_text(
                "failed to connect to websocket: HTTP error: 500 Internal Server Error\n",
                encoding="utf-8",
            )
            return {"ok": False, "exit_code": 1, "signal": None, "timeout": False}

        stdout_path.write_text(
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_path"]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text(
            json.dumps(
                make_final_output(
                    task_id="task-retry",
                    touched_files=["a.py"],
                    retrieved_context_files=["a.py"],
                )
            ),
            encoding="utf-8",
        )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
    )

    task_dir = (output_dir / "task-retry").resolve()

    assert captured["attempt"] == 2
    assert record["status"] == "completed"
    assert record["ok"] is True
    assert record["token_usage"]["input_tokens"] == 5
    assert (task_dir / "stderr.attempt1.log").exists()
    assert (task_dir / "raw-response.attempt1.json").exists()
    assert (task_dir / "codex-events.attempt1.jsonl").exists()
    assert "high demand" in (task_dir / "raw-response.attempt1.json").read_text(encoding="utf-8")


def test_run_coding_agent_task_passes_workspace_key_to_checkout(tmp_path, monkeypatch, make_final_output) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-workspace-key",
        "original_inst_id": "task-workspace-key",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {}

    def fake_checkout(*args, **kwargs):
        captured["workspace_key"] = kwargs.get("workspace_key")
        return str(workspace_path)

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", fake_checkout)
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir)},
    )
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.build_command",
        lambda **kwargs: (
            captured.setdefault("final_output_path", kwargs["final_output_path"]) and ["codex", "exec", "-"],
            "codex-events.jsonl",
        ),
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        stdout_path.write_text(
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 4, "output_tokens": 2}}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_path"]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text(
            json.dumps(
                make_final_output(
                    task_id="task-workspace-key",
                    touched_files=["a.py"],
                    retrieved_context_files=["a.py"],
                )
            ),
            encoding="utf-8",
        )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        workspace_key="suite__task__variant",
    )

    assert captured["workspace_key"] == "suite__task__variant"


def test_run_coding_agent_task_codex_setup_prompt_runs_before_scored_prompt(
    tmp_path,
    monkeypatch,
    make_final_output,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-setup",
        "original_inst_id": "task-setup",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {"final_output_paths": {}, "calls": []}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home"), "EXPERIMENT": "1"},
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")

    def fake_build_codex_command(**kwargs):
        phase = "setup" if kwargs["schema_path"] is None else "main"
        captured["final_output_paths"][phase] = kwargs["final_output_path"]
        return ["codex", "exec", phase], f"{phase}-events.jsonl"

    monkeypatch.setattr("contextbench.agents.codex.runtime.build_command", fake_build_codex_command)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        phase = command[-1]
        assert phase in {"setup", "main"}
        captured["calls"].append(
            {
                "phase": phase,
                "cwd": cwd,
                "stdin_text": stdin_text,
                "timeout": timeout,
                "env": dict(env or {}),
            }
        )
        stdout_path.write_text(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 11 if phase == "setup" else 4, "output_tokens": 3 if phase == "setup" else 2},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        final_output_path = captured["final_output_paths"][phase]
        assert isinstance(final_output_path, Path)
        if phase == "setup":
            (cwd / "setup-ran.txt").write_text("yes", encoding="utf-8")
            final_output_path.write_text("setup complete", encoding="utf-8")
        else:
            assert (cwd / "setup-ran.txt").exists()
            final_output_path.write_text(
                json.dumps(
                    make_final_output(
                        task_id="task-setup",
                        touched_files=["a.py"],
                        retrieved_context_files=["a.py"],
                    )
                ),
                encoding="utf-8",
            )
        return {"ok": True, "exit_code": 0, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        setup={"setup_prompt": "Bootstrap tools", "setup_prompt_timeout": 12},
    )

    task_dir = output_dir / "task-setup"

    assert [call["phase"] for call in captured["calls"]] == ["setup", "main"]
    assert captured["calls"][0]["cwd"] == workspace_path
    assert captured["calls"][1]["cwd"] == workspace_path
    assert captured["calls"][0]["timeout"] == 12
    assert captured["calls"][1]["timeout"] == 30
    assert record["status"] == "completed"
    assert record["token_usage"]["input_tokens"] == 4
    assert record["setup_run"]["status"] == "completed"
    assert record["setup_run"]["token_usage"]["input_tokens"] == 11
    assert record["raw_response_path"] != record["setup_run"]["raw_response_path"]
    assert Path(record["setup_run"]["prompt_path"]).name == "setup-prompt.txt"
    assert Path(record["setup_run"]["stderr_path"]).name == "setup-stderr.log"
    assert Path(record["setup_run"]["raw_response_path"]).exists()
    assert (task_dir / "setup-last-message.txt").read_text(encoding="utf-8") == "setup complete"
    assert Path(record["raw_response_path"]).exists()


def test_run_coding_agent_task_codex_setup_prompt_failure_short_circuits_scored_run(tmp_path, monkeypatch) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = CODEX_OUTPUT_SCHEMA_PATH.resolve()
    task = {
        "bench": "Verified",
        "instance_id": "task-setup-fail",
        "original_inst_id": "task-setup-fail",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "prompt": "Fix the bug.",
        "language": "python",
    }
    captured: dict[str, object] = {"final_output_paths": {}, "calls": []}

    monkeypatch.setattr("contextbench.coding_agents.runtime.checkout", lambda *args, **kwargs: str(workspace_path))
    monkeypatch.setattr("contextbench.coding_agents.runtime.reset_workspace", lambda path: None)
    monkeypatch.setattr(
        "contextbench.agents.codex.runtime.prepare_runtime_env",
        lambda task_dir, **kwargs: {"HOME": str(task_dir / "codex-home")},
    )
    monkeypatch.setattr("contextbench.coding_agents.runtime.git_diff", lambda path: "")

    def fake_build_codex_command(**kwargs):
        phase = "setup" if kwargs["schema_path"] is None else "main"
        captured["final_output_paths"][phase] = kwargs["final_output_path"]
        return ["codex", "exec", phase], f"{phase}-events.jsonl"

    monkeypatch.setattr("contextbench.agents.codex.runtime.build_command", fake_build_codex_command)

    def fake_run_command(command, *, cwd, stdin_text, stdout_path, stderr_path, timeout, env=None):
        phase = command[-1]
        captured["calls"].append(phase)
        stdout_path.write_text(
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 8, "output_tokens": 1}}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("setup failed", encoding="utf-8")
        if phase != "setup":
            pytest.fail("scored prompt should not run after setup failure")
        final_output_path = captured["final_output_paths"][phase]
        assert isinstance(final_output_path, Path)
        final_output_path.write_text("setup failed", encoding="utf-8")
        return {"ok": False, "exit_code": 9, "signal": None, "timeout": False}

    monkeypatch.setattr("contextbench.agents.codex.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
        setup={"setup_prompt": "Bootstrap tools"},
    )

    task_dir = output_dir / "task-setup-fail"

    assert captured["calls"] == ["setup"]
    assert record["status"] == "failed"
    assert record["ok"] is False
    assert record["raw_response_path"] is None
    assert record["token_usage"] is None
    assert record["tool_calls"] == []
    assert record["setup_run"]["status"] == "failed"
    assert record["setup_run"]["exit_code"] == 9
    assert Path(record["setup_run"]["raw_response_path"]).exists()
    assert Path(record["prompt_path"]).exists()
    assert not (task_dir / "raw-response.json").exists()
