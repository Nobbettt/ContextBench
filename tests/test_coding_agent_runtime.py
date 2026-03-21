# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from contextbench.coding_agents import build_prompt
from contextbench.coding_agents.runtime import (
    build_claude_command,
    build_codex_command,
    prepare_codex_runtime_env,
    run_coding_agent_task,
    validate_claude_auth,
    validate_claude_isolation,
)


def test_build_prompt_mentions_local_only_constraints() -> None:
    prompt = build_prompt(
        {
            "bench": "Verified",
            "repo": "psf/requests",
            "instance_id": "psf__requests-1142",
            "prompt": "Fix the bug.",
        },
        "codex",
    )

    assert "Do not use web search or external sources." in prompt
    assert "Fix the bug." in prompt


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


def test_codex_command_uses_json_event_mode(tmp_path, schema_path) -> None:
    command, _ = build_codex_command(
        workspace_path=tmp_path,
        schema_path=schema_path,
        final_output_path=tmp_path / "final-output.json",
        model=None,
        extra_args=[],
    )

    assert "--json" in command
    assert "--verbose" not in command


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


def test_run_coding_agent_task_codex_writes_record_and_diff(tmp_path, monkeypatch, make_final_output) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    output_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    schema_path = Path("contextbench/schemas/coding_agent_output.schema.json").resolve()
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
    monkeypatch.setattr("contextbench.coding_agents.runtime.prepare_codex_runtime_env", lambda task_dir: {"HOME": str(task_dir)})
    monkeypatch.setattr(
        "contextbench.coding_agents.runtime.build_codex_command",
        lambda **kwargs: (
            captured.setdefault("final_output_path", kwargs["final_output_path"]) and ["codex", "exec", "-"],
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

    monkeypatch.setattr("contextbench.coding_agents.runtime.run_command", fake_run_command)

    record = run_coding_agent_task(
        task=task,
        agent="codex",
        output_dir=output_dir,
        cache_dir=cache_dir,
        schema_path=schema_path,
        timeout=30,
    )

    task_dir = output_dir / "task-1"
    record_path = task_dir / "task-1.codex-record.json"

    assert record["agent"] == "codex"
    assert record["final_output"]["task_id"] == "task-1"
    assert record["tool_calls"] == []
    assert record["model_patch"].startswith("diff --git")
    assert Path(record["raw_response_path"]).exists()
    assert Path(record["diff_path"]).exists()
    assert record_path.exists()
    assert (task_dir / "prompt.txt").read_text(encoding="utf-8").startswith("You are running a ContextBench task")
    assert captured["cwd"] == workspace_path
    assert captured["env"] == {"HOME": str(task_dir)}
    assert "Fix the bug." in str(captured["stdin_text"])
