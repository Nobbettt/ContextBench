# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from contextbench.run_suites import RunSuiteConfig, RunSuiteRunner, build_run_suite_variant
from contextbench.coding_agents.files import safe_path_component
from contextbench.coding_agents.constants import (
    CLAUDE_OUTPUT_SCHEMA_PATH,
    CODEX_OUTPUT_SCHEMA_PATH,
)


def _write_task_inputs(tmp_path: Path, *, count: int = 2) -> tuple[Path, Path]:
    task_rows = []
    csv_rows = [
        "bench,instance_id,original_inst_id,language,status,patch_files,patch_blocks,patch_span,gold_context_length,num_agents,repo,commit"
    ]
    for index in range(count):
        instance_id = f"psf__requests-{1000 + index}"
        task_rows.append(
            {
                "instance_id": instance_id,
                "original_inst_id": instance_id,
                "repo_url": "https://github.com/psf/requests.git",
                "base_commit": f"abc12{index}",
                "problem_statement": f"Fix bug {index}.",
                "language": "python",
            }
        )
        csv_rows.append(
            f"Verified,{instance_id},{instance_id},python,pass,1,1,1,10,1,,abc12{index}"
        )

    task_data = tmp_path / "tasks.json"
    task_csv = tmp_path / "tasks.csv"
    task_data.write_text(json.dumps(task_rows), encoding="utf-8")
    task_csv.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    return task_data, task_csv


def _fake_run_coding_agent_task(call_log: list[dict[str, object]]):
    def _run(
        *,
        task,
        agent,
        output_dir,
        cache_dir,
        schema_path,
        timeout,
        model=None,
        reasoning_effort=None,
        agent_args=(),
        env_overrides=None,
        prompt_preamble=None,
        setup=None,
        workspace_key=None,
    ):
        del cache_dir, schema_path, timeout, model
        task_id = safe_path_component(task.get("instance_id") or task.get("original_inst_id") or "task")
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        workspace_path = task_dir / "workspaces" / safe_path_component(workspace_key or f"{agent}-{task_id}")
        workspace_path.mkdir(parents=True, exist_ok=True)
        prompt_text = (prompt_preamble or "") + "\nFix prompt"
        (task_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        record = {
            "agent": agent,
            "bench": task.get("bench"),
            "instance_id": task.get("instance_id"),
            "original_inst_id": task.get("original_inst_id"),
            "repo_url": task.get("repo_url"),
            "commit": task.get("commit") or task.get("base_commit"),
            "task_dir": str(task_dir),
            "workspace_path": str(workspace_path),
            "prompt_path": str(task_dir / "prompt.txt"),
            "started_at": "2026-03-22T00:00:00Z",
            "completed_at": "2026-03-22T00:00:01Z",
            "duration_ms": 1000,
            "timeout": False,
            "exit_code": 0,
            "signal": None,
            "ok": True,
            "status": "completed",
            "final_output": {
                "task_id": task.get("instance_id"),
                "status": "completed",
                "final_answer": "done",
                "touched_files": ["requests/api.py"],
                "retrieval_steps": [
                    {
                        "files": ["requests/api.py"],
                        "spans": {"requests/api.py": [{"start": 1, "end": 4}]},
                        "symbols": {"requests/api.py": ["request"]},
                    }
                ],
                "retrieved_context_files": ["requests/api.py"],
                "retrieved_context_spans": {"requests/api.py": [{"start": 1, "end": 4}]},
                "retrieved_context_symbols": {"requests/api.py": ["request"]},
                "notes": "",
            },
            "token_usage": None,
            "tool_calls": [],
            "raw_response_path": None,
            "diff_path": None,
            "model_patch": "",
        }
        suffix = "codex" if agent == "codex" else "claude"
        record_path = task_dir / f"{task_id}.{suffix}-record.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        call_log.append(
            {
                "task_id": task.get("instance_id"),
                "agent": agent,
                "agent_args": list(agent_args),
                "reasoning_effort": reasoning_effort,
                "env": dict(env_overrides or {}),
                "prompt_preamble": prompt_preamble,
                "setup": dict(setup or {}),
                "workspace_key": workspace_key,
            }
        )
        return record

    return _run


def test_build_run_suite_variant_merges_base_and_variant_overrides(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    config = RunSuiteConfig.model_validate(
        {
                "experiment_name": "suite-codex",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "agent_args": ["--base"],
                "env": {"BASE": "1"},
                "reasoning_effort": "medium",
                "setup": {
                    "copy_paths": [
                        {
                            "source": str(tmp_path),
                            "destination": "base",
                            "target_root": "task_dir",
                        }
                    ]
                },
            },
            "variants": [
                {
                    "name": "with-plugin",
                    "reasoning_effort": "high",
                    "agent_args_add": ["--plugin"],
                    "env_add": {"PLUGIN": "1"},
                    "setup": {
                        "prompt_preamble": "Enable plugin",
                        "setup_prompt": "Bootstrap tools first",
                        "setup_prompt_timeout": 90,
                        "files_to_materialize": [
                            {
                                "path": "plugin.json",
                                "content": {"enabled": True},
                                "format": "json",
                                "target_root": "task_dir",
                            }
                        ],
                    },
                }
            ],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    effective = build_run_suite_variant(config, config.variants[0])

    assert effective.agent_args == ["--base", "--plugin"]
    assert effective.env == {"BASE": "1", "PLUGIN": "1"}
    assert effective.reasoning_effort == "high"
    assert effective.setup.prompt_preamble == "Enable plugin"
    assert effective.setup.setup_prompt == "Bootstrap tools first"
    assert effective.setup.setup_prompt_timeout == 90
    assert len(effective.setup.copy_paths) == 1
    assert len(effective.setup.files_to_materialize) == 1


def test_run_suite_config_rejects_claude_only_invalid_target_roots(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    with pytest.raises(ValueError, match="Agent 'claude' only supports setup target_root values"):
        RunSuiteConfig.model_validate(
            {
                "experiment_name": "claude-invalid-root",
                "agent": "claude",
                "base_run": {
                    "task_data": str(task_data),
                    "task_csv": str(task_csv),
                    "output_root": str(tmp_path / "results"),
                    "repo_cache": str(tmp_path / "cache"),
                    "setup": {
                        "files_to_materialize": [
                            {
                                "path": "settings/plugin.json",
                                "content": {"enabled": True},
                                "format": "json",
                                "target_root": "xdg_config_home",
                            }
                        ]
                    },
                },
                "variants": [{"name": "baseline"}],
                "postprocess": {"convert": False, "evaluate": False},
            }
        )


def test_run_suite_config_rejects_unsupported_claude_reasoning_effort(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    with pytest.raises(ValueError, match="Agent 'claude' only supports reasoning_effort values"):
        RunSuiteConfig.model_validate(
            {
                "experiment_name": "claude-invalid-reasoning",
                "agent": "claude",
                "base_run": {
                    "task_data": str(task_data),
                    "task_csv": str(task_csv),
                    "output_root": str(tmp_path / "results"),
                    "repo_cache": str(tmp_path / "cache"),
                    "reasoning_effort": "minimal",
                },
                "variants": [{"name": "baseline"}],
                "postprocess": {"convert": False, "evaluate": False},
            }
        )


def test_run_suite_config_allows_codex_runtime_target_roots(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-valid-root",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
            },
            "variants": [
                {
                    "name": "plugin",
                    "setup": {
                        "files_to_materialize": [
                            {
                                "path": "settings/plugin.json",
                                "content": {"enabled": True},
                                "format": "json",
                                "target_root": "xdg_config_home",
                            }
                        ]
                    },
                }
            ],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    assert config.variants[0].setup.files_to_materialize[0].target_root == "xdg_config_home"


def test_run_suite_config_defaults_schema_path_per_agent(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)

    codex = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-default-schema",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results-codex"),
                "repo_cache": str(tmp_path / "cache-codex"),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )
    claude = RunSuiteConfig.model_validate(
        {
            "experiment_name": "claude-default-schema",
            "agent": "claude",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results-claude"),
                "repo_cache": str(tmp_path / "cache-claude"),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    assert codex.base_run.schema_path == CODEX_OUTPUT_SCHEMA_PATH
    assert claude.base_run.schema_path == CLAUDE_OUTPUT_SCHEMA_PATH


def test_run_suite_config_normalizes_agent_aliases(tmp_path) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "claude-alias-schema",
            "agent": "claude-code",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results-claude"),
                "repo_cache": str(tmp_path / "cache-claude"),
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": False, "evaluate": False},
        }
    )

    assert config.agent == "claude"
    assert config.base_run.schema_path == CLAUDE_OUTPUT_SCHEMA_PATH


def test_dependency_metadata_includes_pydantic_runtime_requirement() -> None:
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")
    requirements_text = Path("requirements.txt").read_text(encoding="utf-8")

    assert '"pydantic>=2,<3"' in pyproject_text
    assert "pydantic>=2,<3" in requirements_text


def test_run_suite_runner_writes_manifest_and_variant_outputs(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=2)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "codex-variants",
            "description": "Compare baseline and plugin setup.",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "reasoning_effort": "medium",
            },
            "variants": [
                {"name": "baseline"},
                {
                    "name": "with-plugin",
                    "reasoning_effort": "xhigh",
                    "agent_args_add": ["--plugin"],
                    "env_add": {"PLUGIN": "1"},
                    "setup": {
                        "prompt_preamble": "Plugin enabled",
                        "setup_prompt": "Bootstrap plugin",
                        "setup_prompt_timeout": 45,
                    },
                },
            ],
            "parallelism": {"max_workers": 2},
            "postprocess": {"convert": True, "evaluate": False},
        }
    )

    rc = RunSuiteRunner(config).run()

    experiment_dir = tmp_path / "results" / "codex-variants"
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    summary_rows = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert manifest["status"] == "completed"
    assert len(manifest["variants"]) == 2
    assert len(call_log) == 4
    assert len(cleanup_calls) == 4
    assert all(Path(row["pred_path"]).exists() for row in summary_rows)
    assert (experiment_dir / "summary.csv").exists()
    assert [call["task_id"] for call in call_log[:2]] == ["psf__requests-1000", "psf__requests-1000"]
    assert [call["task_id"] for call in call_log[2:]] == ["psf__requests-1001", "psf__requests-1001"]
    assert len({call["workspace_key"] for call in call_log[:2]}) == 2
    assert len({call["workspace_key"] for call in call_log[2:]}) == 2

    plugin_calls = [call for call in call_log if call["prompt_preamble"] == "Plugin enabled"]
    assert len(plugin_calls) == 2
    assert all(call["agent_args"] == ["--plugin"] for call in plugin_calls)
    assert all(call["reasoning_effort"] == "xhigh" for call in plugin_calls)
    assert all(call["env"] == {"PLUGIN": "1"} for call in plugin_calls)
    assert all(call["setup"]["setup_prompt"] == "Bootstrap plugin" for call in plugin_calls)
    assert all(call["setup"]["setup_prompt_timeout"] == 45 for call in plugin_calls)

    with open(experiment_dir / "summary.csv", "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["variant"] for row in rows] == ["baseline", "with-plugin"]


def test_run_suite_runner_resume_skips_completed_tasks(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-run",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False},
        }
    )

    first_rc = RunSuiteRunner(config).run()
    second_rc = RunSuiteRunner(config, resume=True).run()

    manifest = json.loads((tmp_path / "results" / "resume-run" / "manifest.json").read_text(encoding="utf-8"))
    variant = manifest["variants"][0]

    assert first_rc == 0
    assert second_rc == 0
    assert len(call_log) == 1
    assert len(cleanup_calls) == 1
    assert variant["task_counts"]["skipped"] == 1


def test_run_suite_runner_resume_allows_limit_increase(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=2)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config_first = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-limit-change",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "limit": 1,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False},
        }
    )
    config_second = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-limit-change",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "limit": 2,
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False},
        }
    )

    first_rc = RunSuiteRunner(config_first).run()
    second_rc = RunSuiteRunner(config_second, resume=True).run()

    manifest = json.loads((tmp_path / "results" / "resume-limit-change" / "manifest.json").read_text(encoding="utf-8"))
    variant = manifest["variants"][0]

    assert first_rc == 0
    assert second_rc == 0
    assert len(call_log) == 2
    assert [call["task_id"] for call in call_log] == ["psf__requests-1000", "psf__requests-1001"]
    assert len(cleanup_calls) == 2
    assert variant["task_counts"]["total"] == 2
    assert variant["task_counts"]["completed"] == 1
    assert variant["task_counts"]["skipped"] == 1


def test_run_suite_runner_resume_rejects_model_change(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config_first = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-model-change",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "model": "gpt-5.4",
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False},
        }
    )
    config_second = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-model-change",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
                "model": "gpt-5.3-codex",
            },
            "variants": [{"name": "baseline"}],
            "postprocess": {"convert": True, "evaluate": False},
        }
    )

    first_rc = RunSuiteRunner(config_first).run()

    assert first_rc == 0
    with pytest.raises(RuntimeError, match="already exists with a different effective config"):
        RunSuiteRunner(config_second, resume=True).run()


def test_run_suite_runner_resume_reruns_full_task_fanout_when_one_variant_is_missing(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    call_log: list[dict[str, object]] = []
    cleanup_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", _fake_run_coding_agent_task(call_log))
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "resume-partial-task",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}, {"name": "plugin"}],
            "parallelism": {"max_workers": 2},
            "postprocess": {"convert": True, "evaluate": False},
        }
    )

    first_rc = RunSuiteRunner(config).run()

    plugin_record = (
        tmp_path
        / "results"
        / "resume-partial-task"
        / "variants"
        / "plugin"
        / "agent_runs"
        / "codex"
        / "Verified"
        / "psf__requests-1000"
        / "psf__requests-1000.codex-record.json"
    )
    plugin_record.unlink()

    second_rc = RunSuiteRunner(config, resume=True).run()
    manifest = json.loads((tmp_path / "results" / "resume-partial-task" / "manifest.json").read_text(encoding="utf-8"))

    assert first_rc == 0
    assert second_rc == 0
    assert len(call_log) == 4
    assert len(cleanup_calls) == 4
    assert all(variant["task_counts"]["completed"] == 1 for variant in manifest["variants"])
    assert all(variant["task_counts"]["skipped"] == 0 for variant in manifest["variants"])


def test_run_suite_runner_cleans_successful_worktrees_but_keeps_failed_runs(tmp_path, monkeypatch) -> None:
    task_data, task_csv = _write_task_inputs(tmp_path, count=1)
    cleanup_calls: list[tuple[str, str, str]] = []

    def fake_run(
        *,
        task,
        agent,
        output_dir,
        cache_dir,
        schema_path,
        timeout,
        model=None,
        reasoning_effort=None,
        agent_args=(),
        env_overrides=None,
        prompt_preamble=None,
        setup=None,
        workspace_key=None,
    ):
        del cache_dir, schema_path, timeout, model, reasoning_effort, agent_args, env_overrides, prompt_preamble, setup
        task_id = safe_path_component(task.get("instance_id") or task.get("original_inst_id") or "task")
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        workspace_path = task_dir / "workspaces" / safe_path_component(workspace_key or task_id)
        workspace_path.mkdir(parents=True, exist_ok=True)
        suffix = "codex" if agent == "codex" else "claude"
        status = "completed" if "baseline" in str(workspace_key) else "failed"
        timeout_flag = False if status == "completed" else True
        record = {
            "agent": agent,
            "bench": task.get("bench"),
            "instance_id": task.get("instance_id"),
            "original_inst_id": task.get("original_inst_id"),
            "repo_url": task.get("repo_url"),
            "commit": task.get("commit") or task.get("base_commit"),
            "task_dir": str(task_dir),
            "workspace_path": str(workspace_path),
            "prompt_path": str(task_dir / "prompt.txt"),
            "started_at": "2026-03-22T00:00:00Z",
            "completed_at": "2026-03-22T00:00:01Z",
            "duration_ms": 1000,
            "timeout": timeout_flag,
            "exit_code": 0 if status == "completed" else None,
            "signal": None,
            "ok": status == "completed",
            "status": status,
            "final_output": {
                "task_id": task.get("instance_id"),
                "status": status,
                "final_answer": "done",
                "touched_files": ["requests/api.py"],
                "retrieval_steps": [],
                "retrieved_context_files": ["requests/api.py"],
                "retrieved_context_spans": {},
                "retrieved_context_symbols": {},
                "notes": "",
            }
            if status == "completed"
            else None,
            "token_usage": None,
            "tool_calls": [],
            "raw_response_path": None,
            "diff_path": None,
            "model_patch": "",
        }
        record_path = task_dir / f"{task_id}.{suffix}-record.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        return record

    monkeypatch.setattr("contextbench.run_suites_core.runner.run_coding_agent_task", fake_run)
    monkeypatch.setattr(
        "contextbench.run_suites_core.runner.remove_worktree",
        lambda repo_url, cache_dir, worktree_dir: cleanup_calls.append((repo_url, cache_dir, worktree_dir)),
    )

    config = RunSuiteConfig.model_validate(
        {
            "experiment_name": "cleanup-run",
            "agent": "codex",
            "base_run": {
                "task_data": str(task_data),
                "task_csv": str(task_csv),
                "output_root": str(tmp_path / "results"),
                "repo_cache": str(tmp_path / "cache"),
                "timeout": 30,
            },
            "variants": [{"name": "baseline"}, {"name": "plugin"}],
            "parallelism": {"max_workers": 2},
            "postprocess": {"convert": True, "evaluate": False},
        }
    )

    rc = RunSuiteRunner(config).run()
    manifest = json.loads((tmp_path / "results" / "cleanup-run" / "manifest.json").read_text(encoding="utf-8"))

    assert rc == 1
    assert len(cleanup_calls) == 1
    assert "baseline" in cleanup_calls[0][2]
    assert any(variant["status"] == "completed_with_failures" for variant in manifest["variants"])
