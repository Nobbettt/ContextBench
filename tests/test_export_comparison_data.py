# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_comparison_data import ComparisonExportError, build_comparison_payload


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _record(task_dir: Path, duration_ms: int, total_tokens: int, tool_calls: int) -> str:
    record_path = task_dir / f"{task_dir.name}.codex-record.json"
    _write(
        record_path,
        json.dumps(
            {
                "duration_ms": duration_ms,
                "token_usage": {"total_tokens": total_tokens},
                "tool_calls": [{} for _ in range(tool_calls)],
            }
        ),
    )
    return str(record_path)


def test_build_comparison_payload_happy_path(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    _write(
        suite_dir / "experiment.json",
        json.dumps(
            {
                "experiment_name": "demo-suite",
                "description": "A/B comparison",
                "agent": "codex",
                "base_run": {"reasoning_effort": "high"},
            }
        ),
    )
    _write(
        suite_dir / "summary.json",
        json.dumps(
            [
                {"variant": "baseline", "total_tasks": 10, "completed_tasks": 6},
                {"variant": "treatment", "total_tasks": 10, "completed_tasks": 8},
            ]
        ),
    )

    baseline_dir = suite_dir / "variants" / "baseline"
    treatment_dir = suite_dir / "variants" / "treatment"
    _write(
        baseline_dir / "effective-config.json",
        json.dumps(
            {
                "effective_config": {
                    "name": "baseline",
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                    "timeout": 2400,
                    "setup": {"copy_paths": []},
                }
            }
        ),
    )
    _write(
        treatment_dir / "effective-config.json",
        json.dumps(
            {
                "effective_config": {
                    "name": "with-superpowers-mounted",
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                    "timeout": 2400,
                    "setup": {"copy_paths": [{"source": "agent-resources/superpowers"}]},
                }
            }
        ),
    )

    baseline_task_dir = baseline_dir / "agent_runs" / "codex" / "Verified" / "task-a"
    treatment_task_dir = treatment_dir / "agent_runs" / "codex" / "Verified" / "task-a"
    baseline_record = _record(baseline_task_dir, 1000, 1200, 2)
    treatment_record = _record(treatment_task_dir, 2000, 1500, 3)

    _write(
        baseline_dir / "task-results.jsonl",
        "\n".join(
            [
                json.dumps({"status": "completed", "record_path": baseline_record}),
                json.dumps({"status": "partial", "record_path": baseline_record}),
            ]
        ),
    )
    _write(
        treatment_dir / "task-results.jsonl",
        "\n".join(
            [
                json.dumps({"status": "completed", "record_path": treatment_record}),
                json.dumps({"status": "completed", "record_path": treatment_record}),
            ]
        ),
    )

    eval_row = json.dumps(
        {
            "final": {
                "file": {"intersection": 3, "gold_size": 4, "pred_size": 4},
                "symbol": {"intersection": 1, "gold_size": 2, "pred_size": 2},
                "span": {"intersection": 60, "gold_size": 100, "pred_size": 80},
                "line": {"intersection": 3, "gold_size": 4, "pred_size": 6},
            },
            "editloc": {"intersection": 2, "gold_size": 4, "pred_size": 2},
            "trajectory": {
                "auc_coverage": {"file": 0.8, "symbol": 0.5, "span": 0.6},
                "redundancy": {"file": 0.2, "symbol": 0.1, "span": 0.3},
            },
        }
    )
    _write(baseline_dir / "eval.jsonl", eval_row)
    _write(treatment_dir / "eval.jsonl", eval_row)

    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "task_set": {"count": 10},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(baseline_dir / "effective-config.json"),
                        "task_results_path": str(baseline_dir / "task-results.jsonl"),
                        "output_dir": str(baseline_dir),
                    },
                    {
                        "name": "treatment",
                        "effective_config_path": str(treatment_dir / "effective-config.json"),
                        "task_results_path": str(treatment_dir / "task-results.jsonl"),
                        "output_dir": str(treatment_dir),
                    },
                ],
            }
        ),
    )

    payload = build_comparison_payload(suite_dir)

    assert payload["filterOrder"] == ["all", "codex"]
    assert payload["comparisonCards"][0]["title"] == "Baseline vs With Superpowers Mounted"
    assert payload["comparisonCards"][0]["variants"][1]["parameters"][3]["value"] == "Superpowers snapshot"
    assert payload["comparisonCards"][0]["variants"][0]["results"]["quality"]["spanF1"] == "0.667"
    assert payload["comparisonCards"][0]["variants"][0]["results"]["quality"]["avgLineF1"] == "0.600"
    assert payload["comparisonCards"][0]["variants"][0]["results"]["efficiency"]["efficiency"] == "0.633"
    assert payload["leaderboardRows"][0]["model"] == "Baseline"
    assert payload["leaderboardRows"][0]["passAt1"] == "10.0%"
    assert payload["leaderboardRows"][0]["contextF1"] == "0.639"


def test_build_comparison_payload_fails_when_eval_is_missing(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    _write(suite_dir / "experiment.json", json.dumps({"experiment_name": "demo-suite", "agent": "codex"}))
    _write(
        suite_dir / "summary.json",
        json.dumps(
            [
                {"variant": "baseline", "total_tasks": 1, "completed_tasks": 1},
                {"variant": "treatment", "total_tasks": 1, "completed_tasks": 1},
            ]
        ),
    )

    baseline_dir = suite_dir / "variants" / "baseline"
    treatment_dir = suite_dir / "variants" / "treatment"
    _write(baseline_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "setup": {}}}))
    _write(treatment_dir / "effective-config.json", json.dumps({"effective_config": {"name": "treatment", "setup": {}}}))
    _write(baseline_dir / "task-results.jsonl", json.dumps({"status": "completed"}))
    _write(treatment_dir / "task-results.jsonl", json.dumps({"status": "completed"}))
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "task_set": {"count": 1},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(baseline_dir / "effective-config.json"),
                        "task_results_path": str(baseline_dir / "task-results.jsonl"),
                        "output_dir": str(baseline_dir),
                    },
                    {
                        "name": "treatment",
                        "effective_config_path": str(treatment_dir / "effective-config.json"),
                        "task_results_path": str(treatment_dir / "task-results.jsonl"),
                        "output_dir": str(treatment_dir),
                    },
                ],
            }
        ),
    )

    with pytest.raises(ComparisonExportError, match="Missing eval.jsonl"):
        build_comparison_payload(suite_dir)


def test_build_comparison_payload_single_variant_mode(tmp_path: Path) -> None:
    suite_dir = tmp_path / "results" / "run_suites" / "demo-suite"
    _write(
        suite_dir / "experiment.json",
        json.dumps(
            {
                "experiment_name": "demo-suite",
                "description": "Single variant export",
                "agent": "codex",
                "base_run": {"reasoning_effort": "high"},
            }
        ),
    )
    _write(
        suite_dir / "summary.json",
        json.dumps(
            [
                {"variant": "baseline", "total_tasks": 10, "completed_tasks": 6},
                {"variant": "with-superpowers-mounted", "total_tasks": 10, "completed_tasks": 8},
            ]
        ),
    )

    baseline_dir = suite_dir / "variants" / "baseline"
    treatment_dir = suite_dir / "variants" / "with-superpowers-mounted"
    _write(baseline_dir / "effective-config.json", json.dumps({"effective_config": {"name": "baseline", "setup": {}}}))
    _write(treatment_dir / "effective-config.json", json.dumps({"effective_config": {"name": "with-superpowers-mounted", "model": "gpt-5.4", "reasoning_effort": "high", "timeout": 2400, "setup": {"copy_paths": []}}}))
    treatment_task_dir = treatment_dir / "agent_runs" / "codex" / "Verified" / "task-a"
    treatment_record = _record(treatment_task_dir, 2000, 1500, 3)
    _write(treatment_dir / "task-results.jsonl", json.dumps({"status": "completed", "record_path": treatment_record}))
    _write(
        treatment_dir / "eval.jsonl",
        json.dumps(
            {
                "final": {
                    "file": {"intersection": 3, "gold_size": 4, "pred_size": 4},
                    "span": {"intersection": 60, "gold_size": 100, "pred_size": 80},
                },
                "editloc": {"intersection": 2, "gold_size": 4, "pred_size": 2},
            }
        ),
    )
    _write(
        suite_dir / "manifest.json",
        json.dumps(
            {
                "task_set": {"count": 10},
                "variants": [
                    {
                        "name": "baseline",
                        "effective_config_path": str(baseline_dir / "effective-config.json"),
                        "task_results_path": str(baseline_dir / "task-results.jsonl"),
                        "output_dir": str(baseline_dir),
                    },
                    {
                        "name": "with-superpowers-mounted",
                        "effective_config_path": str(treatment_dir / "effective-config.json"),
                        "task_results_path": str(treatment_dir / "task-results.jsonl"),
                        "output_dir": str(treatment_dir),
                    },
                ],
            }
        ),
    )

    payload = build_comparison_payload(suite_dir, variant_name="with-superpowers-mounted")

    assert payload["comparisonCards"][0]["title"] == "With Superpowers Mounted"
    assert len(payload["comparisonCards"][0]["variants"]) == 1
    assert payload["leaderboardRows"][0]["model"] == "With Superpowers Mounted"
