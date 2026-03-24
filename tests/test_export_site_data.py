# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from scripts.export_site_data import build_dashboard


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_build_dashboard_aggregates_results(tmp_path: Path) -> None:
    results_root = tmp_path / "results"

    _write(
        results_root / "codex.metrics.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "alpha__one",
                        "final": {
                            "file": {"coverage": 0.5, "precision": 0.25},
                            "symbol": {"coverage": 0.1},
                            "span": {"coverage": 0.4},
                            "line": {"coverage": 0.3},
                        },
                        "editloc": {"recall": 0.0},
                    }
                ),
                json.dumps(
                    {
                        "instance_id": "beta__two",
                        "final": {
                            "file": {"coverage": 1.0, "precision": 0.75},
                            "symbol": {"coverage": 0.3},
                            "span": {"coverage": 0.6},
                            "line": {"coverage": 0.7},
                        },
                        "editloc": {"recall": 1.0},
                    }
                ),
            ]
        ),
    )

    _write(
        results_root / "run_suites" / "demo-suite" / "summary.json",
        json.dumps(
            [
                {
                    "variant": "baseline",
                    "status": "completed",
                    "total_tasks": 2,
                    "completed_tasks": 2,
                    "failed_tasks": 0,
                    "timeout_tasks": 0,
                    "prediction_count": 2,
                }
            ]
        ),
    )
    _write(
        results_root / "run_suites" / "demo-suite" / "experiment.json",
        json.dumps(
            {
                "experiment_name": "demo-suite",
                "agent": "codex",
                "description": "Demo comparison",
                "variants": [{"name": "baseline", "description": "No bootstrap prompt"}],
            }
        ),
    )

    _write(
        results_root / "agent_runs" / "codex" / "Verified" / "task-a" / "task-a.codex-record.json",
        json.dumps(
            {
                "agent": "codex",
                "bench": "Verified",
                "instance_id": "task-a",
                "repo": "octo/repo",
                "status": "partial",
                "ok": True,
                "duration_ms": 9000,
                "completed_at": "2026-03-21T13:00:00Z",
                "final_output": {
                    "final_answer": "Implemented the fix and verified with local checks.",
                    "touched_files": ["a.py", "b.py"],
                    "retrieval_steps": [{}, {}],
                },
            }
        ),
    )
    _write(
        results_root / "agent_runs" / "claude" / "Pro" / "task-b" / "task-b.claude-record.json",
        json.dumps(
            {
                "agent": "claude",
                "bench": "Pro",
                "instance_id": "task-b",
                "repo": "octo/other",
                "status": "failed",
                "ok": False,
                "duration_ms": 3000,
                "completed_at": "2026-03-20T09:00:00Z",
                "final_output": {"final_answer": "The task failed before producing a patch."},
            }
        ),
    )

    payload = build_dashboard(results_root)

    assert payload["highlights"] == {
        "agentCount": 2,
        "metricsFileCount": 1,
        "runSuiteCount": 1,
        "taskRecordCount": 2,
        "successfulTaskRecordCount": 1,
    }
    assert payload["leaderboard"][0]["label"] == "codex"
    assert payload["leaderboard"][0]["averages"]["fileCoverage"] == 0.75
    assert payload["coverageSeries"][0]["points"][0]["instanceShort"] == "alpha__one"
    assert payload["suiteComparisons"][0]["variants"][0]["description"] == "No bootstrap prompt"
    assert payload["recentRuns"][0]["agent"] == "codex"
    assert payload["recentRuns"][0]["touchedFiles"] == 2
    assert payload["agentSummaries"][0]["agent"] == "claude"
