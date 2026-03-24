# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from contextbench.agents import extract_trajectory
from contextbench.coding_agents import (
    convert_run_record,
    load_predictions_from_path,
    parse_unified_diff,
    record_is_convertible,
)


def test_parse_unified_diff_extracts_new_file_spans() -> None:
    diff = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -10,2 +10,4 @@
 old
+new
"""

    assert parse_unified_diff(diff) == {"foo.py": [{"start": 10, "end": 13}]}


def test_convert_run_record_uses_reported_retrieval(make_final_output, make_record) -> None:
    record = make_record(
        agent="codex",
        instance_id="psf__requests-1142",
        repo_url="https://github.com/psf/requests.git",
        final_output=make_final_output(
            task_id="psf__requests-1142",
            touched_files=["requests/models.py"],
            retrieval_steps=[
                {
                    "files": ["requests/models.py"],
                    "spans": [{"file": "requests/models.py", "start": 1, "end": 20}],
                    "symbols": [{"file": "requests/models.py", "name": "Response"}],
                }
            ],
            retrieved_context_files=["requests/models.py"],
            retrieved_context_spans=[{"file": "requests/models.py", "start": 1, "end": 20}],
            retrieved_context_symbols=[{"file": "requests/models.py", "name": "Response"}],
        ),
    )

    converted = convert_run_record(record)

    assert converted["instance_id"] == "psf__requests-1142"
    assert converted["traj_data"]["pred_files"] == ["requests/models.py"]
    assert converted["traj_data"]["pred_spans"] == {"requests/models.py": [{"start": 1, "end": 20}]}
    assert converted["traj_data"]["pred_symbols"] == {"requests/models.py": ["Response"]}


def test_convert_run_record_falls_back_to_diff_and_touched_files(make_final_output, make_record) -> None:
    record = make_record(
        agent="claude",
        instance_id="psf__requests-1142",
        repo_url="https://github.com/psf/requests.git",
        model_patch="""diff --git a/requests/api.py b/requests/api.py
--- a/requests/api.py
+++ b/requests/api.py
@@ -5,0 +5,3 @@
+x
""",
        final_output=make_final_output(
            task_id="psf__requests-1142",
            touched_files=["requests/api.py"],
        ),
    )

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files"] == ["requests/api.py"]
    assert converted["traj_data"]["pred_spans"] == {"requests/api.py": [{"start": 5, "end": 7}]}


def test_convert_run_record_accepts_minimal_final_output() -> None:
    record = {
        "agent": "codex",
        "instance_id": "psf__requests-1142",
        "repo_url": "https://github.com/psf/requests.git",
        "workspace_path": "/tmp/workspace",
        "model_patch": """diff --git a/requests/api.py b/requests/api.py
--- a/requests/api.py
+++ b/requests/api.py
@@ -5,0 +5,3 @@
+x
""",
        "final_output": {
            "task_id": "psf__requests-1142",
            "status": "completed",
            "final_answer": "done",
            "retrieved_context_files": ["requests/models.py"],
            "retrieved_context_spans": [{"file": "requests/models.py", "start": 1, "end": 20}],
            "retrieved_context_symbols": [],
            "notes": "",
        },
    }

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files"] == ["requests/models.py"]
    assert converted["traj_data"]["pred_spans"]["requests/api.py"][0]["start"] == 5
    assert converted["traj_data"]["pred_spans"]["requests/models.py"][0]["start"] == 1


def test_convert_run_record_merges_inferred_and_reported_retrieval() -> None:
    record = {
        "agent": "codex",
        "instance_id": "task-1",
        "workspace_path": "/tmp/workspace",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "raw_response": {
            "agent": "codex",
            "response_format": "jsonl-events",
            "events": [
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_1",
                        "type": "command_execution",
                        "command": "/bin/zsh -lc 'rg -n \"fill_value\" sklearn/impute/_iterative.py'",
                        "aggregated_output": "sklearn/impute/_iterative.py:120:    fill_value : str or numerical value, default=None\n",
                        "exit_code": 0,
                        "status": "completed",
                    },
                }
            ],
        },
        "model_patch": "",
        "final_output": {
            "task_id": "task-1",
            "status": "completed",
            "final_answer": "done",
            "touched_files": [],
            "retrieval_steps": [
                {
                    "files": ["sklearn/impute/tests/test_impute.py"],
                    "spans": [{"file": "sklearn/impute/tests/test_impute.py", "start": 10, "end": 20}],
                    "symbols": [],
                }
            ],
            "retrieved_context_files": ["sklearn/impute/tests/test_impute.py"],
            "retrieved_context_spans": [{"file": "sklearn/impute/tests/test_impute.py", "start": 10, "end": 20}],
            "retrieved_context_symbols": [],
            "notes": "",
        },
    }

    converted = convert_run_record(record)

    assert converted["traj_data"]["pred_files"] == [
        "sklearn/impute/_iterative.py",
        "sklearn/impute/tests/test_impute.py",
    ]
    assert converted["traj_data"]["pred_spans"]["sklearn/impute/_iterative.py"][0]["start"] == 120
    assert converted["traj_data"]["pred_spans"]["sklearn/impute/tests/test_impute.py"][0]["start"] == 10
    assert len(converted["traj_data"]["pred_steps"]) == 2


def test_load_predictions_from_directory_filters_agent(tmp_path, make_final_output, make_record) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(
            [
                json.dumps(
                    make_record(
                        agent="codex",
                        instance_id="task-codex",
                        final_output=make_final_output(
                            task_id="task-codex",
                            touched_files=["a.py"],
                            retrieved_context_files=["a.py"],
                        ),
                    )
                ),
                json.dumps(
                    make_record(
                        agent="claude",
                        instance_id="task-claude",
                        final_output=make_final_output(
                            task_id="task-claude",
                            touched_files=["b.py"],
                            retrieved_context_files=["b.py"],
                        ),
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    nested = tmp_path / "codex" / "Verified" / "task-codex"
    nested.mkdir(parents=True)
    (nested / "task-codex.codex-record.json").write_text(
        json.dumps(
            make_record(
                agent="codex",
                instance_id="task-codex",
                final_output=make_final_output(
                    task_id="task-codex",
                    touched_files=["a.py"],
                    retrieved_context_files=["a.py"],
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = load_predictions_from_path(tmp_path, expected_agent="codex")

    assert len(predictions) == 1
    assert predictions[0]["instance_id"] == "task-codex"


def test_load_predictions_from_json_list_supports_claude_code_alias(tmp_path, make_final_output, make_record) -> None:
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps(
            [
                make_record(
                    agent="codex",
                    instance_id="task-codex",
                    final_output=make_final_output(task_id="task-codex", retrieved_context_files=["a.py"]),
                ),
                make_record(
                    agent="claude",
                    instance_id="task-claude",
                    final_output=make_final_output(task_id="task-claude", retrieved_context_files=["b.py"]),
                ),
            ]
        ),
        encoding="utf-8",
    )

    predictions = load_predictions_from_path(records_path, expected_agent="claude-code")

    assert len(predictions) == 1
    assert predictions[0]["instance_id"] == "task-claude"


def test_record_is_convertible_accepts_claude_code_alias(make_final_output, make_record) -> None:
    record = make_record(
        agent="claude",
        instance_id="task-claude",
        final_output=make_final_output(task_id="task-claude"),
    )

    assert record_is_convertible(record, expected_agent="claude-code") is True


def test_unified_extractor_dispatch_supports_codex_record_file(tmp_path, make_final_output, make_record) -> None:
    record_path = tmp_path / "task.codex-record.json"
    record_path.write_text(
        json.dumps(
            make_record(
                agent="codex",
                final_output=make_final_output(
                    touched_files=["a.py"],
                    retrieved_context_files=["a.py"],
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    extracted = extract_trajectory(str(record_path))

    assert extracted["pred_files"] == ["a.py"]


def test_unified_extractor_dispatch_supports_claude_record_file(tmp_path, make_final_output, make_record) -> None:
    record_path = tmp_path / "task.claude-record.json"
    record_path.write_text(
        json.dumps(
            make_record(
                agent="claude",
                final_output=make_final_output(
                    touched_files=["b.py"],
                    retrieved_context_files=["b.py"],
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    extracted = extract_trajectory(str(record_path))

    assert extracted["pred_files"] == ["b.py"]


def test_load_predictions_from_missing_path_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_predictions_from_path(tmp_path / "missing-records.json")
