# SPDX-License-Identifier: Apache-2.0

"""Conversion helpers from coding-agent records to ContextBench trajectories."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .files import read_json, read_jsonl
from .records import merge_span_maps, normalize_retrieval_steps, normalize_span_map, normalize_symbol_map, parse_unified_diff
from .types import SymbolMap, TrajectoryData


def record_is_convertible(record: dict[str, object], expected_agent: str | None = None) -> bool:
    if not isinstance(record, dict):
        return False
    agent = str(record.get("agent") or "").strip().lower()
    if expected_agent:
        normalized_expected = "claude" if expected_agent == "claude-code" else expected_agent
        if agent and agent != normalized_expected:
            return False
    final_output = record.get("final_output")
    return isinstance(final_output, dict)


def convert_run_record(record: dict[str, object]) -> dict[str, object]:
    final_output = record.get("final_output") or {}
    retrieval_steps = normalize_retrieval_steps(final_output.get("retrieval_steps"))
    retrieved_context_files = sorted(
        {
            str(item).strip()
            for item in (final_output.get("retrieved_context_files") or [])
            if str(item).strip()
        }
    )
    retrieved_context_spans = normalize_span_map(final_output.get("retrieved_context_spans"))
    retrieved_context_symbols = normalize_symbol_map(final_output.get("retrieved_context_symbols"))

    model_patch = str(record.get("model_patch") or "").strip()
    diff_path = record.get("diff_path")
    if not model_patch and diff_path and Path(diff_path).exists():
        model_patch = Path(diff_path).read_text(encoding="utf-8")

    diff_spans = parse_unified_diff(model_patch)
    merged_step_spans = merge_span_maps(*(step.get("spans") for step in retrieval_steps))
    merged_step_symbols: SymbolMap = {}
    for step in retrieval_steps:
        for file_path, names in normalize_symbol_map(step.get("symbols")).items():
            merged_step_symbols.setdefault(file_path, []).extend(names)
    merged_step_symbols = {
        file_path: sorted(set(names))
        for file_path, names in merged_step_symbols.items()
        if names
    }

    touched_files = sorted(
        {
            str(item).strip()
            for item in (final_output.get("touched_files") or [])
            if str(item).strip()
        }
    )
    pred_files = (
        retrieved_context_files
        or sorted({file for step in retrieval_steps for file in step.get("files", [])})
        or touched_files
    )
    pred_spans = retrieved_context_spans or merged_step_spans or diff_spans
    pred_symbols = retrieved_context_symbols or merged_step_symbols
    pred_steps = (
        retrieval_steps
        or (
            [{"files": pred_files, "spans": pred_spans, "symbols": pred_symbols}]
            if pred_files or pred_spans or pred_symbols
            else []
        )
    )

    traj_data: TrajectoryData = {
        "pred_steps": pred_steps,
        "pred_files": pred_files,
        "pred_spans": pred_spans,
        "pred_symbols": pred_symbols,
    }

    return {
        "instance_id": record.get("instance_id") or record.get("original_inst_id") or "",
        "original_inst_id": record.get("original_inst_id") or None,
        "repo_url": record.get("repo_url") or None,
        "commit": record.get("commit") or None,
        "model_patch": model_patch,
        "traj_data": traj_data,
    }


def convert_records(records: Iterable[dict[str, object]], expected_agent: str | None = None) -> list[dict[str, object]]:
    return [
        convert_run_record(record)
        for record in records
        if record_is_convertible(record, expected_agent=expected_agent)
    ]


def load_predictions_from_path(path: str | Path, expected_agent: str | None = None) -> list[dict[str, object]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Path not found: {source}")

    if source.is_dir():
        aggregate = source / "records.jsonl"
        if aggregate.exists():
            return convert_records(read_jsonl(aggregate), expected_agent=expected_agent)
        records: list[dict[str, object]] = []
        suffixes = ("*.codex-record.json", "*.claude-record.json")
        for pattern in suffixes:
            for record_path in sorted(source.rglob(pattern)):
                records.append(read_json(record_path))
        return convert_records(records, expected_agent=expected_agent)

    if source.suffix == ".jsonl":
        return convert_records(read_jsonl(source), expected_agent=expected_agent)

    loaded = read_json(source)
    if isinstance(loaded, dict):
        loaded_rows = [loaded]
    elif isinstance(loaded, list):
        loaded_rows = list(loaded)
    else:
        raise ValueError(f"Unsupported record payload in {source}")
    return convert_records(loaded_rows, expected_agent=expected_agent)
