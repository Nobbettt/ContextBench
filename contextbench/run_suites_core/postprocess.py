# SPDX-License-Identifier: Apache-2.0

"""Conversion and evaluation helpers for run suites."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..coding_agents.conversion import load_predictions_from_path
from ..coding_agents.files import ensure_dir
from ..evaluate import GoldLoader, aggregate_results, evaluate_instance
from ..extractors import available as treesitter_available
from ..parsers import load_pred


def convert_records_to_jsonl(*, source_dir: Path, expected_agent: str, out_path: Path) -> int:
    predictions = load_predictions_from_path(source_dir, expected_agent=expected_agent) if source_dir.exists() else []
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return len(predictions)


def evaluate_prediction_file(
    *,
    gold_path: Path,
    pred_path: Path,
    cache_dir: Path,
    out_path: Path,
) -> dict[str, object]:
    if not treesitter_available():
        raise RuntimeError("Tree-sitter is not available for evaluation")

    gold_loader = GoldLoader(str(gold_path))
    pred_rows = load_pred(str(pred_path))
    results: list[dict[str, object]] = []
    for pred_data in pred_rows:
        instance_id = pred_data.get("instance_id") or pred_data.get("original_inst_id")
        if not instance_id:
            continue
        gold_ctx = gold_loader.get(instance_id)
        if not gold_ctx:
            results.append({"instance_id": instance_id, "error": "missing_gold"})
            continue
        results.append(evaluate_instance(instance_id, gold_ctx, pred_data, str(cache_dir)))

    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    error_counts = dict(Counter(str(row.get("error")) for row in results if row.get("error")))
    summary = aggregate_results(results)
    summary["error_counts"] = error_counts
    return summary
