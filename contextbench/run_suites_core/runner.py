# SPDX-License-Identifier: Apache-2.0

"""Run suite orchestration."""

from __future__ import annotations

import csv
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..coding_agents.files import append_jsonl, ensure_dir, read_json, safe_path_component, write_json
from ..coding_agents.runtime import run_coding_agent_task
from ..coding_agents.task_data import load_tasks
from ..core.repo import remove_worktree
from .config import build_run_suite_variant
from .helpers import (
    flatten_metrics,
    record_is_resume_complete,
    stable_json_hash,
    task_key,
    task_record_path,
    utc_now,
)
from .postprocess import convert_records_to_jsonl, evaluate_prediction_file
from .types import EffectiveVariantConfig, RunSuiteConfig


@dataclass
class _PreparedVariant:
    variant: EffectiveVariantConfig
    entry: dict[str, object]
    raw_root: Path
    task_results_path: Path
    pred_path: Path
    eval_results_path: Path
    eval_summary_path: Path
    started_monotonic: float


class RunSuiteRunner:
    def __init__(
        self,
        config: RunSuiteConfig,
        *,
        max_workers: int | None = None,
        resume: bool = False,
        skip_convert: bool = False,
        skip_evaluate: bool = False,
    ) -> None:
        self.config = config
        self.resume = resume
        self.skip_convert = skip_convert
        self.skip_evaluate = skip_evaluate
        worker_cap = max_workers if max_workers is not None else config.parallelism.max_workers
        self.max_workers = max(1, int(worker_cap))
        self.experiment_dir = config.base_run.output_root / safe_path_component(config.experiment_name)
        self.manifest_path = self.experiment_dir / "manifest.json"
        self.summary_json_path = self.experiment_dir / "summary.json"
        self.summary_csv_path = self.experiment_dir / "summary.csv"
        self.experiment_config_path = self.experiment_dir / "experiment.json"
        self._run_invocation_key = safe_path_component(f"{time.time_ns()}")

    @staticmethod
    def _resume_compatible_effective_config(
        previous_config: object,
        current_config: dict[str, object],
    ) -> bool:
        if not isinstance(previous_config, dict):
            return False
        previous = dict(previous_config)
        current = dict(current_config)
        previous.pop("limit", None)
        current.pop("limit", None)
        return previous == current

    def _load_tasks(self) -> tuple[list[dict[str, object]], dict[str, object]]:
        base = self.config.base_run
        subset_csv = base.subset_csv or base.task_csv
        tasks = load_tasks(
            base.task_data,
            subset_csv=subset_csv,
            bench_filter=base.bench,
            instance_filter=base.instances,
            limit=base.limit,
        )
        if not tasks:
            raise RuntimeError("No tasks matched the configured task selection")

        task_index = [
            {
                "bench": task.get("bench"),
                "instance_id": task.get("instance_id"),
                "original_inst_id": task.get("original_inst_id"),
                "repo_url": task.get("repo_url"),
                "commit": task.get("commit"),
            }
            for task in tasks
        ]
        bench_counts = Counter(str(task.get("bench") or "Unknown") for task in tasks)
        task_set = {
            "count": len(tasks),
            "hash": stable_json_hash(task_index),
            "bench_counts": dict(sorted(bench_counts.items())),
            "task_ids": [task_key(task) for task in tasks],
        }
        return tasks, task_set

    def _initial_variant_entry(self, variant: EffectiveVariantConfig) -> dict[str, object]:
        variant_dir = self.experiment_dir / "variants" / variant.slug
        return {
            "name": variant.name,
            "slug": variant.slug,
            "description": variant.description,
            "labels": list(variant.labels),
            "notes": variant.notes,
            "status": "pending",
            "config_hash": stable_json_hash(variant.model_dump(mode="json")),
            "effective_config_path": str(variant_dir / "effective-config.json"),
            "output_dir": str(variant_dir),
            "raw_runs_dir": str(variant_dir / "agent_runs"),
            "task_results_path": str(variant_dir / "task-results.jsonl"),
            "pred_path": None,
            "eval_results_path": None,
            "eval_summary_path": None,
            "started_at": None,
            "completed_at": None,
            "duration_ms": None,
            "task_counts": {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "timeout": 0,
                "skipped": 0,
            },
            "metrics": {},
            "errors": [],
        }

    def _write_manifest(
        self,
        *,
        started_at: str,
        completed_at: str | None,
        task_set: dict[str, object],
        variant_entries: list[dict[str, object]],
    ) -> None:
        statuses = {entry["status"] for entry in variant_entries}
        if completed_at is None:
            manifest_status = "running"
        elif statuses <= {"completed"}:
            manifest_status = "completed"
        elif "failed" in statuses or "postprocess_failed" in statuses:
            manifest_status = "failed"
        else:
            manifest_status = "completed_with_failures"
        manifest = {
            "experiment_name": self.config.experiment_name,
            "description": self.config.description,
            "agent": self.config.agent,
            "status": manifest_status,
            "started_at": started_at,
            "completed_at": completed_at,
            "max_workers": self.max_workers,
            "resume": self.resume,
            "task_set": task_set,
            "variants": variant_entries,
        }
        write_json(self.manifest_path, manifest)

    def _write_summary(self, variant_entries: list[dict[str, object]]) -> None:
        rows: list[dict[str, object]] = []
        for entry in variant_entries:
            row = {
                "variant": entry["name"],
                "status": entry["status"],
                "total_tasks": entry["task_counts"]["total"],
                "completed_tasks": entry["task_counts"]["completed"],
                "failed_tasks": entry["task_counts"]["failed"],
                "timeout_tasks": entry["task_counts"]["timeout"],
                "skipped_tasks": entry["task_counts"]["skipped"],
                "pred_path": entry.get("pred_path") or "",
                "eval_results_path": entry.get("eval_results_path") or "",
                "eval_summary_path": entry.get("eval_summary_path") or "",
            }
            row.update(flatten_metrics(entry.get("metrics") or {}))
            rows.append(row)

        write_json(self.summary_json_path, rows)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        ensure_dir(self.summary_csv_path.parent)
        with open(self.summary_csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _prepare_variant_dir(self, variant: EffectiveVariantConfig) -> tuple[Path, dict[str, object]]:
        variant_dir = self.experiment_dir / "variants" / variant.slug
        effective_config = variant.model_dump(mode="json")
        config_payload = {
            "config_hash": stable_json_hash(effective_config),
            "effective_config": effective_config,
        }
        if self.config.base_run.rerun and variant_dir.exists():
            shutil.rmtree(variant_dir)
        elif variant_dir.exists():
            effective_config_path = variant_dir / "effective-config.json"
            if effective_config_path.exists():
                previous = read_json(effective_config_path)
                previous_hash = previous.get("config_hash") if isinstance(previous, dict) else None
                previous_effective_config = (
                    previous.get("effective_config")
                    if isinstance(previous, dict)
                    else None
                )
                if previous_hash != config_payload["config_hash"]:
                    if not (
                        self.resume
                        and self._resume_compatible_effective_config(previous_effective_config, effective_config)
                    ):
                        raise RuntimeError(
                            f"Variant '{variant.name}' already exists with a different effective config. "
                            "Use a new experiment name or enable rerun."
                        )
            if not self.resume:
                raise RuntimeError(
                    f"Variant '{variant.name}' already exists. Re-run with --resume or set base_run.rerun=true."
                )
        ensure_dir(variant_dir)
        write_json(variant_dir / "effective-config.json", config_payload)
        task_results_path = variant_dir / "task-results.jsonl"
        if task_results_path.exists():
            task_results_path.unlink()
        return variant_dir, config_payload

    def _prepare_variant_state(
        self,
        variant: EffectiveVariantConfig,
        entry: dict[str, object],
        *,
        total_tasks: int,
    ) -> _PreparedVariant:
        variant_dir, config_payload = self._prepare_variant_dir(variant)
        raw_root = variant_dir / "agent_runs"
        task_results_path = variant_dir / "task-results.jsonl"
        pred_path = variant_dir / "pred.jsonl"
        eval_results_path = variant_dir / "eval.jsonl"
        eval_summary_path = variant_dir / "eval-summary.json"
        for path in (pred_path, eval_results_path, eval_summary_path):
            if path.exists():
                path.unlink()

        entry.update(
            {
                "status": "running",
                "started_at": utc_now(),
                "completed_at": None,
                "duration_ms": None,
                "task_counts": {
                    "total": total_tasks,
                    "completed": 0,
                    "failed": 0,
                    "timeout": 0,
                    "skipped": 0,
                },
                "metrics": {},
                "errors": [],
                "pred_path": None,
                "eval_results_path": None,
                "eval_summary_path": None,
                "raw_runs_dir": str(raw_root),
                "task_results_path": str(task_results_path),
                "effective_config_path": str(variant_dir / "effective-config.json"),
                "config_hash": config_payload["config_hash"],
            }
        )
        return _PreparedVariant(
            variant=variant,
            entry=entry,
            raw_root=raw_root,
            task_results_path=task_results_path,
            pred_path=pred_path,
            eval_results_path=eval_results_path,
            eval_summary_path=eval_summary_path,
            started_monotonic=time.time(),
        )

    def _task_record_path(self, state: _PreparedVariant, task: dict[str, object]) -> Path:
        return task_record_path(raw_root=state.raw_root, agent=state.variant.agent, task=task)

    def _task_output_dir(self, state: _PreparedVariant, task: dict[str, object]) -> Path:
        task_id = safe_path_component(task_key(task) or "task")
        bench = str(task.get("bench") or "Verified")
        return state.raw_root / state.variant.agent / bench / task_id

    def _clear_task_outputs(self, state: _PreparedVariant, task: dict[str, object]) -> None:
        task_dir = self._task_output_dir(state, task)
        if task_dir.exists():
            shutil.rmtree(task_dir)

    def _task_is_resume_complete(self, states: list[_PreparedVariant], task: dict[str, object]) -> bool:
        return all(record_is_resume_complete(self._task_record_path(state, task)) for state in states)

    def _workspace_key(self, state: _PreparedVariant, task: dict[str, object]) -> str:
        parts = [
            safe_path_component(self.config.experiment_name),
            self._run_invocation_key,
            safe_path_component(task_key(task) or "task"),
            state.variant.slug,
        ]
        return "__".join(part for part in parts if part)

    def _run_variant_task(self, state: _PreparedVariant, task: dict[str, object]) -> dict[str, object]:
        bench = str(task.get("bench") or "Verified")
        record = run_coding_agent_task(
            task=task,
            agent=state.variant.agent,
            output_dir=state.raw_root / state.variant.agent / bench,
            cache_dir=state.variant.repo_cache,
            schema_path=state.variant.schema_path,
            timeout=state.variant.timeout,
            model=state.variant.model,
            reasoning_effort=state.variant.reasoning_effort,
            agent_args=state.variant.agent_args,
            env_overrides=state.variant.env,
            prompt_preamble=state.variant.setup.prompt_preamble,
            setup=state.variant.setup.model_dump(mode="python"),
            workspace_key=self._workspace_key(state, task),
        )
        cleanup_error: str | None = None
        workspace_cleaned = False
        if str(record.get("status") or "") == "completed" and not record.get("timeout"):
            try:
                remove_worktree(
                    str(record.get("repo_url") or task.get("repo_url") or ""),
                    str(state.variant.repo_cache),
                    str(record.get("workspace_path") or ""),
                )
                workspace_cleaned = True
            except Exception as exc:  # pragma: no cover - defensive guard
                cleanup_error = str(exc)
        return {
            "record": record,
            "record_path": self._task_record_path(state, task),
            "workspace_cleaned": workspace_cleaned,
            "cleanup_error": cleanup_error,
        }

    def _record_skipped_task(self, state: _PreparedVariant, task: dict[str, object], task_id: str) -> None:
        counts = state.entry["task_counts"]
        counts["skipped"] += 1
        append_jsonl(
            state.task_results_path,
            {
                "instance_id": task_id,
                "bench": task.get("bench"),
                "status": "skipped",
                "record_path": str(self._task_record_path(state, task)),
            },
        )

    def _record_variant_exception(
        self,
        state: _PreparedVariant,
        task: dict[str, object],
        task_id: str,
        exc: Exception,
    ) -> None:
        counts = state.entry["task_counts"]
        counts["failed"] += 1
        state.entry["errors"].append(f"{task_id}: {exc}")
        append_jsonl(
            state.task_results_path,
            {
                "instance_id": task_id,
                "bench": task.get("bench"),
                "status": "error",
                "error": str(exc),
            },
        )

    def _record_variant_result(
        self,
        state: _PreparedVariant,
        task: dict[str, object],
        task_id: str,
        result: dict[str, object],
    ) -> None:
        record = result["record"]
        record_status = str(record.get("status") or "")
        counts = state.entry["task_counts"]
        if record.get("timeout"):
            counts["timeout"] += 1
        elif record_status == "completed":
            counts["completed"] += 1
        else:
            counts["failed"] += 1

        append_jsonl(
            state.task_results_path,
            {
                "instance_id": task_id,
                "bench": task.get("bench"),
                "status": record_status or ("timeout" if record.get("timeout") else "failed"),
                "record_path": str(result["record_path"]),
                "task_dir": record.get("task_dir"),
                "timeout": bool(record.get("timeout")),
                "ok": bool(record.get("ok")),
                "workspace_cleaned": bool(result["workspace_cleaned"]),
            },
        )
        cleanup_error = result.get("cleanup_error")
        if cleanup_error:
            state.entry["errors"].append(f"{task_id}: workspace cleanup failed: {cleanup_error}")

    def _finalize_variant(self, state: _PreparedVariant) -> None:
        metrics: dict[str, object] = {}
        variant_status = "completed"
        try:
            raw_agent_dir = state.raw_root / state.variant.agent
            if self.config.postprocess.convert and not self.skip_convert:
                pred_count = convert_records_to_jsonl(
                    source_dir=raw_agent_dir,
                    expected_agent=state.variant.agent,
                    out_path=state.pred_path,
                )
                metrics["prediction_count"] = pred_count
            if self.config.postprocess.evaluate and not self.skip_evaluate:
                if not state.pred_path.exists():
                    raise RuntimeError("Prediction file is missing; conversion must succeed before evaluation")
                eval_cache = self.config.postprocess.cache_dir or state.variant.repo_cache
                metrics["evaluation"] = evaluate_prediction_file(
                    gold_path=self.config.postprocess.gold_path,
                    pred_path=state.pred_path,
                    cache_dir=eval_cache,
                    out_path=state.eval_results_path,
                )
                write_json(state.eval_summary_path, metrics["evaluation"])
        except Exception as exc:
            state.entry["errors"].append(f"postprocess: {exc}")
            variant_status = "postprocess_failed"

        counts = state.entry["task_counts"]
        if variant_status == "completed" and (counts["failed"] > 0 or counts["timeout"] > 0):
            variant_status = "completed_with_failures"

        state.entry.update(
            {
                "status": variant_status,
                "completed_at": utc_now(),
                "duration_ms": int((time.time() - state.started_monotonic) * 1000),
                "metrics": metrics,
                "pred_path": str(state.pred_path) if state.pred_path.exists() else None,
                "eval_results_path": str(state.eval_results_path) if state.eval_results_path.exists() else None,
                "eval_summary_path": str(state.eval_summary_path) if state.eval_summary_path.exists() else None,
            }
        )

    def run(self) -> int:
        tasks, task_set = self._load_tasks()
        effective_variants = [
            build_run_suite_variant(self.config, variant)
            for variant in self.config.variants
            if variant.enabled
        ]
        if not effective_variants:
            raise RuntimeError("No enabled variants remain after config filtering")

        ensure_dir(self.experiment_dir)
        write_json(self.experiment_config_path, self.config.model_dump(mode="json"))

        started_at = utc_now()
        variant_entries = [self._initial_variant_entry(variant) for variant in effective_variants]
        states = [
            self._prepare_variant_state(variant, entry, total_tasks=len(tasks))
            for variant, entry in zip(effective_variants, variant_entries, strict=False)
        ]
        self._write_manifest(
            started_at=started_at,
            completed_at=None,
            task_set=task_set,
            variant_entries=variant_entries,
        )

        workers = min(self.max_workers, len(states))
        for index, task in enumerate(tasks, start=1):
            task_id = task_key(task) or f"task-{index}"
            bench = str(task.get("bench") or "Verified")
            if self.resume and self._task_is_resume_complete(states, task):
                print(f"[task {index}/{len(tasks)}] skip {bench} | {task_id}", flush=True)
                for state in states:
                    self._record_skipped_task(state, task, task_id)
                self._write_manifest(
                    started_at=started_at,
                    completed_at=None,
                    task_set=task_set,
                    variant_entries=variant_entries,
                )
                continue

            print(f"[task {index}/{len(tasks)}] run {bench} | {task_id}", flush=True)
            for state in states:
                self._clear_task_outputs(state, task)

            future_map = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for state in states:
                    future = executor.submit(self._run_variant_task, state, task)
                    future_map[future] = state

                for future in as_completed(future_map):
                    state = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        self._record_variant_exception(state, task, task_id, exc)
                    else:
                        self._record_variant_result(state, task, task_id, result)

            self._write_manifest(
                started_at=started_at,
                completed_at=None,
                task_set=task_set,
                variant_entries=variant_entries,
            )

        for state in states:
            self._finalize_variant(state)

        completed_at = utc_now()
        self._write_manifest(
            started_at=started_at,
            completed_at=completed_at,
            task_set=task_set,
            variant_entries=variant_entries,
        )
        self._write_summary(variant_entries)

        bad_statuses = {"failed", "completed_with_failures", "postprocess_failed"}
        return 0 if all(entry["status"] not in bad_statuses for entry in variant_entries) else 1
