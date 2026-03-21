# SPDX-License-Identifier: Apache-2.0

"""Constants for coding-agent integrations."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Default prompt-capable gold/task source for Codex and Claude runs.
DEFAULT_GOLD_PATH = REPO_ROOT / "data" / "full.parquet"
# Default task-ordering/filtering CSV used to mirror the selected upstream slice.
DEFAULT_SUBSET_CSV = REPO_ROOT / "data" / "selected_500_instances.csv"
# Default output root for coding-agent run artifacts.
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "coding_agents"
# Default local repository checkout/worktree cache.
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "repos"
# Default schema used to constrain the benchmark-facing structured agent answer.
DEFAULT_OUTPUT_SCHEMA_PATH = REPO_ROOT / "contextbench" / "schemas" / "coding_agent_output.schema.json"

# Candidate task-data field names, in priority order, for the natural-language issue prompt.
DEFAULT_PROMPT_FIELDS = ["prompt", "question", "task", "problem_statement", "instruction"]
# Required keys used to recognize the benchmark summary object inside raw CLI responses.
FINAL_OUTPUT_REQUIRED_KEYS = [
    "task_id",
    "status",
    "final_answer",
    "touched_files",
    "retrieval_steps",
    "retrieved_context_files",
    "retrieved_context_spans",
    "retrieved_context_symbols",
    "notes",
]
BENCH_LABEL_PREFIXES = [
    ("SWE-Bench-Verified__", "Verified"),
    ("SWE-Bench-Pro__", "Pro"),
    ("SWE-PolyBench__", "Poly"),
    ("Multi-SWE-Bench__", "Multi"),
]
