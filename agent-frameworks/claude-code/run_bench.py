#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Thin ContextBench wrapper around the local Claude Code CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextbench.coding_agents.runtime import run_coding_agent_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Claude Code on one ContextBench task from stdin JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=int)
    parser.add_argument("--model", default=None)
    parser.add_argument("--agent-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        task = json.load(sys.stdin)
        record = run_coding_agent_task(
            task=task,
            agent="claude",
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            schema_path=args.schema,
            timeout=args.timeout,
            model=args.model,
            agent_args=args.agent_arg,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1

    summary = {
        "ok": True,
        "status": record.get("status"),
        "record_path": str(Path(record["task_dir"]) / f"{Path(record['task_dir']).name}.claude-record.json"),
        "task_dir": record.get("task_dir"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
