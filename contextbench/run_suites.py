# SPDX-License-Identifier: Apache-2.0

"""Primary CLI entrypoint for ContextBench run suites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .run_suites_core import (
    RunSuiteConfig,
    RunSuiteRunner,
    build_run_suite_variant,
    load_run_suite_config,
)

__all__ = [
    "RunSuiteConfig",
    "RunSuiteRunner",
    "build_run_suite_variant",
    "load_run_suite_config",
    "main",
    "parse_args",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ContextBench run suite across multiple coding-agent setups")
    parser.add_argument("--config", required=True, type=Path, help="Path to run suite JSON config")
    parser.add_argument("--max-workers", type=int, default=None, help="Override the per-task variant worker cap")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping only tasks where every enabled variant already completed",
    )
    parser.add_argument("--skip-convert", action="store_true", help="Skip record-to-prediction conversion")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation even if enabled in config")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2

    try:
        config = load_run_suite_config(args.config)
        runner = RunSuiteRunner(
            config,
            max_workers=args.max_workers,
            resume=bool(args.resume),
            skip_convert=bool(args.skip_convert),
            skip_evaluate=bool(args.skip_evaluate),
        )
        return runner.run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
