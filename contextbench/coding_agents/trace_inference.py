# SPDX-License-Identifier: Apache-2.0

"""Heuristics for inferring ContextBench trajectory data from raw agent traces."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from .records import merge_span_maps
from .types import RetrievalStep, SpanMap, SymbolMap, TrajectoryData

_PATH_WITH_LINE_RE = re.compile(r"(?P<path>(?:/|\.{0,2}/)?[^\s:]+(?:/[^\s:]+)*\.[A-Za-z0-9_+-]+):(?P<line>\d+)")
_LINE_ARROW_RE = re.compile(r"^\s*(?P<line>\d+)\s*→", re.MULTILINE)
_PLAIN_PATH_RE = re.compile(r"^(?P<path>(?:/|\.{0,2}/)?[^\s]+(?:/[^\s]+)*\.[A-Za-z0-9_+-]+)\s*$")
_SED_RANGE_RE = re.compile(r"(?P<start>\d+),(?P<end>\d+)p")


def normalize_workspace_path(path_value: str, workspace_path: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(workspace_path.resolve()).as_posix()
        except Exception:
            try:
                path_str = str(path.resolve())
            except Exception:
                path_str = str(path)
            try:
                workspace_str = str(workspace_path.resolve())
            except Exception:
                workspace_str = str(workspace_path)
            if path_str.startswith(workspace_str.rstrip("/") + "/"):
                return path_str[len(workspace_str.rstrip("/") + "/") :]
            return path_value
    return path.as_posix()


def infer_read_span_from_text(text: str) -> tuple[int, int] | None:
    matches = [int(match.group("line")) for match in _LINE_ARROW_RE.finditer(text)]
    if not matches:
        return None
    return min(matches), max(matches)


def infer_grep_spans_from_text(text: str, workspace_path: Path) -> SpanMap:
    spans: SpanMap = {}
    for match in _PATH_WITH_LINE_RE.finditer(text):
        file_path = normalize_workspace_path(match.group("path"), workspace_path)
        line_no = int(match.group("line"))
        spans.setdefault(file_path, []).append({"start": line_no, "end": line_no})
    return spans


def infer_file_list_from_text(text: str, workspace_path: Path) -> list[str]:
    files: list[str] = []
    for line in text.splitlines():
        match = _PLAIN_PATH_RE.match(line.strip())
        if not match:
            continue
        files.append(normalize_workspace_path(match.group("path"), workspace_path))
    return sorted(set(files))


def unwrap_shell_command(command: str) -> str:
    try:
        outer = shlex.split(command)
    except Exception:
        return command
    if len(outer) >= 3 and outer[1] == "-lc":
        return outer[2]
    return command


def command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(unwrap_shell_command(command))
    except Exception:
        return []


def _read_like_step(tokens: list[str], output_text: str, workspace_path: Path) -> RetrievalStep | None:
    path_token = None
    for token in tokens:
        if token in {"|", "&&", "||"}:
            continue
        if token.startswith("-"):
            continue
        if "." not in token and "/" not in token:
            continue
        if token.endswith("p") and "," in token:
            continue
        path_token = token
    if not path_token:
        return None
    file_path = normalize_workspace_path(path_token, workspace_path)
    span = infer_read_span_from_text(output_text)
    spans: SpanMap = {file_path: [{"start": span[0], "end": span[1]}]} if span else {}
    return {"files": [file_path], "spans": spans, "symbols": {}}


def infer_retrieval_step_from_command(
    command: str,
    *,
    output_text: str,
    workspace_path: Path,
) -> RetrievalStep | None:
    tokens = command_tokens(command)
    if not tokens:
        return None

    if "Read" in tokens:
        return None

    if "rg" in tokens or "grep" in tokens:
        spans = infer_grep_spans_from_text(output_text, workspace_path)
        files = sorted(spans) or infer_file_list_from_text(output_text, workspace_path)
        if files or spans:
            return {"files": files, "spans": spans, "symbols": {}}
        return None

    if "find" in tokens:
        files = infer_file_list_from_text(output_text, workspace_path)
        if files:
            return {"files": files, "spans": {}, "symbols": {}}
        return None

    if any(token in {"sed", "cat", "head", "tail", "nl"} for token in tokens):
        return _read_like_step(tokens, output_text, workspace_path)

    return None


def infer_read_step(file_path: str, *, output_text: str, workspace_path: Path) -> RetrievalStep:
    normalized = normalize_workspace_path(file_path, workspace_path)
    span = infer_read_span_from_text(output_text)
    spans: SpanMap = {normalized: [{"start": span[0], "end": span[1]}]} if span else {}
    return {"files": [normalized], "spans": spans, "symbols": {}}


def merge_retrieval_steps(*step_lists: list[RetrievalStep]) -> list[RetrievalStep]:
    merged: list[RetrievalStep] = []
    seen: set[tuple[str, str]] = set()
    for steps in step_lists:
        for step in steps:
            key = (
                ",".join(step.get("files", [])),
                repr(step.get("spans", {})),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(step)
    return merged


def trajectory_from_steps(steps: list[RetrievalStep], *, fallback_files: list[str] | None = None) -> TrajectoryData | None:
    fallback_files = sorted(set(fallback_files or []))
    grounded_files = {
        file_path
        for step in steps
        if step.get("spans") or step.get("symbols")
        for file_path in step.get("files", [])
    }
    all_step_files = {file_path for step in steps for file_path in step.get("files", [])}
    files = sorted(grounded_files or all_step_files) or fallback_files
    spans = merge_span_maps(*(step.get("spans") for step in steps))
    symbols: SymbolMap = {}
    for step in steps:
        for file_path, names in step.get("symbols", {}).items():
            symbols.setdefault(file_path, []).extend(names)
    symbols = {file_path: sorted(set(names)) for file_path, names in symbols.items() if names}
    if not steps and not files and not spans and not symbols:
        return None
    return {
        "pred_steps": steps or ([{"files": files, "spans": spans, "symbols": symbols}] if files or spans or symbols else []),
        "pred_files": files,
        "pred_spans": spans,
        "pred_symbols": symbols,
    }
