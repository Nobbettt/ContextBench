<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Fork note: Modified by Norbert Laszlo on 2026-03-16 from upstream ContextBench. -->
<!-- Summary of changes: document fork-specific Codex CLI and Claude Code CLI trajectory extractors. -->

# Agent trajectory extractors

This repo includes trajectory extractors for different coding agents, exposed via a unified API.

## Supported agents

### 1. MiniSWE-agent
- **Format**: `.traj.json` files
- **Location**: `contextbench/agents/minisweagent/extract.py`
- **Notes**:
  - Extracts file views from bash commands in messages
  - Supports `cat`, `sed -n`, `head`, `grep`, `nl | sed` commands
  - Parses `patch_context_data.patch_context` for final context
  - Returns model patch from `info.submission`

### 2. SWE-agent
- **Format**: `.checkpoints.jsonl` files
- **Location**: `contextbench/agents/sweagent/extract.py`
- **Notes**:
  - Extracts from `str_replace_editor view` commands with `--view_range`
  - Only includes steps with explicit line ranges
  - Parses `patch_context` string format (File:/Lines:)

### 3. Codex CLI
- **Format**: `*.codex-record.json`
- **Location**: `contextbench/agents/codex/extract.py`
- **Notes**:
  - Extracts unified trajectories from the wrapper-produced Codex CLI record file
  - Uses reported retrieval steps when present
  - Falls back to touched files and diff-derived spans

### 4. Claude Code CLI
- **Format**: `*.claude-record.json`
- **Location**: `contextbench/agents/claude/extract.py`
- **Notes**:
  - Extracts unified trajectories from the wrapper-produced Claude Code CLI record file
  - Uses reported retrieval steps when present
  - Falls back to touched files and diff-derived spans

### Unified interface

```python
from contextbench.agents import extract_trajectory

result = extract_trajectory("path/to/trajectory.traj.json")
result = extract_trajectory("path/to/trajectory.checkpoints.jsonl")
```

The extractor returns a unified structure:

```python
{
    "pred_steps": [{"files": [...], "spans": {...}}, ...],
    "pred_files": [...],
    "pred_spans": {...},
}
```

## Testing

```bash
python -m contextbench.evaluate \
    --gold Context-dataset/Verified/annots_pass \
    --pred traj_verified-mini/instance/instance.traj.json \
    --out results.jsonl
```
