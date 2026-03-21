# SPDX-License-Identifier: Apache-2.0

"""Prompt construction for coding-agent integrations."""

from __future__ import annotations


def build_prompt(task: dict[str, object], agent_name: str) -> str:
    """Build a small, deterministic prompt for Codex/Claude benchmark runs."""
    lines = [
        f"You are running a ContextBench task in repository {(task.get('repo') or task.get('repo_url') or 'unknown-repo')}.",
        "Use only the local repository and local tools. Do not use web search or external sources.",
        "Work only inside the checked-out repository workspace.",
        "Do not install dependencies or fetch network resources during this run.",
        "If verification tooling is unavailable locally, use offline-safe checks and state the limitation.",
        "Return only a JSON object that matches the required schema.",
        "",
        f"Task ID: {task.get('instance_id') or task.get('original_inst_id')}",
        f"Agent: {agent_name}",
        f"Bench: {task.get('bench')}",
        "",
        "Task:",
        task.get("prompt") or "No task prompt was available.",
        "",
        "Track the repository context you inspect while solving the task.",
        "Report retrieval activity in coarse chronological order using retrieval_steps.",
        "Each retrieval step should include files and, when known, line spans or symbol names.",
        "Set retrieved_context_files and retrieved_context_spans to the final context you relied on most.",
    ]
    return "\n".join(lines)
