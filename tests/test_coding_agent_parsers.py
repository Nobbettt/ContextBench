# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import jsonschema
import pytest

from contextbench.agents.claude import ClaudeAgentParser
from contextbench.agents.codex import CodexAgentParser
from contextbench.coding_agents import (
    build_claude_raw_response,
    build_codex_raw_response,
    extract_structured_output_from_value,
)


def test_extract_structured_output_from_nested_value() -> None:
    payload = {
        "result": {
            "content": json.dumps(
                {
                    "task_id": "task-1",
                    "status": "completed",
                    "final_answer": "done",
                    "touched_files": [],
                    "retrieval_steps": [],
                    "retrieved_context_files": [],
                    "retrieved_context_spans": [],
                    "retrieved_context_symbols": [],
                    "notes": "",
                }
            )
        }
    }

    structured = extract_structured_output_from_value(payload)

    assert structured is not None
    assert structured["task_id"] == "task-1"


def test_extract_structured_output_from_invalid_value_returns_none() -> None:
    assert extract_structured_output_from_value({"result": "not-json"}) is None


def test_codex_parser_extracts_usage_from_turn_completed_event() -> None:
    parser = CodexAgentParser()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {"type": "thread.started"},
            {"type": "turn.completed", "usage": {"input_tokens": 12, "cached_input_tokens": 4, "output_tokens": 3}},
        ],
    }

    usage = parser.extract_token_usage(raw_response)

    assert usage == {
        "source": "codex.turn.completed",
        "input_tokens": 12,
        "output_tokens": 3,
        "cached_input_tokens": 4,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
    }


def test_codex_parser_returns_none_without_turn_completed_usage() -> None:
    parser = CodexAgentParser()

    usage = parser.extract_token_usage({"agent": "codex", "response_format": "jsonl-events", "events": [{"type": "thread.started"}]})

    assert usage is None


def test_codex_parser_extracts_reasoning_tokens_from_output_token_details() -> None:
    parser = CodexAgentParser()
    raw_response = {
        "agent": "codex",
        "response_format": "jsonl-events",
        "events": [
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 20},
                    "output_tokens": 50,
                    "output_tokens_details": {"reasoning_tokens": 30},
                    "total_tokens": 150,
                },
            }
        ],
    }

    usage = parser.extract_token_usage(raw_response)

    assert usage == {
        "source": "codex.turn.completed",
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_input_tokens": 20,
        "total_tokens": 150,
        "cache_read_input_tokens": 20,
        "reasoning_tokens": 30,
    }


def test_claude_parser_extracts_usage_from_response_usage() -> None:
    parser = ClaudeAgentParser()
    raw_response = {
        "agent": "claude",
        "response_format": "json",
        "response": [
            {
                "type": "result",
                "usage": {
                    "input_tokens": 20,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 7,
                    "output_tokens": 9,
                    "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
                },
            }
        ],
    }

    usage = parser.extract_token_usage(raw_response)

    assert usage == {
        "source": "claude.response.usage",
        "input_tokens": 20,
        "output_tokens": 9,
        "total_tokens": 29,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 7,
        "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
    }


def test_claude_parser_returns_none_when_usage_missing() -> None:
    parser = ClaudeAgentParser()

    usage = parser.extract_token_usage({"agent": "claude", "response_format": "json", "response": [{"type": "result", "result": "{}"}]})

    assert usage is None


def test_build_codex_raw_response_reads_events_and_final_output(tmp_path, make_final_output, output_schema) -> None:
    events_path = tmp_path / "codex-events.jsonl"
    final_output_path = tmp_path / "final-output.json"
    events_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "thread.started"}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 1}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    final_output_path.write_text(
        json.dumps(make_final_output(task_id="task-codex", retrieved_context_files=["a.py"])),
        encoding="utf-8",
    )

    raw_response = build_codex_raw_response(events_path, final_output_path)

    assert raw_response["response_format"] == "jsonl-events"
    assert len(raw_response["events"]) == 2
    jsonschema.validate(raw_response["final_message"], output_schema)


def test_build_claude_raw_response_preserves_verbose_event_array(tmp_path) -> None:
    raw_output_path = tmp_path / "claude-output.json"
    raw_output_path.write_text(
        json.dumps(
            [
                {
                    "type": "system",
                    "subtype": "init",
                    "plugins": [],
                    "mcp_servers": {},
                    "slash_commands": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    raw_response = build_claude_raw_response(raw_output_path)

    assert raw_response["agent"] == "claude"
    assert isinstance(raw_response["response"], list)
    assert raw_response["response"][0]["subtype"] == "init"


def test_codex_parser_parses_observed_raw_response_fixture(fixtures_root) -> None:
    parser = CodexAgentParser()
    raw_response = json.loads((fixtures_root / "codex" / "raw_response.json").read_text(encoding="utf-8"))

    structured = parser.extract_structured_output(raw_response)
    usage = parser.extract_token_usage(raw_response)
    tool_calls = parser.extract_tool_calls(raw_response)

    assert structured["task_id"] == "task-1"
    assert structured["retrieved_context_files"] == ["a.py"]
    assert usage == {
        "source": "codex.turn.completed",
        "input_tokens": 12667,
        "output_tokens": 35,
        "cached_input_tokens": 5504,
        "total_tokens": 12702,
        "cache_read_input_tokens": 5504,
    }
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "repo.search"


def test_claude_parser_parses_observed_raw_response_fixture(fixtures_root) -> None:
    parser = ClaudeAgentParser()
    raw_response = json.loads((fixtures_root / "claude" / "raw_response.json").read_text(encoding="utf-8"))

    structured = parser.extract_structured_output(raw_response)
    usage = parser.extract_token_usage(raw_response)
    tool_calls = parser.extract_tool_calls(raw_response)

    assert structured["task_id"] == "task-1"
    assert structured["retrieved_context_files"] == ["a.py"]
    assert usage == {
        "source": "claude.response.usage",
        "input_tokens": 20,
        "output_tokens": 9,
        "total_tokens": 29,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 7,
        "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
    }
    assert tool_calls == [
        {
            "source": "claude.server_tool_use",
            "tool_name": "server_tool_use",
            "payload": {"web_search_requests": 1, "web_fetch_requests": 0},
        }
    ]


def test_observed_fixture_outputs_match_schema(fixtures_root, output_schema) -> None:
    codex_structured = CodexAgentParser().extract_structured_output(
        json.loads((fixtures_root / "codex" / "raw_response.json").read_text(encoding="utf-8"))
    )
    claude_structured = ClaudeAgentParser().extract_structured_output(
        json.loads((fixtures_root / "claude" / "raw_response.json").read_text(encoding="utf-8"))
    )

    jsonschema.validate(codex_structured, output_schema)
    jsonschema.validate(claude_structured, output_schema)


def test_parser_normalize_record_uses_raw_response_path(tmp_path, fixtures_root) -> None:
    raw_path = tmp_path / "raw-response.json"
    raw_path.write_text((fixtures_root / "codex" / "raw_response.json").read_text(encoding="utf-8"), encoding="utf-8")
    record = {
        "agent": "codex",
        "instance_id": "task-1",
        "original_inst_id": "task-1",
        "repo_url": "https://github.com/example/repo.git",
        "commit": "abc123",
        "raw_response_path": str(raw_path),
        "final_output": None,
        "token_usage": None,
        "model_patch": "",
    }

    parser = CodexAgentParser()
    normalized = parser.normalize_record(record)

    assert normalized["final_output"]["task_id"] == "task-1"
    assert normalized["token_usage"]["input_tokens"] == 12667
    assert normalized["tool_calls"][0]["tool_name"] == "repo.search"
