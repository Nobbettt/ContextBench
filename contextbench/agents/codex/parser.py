# SPDX-License-Identifier: Apache-2.0

"""Codex-specific parsing for wrapper-produced records and raw responses."""

from __future__ import annotations

from ..base import BaseCodingAgentParser
from ...coding_agents.response_parsing import extract_structured_output_from_value
from ...coding_agents.types import CodexRawResponse, StructuredOutput, TokenUsage, ToolCall


class CodexAgentParser(BaseCodingAgentParser):
    def extract_structured_output(self, raw_response: CodexRawResponse) -> StructuredOutput | None:
        if not isinstance(raw_response, dict):
            return None
        final_message = raw_response.get("final_message")
        if final_message is not None:
            structured = extract_structured_output_from_value(final_message)
            if structured:
                return structured
        return extract_structured_output_from_value(raw_response)

    def extract_token_usage(self, raw_response: CodexRawResponse) -> TokenUsage | None:
        if not isinstance(raw_response, dict):
            return None
        events = raw_response.get("events")
        if not isinstance(events, list):
            return None
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            if event.get("type") != "turn.completed":
                continue
            usage = event.get("usage")
            if not isinstance(usage, dict):
                continue
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            cached_input_tokens = int(usage.get("cached_input_tokens", 0) or 0)
            if not cached_input_tokens:
                details = usage.get("input_tokens_details")
                if isinstance(details, dict):
                    cached_input_tokens = int(details.get("cached_tokens", 0) or 0)
            reasoning_tokens = 0
            output_details = usage.get("output_tokens_details")
            if isinstance(output_details, dict):
                reasoning_tokens = int(output_details.get("reasoning_tokens", 0) or 0)
            result: TokenUsage = {
                "source": "codex.turn.completed",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": cached_input_tokens,
                "total_tokens": int(usage.get("total_tokens", 0) or (input_tokens + output_tokens)),
            }
            if cached_input_tokens:
                result["cache_read_input_tokens"] = cached_input_tokens
            if reasoning_tokens:
                result["reasoning_tokens"] = reasoning_tokens
            return result
        return None

    def extract_tool_calls(self, raw_response: CodexRawResponse) -> list[ToolCall]:
        if not isinstance(raw_response, dict):
            return []
        events = raw_response.get("events")
        if not isinstance(events, list):
            return []
        calls: list[ToolCall] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "")
            if "tool" not in event_type.lower() and "mcp" not in event_type.lower():
                continue
            calls.append(
                {
                    "source": "codex.event",
                    "tool_name": str(event.get("tool_name") or event.get("type") or "unknown"),
                    "payload": dict(event),
                }
            )
        return calls
