# SPDX-License-Identifier: Apache-2.0

"""Claude-specific parsing for wrapper-produced records and raw responses."""

from __future__ import annotations

from ..base import BaseCodingAgentParser
from ...coding_agents.response_parsing import extract_structured_output_from_value, parse_json_from_text
from ...coding_agents.types import ClaudeRawResponse, StructuredOutput, TokenUsage, ToolCall


class ClaudeAgentParser(BaseCodingAgentParser):
    def extract_structured_output(self, raw_response: ClaudeRawResponse) -> StructuredOutput | None:
        if not isinstance(raw_response, dict):
            return None
        response = raw_response.get("response")
        if isinstance(response, dict):
            result = response.get("result")
            if isinstance(result, str):
                parsed = parse_json_from_text(result)
                if parsed:
                    return parsed
            structured = extract_structured_output_from_value(result)
            if structured:
                return structured
            structured = extract_structured_output_from_value(response)
            if structured:
                return structured
        elif isinstance(response, list):
            for item in reversed(response):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "result":
                    result = item.get("result")
                    if isinstance(result, str):
                        parsed = parse_json_from_text(result)
                        if parsed:
                            return parsed
                structured = extract_structured_output_from_value(item)
                if structured:
                    return structured
        return extract_structured_output_from_value(raw_response)

    def extract_token_usage(self, raw_response: ClaudeRawResponse) -> TokenUsage | None:
        if not isinstance(raw_response, dict):
            return None
        response = raw_response.get("response")
        if isinstance(response, list):
            for item in reversed(response):
                if not isinstance(item, dict):
                    continue
                usage = item.get("usage")
                if isinstance(usage, dict):
                    return self._build_usage(usage)
            return None
        if not isinstance(response, dict):
            return None
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return None
        return self._build_usage(usage)

    def _build_usage(self, usage: dict[str, object]) -> TokenUsage:
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cache_creation_input_tokens = int(usage.get("cache_creation_input_tokens", 0) or 0)
        cache_read_input_tokens = int(usage.get("cache_read_input_tokens", 0) or 0)
        result: TokenUsage = {
            "source": "claude.response.usage",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        }
        server_tool_use = usage.get("server_tool_use")
        if isinstance(server_tool_use, dict):
            result["server_tool_use"] = server_tool_use
        return result

    def extract_tool_calls(self, raw_response: ClaudeRawResponse) -> list[ToolCall]:
        if not isinstance(raw_response, dict):
            return []
        response = raw_response.get("response")
        if isinstance(response, list):
            for item in reversed(response):
                if not isinstance(item, dict):
                    continue
                usage = item.get("usage")
                if isinstance(usage, dict):
                    server_tool_use = usage.get("server_tool_use")
                    if isinstance(server_tool_use, dict):
                        return [
                            {
                                "source": "claude.server_tool_use",
                                "tool_name": "server_tool_use",
                                "payload": dict(server_tool_use),
                            }
                        ]
            return []
        if not isinstance(response, dict):
            return []
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return []
        server_tool_use = usage.get("server_tool_use")
        if isinstance(server_tool_use, dict):
            return [
                {
                    "source": "claude.server_tool_use",
                    "tool_name": "server_tool_use",
                    "payload": dict(server_tool_use),
                }
            ]
        return []
