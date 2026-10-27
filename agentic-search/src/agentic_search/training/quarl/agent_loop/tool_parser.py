"""Search agent-loop tool parser (dual-format).

Accepts both the hermes ``<tool_call>{json}</tool_call>`` format (Qwen3
native, also what the aligned data guidance teaches) and the legacy
``<search>query</search>`` plain-text format the current Qwen2.5 model
still emits — so either output triggers a real tool execution during the
transition period. Registered as ``search``; select via
``multi_turn.format: search``.
"""

import json
import logging
import os

import regex as _regex

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.ray_utils import get_event_loop

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TOOL_CALL_RE = _regex.compile(r"<tool_call>(.*?)</tool_call>", _regex.DOTALL)
_SEARCH_RE = _regex.compile(r"<search>(.*?)</search>", _regex.DOTALL)


@ToolParser.register("search")
class SearchToolParser(ToolParser):
    """Parse tool calls from either the JSON <tool_call> format or the legacy
    <search> plain-text format."""

    tool_call_start_token = "<tool_call>"
    tool_call_end_token = "</tool_call>"

    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list = None
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)

        calls: list[FunctionCall] = []

        # ① hermes-style JSON tool calls: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        if self.tool_call_start_token in text and self.tool_call_end_token in text:
            for match in _TOOL_CALL_RE.findall(text):
                try:
                    function_call = json.loads(match)
                    name, arguments = function_call["name"], function_call["arguments"]
                    calls.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to decode <tool_call> block: {e}")

        # ② legacy plain-text search: <search>query</search>
        if "<search>" in text and "</search>" in text:
            for match in _SEARCH_RE.findall(text):
                calls.append(FunctionCall(name="search", arguments=json.dumps({"query": match.strip()}, ensure_ascii=False)))

        # Content = text minus both kinds of tool-call blocks
        content = _TOOL_CALL_RE.sub("", text)
        content = _SEARCH_RE.sub("", content)
        return content, calls
