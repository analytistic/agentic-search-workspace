"""Search agent-loop tool parser (legacy <search> tag format).

Parses the plain-text ``<search>query</search>`` format the previous
Search-R1 pipeline worked with, wrapping each query into the current
search tool schema (``{"query": ...}``). Select via
``multi_turn.format: search``; switch back to ``hermes`` (or a dual parser)
when the model emits JSON tool calls natively (e.g. Qwen3).
"""

import json
import logging
import os

import regex

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ToolParser.register("search")
class SearchToolParser(ToolParser):
    """Parse <search>..</search> blocks into ``search`` tool calls."""

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<search>"
        self.tool_call_end_token: str = "</search>"
        self.eos_token: str = tokenizer.eos_token
        self.tool_call_regex = regex.compile(r"<search>(.*?)</search>", regex.DOTALL)

    @rollout_trace_op
    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list = None
    ) -> tuple[str, list[FunctionCall]]:
        del tools
        loop = get_event_loop()
        text = await loop.run_in_executor(None, lambda: self.tokenizer.decode(responses_ids, skip_special_tokens=False))
        for token in (self.tokenizer.pad_token, self.eos_token):
            if token:
                text = text.replace(token, "")

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            function_calls.append(
                FunctionCall(
                    name="search",
                    arguments=json.dumps({"query": match.strip()}, ensure_ascii=False),
                )
            )

        # remaining text excluding the search blocks
        content = self.tool_call_regex.sub("", text)
        return content, function_calls
