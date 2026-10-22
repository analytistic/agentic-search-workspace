"""
Search-R1 agentic rollout (vLLM line) on verl 0.9.0.

The multi-turn search/answer agent loop lives here, not in the trainer:
``SearchR1AgentLoop`` implements verl's ``AgentLoopBase`` and registers as
``"search_r1_agent"``; verl's ``AgentLoopWorker`` + ``LLMServerClient`` drive
it against the vLLM server. This mirrors ``sglang_rollout.py`` for the
ssp/kg lines, so template/tool handling shares one pattern.
"""

import json
import logging
import os
import re
from uuid import uuid4
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, ToolListWrap, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.workers.rollout.replica import TokenOutput
from verl.tools.base_tool import ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SEARCH_PREFIX = "<search>"
SEARCH_SUFFIX = "</search>"
ANSWER_PREFIX = "<answer>"
ANSWER_SUFFIX = "</answer>"
INFO_PREFIX = "\n\n<information>"
INFO_SUFFIX = "</information>\n\n"
INVALID_ACTION_MSG = (
    "\nMy previous action is invalid. If I want to search, I should put the query between "
    "<search> and </search>. If I want to give the final answer, I should put the answer "
    "between <answer> and </answer>. Let me try again.\n"
)


@register("search_r1_agent")
class SearchR1AgentLoop(AgentLoopBase):
    """Search-R1 style agent loop: generate -> truncate at ``</search>`` or
    ``</answer>`` -> execute the search tool -> inject observations -> repeat.

    Ported from the old ``LLMGenerationManager`` inside
    ``quarl.trainer.ppo.searchr1_ray_trainer`` (per-sample version of
    ``run_llm_loop``), organized as a verl ``AgentLoopBase`` state machine.
    """

    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        super().__init__(*args, **kwargs)

        mt = self.rollout_config.multi_turn
        self.max_assistant_turns = getattr(mt, "max_assistant_turns", 10)
        self.max_obs_length = getattr(mt, "max_obs_length", 512)
        self.response_length = self.rollout_config.response_length

        tool_list = tools.tools if tools else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [
            tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
        ]
        self._search_tool = self.tools.get("search")

    # ------------------------------------------------------------------
    #  parsing helpers (ported from LLMGenerationManager)
    # ------------------------------------------------------------------
    def _postprocess_response(self, text: str) -> str:
        """Truncate the response at the first complete search/answer block."""
        if SEARCH_SUFFIX in text:
            return text.split(SEARCH_SUFFIX)[0] + SEARCH_SUFFIX
        if ANSWER_SUFFIX in text:
            return text.split(ANSWER_SUFFIX)[0] + ANSWER_SUFFIX
        return text

    def _parse_action(self, text: str) -> tuple[str, str]:
        """Return (action, content) where action is search/answer/invalid."""
        search_match = re.search(rf"{re.escape(SEARCH_PREFIX)}(.*?){re.escape(SEARCH_SUFFIX)}", text, re.DOTALL)
        if search_match:
            return "search", search_match.group(1).strip()
        answer_match = re.search(rf"{re.escape(ANSWER_PREFIX)}(.*?){re.escape(ANSWER_SUFFIX)}", text, re.DOTALL)
        if answer_match:
            return "answer", answer_match.group(1).strip()
        return "invalid", ""

    # ------------------------------------------------------------------
    #  agent loop state machine
    # ------------------------------------------------------------------
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics={},
            request_id=uuid4().hex,
            tools_kwargs=kwargs.get("tools_kwargs", {}),
        )
        agent_data.extra_fields.update(kwargs.get("extra_info", {}) or {})

        # verl 0.9.0 does not carry the validate flag past the agent loop
        # (DataProto.pop and the loop output rebuild meta_info). Detect it
        # from the sampling params (validate runs use config.val_kwargs) and
        # attach it to extra_fields -> non_tensor_batch["validate"] so
        # SearchR1RewardManager can dispatch train vs validation scoring.
        vc = self.rollout_config.val_kwargs
        agent_data.extra_fields["validate"] = bool(
            sampling_params.get("top_p") == vc.top_p
            and sampling_params.get("top_k") == vc.top_k
            and sampling_params.get("temperature") == vc.temperature
        )

        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_search(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output (same layout as ToolAgentLoop.run)
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        mm_out = {}
        if agent_data.image_data is not None:
            mm_out["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            mm_out["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            mm_out["audios"] = agent_data.audio_data

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=mm_out,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields=agent_data.extra_fields,
        )

    async def _handle_pending(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Render the current message list and start generation."""
        if agent_data.messages:
            prompt_ids = await self.apply_chat_template(agent_data.messages)
            agent_data.prompt_ids = list(prompt_ids)
            agent_data.response_mask = [0] * len(agent_data.prompt_ids)
        return AgentState.GENERATING

    async def _handle_generating(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Generate one response, truncate at search/answer block, route."""
        output: TokenOutput = await self.server_manager.generate(
            request_id=agent_data.request_id,
            prompt_ids=agent_data.prompt_ids,
            sampling_params=sampling_params,
            image_data=agent_data.image_data,
            video_data=agent_data.video_data,
            audio_data=agent_data.audio_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
        )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += output.token_ids
        agent_data.response_mask += [1] * len(output.token_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Termination by length / turn budget
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED

        text = self._postprocess_response(
            self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        )
        action, content = self._parse_action(text)

        if action == "answer":
            return AgentState.TERMINATED
        if action == "search":
            # Re-tokenize the truncated text so only the <search>..</search>
            # block is appended (observations go into the prompt region).
            truncated_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            agent_data.prompt_ids = agent_data.prompt_ids[: -len(output.token_ids)] + truncated_ids
            agent_data.response_mask = agent_data.response_mask[: -len(output.token_ids)] + [1] * len(truncated_ids)
            if output.log_probs:
                agent_data.response_logprobs = agent_data.response_logprobs[: -len(output.token_ids)]
            agent_data.tool_calls = [
                {"name": "search", "arguments": json.dumps({"query_list": [content]})}
            ]
            return AgentState.PROCESSING_TOOLS

        # Invalid action: inject the retry hint and continue
        agent_data.messages.append({"role": "assistant", "content": text})
        agent_data.messages.append({"role": "user", "content": INVALID_ACTION_MSG})
        return AgentState.PENDING

    async def _handle_search(self, agent_data: AgentData) -> AgentState:
        """Execute the search tool and inject the observation, then continue."""
        tool_call = agent_data.tool_calls[0]
        obs = ""
        if self._search_tool is not None:
            try:
                tool_args = json.loads(tool_call["arguments"])
                instance_id, _ = await self._search_tool.create()
                tool_response: ToolResponse = (
                    await self._search_tool.execute(instance_id, tool_args, agent_data=agent_data)
                )[0]
                obs = tool_response.text or ""
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Search tool failed: {e}")
                obs = ""
            finally:
                await self._search_tool.release(instance_id)

        obs_text = f"{INFO_PREFIX}{obs.strip()}{INFO_SUFFIX}" if obs else INVALID_ACTION_MSG
        obs_ids = self.tokenizer(
            obs_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].tolist()
        if len(obs_ids) > self.max_obs_length:
            obs_ids = obs_ids[: self.max_obs_length]

        agent_data.prompt_ids += obs_ids
        agent_data.response_mask += [0] * len(obs_ids)
        agent_data.tool_calls = []
        return AgentState.GENERATING
