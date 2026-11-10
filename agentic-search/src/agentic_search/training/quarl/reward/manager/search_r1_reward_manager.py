"""Search-R1 reward manager for verl 0.9.0.

The scoring function comes entirely from the config: ``load_reward_manager``
resolves ``reward.custom_reward_function`` and injects it as ``compute_score``
into this manager, so training and validation share the same function --
verl 0.9.0 has a single custom_reward_function slot, and the validate flag
does not survive the agent loop anyway.
"""

import inspect

import torch

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

ASSISTANT_MARKER = "<|im_start|>assistant"


class SearchR1RewardManager(RewardManagerBase):
    """Scores every trajectory with the config-provided compute_score."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        # reward_router_address / reward_model_tokenizer are passed by
        # RewardLoopWorker.load_reward_manager for reward-model setups;
        # accepted (and ignored) here to stay compatible.
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    def _assistant_turn_start(self, ids):
        """Index of the first ``<|im_start|>assistant`` in the full sequence.

        ``input_ids`` is ``prompt + response``. The prompt ends with the
        generation prefix ``<|im_start|>assistant\\n``; the response contains
        every later turn boundary (``<|im_end|>``, ``<|im_start|>user``, ...).
        Starting at that first assistant turn gives the format state machine
        exactly the sequence it was written for, without hard-coding how many
        tokens the prefix occupies or relying on a possibly-truncated prompt
        length.
        """
        marker = torch.tensor(
            self.tokenizer.encode(ASSISTANT_MARKER, add_special_tokens=False),
            dtype=ids.dtype,
            device=ids.device,
        )
        if marker.numel() == 0:
            return 0
        k = marker.numel()
        for pos in (ids == marker[0]).nonzero().flatten().tolist():
            if pos + k <= ids.numel() and torch.equal(ids[pos : pos + k], marker):
                return pos
        return 0

    async def run_single(self, data):
        data = data[-1:]  # multi-sequence outputs: score the last sequence
        data_item = data[0]

        attention_mask = data_item.batch["attention_mask"]
        if "input_ids" in data_item.batch.keys():
            # _compute_score builds input_ids = cat(prompts, responses) and the
            # rollout batch carries the same full sequence; filter padding with
            # the mask so either padding direction works.
            full_ids = data_item.batch["input_ids"][attention_mask.bool()]
        else:
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            prompt_length = prompt_ids.shape[-1]
            full_ids = torch.cat(
                [
                    prompt_ids[attention_mask[:prompt_length].bool()],
                    response_ids[attention_mask[-response_ids.shape[-1] :].bool()],
                ]
            )

        start = self._assistant_turn_start(full_ids)
        trace_ids = full_ids[start:]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        if num_turns is not None:
            extra_info["num_turns"] = num_turns

        # Keep chat markers: both format checks are state machines driven by
        # <|im_start|>assistant / <|im_end|>. Decoding with skip_special_tokens
        # =True turns "<|im_start|>user" into the bare word "user" and drops
        # <|im_end|> entirely, so every trajectory was pinned to the broken
        # format score (0.1) and the +0.1 format signal carried zero gradient
        # under GRPO's group normalisation.
        solution_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(trace_ids, skip_special_tokens=False)
        )

        validate = bool(data_item.non_tensor_batch.get("validate", False))
        fn = self.compute_score
        if inspect.iscoroutinefunction(fn):
            result = await fn(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                validate=validate,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: fn(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    validate=validate,
                ),
            )

        reward_extra_info = {}
        if isinstance(result, dict):
            score = result["score"]
            reward_extra_info.update(result)
        else:
            score = result
            reward_extra_info["acc"] = score
        return {"reward_score": float(score), "reward_extra_info": reward_extra_info}
