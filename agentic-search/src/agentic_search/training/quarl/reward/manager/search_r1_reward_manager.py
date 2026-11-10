"""Search-R1 reward manager for verl 0.9.0.

The scoring function comes entirely from the config: ``load_reward_manager``
resolves ``reward.custom_reward_function`` and injects it as ``compute_score``
into this manager, so training and validation share the same function --
verl 0.9.0 has a single custom_reward_function slot, and the validate flag
does not survive the agent loop anyway.
"""

import inspect

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


class SearchR1RewardManager(RewardManagerBase):
    """Scores every trajectory with the config-provided compute_score."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        # reward_router_address / reward_model_tokenizer are passed by
        # RewardLoopWorker.load_reward_manager for reward-model setups;
        # accepted (and ignored) here to stay compatible.
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data):
        data = data[-1:]  # multi-sequence outputs: score the last sequence
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        if num_turns is not None:
            extra_info["num_turns"] = num_turns

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        validate = bool(data_item.non_tensor_batch.get("validate", False))
        fn = self.compute_score
        if inspect.iscoroutinefunction(fn):
            result = await fn(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                validate=validate,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: fn(
                    data_source=data_source,
                    solution_str=response_str,
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
