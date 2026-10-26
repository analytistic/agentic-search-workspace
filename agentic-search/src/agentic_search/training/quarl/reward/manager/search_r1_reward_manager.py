"""Search-R1 reward manager for verl 0.9.0.

Training scores with ``qa_em.compute_score`` (exact match); validation scores
with the richer ``search_eval_score.compute_score`` (EM + F1 + judge),
preserving the legacy ``reward_fn`` / ``val_reward_fn`` split.

NOTE: with verl's stock ``ToolAgentLoop`` (``agent.default_agent_loop:
tool_agent``) the validate flag is no longer attached by a custom agent loop
(verl 0.9.0 does not carry ``meta_info["validate"]`` past the agent loop), so
``non_tensor_batch["validate"]`` is always absent -> training scoring
(``qa_em``) applies to every sample, including validation runs. To restore
the train/validation split, either re-attach the flag (a thin loop wrapper /
dataset ``extra_info`` field) or run validation with a separate reward config.
"""

import inspect

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from quarl.reward.score.qa_em import compute_score as _train_compute_score
from quarl.reward.score.search_eval_score import compute_score as _val_compute_score


class SearchR1RewardManager(RewardManagerBase):
    """Dispatch on the validate flag: training -> qa_em, validation -> search_eval_score."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        # reward_router_address / reward_model_tokenizer are passed by
        # RewardLoopWorker.load_reward_manager for reward-model setups;
        # accepted (and ignored) here to stay compatible.
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self._train_score = compute_score or _train_compute_score
        self._val_score = _val_compute_score

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
        fn = self._val_score if validate else self._train_score
        if inspect.iscoroutinefunction(fn):
            result = await fn(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: fn(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
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
