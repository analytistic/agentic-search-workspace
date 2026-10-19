"""Project vLLM rollout wrapper.

This is the project-level vLLM backend used by Search-R1, SSP, and KG
training. It wraps verl's synchronous vLLM implementation and normalizes its
``DataProto`` output to the fields expected by the project's rollout code.
Algorithm-specific multi-turn orchestration can call this one-turn backend
repeatedly; the wrapper itself does not create a second engine or a remote
vLLM server.

The worker/sharding manager still owns model loading and actor-to-rollout
weight synchronization. In particular, this class must not create a second
AsyncLLM or a standalone HTTP vLLM server.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch


class vLLMRollout:
    """Project adapter around verl's synchronous ``vLLMRollout``.

    Constructor arguments are intentionally identical to verl's modern
    ``vLLMRollout`` constructor. The import is lazy so configuration and
    tooling can import this module on a machine where vLLM is not installed.
    Once constructed, all engine lifecycle and weight synchronization methods
    are delegated to the wrapped rollout.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Keep the two classes unambiguous: this module exports the project
        # wrapper under the same public name as verl's backend.
        from verl.workers.rollout.vllm_rollout import vLLMRollout as VerlVLLMRollout

        self._rollout = VerlVLLMRollout(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # ``__getattr__`` is only called after normal attribute lookup fails.
        # Guard the bootstrap case to avoid recursive lookup if construction
        # itself raises before ``_rollout`` is assigned.
        rollout = self.__dict__.get("_rollout")
        if rollout is None:
            raise AttributeError(name)
        return getattr(rollout, name)

    @contextmanager
    def update_sampling_params(self, **kwargs: Any):
        """Temporarily override one-turn vLLM sampling parameters."""
        with self._rollout.update_sampling_params(**kwargs):
            yield

    @staticmethod
    def _add_sglang_compatible_fields(output):
        """Normalize one-turn output fields expected by PPO and SGLang.

        The current verl vLLM rollout already returns ``prompts``,
        ``responses``, ``input_ids``, ``attention_mask`` and ``position_ids``.
        SGLang additionally exposes ``response_mask``. A one-turn vLLM
        response has no observation tokens, so every non-padding response
        token is a trainable model token and receives mask value 1.
        """
        response = output.batch["responses"]
        attention_mask = output.batch["attention_mask"]
        prompt_length = output.batch["prompts"].shape[-1]
        response_attention = attention_mask[:, prompt_length : prompt_length + response.shape[-1]]

        if "response_mask" not in output.batch:
            output.batch["response_mask"] = response_attention.to(dtype=torch.long)
        else:
            output.batch["response_mask"] = output.batch["response_mask"].to(dtype=torch.long)

        if "position_ids" not in output.batch:
            full_attention = attention_mask.to(dtype=torch.long)
            output.batch["position_ids"] = (full_attention.cumsum(dim=-1) - 1).clamp_min(0) * full_attention

        if "rollout_log_probs" in output.batch:
            output.batch["rollout_log_probs"] = output.batch["rollout_log_probs"].to(dtype=torch.float32)

        return output

    @torch.no_grad()
    def generate_sequences(self, prompts, **kwargs):
        """Generate one response turn synchronously.

        Search-R1's generation manager, or an SSP/KG rollout loop, can perform
        the following sequence around this method:

        1. call this method for the active prompts;
        2. stop at ``</search>`` or ``</answer>``;
        3. call the retriever and append ``<information>`` tokens;
        4. call this method again for unfinished samples.

        Keeping one-turn generation here is what lets the worker enter
        ``FSDPVLLMShardingManager`` before every call and synchronize the
        latest actor weights in the same way as the stock verl rollout.
        """
        output = self._rollout.generate_sequences(prompts, **kwargs)
        return self._add_sglang_compatible_fields(output)

    def generate_one_turn(self, prompts, **kwargs):
        """Explicit alias used by external Search-R1-style managers."""
        return self.generate_sequences(prompts, **kwargs)


__all__ = ["vLLMRollout"]
