# Importing this package registers the agent loop classes with verl's
# _agent_loop_registry (via @register). It must be imported before
# AgentLoopWorker assembles agent loops from
# config.actor_rollout_ref.agent.default_agent_loop.
#
# NOTE: sglang_rollout is intentionally NOT imported here — the ssp/kg
# (sglang) line is migrated separately and its module needs sglang 0.5.8.

from .vllm_rollout import SearchR1AgentLoop

__all__ = [
    "SearchR1AgentLoop",
]
