"""Search-R1 PPO trainer on verl 0.9.0.

``Searchr1RayPPOTrainer`` now inherits verl's ``RayPPOTrainer``: the
multi-turn search/answer agent loop runs in the rollout via verl's stock
``ToolAgentLoop`` (``agent.default_agent_loop: tool_agent``, search is the
only tool, JSON function-calling format ``hermes``) and reward is
computed by verl's ``RewardLoopManager`` driven by ``config.reward``, so the
trainer itself carries no Search-R1-specific generation or reward logic.
"""

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class Searchr1RayPPOTrainer(RayPPOTrainer):
    """Search-R1 trainer: inherits the verl 0.9.0 PPO loop end to end.
    """

    pass
