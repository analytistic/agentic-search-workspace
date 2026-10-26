# SSP-KG components are imported lazily: quarl.trainer.ppo.ssp_ray_trainer
# still targets pre-0.9.0 verl APIs (compute_reward_async, utils.debug) and
# would break ``import quarl`` for search_r1 runs. Load on access only;
# the trainers themselves are already lazy-loaded via
# quarl.algorithm_factory.get_trainer_class.


def __getattr__(name):
    if name == "SSPRayPPOTrainer":
        from quarl.trainer.ppo.ssp_ray_trainer import SSPRayPPOTrainer

        return SSPRayPPOTrainer
    if name == "SSPDataManager":
        from quarl.utils.ssp_data_manager import SSPDataManager

        return SSPDataManager
    if name == "ProblemExtractor":
        from quarl.utils.problem_extraction import ProblemExtractor

        return ProblemExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SSPRayPPOTrainer", "SSPDataManager", "ProblemExtractor"]
