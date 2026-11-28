from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking


class SSPRayPPOTrainer(RayPPOTrainer):
    """SSP self-play trainer on verl 0.9.0."""

    def fit(self):
        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        current_epoch = self.global_steps // len(self.train_dataloader)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            print(f"Epoch {epoch}")
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                with marked_timer("self_play_step", timing_raw):
                    # proposer -> solver -> reward -> update
                    pass
