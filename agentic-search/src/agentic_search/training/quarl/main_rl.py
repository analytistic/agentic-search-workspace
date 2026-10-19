import logging
import os
from functools import partial
from pprint import pprint

import hydra
import ray
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from quarl.algorithm_factory import build_trainer


def setup_logging():
    log_level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
    log_level = getattr(logging, log_level_str, logging.WARNING)

    logging.getLogger().setLevel(log_level)

    logging.getLogger("quarl").setLevel(log_level)
    logging.getLogger("verl").setLevel(log_level)

    for logger_name in [
        "verl.workers.rollout.sglang_rollout",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.propagate = True

    print(f"Logging level set to: {log_level_str} ({log_level})")

setup_logging()


@hydra.main(config_path="config", config_name="kg_self_play_config", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config

        from omegaconf import OmegaConf

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # used for multimodal LLM, could be none
        processor = hf_processor(local_path, use_fast=True)

  
        from quarl.utils.patch.metric_patch import apply_metric_patch
        apply_metric_patch(tokenizer)

        # cls should be imported after apply_metric_patch
        trainer = build_trainer(config, tokenizer, processor)

        trainer.init_workers()
        trainer.fit()


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
