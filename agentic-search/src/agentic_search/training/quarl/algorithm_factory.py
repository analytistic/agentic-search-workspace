"""Algorithm-specific assembly helpers for QUARL training.

The trainer orchestration remains in :mod:`quarl.main_rl`; this module owns
the dataset/chat-template and reward-manager construction that varies by
training algorithm.
"""
from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager

def get_trainer_class(config):
    """Return the PPO trainer class for an algorithm name."""
    from quarl.trainer.ppo.searchr1_ray_trainer import Searchr1RayPPOTrainer
    from quarl.trainer.ppo.kgssp_ray_trainer import KGSSPRayPPOTrainer
    from quarl.trainer.ppo.ssp_ray_trainer import SSPRayPPOTrainer
    algorithm_name = _resolve_algorithm_name(config)
    trainer_registry = {
        "search_r1": Searchr1RayPPOTrainer,
        "ssp": SSPRayPPOTrainer,
        "kg_self_play": KGSSPRayPPOTrainer,
    }
    try:
        print(f"Selected trainer class: {trainer_registry[algorithm_name].__name__}")
        return trainer_registry[algorithm_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported algorithm.name={algorithm_name!r}; "
            f"expected one of: {sorted(trainer_registry)}"
        ) from exc


def get_role_worker_mapping(config):
    """Build the role-to-worker mapping used by the PPO trainer."""
    import ray
    from verl.trainer.ppo.ray_trainer import Role

    from verl.workers.fsdp_workers import (
        ActorRolloutRefWorker as FSDPActorRolloutRefWorker,
        AsyncActorRolloutRefWorker as FSDPAsyncActorRolloutRefWorker,
    )
    from quarl.worker.fsdp_worker import SSPActorRolloutRefWorker

    strategy = config.actor_rollout_ref.actor.strategy
    actor_worker_registry = {
        ("fsdp", "search_r1"): FSDPActorRolloutRefWorker,
        ("fsdp", "ssp"): SSPActorRolloutRefWorker,
        ("fsdp", "kg_self_play"): SSPActorRolloutRefWorker,
        ("fsdp2", "search_r1"): FSDPActorRolloutRefWorker,
        ("fsdp2", "ssp"): SSPActorRolloutRefWorker,
        ("fsdp2", "kg_self_play"): SSPActorRolloutRefWorker,
    }
    algorithm_name = _resolve_algorithm_name(config)
    registry_key = (strategy, algorithm_name)
    try:
        actor_rollout_cls = actor_worker_registry[registry_key]
    except KeyError as exc:
        if strategy == "megatron":
            # megatron workers imported lazily (unused in fsdp configs)
            from verl.workers.megatron_workers import ActorRolloutRefWorker as MegatronActorRolloutRefWorker

            actor_rollout_cls = MegatronActorRolloutRefWorker
        else:
            raise ValueError(f"Unsupported actor strategy/algorithm: {registry_key}") from exc

    if strategy in ["fsdp", "fsdp2"]:
        from verl.workers.fsdp_workers import CriticWorker

        critic_worker_cls = CriticWorker
    elif strategy == "megatron":
        from verl.workers.megatron_workers import CriticWorker

        critic_worker_cls = CriticWorker
    else:
        raise NotImplementedError(f"Unsupported actor strategy: {strategy}")

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(critic_worker_cls),
    }

    if config.reward_model.enable:
        if config.reward_model.strategy in ["fsdp", "fsdp2"]:
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError(
                f"Unsupported reward model strategy: {config.reward_model.strategy}"
            )
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

    if config.algorithm.get("use_ref_policy", False) or config.algorithm.get("use_kl", False):
        if strategy in ["fsdp", "fsdp2"]:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker as RefPolicyWorker
        else:
            from verl.workers.megatron_workers import ActorRolloutRefWorker as RefPolicyWorker

        role_worker_mapping[Role.RefPolicy] = ray.remote(RefPolicyWorker)

    return role_worker_mapping


def get_resource_pool_manager(config):
    """Select the Ray worker-group implementation from actor strategy."""
    from verl.single_controller.ray import RayWorkerGroup
    # from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

    worker_group_registry = {
        "fsdp": RayWorkerGroup,
        "fsdp2": RayWorkerGroup,
        # "megatron": NVMegatronRayWorkerGroup,
    }
    strategy = config.actor_rollout_ref.actor.strategy
    try:
        return worker_group_registry[strategy]
    except KeyError as exc:
        raise NotImplementedError(f"Unsupported actor strategy: {strategy}") from exc


def _resolve_algorithm_name(config):
    """Resolve the algorithm name from the top-level ``algorithm.name`` field
    (local kg/search_r1 configs) or the legacy ``quark.task_type`` namespace
    used by the upstream SSP config."""
    name = config.get("algorithm", {}).get("name")
    if name:
        return name
    task_type = config.get("quark", {}).get("task_type", "")
    if task_type in ("origin", "quark_deep_search"):
        return "ssp"
    return "search_r1"


def _resolve_diff_val_reward_fn_config(config):
    """diff_val_reward_fn_config lives at top level in local configs and under
    ``quark.*`` in the upstream SSP config."""
    if config.get("diff_val_reward_fn_config"):
        return config.diff_val_reward_fn_config
    return config.get("quark", {}).get("diff_val_reward_fn_config")


def create_rl_dataset(data_paths, config, tokenizer=None, processor=None):
    """Create the RL dataset and apply algorithm-specific chat templates."""
    import os
    import pandas as pd
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)

    if tokenizer is None:
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    if processor is None:
        processor = hf_processor(local_path, use_fast=True)

    # RLHFDataset needs parquet; convert .jsonl inputs to .parquet on demand.
    converted = []
    for p in [str(x) for x in data_paths]:
        if p.endswith(".jsonl"):
            pq_path = p[: -len(".jsonl")] + ".parquet"
            if not os.path.exists(pq_path):
                print(f"[create_rl_dataset] converting {p} -> {pq_path}")
                pd.read_json(p, lines=True).to_parquet(pq_path)
            converted.append(pq_path)
        else:
            converted.append(p)

    from quarl.utils.chat_template import SearchChatTemplate as QSCT
    from quarl.utils.patch.chat_template import switch_apply_chat_template

    template_name = config.algorithm.get("chat_template", "default")
    if not QSCT.set_template(template_name):
        raise ValueError(
            f"Unknown algorithm.chat_template={template_name!r}; "
            f"available templates: {QSCT.get_available_templates()}"
        )
    print("[create_rl_dataset]: SearchChatTemplate._current_template =", QSCT._current_template)
    tokenizer = switch_apply_chat_template(tokenizer, QSCT.apply_chat_template)
    processor = switch_apply_chat_template(processor, QSCT.apply_chat_template)

    dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")
    return dataset_cls(
        data_files=converted,
        tokenizer=tokenizer,
        processor=processor,
        config=config.data,
    )

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

def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """Build the reward manager selected by the training configuration."""
    from verl.trainer.ppo.reward import get_custom_reward_fn

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    print(f"Loading reward manager: {reward_manager_name}")

    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        return NaiveRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=get_custom_reward_fn(config),
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        return PrimeRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=get_custom_reward_fn(config),
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )
    elif reward_manager_name == "batch":
        from verl.workers.reward_manager import BatchRewardManager

        return BatchRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=get_custom_reward_fn(config),
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )
    elif reward_manager_name == "dapo":
        from verl.workers.reward_manager import DAPORewardManager

        return DAPORewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=get_custom_reward_fn(config),
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )
    elif reward_manager_name in ["ssp_kg", "search_r1", "ssp"]:
        from quarl.reward import get_custom_reward_fns
        from quarl.reward.manager import SSPRewardManager

        return SSPRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            reward_fns=get_custom_reward_fns(config),
            reward_fn_key=config.data.reward_fn_key,
            save_records=False,
            save_path=config.reward_model.get("save_path", None),
        )
    elif reward_manager_name == "naive_with_prompt":
        from quarl.reward.manager import NaiveRewardManagerComputeWithPrompt

        return NaiveRewardManagerComputeWithPrompt(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=get_custom_reward_fn(config),
            reward_fn_key=config.data.reward_fn_key,
            save_records=False,
            save_path=config.reward_model.get("save_path", None),
            **reward_kwargs,
        )
    else:
        raise NotImplementedError(f"Unsupported reward manager: {reward_manager_name}")

    
def build_trainer(
    config,
    tokenizer,
    processor
): 

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }

    trainer_cls = get_trainer_class(config)
    role_worker_mapping = get_role_worker_mapping(config)
    mapping = {
        role: global_pool_id
        for role in role_worker_mapping
    }
    ray_worker_group_cls = get_resource_pool_manager(config)
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    val_reward_fn = load_reward_manager(_resolve_diff_val_reward_fn_config(config), tokenizer, num_examine=1)
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    from verl.utils.dataset.rl_dataset import collate_fn

    train_dataset = create_rl_dataset(config.data.train_files, config, tokenizer=tokenizer, processor=processor)
    val_dataset = create_rl_dataset(config.data.val_files, config, tokenizer=tokenizer, processor=processor)
    train_sampler = create_rl_sampler(config.data, train_dataset)
    return trainer_cls(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
    )
