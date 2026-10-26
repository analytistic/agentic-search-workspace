"""Algorithm-specific assembly helpers for QUARL training.

The trainer orchestration remains in :mod:`quarl.main_rl`; this module owns
the dataset/chat-template and reward-manager construction that varies by
training algorithm.
"""
import re

from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager

def _lazy_import(module_name: str, class_name: str):
    """Import a class on demand (keeps unrelated trainer modules out of the
    import path, e.g. ssp/kgssp while running search_r1)."""
    import importlib

    return getattr(importlib.import_module(module_name), class_name)


# Legacy Search-R1 data prompts teach the <search> tag protocol; with the
# stock ToolAgentLoop (hermes <tool_call> JSON calling) the model must emit
# a JSON tool call instead. Covers the two known legacy phrasings (nq_search /
# qa_search_* and problem_extraction.py). Rewritten in create_rl_dataset,
# idempotent, marker-based.
_LEGACY_SEARCH_GUIDE_RE = re.compile(
    r"you can call a search engine by <search> query </search>,?\s*"
    r"and it will return the top searched results between <information> and </information>",
    re.IGNORECASE,
)
_TOOL_CALL_GUIDE = (
    "you can search by calling the search tool in this format: "
    '<tool_call>{"name": "search", "arguments": {"query_list": ["<your query>"]}}</tool_call>. '
    "Search results will be returned to you"
)


def _align_searchr1_prompt(prompt):
    """Rewrite legacy <search>-tag guidance in user messages to JSON
    tool-calling guidance (hermes <tool_call> format). Pure: returns a new
    prompt when changed, the original object when not.
    """
    if not isinstance(prompt, list):
        return prompt
    new_prompt = None
    for i, msg in enumerate(prompt):
        if (
            isinstance(msg, dict)
            and msg.get("role") == "user"
            and isinstance(msg.get("content"), str)
        ):
            content = _LEGACY_SEARCH_GUIDE_RE.sub(_TOOL_CALL_GUIDE, msg["content"], count=1)
            if content != msg["content"]:
                if new_prompt is None:
                    new_prompt = list(prompt)
                new_prompt[i] = {**msg, "content": content}
    return new_prompt if new_prompt is not None else prompt


def get_trainer_class(config):
    """Return the PPO trainer class for an algorithm name (lazy imports)."""
    algorithm_name = _resolve_algorithm_name(config)
    trainer_registry = {
        "search_r1": ("quarl.trainer.ppo.searchr1_ray_trainer", "Searchr1RayPPOTrainer"),
        "ssp": ("quarl.trainer.ppo.ssp_ray_trainer", "SSPRayPPOTrainer"),
        "kg_self_play": ("quarl.trainer.ppo.kgssp_ray_trainer", "KGSSPRayPPOTrainer"),
    }
    try:
        trainer_cls = _lazy_import(*trainer_registry[algorithm_name])
        print(f"Selected trainer class: {trainer_cls.__name__}")
        return trainer_cls
    except KeyError as exc:
        raise ValueError(
            f"Unsupported algorithm.name={algorithm_name!r}; "
            f"expected one of: {sorted(trainer_registry)}"
        ) from exc


def get_role_worker_mapping(config):
    """Build the role-to-worker mapping used by the PPO trainer (verl 0.9.0).

    Follows verl's own ``trainer/main_ppo_v0.py``: a single
    ``ActorRolloutRefWorker`` covers actor+rollout(+ref) and ``TrainingWorker``
    covers critic; reward models are handled by RewardLoopManager, not by a
    worker role.
    """
    import ray
    from verl.trainer.ppo.ray_trainer import Role
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

    lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

    # Ref policy is fused into ActorRolloutRefWorker unless LoRA is used with
    # a dedicated ref model.
    if need_reference_policy(config) and not ref_in_actor:
        role = Role.ActorRolloutRef
    else:
        role = Role.ActorRollout
    role_worker_mapping = {role: ray.remote(ActorRolloutRefWorker)}

    if need_critic(config):
        role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)

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

    # Legacy <search>-tag guidance -> hermes <tool_call> JSON, rewritten in
    # place (idempotent: _align_searchr1_prompt is a no-op once the marker
    # is gone, so this also covers already-aligned files cheaply).
    for pq in converted:
        df = pd.read_parquet(pq)
        if "prompt" not in df.columns:
            continue
        aligned = df["prompt"].apply(_align_searchr1_prompt)
        if not df["prompt"].equals(aligned):
            print(f"[create_rl_dataset] aligning legacy search prompt: {pq}")
            df["prompt"] = aligned
            df.to_parquet(pq)

    # NOTE: no QSCT/chat-template injection here — verl 0.9.0's RLHFDataset
    # only returns raw_prompt (apply_chat_template moved to AgentLoop), and
    # the rollout renders messages with the model's default chat template
    # (format guidance lives in the data prompt).

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
    # verl 0.9.0: reward is configured via config.reward (custom_reward_function
    # + reward_manager) and computed by RewardLoopManager inside the trainer.
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
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
    )
