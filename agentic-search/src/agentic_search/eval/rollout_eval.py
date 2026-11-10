#!/usr/bin/env python3
"""Standalone multi-mode rollout evaluation (Search-R1 / SSP baselines).

Three modes over the same test set:

  agentic   -- ``tool_agent`` loop, full multi-turn search + answer
               (max_assistant_turns=10, search tool registered)
  single    -- ``tool_agent`` loop with max_assistant_turns=1: exactly one
               search round, then answer
  no_tool   -- ``single_turn_agent`` loop: plain generation, no tools

Everything else is shared with training: the hermes chat template
(qwen3_hermes.jinja, same as the hermes launchers), the search function tool
(quarl/tool/search_tool_fn.py -> retrieval service), and the EM/F1/judge
scorer (quarl/reward/score/search_eval_score.py).

It reuses verl's AgentLoopManager: only vLLM rollout workers are started
(no FSDP actor/ref, no reward loop), so startup is a fraction of the full
RayPPOTrainer, and the three modes run in one process / one Ray cluster.

Launched via hydra + shell script (same style as the training launchers):

    MODE=no_tool bash run_rollout_eval.sh          # or agentic / single

Per-sample trajectories (decoded text incl. tool messages, rm_score and
reward components) go to <output-dir>/{mode}_trajectories.jsonl; aggregated
metrics go to mlflow (val-core / val-aux, one experiment per mode).
"""

import json
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path

# ---- path bootstrap (mirrors the training launchers' PYTHONPATH) ----
# WORKSPACE must point at the training dir: search_r1_config's hydra
# searchpath and function_tool_path resolve ${oc.env:WORKSPACE} against it.
EVAL_DIR = Path(__file__).resolve().parent
SRC_DIR = EVAL_DIR.parent  # agentic_search/
TRAINING_DIR = SRC_DIR / "training"
os.environ.setdefault("WORKSPACE", str(TRAINING_DIR))
for p in (str(TRAINING_DIR), str(TRAINING_DIR / "verl"), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import ray  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torchdata.stateful_dataloader import StatefulDataLoader  # noqa: E402

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto  # noqa: E402
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup  # noqa: E402
from verl.single_controller.ray.base import create_colocated_worker_cls  # noqa: E402
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role  # noqa: E402
from verl.trainer.ppo.reward import extract_reward  # noqa: E402
from verl.trainer.ppo.metric_utils import process_validation_metrics  # noqa: E402
from verl.trainer.ppo.utils import create_rl_dataset  # noqa: E402
from verl.utils import hf_processor, hf_tokenizer  # noqa: E402
from verl.utils.dataset.rl_dataset import collate_fn  # noqa: E402
from verl.utils.tracking import Tracking, ValidationGenerationsLogger  # noqa: E402
from verl.workers.engine_workers import ActorRolloutRefWorker  # noqa: E402

from quarl.algorithm_factory import _clean_prompts_for_hermes  # noqa: E402


def init_agent_loop_manager(config):
    """Start vLLM AgentLoopWorkers + AgentLoopManager (verl official test pattern).

    Only the rollout half of the hybrid worker is created: role="rollout" makes
    init_model skip the FSDP actor/ref branches (engine_workers.py checks
    `"actor" in self.role` / `"ref" in self.role`), so no actor weights are
    loaded or sharded -- each worker just holds one vLLM engine; the agent loop
    (tool_agent / single_turn_agent) runs inside the worker.
    """
    role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
    global_pool_id = "global_pool"
    # env interpolation / Hydra CLI overrides deliver strings ("1"); the
    # resource pool spec requires ints
    n_gpus_per_node = int(config.trainer.n_gpus_per_node)
    nnodes = int(config.trainer.nnodes)
    resource_pool_spec = {
        global_pool_id: [n_gpus_per_node] * nnodes,
    }
    mapping = {Role.ActorRollout: global_pool_id}
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout], config=config.actor_rollout_ref, role="rollout"
    )
    resource_pool_to_cls[resource_pool]["rollout"] = rollout_cls

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
    rollout_wg = all_wg["rollout"]
    rollout_wg.init_model()

    from verl.experimental.agent_loop import AgentLoopManager
    from verl.experimental.reward_loop import RewardLoopManager
    from verl.workers.rollout.llm_server import LLMServerManager

    # vLLM servers are managed by LLMServerManager on top of the rollout workers;
    # AgentLoopManager builds num_workers CPU AgentLoopWorkers that hold the client.
    llm_server_manager = LLMServerManager.create(
        config=config,
        worker_group=rollout_wg,
        rollout_resource_pool=resource_pool,
    )
    llm_client = llm_server_manager.get_client()
    # reward loop workers (rule-based qa_em, CPU) -- same source as training;
    # handles are passed through so the agent loop scores inline (rm_scores in output)
    reward_loop_manager = RewardLoopManager(config=config, rm_resource_pool=None)  # reward_model.enable=false
    return AgentLoopManager.create(
        config=config,
        llm_client=llm_client,
        reward_loop_worker_handles=reward_loop_manager.reward_loop_worker_handles,
    )


def _get_gen_batch(batch: DataProto) -> DataProto:
    """RayPPOTrainer._get_gen_batch: strip non-generation fields, keep reward keys.

    ``pop`` removes the keys from ``batch`` into ``gen_batch``; the reward
    fields (data_source/reward_model/extra_info/uid) are then re-added to
    gen_batch because the agent loop may compute scores inline.
    """
    reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
    non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
    gen_batch = batch.pop(
        batch_keys=[],
        non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
    )
    gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
    return gen_batch


def _val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
    """RayPPOTrainer._val_metrics_update: aggregate per-sample val data into
    layered scalar metrics (val-core / val-aux per data source)."""
    data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
    metric_dict = {}
    for data_source, var2metric2val in data_src2var2metric2val.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
            for metric_name, metric_val in metric2val.items():
                if (
                    (var_name == core_var)
                    and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                    and (f"@{n_max}" in metric_name)
                ):
                    metric_sec = "val-core"
                else:
                    metric_sec = "val-aux"
                metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = metric_val

    if len(sample_turns) > 0:
        sample_turns = np.concatenate(sample_turns)
        metric_dict["val-aux/num_turns/min"] = sample_turns.min()
        metric_dict["val-aux/num_turns/max"] = sample_turns.max()
        metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

    return metric_dict


@hydra.main(config_path="config", config_name="rollout_eval_agentic", version_base=None)
def main(config):
    # hermes template: launcher exports HERMES_JINJA (same as training); fall
    # back to the repo template so the script also runs without a launcher.
    if not os.environ.get("HERMES_JINJA"):
        os.environ["HERMES_JINJA"] = (TRAINING_DIR / "quarl/template/qwen3_hermes.jinja").read_text(encoding="utf-8")

    mode = config.mode  # agentic | single | no_tool (chosen via hydra config-name)
    loop = config.actor_rollout_ref.rollout.agent.default_agent_loop
    n = int(config.actor_rollout_ref.rollout.val_kwargs.n)
    print(
        f"[eval] mode={mode} loop={loop} model={config.model_path} "
        f"n={n} temp={config.actor_rollout_ref.rollout.val_kwargs.temperature}"
    )

    # ---- Data: reuse the verl val pipeline (create_rl_dataset + StatefulDataLoader) ----
    # Same loader as RayPPOTrainer._validate; filter_overlong_prompts=false so
    # the test set stays 1:1 (no rows dropped). RLHFDataset yields raw_prompt
    # (original messages) + reward_model (ground_truth) + extra_info (question).
    tokenizer = hf_tokenizer(config.model_path)
    processor = hf_processor(config.model_path)
    dataset = create_rl_dataset(
        list(config.data.val_files),
        config.data,
        tokenizer,
        processor,
        is_train=False,
        max_samples=int(config.max_samples),
    )
    if len(dataset) == 0:
        print("[eval] empty dataset; nothing to do", file=sys.stderr)
        sys.exit(1)
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=int(config.data.val_batch_size or len(dataset)),
        num_workers=int(config.data.dataloader_num_workers),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    print(f"[eval] dataset size={len(dataset)}, val_batch_size={dataloader.batch_size}, n={config.n}")

    # ---- Ray + vLLM agent loop workers (same env as the verl official tests) ----
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )
    manager = init_agent_loop_manager(config)

    # ---- Generate per batch ----
    total_samples = 0
    os.makedirs(config.output_dir, exist_ok=True)
    traj_path = Path(config.output_dir) / f"{mode}_trajectories.jsonl"
    traj_path.write_text("", encoding="utf-8")  # truncate: rows are appended per batch

    # metrics tracking (same Tracking backend stack as training; one run per mode)
    tracker = Tracking(
        project_name="rollout_eval",
        experiment_name=f"{mode}_n{n}_t{config.actor_rollout_ref.rollout.val_kwargs.temperature}",
        default_backend=["mlflow"],
        config=OmegaConf.to_container(config, resolve=True),
    )
    val_gen_logger = ValidationGenerationsLogger(
        project_name="rollout_eval",
        experiment_name=f"{mode}_n{n}_t{config.actor_rollout_ref.rollout.val_kwargs.temperature}",
    )
    sample_inputs, sample_outputs, sample_scores = [], [], []
    sample_uids, sample_turns, data_source_lst = [], [], []
    reward_extra_infos_dict = defaultdict(list)

    for batch_dict in dataloader:
        # 1. wrap: same as RayPPOTrainer._validate (DataProto.from_single_dict)
        test_batch = DataProto.from_single_dict(batch_dict)
        test_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
        )

        # hermes-clean prompts in-memory (training does it at parquet level via
        # clean_question=true; here the loader yields the raw messages)
        test_batch.non_tensor_batch["raw_prompt"] = np.array(
            [_clean_prompts_for_hermes(m) for m in test_batch.non_tensor_batch["raw_prompt"]], dtype=object
        )

        # 2. repeat (val_kwargs.n) and align per-sample metadata
        test_batch = test_batch.repeat(repeat_times=n, interleave=True)
        ground_truths = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch]
        data_sources = [item.non_tensor_batch.get("data_source", None) for item in test_batch]
        extra_infos = [item.non_tensor_batch.get("extra_info", None) for item in test_batch]

        # 3. gen batch + meta_info (same keys as RayPPOTrainer._validate)
        gen_batch = _get_gen_batch(test_batch)
        gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
            "global_steps": 0,
        }
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(
            gen_batch, int(config.actor_rollout_ref.rollout.agent.num_workers)
        )
        result = manager.generate_sequences(prompts=gen_batch_padded)
        result = unpad_dataproto(result, pad_size=pad_size)

        # 4. decode + union back (same as RayPPOTrainer._validate L660-666)
        output_texts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in result.batch["responses"]]
        sample_outputs.extend(output_texts)
        test_batch = test_batch.union(result)
        test_batch.meta_info["validate"] = True

        # Store original inputs (same as training: decode the rendered prompt tensor)
        input_ids = test_batch.batch["prompts"]
        input_texts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in input_ids]
        sample_inputs.extend(input_texts)
        sample_uids.extend(test_batch.non_tensor_batch["uid"])

        # 5. evaluate using reward_function (same as training: worker-side rm_scores)
        reward_tensor, reward_extra_info = extract_reward(test_batch)
        scores = reward_tensor.sum(-1).cpu().tolist()
        sample_scores.extend(scores)
        reward_extra_infos_dict["reward"].extend(scores)
        for key, values in reward_extra_info.items():
            if key not in reward_extra_infos_dict:
                reward_extra_infos_dict[key] = []
            if isinstance(values, np.ndarray):
                reward_extra_infos_dict[key].extend(values.tolist())
            else:
                reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

        # jsonl rows: rm_score (raw per-sample reward, not group advantage)
        # plus every reward component from reward_extra_info (EM/format/answer/...)
        num_turns = test_batch.non_tensor_batch["__num_turns__"]
        reward_component_keys = list(reward_extra_info.keys())
        batch_rows = []
        for i, text in enumerate(output_texts):
            extra = extra_infos[i] if isinstance(extra_infos[i], dict) else {}
            turns = int(num_turns[i])
            row = {
                "index": extra.get("index", i),
                "mode": mode,
                "question": extra.get("question"),
                "num_turns": turns,
                "ground_truth": (ground_truths[i] or {}).get("target") if ground_truths[i] else None,
                "rm_score": scores[i],
                "trajectory": text,
            }
            for key in reward_component_keys:
                val = reward_extra_info[key][i]
                row[key] = val.item() if hasattr(val, "item") else val
            batch_rows.append(row)
            total_samples += 1
        with open(traj_path, "a", encoding="utf-8") as f:
            for t in batch_rows:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        # collect num_turns / data_source (same as training L679-682)
        if "__num_turns__" in test_batch.non_tensor_batch:
            sample_turns.append(np.asarray(test_batch.non_tensor_batch["__num_turns__"]))
        data_source_lst.append(np.asarray(data_sources))

        print(f"[eval] batch done: {len(batch_rows)} samples")

    # ---- Summary ----
    if total_samples == 0:
        print("[eval] no samples generated", file=sys.stderr)
        sys.exit(1)

    # validation generation samples -> mlflow artifact (same as RayPPOTrainer._maybe_log_val_generations)
    if int(config.log_val_generations) > 0:
        samples = list(zip(sample_inputs, sample_outputs, sample_scores, strict=True))
        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[: int(config.log_val_generations)]
        val_gen_logger.log(["mlflow"], samples, step=0)

    # layered validation metrics (same as RayPPOTrainer._val_metrics_update -> logger.log)
    val_metrics = _val_metrics_update(
        np.concatenate(data_source_lst), sample_uids, reward_extra_infos_dict, sample_turns
    )
    tracker.log(data=val_metrics, step=0)
    tracker.finish()

    print(f"[eval] done: {total_samples} samples; val metrics (also -> mlflow {mode}):")
    print(json.dumps(val_metrics, indent=2, ensure_ascii=False))
    print(f"[eval] trajectories -> {traj_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
