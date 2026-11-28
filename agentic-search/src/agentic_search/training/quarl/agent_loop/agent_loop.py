import asyncio
from typing import Any
from uuid import uuid4

import numpy as np
import ray
import ray.util.scheduling_strategies
from omegaconf import DictConfig

from verl.experimental.agent_loop import AgentLoopManager, AgentLoopWorker, get_trajectory_info
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import RolloutTraceConfig
from verl.utils.skip import SkipManager
from verl.workers.rollout.llm_server import LLMServerClient




class SearchAgentloopWorker(AgentLoopWorker):
    """Agent Loop Worker that manages the agent loop."""

    
    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
        role: str = "solver",
    ):
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)
        self.role = role
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
            stop=config.custom[self.role]["stop"],
            no_stop_trim=config.custom[self.role]["no_stop_trim"],
        )

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # NOTE: __do_sample__ is an internal per-sample override used by REMAX combined rollout.
        # Do not forward it to concrete agent loops, which may reject unknown kwargs.
        per_sample_do_sample = batch.non_tensor_batch.get("__do_sample__")
        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(
            outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
        return output


class SearchAgentloopManager(AgentLoopManager):
    """Search Agent Loop Manager that manages the search agent loop."""
    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.agent_loop_workers_class = ray.remote(SearchAgentloopWorker)
        super().__init__(*args, **kwargs)


class SearchSPAgentloopManager(AgentLoopManager):
    """Search Agent Self-Play Loop Manager that manages proposer and solver agents.
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
       self.agent_loop_workers_class = ray.remote(SearchAgentloopWorker)
       super().__init__(*args, **kwargs)


    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.solver_agent_loop_workers = []
        self.proposer_agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.solver_agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                    "solver"
                )
            )
            self.proposer_agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                    "proposer"
                )
            )

    @auto_await
    @SkipManager.annotate(role="proposer")
    async def generate_proposer_sequences(self, prompts: DataProto) -> DataProto:
        """Generate proposer trajectories (question construction phase).

        Both phases share the same ``llm_client`` / ``teacher_client`` (one actor,
        one sglang service); this entry only picks the proposer worker group.
        Timing keys are prefixed so the two phases do not overwrite each other
        when ``timing_raw.update(...)`` is called in fit.
        """
        workers = self.proposer_agent_loop_workers
        if not workers:
            raise RuntimeError("No proposer agent loop workers available; call create() first.")

        # The base worker rebuilds meta_info, so re-attach the phase tag to the output.
        prompts.meta_info["role"] = "proposer"

        # Assign globally-unique priorities before chunking so chunks own disjoint ranges.
        if "priority" not in prompts.non_tensor_batch:
            prompts.non_tensor_batch["priority"] = np.arange(len(prompts), dtype=np.int64)

        chunkes = prompts.chunk(len(workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        metrics = [chunk_output.meta_info.pop("metrics") for chunk_output in outputs]
        timing = self._performance_metrics(metrics, output)
        timing = {f"proposer/{key}": value for key, value in timing.items()}

        output.meta_info = {**outputs[0].meta_info, "timing": timing, "role": "proposer"}
        return output

    @auto_await
    @SkipManager.annotate(role="solver")
    async def generate_solver_sequences(self, prompts: DataProto) -> DataProto:
        """Generate solver trajectories (question answering phase)."""
        workers = self.solver_agent_loop_workers
        if not workers:
            raise RuntimeError("No solver agent loop workers available; call create() first.")

        # The base worker rebuilds meta_info, so re-attach the phase tag to the output.
        prompts.meta_info["role"] = "solver"

        # Assign globally-unique priorities before chunking so chunks own disjoint ranges.
        if "priority" not in prompts.non_tensor_batch:
            prompts.non_tensor_batch["priority"] = np.arange(len(prompts), dtype=np.int64)

        chunkes = prompts.chunk(len(workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        metrics = [chunk_output.meta_info.pop("metrics") for chunk_output in outputs]
        timing = self._performance_metrics(metrics, output)
        timing = {f"solver/{key}": value for key, value in timing.items()}

        output.meta_info = {**outputs[0].meta_info, "timing": timing, "role": "solver"}
        return output
