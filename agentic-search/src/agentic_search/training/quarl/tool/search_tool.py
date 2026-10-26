import json
import logging
import os
import threading
from contextlib import ExitStack
from typing import Any, Callable, Optional, Tuple, TypeVar
from uuid import uuid4

import ray
import ray.actor
from verl.tools.base_tool import BaseTool, ToolResponse
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op

from quarl.tool.search_r1_like_utils import perform_single_search_batch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class TokenBucketWorker:
    """Simple token-bucket rate limiter running as a Ray actor."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class SearchExecutionWorker:
    """Ray actor that executes search functions with global rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.remote(rate_limit)

    def ping(self):
        return 0

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                return ray.get(fn.remote(*fn_args, **fn_kwargs))
        else:
            return ray.get(fn.remote(*fn_args, **fn_kwargs))


def init_search_execution_pool(num_workers: int = 120, enable_global_rate_limit: bool = True, rate_limit: int = 10):
    """Initialize a Ray execution pool for search operations.

    Args:
        num_workers: Number of workers in the pool.
        enable_global_rate_limit: Enable global rate limiting.
        rate_limit: Rate limit for the global rate limiter.

    Returns:
        Ray actor handle to the execution pool.
    """
    from ray.util import ActorPool

    workers = [
        SearchExecutionWorker.options(num_cpus=0, max_concurrency=num_workers).remote(
            enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit
        )
        for _ in range(num_workers)
    ]
    return ActorPool(workers)


class QuarkSearchTool(BaseTool):
    """Search tool for retrieval-augmented RL (migrated from verl.tools.search_tool
    which was removed in verl 0.9.0; now a direct BaseTool subclass).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
        )

        # Retrieval service configuration (flat URL string, resolved from
        # ${oc.env:SEARCH_IP}/${oc.env:SEARCH_PORT} interpolation in the tool yaml)
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        self.topk = config.get("topk", 3)

        logger.info(f"Initialized QuarkSearchTool with config: {config}")

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance for a trajectory."""
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    def execute_search(self, instance_id: str, query_list: list, retrieval_service_url: str, topk: int, timeout: int):
        """Execute search operation using retrieval service."""
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_text, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        query_list_from_params = parameters.get("query_list")

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        # Execute search using Ray execution pool
        try:
            result_text, metadata = await self.execution_pool.execute.remote(
                self.execute_search,
                instance_id,
                query_list_from_params,
                self.retrieval_service_url,
                self.topk,
                self.timeout,
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
            }

            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> list:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
