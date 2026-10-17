import json
import logging
import os
from typing import Any, Callable, Optional, Tuple, TypeVar

from verl.tools.search_tool import SearchTool as _BaseSearchTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

USE_WIKI = [
    "searchR1_nq",
    "searchR1_triviaqa",
    "searchR1_popqa",
    "searchR1_hotpotqa",
    "searchR1_2wikimultihopqa",
    "searchR1_musique",
    "searchR1_bamboogle",
    "ssp_kg_selfplay_en",
    "self_generated",
    "ssp_kg_random_v1",
    "ssp_kg_llm_v1",
    "ssp_kg_llm_cl_v1",
    "ssp_kg_random_v3",
    "ssp_kg_random_v4",
    "ssp_kg_random_v4p2",
]


class SSPSearchTool(_BaseSearchTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        timeout = self.timeout
        query_list_from_params = parameters.get("query_list")
        datasource = parameters.get("data_source", None)
        # print("[DebugSearchToolDataSource] data_source:", datasource)

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return json.dumps({"result": error_msg}), 0.0, {}

        # Execute search using Ray execution pool
        try:
            # Always use wiki URL regardless of data_source to avoid routing
            # to the default (0.0.0.0:8000) which may not have a running retriever.
            retrieval_service_url_exe = self.retrieval_service_url.get("wiki")

            result_text, metadata = await self.execution_pool.execute.remote(
                self.execute_search, instance_id, query_list_from_params, retrieval_service_url_exe, self.topk, timeout
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

            return result_text, 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return error_result, 0.0, {"error": str(e)}
