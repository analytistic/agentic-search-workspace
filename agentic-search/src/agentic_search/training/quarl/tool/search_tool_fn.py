"""Search tool for verl 0.9.x, following the official ``@function_tool`` pattern.

Registered via ``rollout.multi_turn.function_tool_path``; the OpenAI tool
schema is inferred from the signature + Google-style docstring by
``transformers.utils.get_json_schema``. The function is stateless: each call
hits the retrieval service (with retry logic) and returns the raw result
JSON string (``{"result": "Doc 1 (Title: ...)..."}``).
"""

import json
import logging
import os
import uuid
from typing import Any, Optional

import requests
from verl.tools.function_tool import function_tool

# Importing this registers the "search" tool parser (legacy <search>query</search>
# tag format) with verl's ToolParser registry — this file is executed by
# load_function_tools_from_path in AgentLoopWorker.__init__, before
# ToolAgentLoop picks the parser.
from quarl.agent_loop.tool_parser import SearchToolParser  # noqa: F401

# 3 s is right when the retriever answers in milliseconds, which is the case on
# GPU-backed BM25/dense services. On a CPU-only faiss index (e.g. an eval box with
# no faiss-gpu) the server serialises requests, so a 32-way rollout queues for
# seconds per call and every search times out -- the agent then answers from
# memory and the trajectory contains no <search> at all. Override per host.
TIMEOUT = float(os.environ.get("SEARCH_TIMEOUT", "3"))
MAX_RETRIES = int(os.environ.get("SEARCH_MAX_RETRIES", "2"))  # initial attempt + 1 retry
INITIAL_RETRY_DELAY = 1
TOPK = 3

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Retrieval service is co-located with the trainer (single-node launcher);
# hard-code the loopback URL so a stray SEARCH_IP env override (the server
# shell exports one) cannot reroute requests to an unreachable host.
RETRIEVAL_URL = "http://127.0.0.1:8000/retrieve"


def _retrieval_url() -> str:
    return RETRIEVAL_URL


def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = TIMEOUT,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Call the remote search API with retry logic.

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        return_scores: Whether to return scores.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (response_json, error_message). On success response_json is
        the API's returned JSON object and error_message is None; after all
        retries are exhausted response_json is None and error_message holds
        the last error.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    # 先申请返回变量，后续所有分支只改它
    result = None

    # 格式校验：请求是否符合 /retrieve 的期望格式，不符合直接返回格式错误
    if not isinstance(query_list, list) or len(query_list) == 0:
        return None, "Search error: query_list must be a non-empty list"
    if not all(isinstance(q, str) and q.strip() for q in query_list):
        return None, "Search error: all queries must be non-empty strings"
    if not isinstance(topk, int) or topk <= 0:
        return None, f"Search error: topk must be a positive integer, got {topk!r}"

    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
            )
            response = requests.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
                # The retrieval service is local (SEARCH_IP:SEARCH_PORT); never
                # route it through the environment's proxy, which cannot reach
                # 127.0.0.1 and hangs the request.
                proxies={"http": None, "https": None},
            )

            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                continue

            response.raise_for_status()
            logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")
            result = response.json()
            return result, None

        except Exception as e:
            last_error = f"{log_prefix}{type(e).__name__}: {str(e)[:200]}"
            logger.warning(last_error)
            continue

    # 失败：error 字符串 200 截断
    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return result, (last_error or "API Call Failed after retries").replace(log_prefix, "")[:200]


def _passages2string(retrieval_result) -> str:
    """Convert retrieval results to formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


def perform_single_search_batch(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    timeout: int = TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    """Perform a single batch search for multiple queries.

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (result_text, metadata). result_text is a JSON string with
        the formatted passages under the "result" key; metadata carries the
        query count, status and totals for the batch.
    """
    logger.info(f"Starting batch search for {len(query_list)} queries.")

    api_response = None
    error_msg = None

    api_response, error_msg = call_search_api(
        retrieval_service_url=retrieval_service_url,
        query_list=query_list,
        topk=topk,
        return_scores=True,
        timeout=timeout,
    )

    metadata = {
        "query_count": len(query_list),
        "queries": query_list,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "total_results": 0,
        "formatted_result": None,
    }

    result_text = json.dumps({"result": "Search request failed or timed out after retries."})

    if error_msg:
        metadata["status"] = "api_error"
        # error_msg carries the full exception repr (nested quotes/parens,
        # chained "(Caused by ...)" tail). Strip the repr shell so the model
        # gets a clean one-liner; the full error stays in logger.error below.
        text = str(error_msg) if error_msg else "unknown error"
        if " (Caused by " in text:
            text = text.split(" (Caused by ")[0]
        first_line = text.splitlines()[0] if text else "unknown error"
        result_text = json.dumps({"result": f"Search error: {first_line[:200]}"})
        logger.error(f"Batch search: API error occurred: {error_msg}")
    elif api_response:
        logger.debug(f"Batch search: API Response: {api_response}")
        metadata["api_response"] = api_response

        try:
            raw_results = api_response.get("result", [])
            if raw_results:
                pretty_results = []
                total_results = 0

                # 每个 query 的结果前标注它自己的检索词（服务端按 query_list 顺序返回）
                for query, retrieval in zip(query_list, raw_results):
                    formatted = f"Query: {query}\n{_passages2string(retrieval)}"
                    pretty_results.append(formatted)
                    total_results += len(retrieval) if isinstance(retrieval, list) else 1

                final_result = "\n---\n".join(pretty_results)
                result_text = json.dumps({"result": final_result})
                metadata["status"] = "success"
                metadata["total_results"] = total_results
                metadata["formatted_result"] = final_result
                logger.info(f"Batch search: Successful, got {total_results} total results")
            else:
                result_text = json.dumps({"result": "No search results found."})
                metadata["status"] = "no_results"
                metadata["total_results"] = 0
                logger.info("Batch search: No results found")
        except Exception as e:
            error_msg = f"Error processing search results: {e}"
            result_text = json.dumps({"result": error_msg})
            metadata["status"] = "processing_error"
            logger.error(f"Batch search: {error_msg}")
    else:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps({"result": "Unknown API state (no response and no error message)."})
        logger.error("Batch search: Unknown API state.")

    return result_text, metadata


@function_tool
def search(query_list: list[str]) -> str:
    """Searches the web for relevant information based on the given queries.

    Args:
        query_list: A list of fully-formed semantic queries. The tool will return search results for each query.
    """
    result_text, metadata = perform_single_search_batch(
        retrieval_service_url=_retrieval_url(),
        query_list=query_list,
        topk=TOPK,
        timeout=TIMEOUT,
    )
    # Protocol tags are the template's job: search_r1.jinja wraps tool
    # messages in <information>...</information>, qwen*_hermes.jinja in
    # <tool_response>...</tool_response>. Return the plain text result.
    formatted = metadata.get("formatted_result") or result_text
    return formatted
