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
import time
import traceback
import uuid
from typing import Any, Optional

import requests
from verl.tools.function_tool import function_tool

# Importing this registers the "search" tool parser (legacy <search>query</search>
# tag format) with verl's ToolParser registry — this file is executed by
# load_function_tools_from_path in AgentLoopWorker.__init__, before
# ToolAgentLoop picks the parser.
from quarl.agent_loop.tool_parser import SearchToolParser  # noqa: F401

DEFAULT_TIMEOUT = 2  # Default search request timeout
MAX_RETRIES = 2  # initial attempt + 1 retry (worst case ~5s, fail fast instead of hanging the trajectory)
INITIAL_RETRY_DELAY = 1
TOPK = 3
TIMEOUT = 5000

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
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Call the remote search API with retry logic and increasing delays.

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
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            response.raise_for_status()
            logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break

    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


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
    timeout: int = DEFAULT_TIMEOUT,
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

    try:
        api_response, error_msg = call_search_api(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            return_scores=True,
            timeout=timeout,
        )
    except Exception as e:
        error_msg = f"API Request Exception during batch search: {e}"
        logger.error(f"Batch search: {error_msg}")
        traceback.print_exc()

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
        # error_msg carries the full exception repr (nested quotes/parens like
        # NewConnectionError("... ')))"}")); pass the model only the first
        # line, truncated, so the injected <information> text stays readable.
        first_line = str(error_msg).splitlines()[0] if error_msg else "unknown error"
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

                for retrieval in raw_results:
                    formatted = _passages2string(retrieval)
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
    # Parser may degrade an array parameter to a plain string when the model
    # skips the list syntax (e.g. literal_eval fails); normalize to a list.
    if isinstance(query_list, str):
        query_list = [query_list]

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
