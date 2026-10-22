from quarl.reward.score.search_r1_like_qa_em import search_r1_like_qa_em
from quarl.reward.score.process_score import (
    compute_think_length_penalty,
    extract_think_text,
    has_broken_format,
    has_format_error,
    is_overlength_rollout,
)


def compute_score(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


def compute_score_batch(
    batch_data,
    score_coef: float = 1.0,
    think_only: bool = True,
    max_search_calls: int = 0,
    max_response_tokens: int = 0,
    think_length_penalty_beta: float = 0.0,
):
    """Compute Search-R1 answer reward with format and length controls.

    Unlike ``process_score``, this function has no KG-node component. A valid
    exact-match answer receives one point; invalid or incorrect answers receive
    zero, optionally scaled by the group-normalized think-length penalty.
    """
    results = []
    group_lengths = {}
    for batch_idx, data in enumerate(batch_data):
        solution_str = data["response_str"]
        ground_truth = data["ground_truth"]
        think_text = extract_think_text(solution_str) if think_only else solution_str
        uid = str(data.get("uid") or data.get("extra_info", {}).get("index", batch_idx))
        group_lengths.setdefault(uid, []).append(len(think_text))

    for batch_idx, data in enumerate(batch_data):
        solution_str = data["response_str"]
        ground_truth = data["ground_truth"]
        think_text = extract_think_text(solution_str) if think_only else solution_str
        uid = str(data.get("uid") or data.get("extra_info", {}).get("index", batch_idx))
        format_error = has_broken_format(solution_str) or has_format_error(solution_str)
        overlength = is_overlength_rollout(
            solution_str,
            valid_response_length=int(data.get("valid_response_length", 0)),
            max_search_calls=max_search_calls,
            max_response_tokens=max_response_tokens,
        )
        answer = compute_score(
            data_source=data.get("data_source", ""),
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=data.get("extra_info", {}),
        )
        base_score = 1.0 if answer > 0 and not format_error and not overlength else 0.0
        length_factor, length_normalized = compute_think_length_penalty(
            len(think_text), group_lengths[uid], think_length_penalty_beta
        )
        results.append(
            {
                "score": float(base_score * length_factor * score_coef),
                "idx": data["idx"],
                "answer_correct": bool(answer > 0),
                "format_error": format_error,
                "overlength": overlength,
                "think_length": len(think_text),
                "think_length_normalized": float(length_normalized),
                "think_length_penalty_factor": float(length_factor),
            }
        )
    return results
