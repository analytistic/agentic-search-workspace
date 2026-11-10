"""Search-R1 scoring entry point, config-injected into SearchR1RewardManager.

Layered combination (assembled here):
    score = format score + answer score
      format: ``qa_format_compute_score`` -- structure valid -> structure_format_score,
              broken -> final_format_score
      answer: EM hit -> 1.0; EM miss -> LLM judge semantic check -> 1.0 / 0.0

The protocol format check is selected via config
(``custom_reward_function.reward_kwargs.format_check``: "search_r1" | "qwen3" | "qwen3_5").
"""

from quarl.reward.score.utils import (
    MATH_DATA_SOURCES,
    em_check,
    extract_math_solution,
    extract_solution,
    is_valid_sequence_qwen3,
    is_valid_sequence_qwen3_5,
    is_valid_sequence_search_r1,
    qa_format_compute_score,
    qa_judge_compute_score,
)

_FORMAT_CHECKS = {
    "search_r1": is_valid_sequence_search_r1,
    "qwen3": is_valid_sequence_qwen3,
    "qwen3_5": is_valid_sequence_qwen3_5,
}


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    validate=False,
    format_check="search_r1",
):
    """Search-R1 layered score: format compliance + answer correctness.

    Args:
        data_source: data source (math datasets use \\boxed extraction)
        solution_str: the decoded trajectory text
        ground_truth: {"target": [...]}
        extra_info: per-sample metadata (question etc.)
        validate: validation-mode flag (kept for the manager contract)
        format_check: protocol name for the structure check, from config
            (custom_reward_function.reward_kwargs.format_check)

    Returns:
        dict with "score" (format + answer) plus components for
        reward_extra_info (EM / format / answer).
    """
    check = _FORMAT_CHECKS.get(format_check, is_valid_sequence_search_r1)

    # 1. format score: structure compliance only
    fmt = qa_format_compute_score(solution_str, format_check=check)

    # 2. answer score: EM first, LLM judge for the misses
    if data_source in MATH_DATA_SOURCES:
        answer = extract_math_solution(solution_str=solution_str)
    else:
        answer = extract_solution(solution_str=solution_str)

    em = 0.0
    if answer is None:
        ans = 0.0
    elif em_check(answer, ground_truth["target"]):
        em = 1.0
        ans = 1.0
    else:
        question = (extra_info or {}).get("question")
        ans = qa_judge_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            prompt_str=question,
            extra_info=extra_info,
        )

    return {
        "score": float(fmt + ans),
        "EM": float(em),
        "format": float(fmt),
        "answer": float(ans),
    }
