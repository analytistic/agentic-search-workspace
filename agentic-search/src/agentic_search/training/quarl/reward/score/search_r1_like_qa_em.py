"""Search-R1 style exact-match scorer.

Vendored into quarl for the verl 0.9.0 migration: the original home of
``search_r1_like_qa_em`` (``verl.utils.reward_score``) was never present in
any verl release, so the module is implemented here with the same interface
(``compute_score(solution_str, ground_truth)`` -> scalar).
"""

import re

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer(solution_str: str) -> str:
    """Take the content of the last <answer>..</answer> block."""
    matches = _ANSWER_RE.findall(solution_str)
    return (matches[-1] if matches else solution_str).strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip('"').strip("'")


def _targets(ground_truth) -> list[str]:
    """Normalize ground truth into a list of candidate answers."""
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target", "")
    else:
        target = ground_truth
    if isinstance(target, (list, tuple)):
        return [str(t) for t in target]
    return [str(target)]


def compute_score(solution_str, ground_truth) -> float:
    """Exact-match score: 1.0 if the extracted answer matches any candidate."""
    answer = _normalize(_extract_answer(solution_str))
    return 1.0 if any(answer == _normalize(t) for t in _targets(ground_truth)) else 0.0
