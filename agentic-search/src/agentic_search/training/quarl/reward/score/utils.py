import logging
import os
import random
import re
import string
import time

logger = logging.getLogger(__name__)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


MATH_DATA_SOURCES = {
    "Algebra",
    "Intermediate Algebra",
    "Prealgebra",
    "Number Theory",
    "Geometry",
    "Precalculus",
    "Counting & Probability",
}


def extract_math_solution(solution_str):
    """Extract the answer from the solution string within <answer></answer> tags,
    returning boxed content if present, otherwise the raw content.
    If no <answer> tags are present, extract and return the last boxed content."""
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if len(answer_matches) < 1:
        boxed_pattern = r"\\boxed\{(.*?)\}"
        boxed_matches = list(re.finditer(boxed_pattern, solution_str, re.DOTALL))
        if boxed_matches:
            return boxed_matches[-1].group(1).strip()
        else:
            return None

    answer_content = answer_matches[-1].group(1).strip()

    boxed_pattern = r"\\boxed\{(.*?)\}"
    boxed_match = re.search(boxed_pattern, answer_content)

    if boxed_match:
        return boxed_match.group(1).strip()
    else:
        return answer_content


def qa_em_compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            return format_score


def qa_subem_compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score


# ---- format checks (adapted from Search-R1 qa_em_format.py) ----


def is_valid_sequence_search_r1(text):
    """Search-R1 protocol structural check (think -> search -> information ->
    ... -> answer, balanced tags, only whitespace between tags).

    Returns:
        (is_valid: bool, reason: str)
    """
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)

    if not assistant_match:
        return False, "Missing assistant marker"

    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]

    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # Now check for proper sequence pattern and no extraneous content

    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end

    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue

        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


# ---- hermes protocol checks (qwen3 / qwen3.5 tool-calling trajectories) ----

_HERMES_SPLIT_RE = re.compile(
    r"(</?(?:think|tool_call|tool_response|answer)>|<\|im_start\|>(?:assistant|user)|<\|im_end\|>)"
)


def _check_hermes_sequence(text: str, tool_call_ok) -> tuple[bool, str]:
    """Shared state machine for qwen3 / qwen3.5 hermes trajectories:

        <|im_start|>assistant -> think -> tool_call -> <|im_end|>
          -> <|im_start|>user -> tool_response -> <|im_end|>
          -> <|im_start|>assistant (loop)
          -> ... -> think -> answer -> <|im_end|> -> end

    ``tool_call_ok`` validates the tool_call content (JSON for qwen3,
    <function=..> for qwen3.5).

    Returns:
        (is_valid: bool, reason: str)
    """
    parts = re.split(_HERMES_SPLIT_RE, text)
    state = "start"
    tool_call_content = []

    def bad(reason):
        return False, reason

    for part in parts:
        if not part.strip():
            continue
        if re.fullmatch(r"</?(?:think|tool_call|tool_response|answer)>", part):
            if part == "<think>" and state in ("in_assistant", "after_tool_response"):
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<tool_call>" and state == "after_think":
                state = "in_tool_call"
                tool_call_content = []
            elif part == "</tool_call>" and state == "in_tool_call":
                if not tool_call_ok("\n".join(tool_call_content).strip()):
                    return bad(f"bad tool_call content: {' '.join(tool_call_content)[:80]}")
                state = "after_tool_call"
            elif part == "<tool_response>" and state == "in_user":
                state = "in_tool_response"
            elif part == "</tool_response>" and state == "in_tool_response":
                state = "after_tool_response"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "after_answer"
            else:
                return bad(f"Unexpected tag {part} in state {state}")
        elif re.fullmatch(r"<\|im_start\|>(?:assistant|user)", part):
            if part == "<|im_start|>assistant" and state in ("start", "after_tool_response", "after_tool_response_end"):
                state = "in_assistant"
            elif part == "<|im_start|>user" and state in ("after_tool_call", "after_tool_call_end"):
                state = "in_user"
            else:
                return bad(f"Unexpected marker {part} in state {state}")
        elif part == "<|im_end|>":
            if state == "after_tool_call":
                state = "after_tool_call_end"
            elif state == "after_tool_response":
                state = "after_tool_response_end"
            elif state == "after_answer":
                state = "end"
            else:
                return bad(f"Unexpected <|im_end|> in state {state}")
        else:
            # plain content
            if state in ("in_think", "in_tool_call", "in_tool_response", "in_answer"):
                if state == "in_tool_call":
                    tool_call_content.append(part)
                continue  # content allowed inside tags
            return bad(f"Unexpected content '{part.strip()}' in state {state}")

    if state != "end":
        return bad(f"Incomplete sequence, ended in state {state}")
    return True, "Valid sequence format"


def is_valid_sequence_qwen3(text):
    """Qwen3 hermes protocol check: tool_call bodies are JSON
    (``{"name": ..., "arguments": ...}``)."""
    return _check_hermes_sequence(
        text,
        tool_call_ok=lambda content: bool(content) and ("{" in content and '"name"' in content),
    )


def is_valid_sequence_qwen3_5(text):
    """Qwen3.5 hermes protocol check: tool_call bodies are XML
    (``<function=...>``)."""
    return _check_hermes_sequence(
        text,
        tool_call_ok=lambda content: bool(content) and "<function=" in content,
    )


_ASSISTANT_PREFIX = "<|im_start|>assistant"


def ensure_assistant_prefix(text: str) -> str:
    """Put the assistant marker back in front of a response-only slice.

    The reward path decodes the `responses` tensor, and a response starts at
    the first *generated* token -- i.e. immediately after
    ``<|im_start|>assistant\\n``. That marker belongs to the prompt for turn 1
    and to the rendered observation for later turns, so it is absent from the
    string the format checks receive. Both checks are driven by it
    (``is_valid_sequence_search_r1`` looks for it explicitly, and
    ``_check_hermes_sequence`` cannot leave its initial ``start`` state without
    it), so without this every well-formed trajectory is reported as broken.

    Idempotent: if the text already opens with the marker (e.g. a full
    conversation was passed in) it is returned unchanged.
    """
    body = text.lstrip()
    if body.startswith(_ASSISTANT_PREFIX):
        return body
    return f"{_ASSISTANT_PREFIX}\n{body}"


def qa_format_compute_score(
    solution_str,
    structure_format_score=0.2,
    final_format_score=0.1,
    format_check=None,
):
    """Format-only scoring: run the configured format check and return the
    format score.

    The layered combination (EM score / retrieval score / format weights)
    is assembled by the caller; this function only judges format compliance.

    Args:
        solution_str: the solution text
        structure_format_score: score when the structure is valid (Search-R1
            paper default 0.2)
        final_format_score: score when the structure is broken (paper 0.1)
        format_check: (text) -> (bool, reason) callable, selected from the
            config by the caller: is_valid_sequence_search_r1 /
            is_valid_sequence_qwen3 / is_valid_sequence_qwen3_5.
            Defaults to the Search-R1 protocol check.

    Returns:
        structure_format_score if the trajectory follows the expected
        structure, else final_format_score.
    """
    check = format_check or is_valid_sequence_search_r1
    is_valid_format, reason = check(ensure_assistant_prefix(solution_str))
    if is_valid_format:
        return structure_format_score
    logger.debug("[format] broken: %s", reason)
    return final_format_score


def qa_judge_compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    prompt_str=None,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    """Pure LLM-judge scoring (inline judge client).

    Extract the answer and let the judge (OpenAI-compatible chat endpoint at
    JUDGE_BASE_URL) decide semantic equivalence with the golden answer -- no
    EM pre-check. Answer-tag spam (>10 tags) quarters the score.

    Requires JUDGE_BASE_URL / JUDGE_MODEL (raises otherwise). Retries on
    transient errors with backoff; 429 waits exponentially.
    """
    import requests

    base_url = os.getenv("JUDGE_BASE_URL")
    api_key = os.getenv("JUDGE_API_KEY", "dummy_api_key")
    model = os.getenv("JUDGE_MODEL")
    if not base_url or not model:
        raise ValueError("JUDGE_BASE_URL or JUDGE_MODEL is not set")

    if data_source is not None and data_source in MATH_DATA_SOURCES:
        answer = extract_math_solution(solution_str=solution_str)
    else:
        answer = extract_solution(solution_str=solution_str)

    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if answer is None:
        print(f"[qa_judge]: No answer extracted, data_source: {data_source}")
        return 0.0

    question = ""
    if prompt_str:
        if "Question: " in prompt_str:
            question = prompt_str.split("Question: ", 1)[1]
        elif "<｜User｜>" in prompt_str:
            question = prompt_str.replace("<｜User｜>", "")
        else:
            question = prompt_str

    prompt = f"""
Please determine whether the model's answer is consistent with the reference answer:

Question: {question}
Model Answer: {answer}
Reference Answer: {ground_truth['target']}

Evaluation Criteria:
1. The model answer must accurately respond to the question and be consistent with the reference answer in meaning.
2. For numerical questions, the values must be equal or very close.
3. For textual questions, the core meaning must be correct.
4. Differences in wording or language are allowed as long as the core answer is the same.
5. If the model answer includes the correct answer and does not contain conflicting information, it is also considered correct.

Please respond only with "Correct" or "Wrong". Do not provide any additional explanation.
"""
    messages = [
        {"role": "system", "content": "You are a professional judge who evaluates the correctness of answers."},
        {"role": "user", "content": prompt},
    ]
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 2048, "stream": False}

    for attempt in range(5):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30.0)
            if response.status_code == 429:
                time.sleep(min(2 ** attempt * 2, 60))  # exponential backoff
                continue
            response.raise_for_status()
            data = response.json()
            if "choices" in data:
                message = data["choices"][0]["message"]
            else:
                # volc/volcengine compatible endpoints return the message at
                # the top level: {"id", "role", "model", "content", ...}
                message = data
            content = message.get("content") or ""
            # volc/volcengine compatible endpoints may return content as a
            # list of segments: [{"type": "thinking", ...}, {"type": "text", "text": ...}]
            if isinstance(content, list):
                text_parts = [
                    seg.get("text", "")
                    for seg in content
                    if isinstance(seg, dict) and seg.get("type") == "text"
                ]
                content = "".join(text_parts) or ""
            # qwen3 models may put the answer in reasoning_content
            if not content and "reasoning_content" in message:
                reasoning = message.get("reasoning_content", "")
                if "Answer:" in reasoning:
                    content = reasoning.split("Answer:")[-1].strip()
                elif "answer:" in reasoning:
                    content = reasoning.split("answer:")[-1].strip()
                else:
                    content = reasoning.strip().split("\n")[-1].strip()
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>")[-1].strip()

            matched = "correct" in content.lower()
            if matched and (open_count > 10 or close_count > 10):  # answer-tag spam
                return 0.25
            return 1.0 if matched else 0.0
        except Exception:
            time.sleep(1)
    return 0.0
