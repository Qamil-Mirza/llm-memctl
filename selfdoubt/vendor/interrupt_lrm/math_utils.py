# Vendored from https://github.com/dynamic-lm/interrupt-lrm at commit b87d61bd2e5e998f34878fb0ee5fb2f5550f98c6
# (eval/math/utils.py), Copyright the interrupt-lrm authors (Wu, Miroyan et al.), Apache License 2.0 (see LICENSE here).
# Function bodies below the VERBATIM marker are copied unchanged (checked by selfdoubt/tests/test_stage0.py).
# Only the functions needed for Qwen3 math runs are kept; nothing else is modified.
# The original also builds a global ProcessPoolExecutor at import and a 5 s timeout wrapper
# (score_response_with_timeout); selfdoubt/scoring.py re-implements that wrapper without the import side effect.
from .math_parsing_util import strip_answer_string, extract_answer, math_equal
import re

# ---- VERBATIM below ----
def extract_gsm8k_gt_answer(ground_truth):
    gt_re = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = gt_re.search(ground_truth)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return "[invalid]"


def extract_aime2024_gt_answer(ground_truth):
    gt_re = re.compile(r"\\boxed{(.*)}")
    match = gt_re.search(ground_truth)
    if match:
        return match.group(1).strip()
    else:
        return "[invalid]"


def sanitize_answer(answer):
    ans_re = re.compile(r"((-?[$0-9.,]{2,})|(-?[0-9]+))")
    patterns_to_remove = [
        ",",  # Remove commas
        r"\$",  # Remove dollar signs
        r"\.$" r"\*",  # Remove trailing period  # Remove asterisks
    ]
    for pattern in patterns_to_remove:
        answer = re.sub(pattern, "", answer)

    matches = ans_re.findall(answer)
    if matches:
        # get the last match (i.e final response) and the first / outer capturing group
        match_str = matches[-1][0].strip()
        return match_str
    else:
        return "[invalid]"


def score_response(task, ground_truth, prediction):
    if task == "gsm8k":
        gt_answer = extract_gsm8k_gt_answer(ground_truth)
        pred = extract_answer(prediction)
        pred = sanitize_answer(pred)
        return pred == gt_answer
    elif task == "aime2024":
        gt_answer = extract_aime2024_gt_answer(ground_truth)
        gt_answer = strip_answer_string(gt_answer)
        pred = extract_answer(prediction)
        pred = sanitize_answer(pred)
        return pred == gt_answer
    else:
        answer = strip_answer_string(ground_truth)
        pred = extract_answer(prediction)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)
