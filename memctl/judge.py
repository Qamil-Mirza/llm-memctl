"""An LLM judge for QA answers, for benchmarks whose official metric is a judged one.

Optional: an environment uses it when its config has a `judge` model. Without
one, answers are scored locally (memctl/metrics.py).
"""

from __future__ import annotations

from memctl.llm import build_llm

PROMPT = (
    "You are grading an answer to a question about a past conversation.\n"
    "Question: {question}\nCorrect answer: {gold}\nAnswer to grade: {answer}\n\n"
    "Does the answer to grade state the same thing as the correct answer? "
    "Small differences in wording or format are fine. Reply with exactly one word: yes or no."
)

# LongMemEval's official per-type judge prompts, verbatim from src/evaluation/evaluate_qa.py
# (get_anscheck_prompt) of https://github.com/xiaowu0162/LongMemEval. Abstention questions are scored
# locally by refusal here, as for every benchmark, and reported apart.
_LME_HEAD = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
    "response contains the correct answer. Otherwise, answer no. "
)
_LME_STEPS = (
    "If the response is equivalent to the correct answer or contains all the intermediate steps to get the "
    "correct answer, you should also answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. "
)
_LME_TAIL = "\n\nQuestion: {question}\n\nCorrect Answer: {gold}\n\nModel Response: {answer}\n\nIs the model response correct? Answer yes or no only."
LONGMEMEVAL_PROMPTS = {
    "default": _LME_HEAD + _LME_STEPS + _LME_TAIL,
    "temporal-reasoning": _LME_HEAD + _LME_STEPS + (
        "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the "
        "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
        "the answer is 18), the model's response is still correct. "
    ) + _LME_TAIL,
    "knowledge-update": _LME_HEAD + (
        "If the response contains some previous information along with an updated answer, the response should "
        "be considered as correct as long as the updated answer is the required answer."
    ) + _LME_TAIL,
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response from a model. "
        "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does "
        "not need to reflect all the points in the rubric. The response is correct as long as it recalls and "
        "utilizes the user's personal information correctly.\n\nQuestion: {question}\n\nRubric: {gold}\n\n"
        "Model Response: {answer}\n\nIs the model response correct? Answer yes or no only."
    ),
}


class Judge:
    """`style: longmemeval` in the judge config uses the official per-type prompts; the default is PROMPT."""

    def __init__(self, config: dict, llm=None) -> None:
        self.style = config.get("style", "default")
        if self.style not in ("default", "longmemeval"):
            raise ValueError(f"unknown judge style {self.style!r}")
        self.llm = llm or build_llm({k: v for k, v in config.items() if k != "style"})
        self.name = self.llm.name
        self.calls = 0

    def prompt(self, question: str, gold: str, answer: str, category: str | None = None) -> str:
        if self.style == "longmemeval":
            template = LONGMEMEVAL_PROMPTS.get(category or "", LONGMEMEVAL_PROMPTS["default"])
        else:
            template = PROMPT
        return template.format(question=question, gold=gold, answer=answer)

    def is_correct(self, question: str, gold: str, answer: str, category: str | None = None) -> bool:
        self.calls += 1
        reply = self.llm.generate(self.prompt(question, gold, answer, category), max_new_tokens=4)
        return reply.strip().lower().startswith("yes")
