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


class Judge:
    def __init__(self, config: dict, llm=None) -> None:
        self.llm = llm or build_llm(config)
        self.name = self.llm.name
        self.calls = 0

    def is_correct(self, question: str, gold: str, answer: str) -> bool:
        self.calls += 1
        reply = self.llm.generate(PROMPT.format(question=question, gold=gold, answer=answer), max_new_tokens=4)
        return reply.strip().lower().startswith("yes")
