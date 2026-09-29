"""The LLM judge: asks a model whether an answer matches the gold answer.

The judge model is set in the config (`judge:`) and its outputs are cached on
disk like every other generation. A small local judge makes mistakes, so F1 and
BLEU-1 are always reported next to it.
"""

from __future__ import annotations

from memctl.llm import LLM
from memctl.metrics import is_refusal

JUDGE_PROMPT = """You are a strict grader. Decide whether the student's answer states the same fact as the reference answer.

Rules:
- CORRECT: the student's answer contains the key fact of the reference answer (same person, thing, place, number or date). Different wording or extra words are fine.
- WRONG: the key fact is missing or different, the answer is vague, the answer is only an id such as [D3:4], or the answer is a copied message that does not state the fact.
- For dates and times, the student's answer must point to the same day, month or year as the reference.

Examples:
Question: What pet does Sam have? | Reference: a beagle | Student: He has a beagle named Max. -> CORRECT
Question: When did Ana move? | Reference: June 2022 | Student: [D4:2] -> WRONG
Question: When did Ana move? | Reference: June 2022 | Student: (Ana, 3:10 pm on 5 March, 2023) -> WRONG
Question: What did Lee go through? | Reference: a divorce and job loss | Student: Lee went through a lot. -> WRONG
Question: Where did Kim study? | Reference: Boston | Student: Chicago -> WRONG
Question: How many cats does Jo have? | Reference: two | Student: 2 cats -> CORRECT

Now grade this one. Reply with one word, CORRECT or WRONG.
Question: {question} | Reference: {gold} | Student: {answer} ->"""


class Judge:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.name = llm.name

    def is_correct(self, question: str, gold: str, answer: str, category: str) -> bool:
        if category == "adversarial":  # no model needed: right only if the agent declined
            return is_refusal(answer)
        if not answer.strip():
            return False
        prompt = JUDGE_PROMPT.format(question=question, gold=gold, answer=answer)
        verdict = self.llm.generate(prompt, max_new_tokens=6).upper()
        return "CORRECT" in verdict and "WRONG" not in verdict
