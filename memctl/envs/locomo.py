"""LoCoMo (Maharana et al., 2024): 10 long two-person conversations with QA pairs.

One episode is one conversation: seed s plays conversation s mod 10. Every QA
pair lists its evidence turns, which become the ground-truth dependencies.
There is no train/test split in the dataset.

Known from the earlier work on this repository: 5,882 turns and 1,986 questions
in total, and evidence is about 24% of turns and 31% of tokens, so a budget of
50% is the first at which a perfect delete-only policy could hold all evidence.
"""

from __future__ import annotations

import json
import re
import urllib.request
from functools import lru_cache
from pathlib import Path

from memctl.envs.qa import QAEnvironment, QAEpisode, QAItem
from memctl.memory.items import SourceType
from memctl.task import Observation

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
DIALOGUE_ID = re.compile(r"D\d+:\d+")


@lru_cache(maxsize=2)
def _load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def parse_conversation(sample: dict, include_adversarial: bool = True) -> QAEpisode:
    dialogue = sample["conversation"]
    sessions = sorted(
        int(key.split("_")[1]) for key in dialogue if re.fullmatch(r"session_\d+", key) and dialogue[key]
    )
    turns, text_of = [], {}
    for number in sessions:
        date = dialogue.get(f"session_{number}_date_time", "")
        for turn in dialogue[f"session_{number}"]:
            text = turn.get("text", "")
            if turn.get("blip_caption"):
                text += f" [shares a photo: {turn['blip_caption']}]"
            metadata = {"speaker": turn["speaker"], "session": number, "date": date}
            turns.append(Observation(turn["dia_id"], text, SourceType.USER, metadata=metadata))
            text_of[turn["dia_id"]] = text
    questions = []
    for number, qa in enumerate(sample["qa"]):
        category = CATEGORIES[qa["category"]]
        if category == "adversarial" and not include_adversarial:
            continue
        # Evidence lists contain a few typos such as "D8:6; D9:17", so pull ids out with a pattern.
        evidence = [i for entry in qa.get("evidence", []) for i in DIALOGUE_ID.findall(str(entry))]
        evidence = tuple(dict.fromkeys(i for i in evidence if i in text_of))
        unanswerable = category == "adversarial"
        gold = "unknown" if unanswerable else str(qa.get("answer", ""))
        questions.append(QAItem(f"{sample['sample_id']}_q{number}", qa["question"], gold, category, evidence, unanswerable))
    return QAEpisode(str(sample["sample_id"]), turns, questions, text_of)


class LoCoMoEnv(QAEnvironment):
    name = "locomo"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.path = config.get("path", "data/locomo/locomo10.json")
        self.include_adversarial = bool(config.get("include_adversarial", True))
        if not Path(self.path).exists():
            if not config.get("download", False):
                raise FileNotFoundError(
                    f"LoCoMo not found at {self.path}. Set env.download: true to fetch it (2.8 MB) from {URL}"
                )
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL, self.path)

    def load_episode(self, seed: int) -> QAEpisode:
        samples = _load(self.path)
        return parse_conversation(samples[seed % len(samples)], self.include_adversarial)
