"""LoCoMo loader (Maharana et al., 2024): 10 long two-person conversations with QA pairs.

Each QA pair lists its evidence turns (`dia_id`s such as "D3:12"). LoCoMo asks
all questions after the conversation has ended, so every evidence item is
"needed at" the end. There is no official train/test split: all 10 conversations
are the test set.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from memctl.benchmarks.types import Conversation, Event, Question
from memctl.items import make_item

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
NOT_MENTIONED = "Not mentioned in the conversation"
DIALOGUE_ID = re.compile(r"D\d+:\d+")


def load_locomo(path: str = "data/locomo/locomo10.json") -> list[Conversation]:
    file = Path(path)
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, file)  # 2.8 MB
    return [_conversation(sample) for sample in json.loads(file.read_text())]


def _conversation(sample: dict) -> Conversation:
    dialogue = sample["conversation"]
    session_numbers = sorted(
        int(key.split("_")[1]) for key in dialogue if re.fullmatch(r"session_\d+", key) and dialogue[key]
    )
    events: list[Event] = []
    for number in session_numbers:
        date = dialogue.get(f"session_{number}_date_time", "")
        for turn in dialogue[f"session_{number}"]:
            text = turn.get("text", "")
            if turn.get("blip_caption"):
                text += f" [shares a photo: {turn['blip_caption']}]"
            source = {"speaker": turn["speaker"], "session": number, "date": date}
            events.append(Event("item", item=make_item(turn["dia_id"], text, len(events) + 1, source=source)))
        events.append(Event("session_end"))

    known_ids = {event.item.id for event in events if event.kind == "item"}
    for number, qa in enumerate(sample["qa"]):
        category = CATEGORIES[qa["category"]]
        # Evidence lists contain a few typos such as "D8:6; D9:17", so pull ids out with a pattern.
        evidence = [i for entry in qa.get("evidence", []) for i in DIALOGUE_ID.findall(str(entry))]
        evidence = tuple(dict.fromkeys(i for i in evidence if i in known_ids))
        gold = NOT_MENTIONED if category == "adversarial" else str(qa.get("answer", ""))
        question = Question(f"{sample['sample_id']}_q{number}", qa["question"], gold, category, evidence)
        events.append(Event("question", question=question))
    return Conversation(sample["sample_id"], events)
