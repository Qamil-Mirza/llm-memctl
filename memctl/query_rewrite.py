"""Query rewriting for the §19 head's candidate pool (Experiment 28).

Two ways to change which 32 BM25 candidates the head chooses its 8 from, with the head itself unchanged:

- `rule`: no model. A hand-written parser reads a time phrase in the question ("two weeks ago", "last
  Saturday", "in March", "yesterday") and, given the date the question was asked, turns it into a date range.
  The parser was written from general English phrasing before any LongMemEval question was read for this
  experiment; its two settings (action, slack) are chosen per fold on that fold's training part.
- `llm`: one call per question. The model writes extra search words and, for a time question, a date range.
  The output is cached (memctl/llm.py), so a reader run replays it with `cache_only`.

A range changes the pool, never the question the head or the reader sees:
- `drop`: only candidates whose session date is in the range (the pool may hold fewer than 32);
- `demote`: in-range candidates first, then the others, each in BM25 order, up to 32.
Both rank the whole archive once, so BM25's document frequencies are the same as without a range.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from pathlib import Path

DATE = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")

MONTHS = {name: number for number, name in enumerate(calendar.month_name) if name}
WEEKDAYS = {name.lower(): number for number, name in enumerate(calendar.day_name)}
NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "a couple": 2, "couple of": 2,
    "a few": 3, "few": 3, "several": 3,
}
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
# Half-width of the window around a point in time, per unit, at slack 1. Multiplied by the `slack` setting.
UNIT_HALF = {"day": 2, "week": 4, "month": 15, "year": 120}

_NUMBER = r"(\d+|" + "|".join(sorted((re.escape(k) for k in NUMBERS), key=len, reverse=True)) + r")"
_UNIT = r"(day|week|month|year)s?"
_AGO = re.compile(rf"\b{_NUMBER}\s+{_UNIT}\s+ago\b")
_PAST_N = re.compile(rf"\b(?:the\s+)?(?:past|last|previous)\s+{_NUMBER}\s+{_UNIT}\b")
_LAST_UNIT = re.compile(r"\b(?:last|previous|past)\s+(week|month|year|weekend)\b")
_THIS_UNIT = re.compile(r"\bthis\s+(week|month|year|weekend)\b")
_DAY_WORDS = re.compile(r"\b(yesterday|today|tonight|this morning|this afternoon|this evening|last night)\b")
_WEEKDAY = re.compile(r"\b(last|this past|past|this|on)\s+(" + "|".join(WEEKDAYS) + r")\b")
_MONTH_NAMES = "|".join(MONTHS)
_ORD = r"(\d{1,2})(?:st|nd|rd|th)?"
# Month names are matched case-sensitively, so "may" and "march" as words do not count; "May" also needs a
# preposition, a day or a year beside it (a sentence can start with "May I").
_MONTH = re.compile(rf"(?:\b(in|during|of|on|since|early|mid|late|end of|beginning of)\s+)?\b({_MONTH_NAMES})\b"
                    rf"(?:\s+{_ORD}\b)?(?:,?\s+(\d{{4}}))?")
_DAY_OF_MONTH = re.compile(rf"\b{_ORD}\s+of\s+({_MONTH_NAMES})\b(?:,?\s+(\d{{4}}))?")


def parse_date(text: str) -> date | None:
    """The first YYYY/MM/DD (or YYYY-MM-DD) in the text, as a date."""
    found = DATE.search(text or "")
    if not found:
        return None
    try:
        return date(int(found.group(1)), int(found.group(2)), int(found.group(3)))
    except ValueError:
        return None


def split_question(content: str) -> tuple[date | None, str]:
    """(question date, question text) from the LongMemEval observation "(asked on DATE) question"."""
    match = re.match(r"^\(asked on (.*?\d{1,2}:\d{2})\)\s*", content) or re.match(r"^\(asked on ([^)]*)\)\s*", content)
    if not match:
        return None, content
    return parse_date(match.group(1)), content[match.end():]


def _number(text: str) -> int:
    return int(text) if text.isdigit() else NUMBERS[text]


def _around(center: date, unit: str, slack: float) -> tuple[date, date]:
    half = timedelta(days=max(1, round(UNIT_HALF[unit] * slack)))
    return center - half, center + half


def _month_range(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _latest_year(asked: date, month: int, day: int = 1) -> int:
    """The year of the latest such month (and day) not after the question date."""
    return asked.year if (month, day) <= (asked.month, asked.day) else asked.year - 1


def rule_range(question: str, asked: date | None, slack: float = 1.0) -> tuple[date, date] | None:
    """The date range a time phrase in the question points to, or None. Several phrases give their union.
    The end is never after the question date."""
    if asked is None:
        return None
    lower = question.lower()
    ranges: list[tuple[date, date]] = []
    pad = timedelta(days=max(1, round(UNIT_HALF["day"] * slack)))
    for match in _PAST_N.finditer(lower):
        n, unit = _number(match.group(1)), match.group(2)
        ranges.append((asked - timedelta(days=n * UNIT_DAYS[unit]) - pad, asked))
    spans = [m.span() for m in _PAST_N.finditer(lower)]
    for match in _AGO.finditer(lower):
        n, unit = _number(match.group(1)), match.group(2)
        ranges.append(_around(asked - timedelta(days=n * UNIT_DAYS[unit]), unit, slack))
    for match in _LAST_UNIT.finditer(lower):
        if any(start <= match.start() < end for start, end in spans):
            continue
        unit = match.group(1)
        if unit == "weekend":
            saturday = asked - timedelta(days=(asked.weekday() - 5) % 7 or 7)
            ranges.append((saturday - pad, saturday + timedelta(days=1) + pad))
        else:
            ranges.append(_around(asked - timedelta(days=UNIT_DAYS[unit]), unit, slack))
    for match in _THIS_UNIT.finditer(lower):
        unit = match.group(1)
        start = {"week": asked - timedelta(days=7), "weekend": asked - timedelta(days=3),
                 "month": asked.replace(day=1), "year": asked.replace(month=1, day=1)}[unit]
        ranges.append((start - pad, asked))
    for match in _DAY_WORDS.finditer(lower):
        back = 1 if match.group(1) in ("yesterday", "last night") else 0
        ranges.append(_around(asked - timedelta(days=back), "day", slack))
    for match in _WEEKDAY.finditer(lower):
        target = WEEKDAYS[match.group(2)]
        back = (asked.weekday() - target) % 7
        if match.group(1) != "this" and back == 0:
            back = 7
        ranges.append(_around(asked - timedelta(days=back), "day", slack))
    taken = [m.span() for m in _DAY_OF_MONTH.finditer(question)]
    for match in _MONTH.finditer(question):
        if any(start <= match.start(2) < end for start, end in taken):
            continue
        word, name, day, year = match.group(1), match.group(2), match.group(3), match.group(4)
        if name == "May" and not (word or day or year):
            continue
        month = MONTHS[name]
        try:
            if day:
                y = int(year) if year else _latest_year(asked, month, int(day))
                ranges.append(_around(date(y, month, int(day)), "day", slack))
            else:
                y = int(year) if year else _latest_year(asked, month)
                start, end = _month_range(y, month)
                ranges.append((start - pad, end + pad))
        except ValueError:
            continue
    for match in _DAY_OF_MONTH.finditer(question):
        day, month, year = int(match.group(1)), MONTHS[match.group(2)], match.group(3)
        try:
            y = int(year) if year else _latest_year(asked, month, day)
            ranges.append(_around(date(y, month, day), "day", slack))
        except ValueError:
            continue
    for match in DATE.finditer(question):
        found = parse_date(match.group(0))
        if found:
            ranges.append(_around(found, "day", slack))
    if not ranges:
        return None
    start, end = min(r[0] for r in ranges), min(max(r[1] for r in ranges), asked)
    return (start, end) if start <= end else None


# ---- the LLM rewrite --------------------------------------------------------------------------------------

REWRITE_MAX_TOKENS = 96


def load_prompt(path: str | Path) -> str:
    """A rewrite prompt template with {asked} and {question} slots (configs/sweeps/exp28/prompts/vN.txt)."""
    return Path(path).read_text()


def rewrite_prompt(template: str, asked: date | None, question: str) -> str:
    when = f"{asked:%Y/%m/%d} ({calendar.day_name[asked.weekday()]})" if asked else "unknown"
    return template.replace("{asked}", when).replace("{question}", question.strip())


def parse_rewrite(output: str) -> tuple[str, tuple[date, date] | None, bool]:
    """(extra query words, range or None, well-formed) from the model's two lines "QUERY: ..." and
    "RANGE: YYYY/MM/DD - YYYY/MM/DD" (or "RANGE: none"). A malformed answer gives ("", None, False):
    the raw question alone, as without a rewrite."""
    query, span, has_query, has_range = "", None, False, False
    for line in (output or "").splitlines():
        line = line.strip().strip("*").strip()
        head, _, rest = line.partition(":")
        key = head.strip().strip("*").upper()
        if key == "QUERY":
            query, has_query = rest.strip(), True
        elif key == "RANGE":
            has_range = True
            dates = [parse_date(m.group(0)) for m in DATE.finditer(rest)]
            dates = [d for d in dates if d]
            if len(dates) >= 2:
                span = (min(dates[0], dates[1]), max(dates[0], dates[1]))
            elif len(dates) == 1:
                span = (dates[0], dates[0])
    if not (has_query and has_range):
        return "", None, False
    return query, span, True


def widen(span: tuple[date, date] | None, days: int, asked: date | None) -> tuple[date, date] | None:
    if span is None:
        return None
    start, end = span[0] - timedelta(days=days), span[1] + timedelta(days=days)
    if asked is not None:
        end = min(end, asked)
    return (start, end) if start <= end else None


# ---- the pool ---------------------------------------------------------------------------------------------

def in_range(item_date: str | None, span: tuple[date, date]) -> bool:
    found = parse_date(item_date or "")
    return found is not None and span[0] <= found <= span[1]


def pool(retriever, query: str, items, k: int, span: tuple[date, date] | None, action: str = "demote"):
    """The k candidates for the head: BM25 over the whole archive, then the range applied (`drop` or
    `demote`). Without a range this is exactly retriever.search(query, items, k)."""
    if span is None:
        return retriever.search(query, items, k)
    ranked = retriever.search(query, items, len(items))
    inside = [pair for pair in ranked if in_range(pair[0].metadata.get("date"), span)]
    if action == "drop":
        return inside[:k]
    if action == "demote":
        outside = [pair for pair in ranked if not in_range(pair[0].metadata.get("date"), span)]
        return (inside + outside)[:k]
    raise ValueError(f"unknown range action {action!r} (drop, demote)")


class QueryRewriter:
    """Built from the controller's `query_rewrite` config:
    {mode: rule, action: drop|demote, slack: float}
    {mode: llm, action: drop|demote, pad_days: int, prompt_file: path, model: {build_llm config}}"""

    def __init__(self, config: dict) -> None:
        self.mode = config["mode"]
        if self.mode not in ("rule", "llm"):
            raise ValueError(f"unknown query_rewrite mode {self.mode!r} (rule, llm)")
        self.action = config.get("action", "demote")
        if self.action not in ("drop", "demote"):
            raise ValueError(f"unknown range action {self.action!r} (drop, demote)")
        self.slack = float(config.get("slack", 1.0))
        self.pad_days = int(config.get("pad_days", 3))
        self.llm = None
        if self.mode == "llm":
            from memctl.llm import build_llm

            self.template = load_prompt(config["prompt_file"])
            self.llm = build_llm(dict(config["model"]))
        self.last: dict = {}

    def query_and_range(self, content: str) -> tuple[str, tuple[date, date] | None]:
        asked, question = split_question(content)
        if self.mode == "rule":
            span = rule_range(question, asked, self.slack)
            self.last = {"range": _show(span)}
            return content, span
        output = self.llm.generate(rewrite_prompt(self.template, asked, question), REWRITE_MAX_TOKENS)
        from memctl.llm import CACHE_MISS

        if output == CACHE_MISS:
            raise RuntimeError("query rewrite not in the cache (cache_only): run the rewrite stage first")
        extra, span, ok = parse_rewrite(output)
        span = widen(span, self.pad_days, asked)
        self.last = {"range": _show(span), "extra": extra, "well_formed": ok}
        return (f"{content} {extra}" if extra else content), span

    def search(self, retriever, content: str, items, k: int):
        query, span = self.query_and_range(content)
        return pool(retriever, query, items, k, span, self.action)


def _show(span):
    return None if span is None else [f"{span[0]:%Y/%m/%d}", f"{span[1]:%Y/%m/%d}"]
