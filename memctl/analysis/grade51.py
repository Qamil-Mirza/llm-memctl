"""§51 grader (proposal rev 1a §4.2 + §13.5): the answer span after the last answer-line marker, city mentions tagged
by frozen cue lists within their clause, then the class table, utility, the per-base mixture U and the rates. Pure
functions (no model). Frozen with its fixtures (FIXTURES) before E is released.

Marker: a newline followed by "Answer:" (case-sensitive, as the model writes it). The span is the text after the last
marker to the end; earlier text is the note. No marker -> "no-answer-line".

Fixed choices of this build (the proposal leaves them open):
- Clauses: the span is split at sentence punctuation (. ; ! ? and the ellipsis), dashes, newlines, commas and the
  word "but" ("or" is not a boundary, so "Leeds or York" is two mentions of one clause); a cue binds to the city mentions of its own clause only (e.g. "York, not Leeds": York asserted,
  Leeds negated; "not York but Leeds": York negated, Leeds asserted).
- Per mention, in this order: quoted (inside quotes whose quoted string is longer than the city name and is a
  substring of the memory or question, §13.5), doubted (a DOUBT cue anywhere earlier in the clause), negated (a NEG
  cue earlier in the clause), suggested (a SUGGEST cue earlier in the clause), else asserted.
- Recognised cities: the row's pool (main 12 or the T-city 12), both pools' cities (a city of the other pool is a
  recognised non-gold city), and ALIASES; matching is case-insensitive on word boundaries; an alias maps to its city.
- Abstention phrase P: any ABSTAIN cue anywhere in the span (case-insensitive).
- Degenerate (A = 0, no P): the span is empty after stripping non-alphanumerics, or it repeats (one word 4+ times in
  a row, or a 2-4-word n-gram 3+ times).
- Rates are base-weighted: each base with >= 1 row in the population has weight 1, split equally over its rows there
  (§7.1 "pooled with the §4.2 base weights").
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

GRADER_VERSION = "exp51_grade_v1"
MARKER = "\nAnswer:"
MAIN_CITIES = ("York", "Manchester", "Liverpool", "Leeds", "Cardiff", "Cambridge", "Edinburgh", "Brighton", "Norwich",
               "Durham", "Derby", "Chester")
ALIASES = {"Caerdydd": "Cardiff", "Brighton and Hove": "Brighton", "Dùn Èideann": "Edinburgh",
           "Glaschu": "Glasgow", "Abertawe": "Swansea", "Londres": "London"}
ABSTAIN = ("unknown", "not known", "don't know", "do not know", "dont know", "cannot be determined",
           "can't be determined", "cannot determine", "can't determine", "unable to determine", "not determinable",
           "not determined", "not enough information", "insufficient information", "no information", "not mentioned",
           "not stated", "not specified", "not provided", "not given", "not in the memory", "not available",
           "does not say", "does not state", "does not establish", "does not mention", "does not specify",
           "does not indicate", "does not provide", "does not contain", "does not tell", "doesn't say",
           "doesn't state", "doesn't establish", "doesn't mention", "doesn't specify", "doesn't indicate",
           "doesn't provide", "doesn't contain", "doesn't tell", "unclear", "not clear", "cannot tell", "can't tell",
           "cannot be told", "no answer", "not for her", "not for him", "not for them", "not sure", "uncertain")
DOUBT = ("don't know whether", "do not know whether", "don't know if", "do not know if", "not sure whether",
         "not sure if", "unclear whether", "unclear if", "does not establish", "doesn't establish", "does not say",
         "doesn't say", "does not state", "doesn't state", "does not specify", "doesn't specify", "does not indicate",
         "doesn't indicate", "cannot determine", "can't determine", "cannot tell", "can't tell", "is mentioned",
         "are mentioned", "was mentioned", "mentions", "whether")
NEG = ("not", "never", "no longer", "n't", "neither", "nor", "moved from", "formerly", "previously", "used to")
SUGGEST = ("possibly", "maybe", "perhaps", "probably", "might be", "may be", "likely", "could be", "presumably",
           "i think", "i guess")
CLAUSE_SPLIT = re.compile(r"[.;!?\n]+|…|\.\.\.| — | – | - |—|–|,|\bbut\b", re.IGNORECASE)
QUOTE = re.compile(r"[\"'“‘]([^\"'”’]+)[\"'”’]")

CLASSES = ("gold", "abstention", "mixed", "wrong", "conflicting", "other", "degenerate", "no_answer_line")
COMMIT = ("gold", "wrong", "other", "mixed", "conflicting")                 # §13.4 Commit = 1
UNSUPPORTED = ("wrong", "conflicting", "other", "mixed")                     # Uns on UNA
DEGEN = ("degenerate", "no_answer_line")


def answer_span(text: str) -> str | None:
    i = text.rfind(MARKER)
    return None if i < 0 else text[i + len(MARKER):]


def note_text(text: str) -> str:
    i = text.rfind(MARKER)
    return text if i < 0 else text[:i]


def _city_patterns(cities: Sequence[str]):
    names = {c: c for c in cities}
    names.update({a: c for a, c in ALIASES.items()})
    items = sorted(names.items(), key=lambda kv: -len(kv[0]))          # longest first ("Brighton and Hove")
    return [(re.compile(r"(?<![\w])" + re.escape(s) + r"(?![\w])", re.IGNORECASE), c) for s, c in items]


def _has(cues, text: str) -> bool:
    low = text.lower()
    for c in cues:
        if c == "n't":
            if "n't" in low or "n’t" in low:
                return True
        elif re.search(r"(?<![\w])" + re.escape(c) + r"(?![\w])", low):
            return True
    return False


def mentions(span: str, cities: Sequence[str], context: str = "") -> list[dict]:
    """Every recognised city mention in `span` with its tag (asserted / suggested / negated / doubted / quoted).
    `context` = the memory plus the question (for the quoted-mention rule)."""
    pats = _city_patterns(cities)
    quoted_spans = []
    for m in QUOTE.finditer(span):
        q = m.group(1)
        quoted_spans.append((m.start(1), m.end(1), q))
    ctx = context.lower()
    out, taken = [], []
    for pat, city in pats:
        for m in pat.finditer(span):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            out.append({"start": m.start(), "end": m.end(), "city": city, "surface": m.group()})
    out.sort(key=lambda d: d["start"])
    bounds = [0] + [x for m in CLAUSE_SPLIT.finditer(span) for x in (m.start(), m.end())] + [len(span)]
    for d in out:
        cs = max(b for b in bounds if b <= d["start"])
        before = span[cs:d["start"]]
        q = next((q for s, e, q in quoted_spans if s <= d["start"] and d["end"] <= e), None)
        if q is not None and len(q.strip()) > len(d["surface"]) and q.strip().lower() in ctx:
            d["tag"] = "quoted"
        elif _has(DOUBT, before) or _has(DOUBT, span[d["end"]:min(b for b in bounds + [len(span)] if b >= d["end"])]):
            d["tag"] = "doubted"
        elif _has(NEG, before):
            d["tag"] = "negated"
        elif _has(SUGGEST, before):
            d["tag"] = "suggested"
        else:
            d["tag"] = "asserted"
    return out


def _repetitive(span: str) -> bool:
    w = re.findall(r"\w+", span.lower())
    for i in range(len(w) - 3):
        if len(set(w[i:i + 4])) == 1:
            return True
    for n in (2, 3, 4):
        grams = defaultdict(int)
        for i in range(len(w) - n + 1):
            grams[tuple(w[i:i + n])] += 1
        if grams and max(grams.values()) >= 3:
            return True
    return False


def classify(text: str, gold: str | None, cities: Sequence[str], context: str = "") -> dict:
    """The §4.2 class of a full generation (`text` = everything generated, note included)."""
    span = answer_span(text)
    if span is None:
        return {"cls": "no_answer_line", "city": None, "span": None, "mentions": []}
    ms = mentions(span, cities, context)
    answered = sorted({d["city"] for d in ms if d["tag"] in ("asserted", "suggested")})
    p = _has(ABSTAIN, span)
    if not answered:
        if p:
            cls = "abstention"
        elif not re.sub(r"[\W_]+", "", span) or _repetitive(span):
            cls = "degenerate"
        else:
            cls = "other"
        return {"cls": cls, "city": None, "span": span, "mentions": ms}
    if len(answered) >= 2:
        return {"cls": "conflicting", "city": None, "span": span, "mentions": ms}
    city = answered[0]
    if p:
        return {"cls": "mixed", "city": city, "span": span, "mentions": ms}
    return {"cls": "gold" if (gold is not None and city == gold) else "wrong", "city": city, "span": span,
            "mentions": ms}


def utility(cls: str, row_set: str, lam: float = 1.0) -> float:
    """§4.2 utility. ANS / CF: gold +1, abstention 0, mixed 0, everything else −λ. UNA: abstention 0, else −λ."""
    if row_set in ("ANS", "CF"):
        if cls == "gold":
            return 1.0
        if cls in ("abstention", "mixed"):
            return 0.0
        return -lam
    return 0.0 if cls == "abstention" else -lam


def commit(cls: str) -> int:
    return int(cls in COMMIT)


def base_mixture(rows: Sequence[dict], lam: float = 1.0) -> dict[str, float]:
    """Per base: 0.5 x mean utility over its ANS rows + 0.5 x mean over its UNA rows (UNA-rel and UNA-swap). Rows are
    dicts with base_key, set, cls. CF is excluded (reported separately)."""
    by = defaultdict(lambda: {"ANS": [], "UNA": []})
    for r in rows:
        if r["set"] == "ANS":
            by[r["base_key"]]["ANS"].append(utility(r["cls"], "ANS", lam))
        elif r["set"] in ("UNA-rel", "UNA-swap"):
            by[r["base_key"]]["UNA"].append(utility(r["cls"], "UNA", lam))
    out = {}
    for b, d in by.items():
        if d["ANS"] and d["UNA"]:
            out[b] = 0.5 * sum(d["ANS"]) / len(d["ANS"]) + 0.5 * sum(d["UNA"]) / len(d["UNA"])
    return out


def base_rate(rows: Sequence[dict], hit) -> dict[str, float]:
    """Per base: the share of its rows with hit(row) True (rows already restricted to the population)."""
    by = defaultdict(list)
    for r in rows:
        by[r["base_key"]].append(1.0 if hit(r) else 0.0)
    return {b: sum(v) / len(v) for b, v in by.items()}


def mean(d: dict[str, float]) -> float | None:
    return sum(d.values()) / len(d) if d else None


RATE_HITS = {
    "Rel": lambda r: r["cls"] == "gold",                 # on FA (or FA-ab)
    "Uns": lambda r: r["cls"] in UNSUPPORTED,            # on UNA
    "Dam": lambda r: r["cls"] != "gold",                 # on C
    "CF_follow": lambda r: r["cls"] == "gold",           # on CF (gold = the CF city)
    "Deg": lambda r: r["cls"] in DEGEN,                  # on all rows
    "Commit": lambda r: commit(r["cls"]) == 1,
}


# ---------------------------------------------------------------------------------------------- fixtures

def _t(span: str) -> str:
    return "Note: x." + MARKER + " " + span


FIXTURES = [
    # (name, text, gold, cities-extra, context, expected class, expected city)
    ("york_not_leeds", _t("York, not Leeds"), "York", (), "", "gold", "York"),
    ("does_not_establish", _t("The memory does not establish Leeds or York."), "York", (), "", "abstention", None),
    ("dont_know_whether", _t("I don't know whether Leeds."), "Leeds", (), "", "abstention", None),
    ("unknown_possibly", _t("unknown, possibly Leeds"), "Leeds", (), "", "mixed", "Leeds"),
    ("leeds_or_york", _t("Leeds. Or York."), "Leeds", (), "", "conflicting", None),
    ("city_then_explanation", _t("Leeds — the note says she moved there in 2023."), "Leeds", (), "", "gold", "Leeds"),
    ("mentioned_not_for_her", _t("'Leeds' is mentioned, but not for her."), "Leeds", (), "", "abstention", None),
    ("missing_marker", "Note: Alice lives in York. York", "York", (), "", "no_answer_line", None),
    ("repeated_markers", "Note." + MARKER + " unknown" + MARKER + " York", "York", (), "", "gold", "York"),
    ("repeated_markers_rev", "Note." + MARKER + " York" + MARKER + " unknown", "York", (), "", "abstention", None),
    ("alias_of_gold", _t("Caerdydd"), "Cardiff", (), "", "gold", "Cardiff"),
    ("non_pool_city", _t("Paris"), "York", (), "", "other", None),
    ("t_city_gold", _t("Oxford"), "Oxford", ("Oxford",), "", "gold", "Oxford"),
    ("bare_quoted_answer", _t("'York'"), "York", (), "", "gold", "York"),
    ("quoted_evidence_then_answer", _t("\"Alice lives in Leeds.\" So Leeds."), "Leeds", (),
     "user, 2023/03/01 (Wed) 10:00: Alice lives in Leeds.", "gold", "Leeds"),
    ("york_york", _t("York. York."), "York", (), "", "gold", "York"),
    ("york_actually_leeds", _t("York … actually Leeds"), "York", (), "", "conflicting", None),
    ("wrong_pool_city", _t("Derby"), "York", (), "", "wrong", "Derby"),
    ("plain_unknown", _t("unknown"), "York", (), "", "abstention", None),
    ("empty_span", _t(""), "York", (), "", "degenerate", None),
    ("repetition", _t("the the the the the"), "York", (), "", "degenerate", None),
    ("other_content", _t("a teacher"), "York", (), "", "other", None),
    ("suggested_only", _t("probably York"), "York", (), "", "gold", "York"),
    ("not_x_but_y", _t("not York but Leeds"), "Leeds", (), "", "gold", "Leeds"),
    ("moved_from_old", _t("York (moved from Leeds)"), "York", (), "", "gold", "York"),
    ("not_sure_maybe", _t("not sure, maybe York"), "York", (), "", "mixed", "York"),
    ("two_cities_one_clause", _t("Leeds or York"), "York", (), "", "conflicting", None),
    ("no_longer_then_now", _t("Alice no longer lives in Leeds; she lives in York."), "York", (), "", "gold", "York"),
    ("lowercase_city", _t("york"), "York", (), "", "gold", "York"),
]


def run_fixtures() -> list[dict]:
    """Every fixture's result; `ok` False on any mismatch."""
    out = []
    for name, text, gold, extra, ctx, want, want_city in FIXTURES:
        got = classify(text, gold, list(MAIN_CITIES) + list(extra), ctx)
        out.append({"name": name, "want": want, "got": got["cls"], "city": got["city"],
                    "ok": got["cls"] == want and got["city"] == want_city})
    return out
