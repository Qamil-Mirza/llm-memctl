"""Turns the rows of one run (decisions and answers) into metrics.json."""

from __future__ import annotations

from collections import Counter

from memctl.attribution import LABELS, MEMORY_FAILURES
from memctl.metrics import mean


def share(rows: list, test) -> float | None:
    return round(sum(1 for row in rows if test(row)) / len(rows), 4) if rows else None


def answer_quality(answers: list[dict]) -> dict:
    """Mean of every score, overall and per question category."""
    names = sorted({name for row in answers for name in row["scores"]})
    categories = sorted({row["category"] for row in answers})
    quality = {"accuracy": share(answers, lambda r: r["correct"])}
    for name in names:
        quality[name] = mean([r["scores"][name] for r in answers])
    quality["by_category"] = {
        category: {
            "questions": sum(1 for r in answers if r["category"] == category),
            "accuracy": share([r for r in answers if r["category"] == category], lambda r: r["correct"]),
            **{n: mean([r["scores"][n] for r in answers if r["category"] == category]) for n in names},
        }
        for category in categories
    }
    return quality


def failure_attribution(answers: list[dict]) -> dict:
    """How many wrong answers got each label, and the memory / reasoning split."""
    wrong = [r for r in answers if not r["correct"]]
    counts = Counter(r["failure_label"] for r in wrong)
    memory = sum(counts[label] for label in MEMORY_FAILURES)
    return {
        "wrong_answers": len(wrong),
        "counts": {label: counts[label] for label in LABELS},
        "memory_failures": memory,
        "reasoning_failures": counts["evidence_present_model_wrong"],
        "unknown": counts["no_evidence_label"],
        "memory_failure_share_of_wrong": round(memory / len(wrong), 4) if wrong else None,
    }


def evidence_availability(answers: list[dict]) -> dict:
    evidence = [e for row in answers for e in row["evidence"]]
    with_evidence = [row for row in answers if row["evidence"]]
    return {
        "evidence_items": len(evidence),
        "in_context": share(evidence, lambda e: e["place"] == "CONTEXT"),
        "in_store_and_retrieved": share(evidence, lambda e: e["place"] == "STORE" and e["retrieved_from_store"]),
        "in_store_not_retrieved": share(evidence, lambda e: e["place"] == "STORE" and not e["retrieved_from_store"]),
        "in_archive_and_recalled": share(evidence, lambda e: e["place"] == "ARCHIVE" and e["recalled_from_archive"]),
        "in_archive_not_recalled": share(evidence, lambda e: e["place"] == "ARCHIVE" and not e["recalled_from_archive"]),
        "dropped": share(evidence, lambda e: e["place"] == "DROPPED"),
        "questions_with_all_evidence_in_prompt": share(with_evidence, lambda r: all(e["in_prompt"] for e in r["evidence"])),
    }


def oracle_agreement(decisions: list[dict]) -> dict:
    """How often the controller chose what the oracle would have chosen, for moves AND keeps."""
    graded: list[tuple[str, str]] = []  # (place chosen, oracle's place), one per item per decision
    for d in decisions:
        if d["kind"] == "keep":
            moved = d["oracle_would_move"]
            graded += [("CONTEXT", moved.get(item_id, "CONTEXT")) for item_id in d["kept_item_ids"]]
        elif d["oracle_place"] is not None:
            graded.append((d["place"], d["oracle_place"]))
    moves = [g for g in graded if g[0] != "CONTEXT"]
    keeps = [g for g in graded if g[0] == "CONTEXT"]
    by_oracle_place = {}
    for place in sorted({oracle for _, oracle in graded}):
        rows = [g for g in graded if g[1] == place]
        by_oracle_place[place] = {"decisions": len(rows), "agreement": share(rows, lambda g: g[0] == g[1])}
    return {
        "decisions": len(graded),
        "agreement": share(graded, lambda g: g[0] == g[1]),
        "moves": {"decisions": len(moves), "agreement": share(moves, lambda g: g[0] == g[1])},
        "keeps": {"decisions": len(keeps), "agreement": share(keeps, lambda g: g[0] == g[1])},
        "when_oracle_chose": by_oracle_place,
    }


def summarize(config, display_name, conversations, answers, decisions, context_tokens, places, totals, jev_log) -> dict:
    return {
        "controller": display_name,
        "fake_jev": display_name.endswith("FAKE"),
        "benchmark": config["benchmark"]["name"],
        "budget": config["budget"],
        "budget_tokens_per_conversation": totals.pop("budgets"),
        "model": config.get("model", {}).get("name"),
        "judge": (config.get("judge") or {}).get("name"),
        "seed": config.get("seed", 0),
        "conversations": len(conversations),
        "questions": len(answers),
        "answer_quality": answer_quality(answers),
        "failure_attribution": failure_attribution(answers),
        "evidence_availability": evidence_availability(answers),
        "oracle_agreement": oracle_agreement(decisions),
        "cost": {
            "avg_context_tokens": mean(context_tokens),
            "peak_context_tokens": max(context_tokens, default=0),
            "avg_prompt_tokens": mean([row["prompt_tokens"] for row in answers]),
            "peak_prompt_tokens": max((row["prompt_tokens"] for row in answers), default=0),
            **totals,
            "jev_calls": len(jev_log),
            "jev_cost_usd": round(sum(call["cost_usd"] for call in jev_log), 6),
            "api_cost_usd": round(sum(call["cost_usd"] for call in jev_log), 6),
        },
        "final_places": dict(places),
    }
