"""Check that every tagged number in a draft matches the committed result file it cites.

Tag a number by putting an HTML comment right after it (invisible when the Markdown is rendered):

    the answer margin was −11.17<!-- src: selfdoubt/results/stage2_heldout_report.json#/tables/ans|128/stale/m_base -->

The part after ``#`` is a JSON Pointer (RFC 6901): ``/``-separated keys, list indices as integers, ``~1`` for ``/``
and ``~0`` for ``~``. A number followed by ``%`` is compared with 100 × the file value. The comparison uses the
precision shown in the text (``-11.2`` matches ``-11.1703``; ``-11.18`` does not).

Each source must be tracked by git and unchanged since its last commit, so the number traces to a commit hash.
Numbers with a decimal point or a ``%`` that carry no tag are listed as warnings (errors with ``--strict``).

    python -m selfdoubt.check_numbers DRAFT.md [--repo ROOT] [--strict]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

NUM = r"[-−+]?\d[\d,]*(?:\.\d+)?"
TAG_RE = re.compile(rf"(?P<num>{NUM})(?P<pct>%?)\s*<!--\s*src:\s*(?P<file>[^#\s]+)#(?P<ptr>[^\s]*)\s*-->")
UNTAGGED_RE = re.compile(r"(?<![\w.\-−])(?:\d+\.\d+%?|\d+%)(?!\w|\.\d)")
ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")


@dataclass
class Checked:
    text: str
    source: str
    pointer: str
    value: float
    commit: str


@dataclass
class Problem:
    kind: str  # mismatch | missing-file | bad-pointer | untracked-source | dirty-source
    text: str
    detail: str


@dataclass
class Untagged:
    text: str
    line: int


@dataclass
class Report:
    checked: list[Checked] = field(default_factory=list)
    errors: list[Problem] = field(default_factory=list)
    untagged: list[Untagged] = field(default_factory=list)


def resolve_pointer(doc, pointer: str):
    if pointer in ("", "/"):
        return doc
    cur = doc
    for raw in pointer.lstrip("/").split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            try:
                cur = cur[int(key)]
            except (ValueError, IndexError) as exc:
                raise KeyError(key) from exc
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            raise KeyError(key)
    return cur


def _to_decimal(text: str) -> Decimal:
    return Decimal(text.replace("−", "-").replace(",", "").lstrip("+"))


def matches(text: str, value: float, pct: bool) -> bool:
    shown = _to_decimal(text)
    places = -shown.as_tuple().exponent if shown.as_tuple().exponent < 0 else 0
    actual = Decimal(repr(float(value))) * (100 if pct else 1)
    half_step = Decimal(5) * Decimal(10) ** (-(places + 1))
    return abs(actual - shown) <= half_step + Decimal("1e-12")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _source_state(repo: Path, rel: str) -> tuple[str | None, str]:
    """Return (problem kind or None, short commit hash of the last commit touching the file)."""
    if _git(repo, "ls-files", "--error-unmatch", "--", rel).returncode != 0:
        return "untracked-source", ""
    if _git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        return "dirty-source", ""
    return None, _git(repo, "log", "-1", "--format=%h", "--", rel).stdout.strip()


def _strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"`[^`\n]*`", "", text)


def check(draft: Path, repo_root: Path) -> Report:
    draft, repo_root = Path(draft), Path(repo_root)
    text = draft.read_text()
    report = Report()
    cache: dict[str, object] = {}
    for m in TAG_RE.finditer(text):
        num, pct, rel, ptr = m["num"], m["pct"] == "%", m["file"], m["ptr"]
        shown = num + ("%" if pct else "")
        path = repo_root / rel
        if not path.is_file():
            report.errors.append(Problem("missing-file", shown, rel))
            continue
        state, commit = _source_state(repo_root, rel)
        if state:
            report.errors.append(Problem(state, shown, rel))
            continue
        if rel not in cache:
            cache[rel] = json.loads(path.read_text())
        try:
            value = resolve_pointer(cache[rel], ptr)
        except KeyError as exc:
            report.errors.append(Problem("bad-pointer", shown, f"{rel}#{ptr} (no key {exc})"))
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            report.errors.append(Problem("bad-pointer", shown, f"{rel}#{ptr} is not a number"))
            continue
        if not matches(num, value, pct):
            report.errors.append(Problem("mismatch", shown, f"{rel}#{ptr} = {value!r}"))
            continue
        report.checked.append(Checked(shown, rel, ptr, float(value), commit))

    rest = _strip_code(TAG_RE.sub("", text))
    for lineno, line in enumerate(rest.splitlines(), 1):
        line = ARXIV_ID_RE.sub("", line)
        for u in UNTAGGED_RE.finditer(line):
            report.untagged.append(Untagged(u.group(0), lineno))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("draft")
    ap.add_argument("--repo", default=".", help="repository root that tag paths are relative to")
    ap.add_argument("--strict", action="store_true", help="untagged numbers are errors")
    args = ap.parse_args(argv)
    rep = check(Path(args.draft), Path(args.repo))
    for c in rep.checked:
        print(f"ok        {c.text:>10}  {c.source}#{c.pointer} @ {c.commit}")
    for e in rep.errors:
        print(f"ERROR     {e.text:>10}  {e.kind}: {e.detail}")
    for u in rep.untagged:
        print(f"{'ERROR' if args.strict else 'untagged':9} {u.text:>10}  line {u.line}")
    print(f"{len(rep.checked)} traced, {len(rep.errors)} errors, {len(rep.untagged)} untagged")
    return 1 if rep.errors or (args.strict and rep.untagged) else 0


if __name__ == "__main__":
    sys.exit(main())
