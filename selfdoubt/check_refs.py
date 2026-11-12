"""Check that every reference in a draft resolves to a real record.

Finds arXiv IDs, DOIs and OpenReview forum IDs in a Markdown draft, or reads a BibTeX file (where an entry with no
``eprint``/``doi``/OpenReview ``url`` is itself an error). Each ID is looked up:

- arXiv via export.arxiv.org (also warns when the ID's YYMM does not match the first-version date);
- DOIs via api.crossref.org;
- OpenReview via its API, which usually answers automated clients with a challenge. Those are reported as
  "check by hand", never as passing.

When a BibTeX entry has a title, the fetched title must match it. Only the IDs are sent; the client identifies
itself generically.

    python -m selfdoubt.check_refs DRAFT.md|REFS.bib
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

USER_AGENT = "selfdoubt-check-refs/0.1"
ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|arXiv:\s*|arXiv\s+)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\])>\"'},]+)")
OPENREVIEW_RE = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([\w-]+)")
ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass
class Ref:
    kind: str  # arxiv | doi | openreview
    ident: str
    title: str | None = None
    key: str | None = None


@dataclass
class Resolution:
    ref: Ref
    ok: bool | None  # None = could not be checked automatically
    title: str = ""
    problem: str = ""
    warnings: list[str] = field(default_factory=list)


def http_fetch(url: str) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, str(exc)


def extract_refs(text: str) -> list[Ref]:
    found: list[tuple[int, Ref]] = []
    for m in ARXIV_RE.finditer(text):
        found.append((m.start(), Ref("arxiv", m.group(1))))
    for m in DOI_RE.finditer(text):
        found.append((m.start(), Ref("doi", m.group(1).rstrip(".;"))))
    for m in OPENREVIEW_RE.finditer(text):
        found.append((m.start(), Ref("openreview", m.group(1))))
    seen, out = set(), []
    for _, ref in sorted(found, key=lambda p: p[0]):
        if (ref.kind, ref.ident) not in seen:
            seen.add((ref.kind, ref.ident))
            out.append(ref)
    return out


def _bib_field(body: str, name: str) -> str | None:
    m = re.search(rf"\b{name}\s*=\s*[{{\"](.*?)[}}\"]\s*(?:,|$)", body, re.I | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip("{} ") if m else None


def extract_bib(text: str) -> tuple[list[Ref], list[str]]:
    refs, missing = [], []
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)(?=\n\s*@|\Z)", text, re.S):
        key, body = m.group(1), m.group(2)
        title = _bib_field(body, "title")
        eprint, doi, url = _bib_field(body, "eprint"), _bib_field(body, "doi"), _bib_field(body, "url") or ""
        if eprint and re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", eprint):
            refs.append(Ref("arxiv", eprint.split("v")[0], title, key))
        elif doi:
            refs.append(Ref("doi", doi, title, key))
        elif OPENREVIEW_RE.search(url):
            refs.append(Ref("openreview", OPENREVIEW_RE.search(url).group(1), title, key))
        elif ARXIV_RE.search(url):
            refs.append(Ref("arxiv", ARXIV_RE.search(url).group(1), title, key))
        else:
            missing.append(key)
    return refs, missing


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _titles_agree(expected: str, got: str) -> bool:
    a, b = _words(expected), _words(got)
    return bool(a and b) and len(a & b) / len(a | b) >= 0.6


def resolve(ref: Ref, fetch=http_fetch) -> Resolution:
    if ref.kind == "arxiv":
        status, body = fetch(f"https://export.arxiv.org/api/query?id_list={ref.ident}")
        if status != 200:
            return Resolution(ref, False, problem=f"arXiv API status {status}")
        try:
            entries = ET.fromstring(body).findall(f"{ATOM}entry") or ET.fromstring(body).findall("entry")
        except ET.ParseError:
            return Resolution(ref, False, problem="arXiv API returned unreadable XML")
        res = None
        for e in entries:
            eid = (e.findtext(f"{ATOM}id") or e.findtext("id") or "")
            if f"/abs/{ref.ident}" in eid:
                title = " ".join((e.findtext(f"{ATOM}title") or e.findtext("title") or "").split())
                published = e.findtext(f"{ATOM}published") or e.findtext("published") or ""
                res = Resolution(ref, True, title=title)
                yymm = published[2:4] + published[5:7]
                if published and yymm != ref.ident[:4]:
                    res.warnings.append(f"ID month {ref.ident[:4]} but first version dated {published[:10]}")
        if res is None:
            return Resolution(ref, False, problem="no such arXiv record")
    elif ref.kind == "doi":
        status, body = fetch(f"https://api.crossref.org/works/{urllib.parse.quote(ref.ident, safe='/')}")
        if status != 200:
            return Resolution(ref, False, problem=f"DOI not found (Crossref status {status})")
        titles = json.loads(body).get("message", {}).get("title") or [""]
        res = Resolution(ref, True, title=titles[0])
    elif ref.kind == "openreview":
        status, body = fetch(f"https://api2.openreview.net/notes?id={ref.ident}")
        if status == 403:
            return Resolution(ref, None, problem="OpenReview blocks automated checks; check by hand")
        if status != 200:
            return Resolution(ref, False, problem=f"OpenReview status {status}")
        notes = json.loads(body).get("notes") or []
        if not notes:
            return Resolution(ref, False, problem="no such OpenReview note")
        t = notes[0].get("content", {}).get("title", "")
        res = Resolution(ref, True, title=t.get("value", "") if isinstance(t, dict) else t)
    else:
        raise ValueError(ref.kind)
    if ref.title and not _titles_agree(ref.title, res.title):
        res.ok, res.problem = False, f"title differs: cited {ref.title!r}, record {res.title!r}"
    return res


def main(argv: list[str] | None = None, sleep=time.sleep) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    path = Path(argv[0])
    text = path.read_text()
    if path.suffix == ".bib":
        refs, missing = extract_bib(text)
    else:
        refs, missing = extract_refs(text), []
    failed = len(missing)
    for key in missing:
        print(f"ERROR   {key}: no arXiv ID, DOI or OpenReview link")
    by_hand = 0
    for i, ref in enumerate(refs):
        if i and ref.kind == "arxiv" and fetch_is_live():
            sleep(3)  # arXiv API asks for 3 s between calls
        r = resolve(ref, http_fetch)
        label = f"{ref.kind}:{ref.ident}" + (f" [{ref.key}]" if ref.key else "")
        if r.ok:
            print(f"ok      {label}  {r.title}")
        elif r.ok is None:
            by_hand += 1
            print(f"BY HAND {label}  {r.problem}")
        else:
            failed += 1
            print(f"ERROR   {label}  {r.problem}")
        for w in r.warnings:
            print(f"warning {label}  {w}")
    print(f"{len(refs)} references, {failed} errors, {by_hand} to check by hand")
    return 1 if failed else 0


def fetch_is_live() -> bool:
    return http_fetch is _REAL_FETCH


_REAL_FETCH = http_fetch

if __name__ == "__main__":
    sys.exit(main())
