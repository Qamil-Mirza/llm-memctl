"""Tests for selfdoubt.check_refs: every reference must resolve to a real record (network faked)."""
from selfdoubt import check_refs as cr

ARXIV_OK = """<feed><entry><id>http://arxiv.org/abs/2510.11713v4</id>
<published>2025-10-13T17:00:00Z</published><title>Are Large Reasoning
  Models Interruptible?</title></entry></feed>"""
ARXIV_BAD_MONTH = """<feed><entry><id>http://arxiv.org/abs/2606.15497v1</id>
<published>2026-09-02T00:00:00Z</published><title>Some Paper</title></entry></feed>"""
ARXIV_MISSING = """<feed><entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_9999.99999</id>
<title>Error</title></entry></feed>"""


def fake_fetch(table):
    calls = []

    def fetch(url):
        calls.append(url)
        for key, resp in table.items():
            if key in url:
                return resp
        return 404, ""
    fetch.calls = calls
    return fetch


def test_extracts_ids_from_markdown():
    text = ("See [Wu](https://arxiv.org/abs/2510.11713v4), arXiv:2502.14297, "
            "https://doi.org/10.1038/s41586-024-07566-y). And https://openreview.net/forum?id=qXmWvPFkZP "
            "plus arxiv.org/html/2510.11713v4 again.")
    refs = cr.extract_refs(text)
    assert [(r.kind, r.ident) for r in refs] == [
        ("arxiv", "2510.11713"), ("arxiv", "2502.14297"),
        ("doi", "10.1038/s41586-024-07566-y"), ("openreview", "qXmWvPFkZP")]


def test_bib_entries_without_identifier_are_flagged():
    bib = """@article{wu2025, title={Are Large Reasoning Models Interruptible?}, eprint={2510.11713}}
@inproceedings{ghost, title={A Paper With No Identifier}, year={2025}}"""
    refs, missing = cr.extract_bib(bib)
    assert [(r.kind, r.ident, r.title) for r in refs] == [
        ("arxiv", "2510.11713", "Are Large Reasoning Models Interruptible?")]
    assert missing == ["ghost"]


def test_resolves_arxiv_and_checks_title_and_month():
    fetch = fake_fetch({"id_list=2510.11713": (200, ARXIV_OK), "id_list=2606.15497": (200, ARXIV_BAD_MONTH)})
    good = cr.resolve(cr.Ref("arxiv", "2510.11713", title="Are large reasoning models interruptible"), fetch)
    assert good.ok and good.title == "Are Large Reasoning Models Interruptible?" and good.warnings == []
    wrong_title = cr.resolve(cr.Ref("arxiv", "2510.11713", title="Attention is all you need"), fetch)
    assert not wrong_title.ok and "title" in wrong_title.problem
    odd = cr.resolve(cr.Ref("arxiv", "2606.15497"), fetch)
    assert odd.ok and any("month" in w for w in odd.warnings)


def test_unresolved_ids_fail():
    fetch = fake_fetch({"id_list=9999.99999": (200, ARXIV_MISSING)})
    assert not cr.resolve(cr.Ref("arxiv", "9999.99999"), fetch).ok
    assert not cr.resolve(cr.Ref("doi", "10.0000/nope"), fake_fetch({})).ok


def test_resolves_doi_and_openreview():
    fetch = fake_fetch({
        "api.crossref.org/works/10.1038": (200, '{"message": {"title": ["A Nature Paper"]}}'),
        "api2.openreview.net/notes?id=qXmWvPFkZP": (
            200, '{"notes": [{"content": {"title": {"value": "Hypothesis Anchoring"}}}]}'),
    })
    doi = cr.resolve(cr.Ref("doi", "10.1038/s41586-024-07566-y"), fetch)
    assert doi.ok and doi.title == "A Nature Paper"
    orv = cr.resolve(cr.Ref("openreview", "qXmWvPFkZP"), fetch)
    assert orv.ok and orv.title == "Hypothesis Anchoring"


def test_openreview_challenge_means_check_by_hand_not_pass():
    # OpenReview's API answers automated clients with a 403 challenge; that is "unverified", never "ok".
    fetch = fake_fetch({"api2.openreview.net": (403, '{"name":"ChallengeRequiredError"}')})
    res = cr.resolve(cr.Ref("openreview", "qXmWvPFkZP"), fetch)
    assert res.ok is None and "by hand" in res.problem


def test_main_exit_code(tmp_path, monkeypatch):
    draft = tmp_path / "d.md"
    draft.write_text("arXiv:2510.11713 and arXiv:9999.99999\n")
    monkeypatch.setattr(cr, "http_fetch", fake_fetch(
        {"id_list=2510.11713": (200, ARXIV_OK), "id_list=9999.99999": (200, ARXIV_MISSING)}))
    assert cr.main([str(draft)]) == 1
    draft.write_text("arXiv:2510.11713\n")
    assert cr.main([str(draft)]) == 0
