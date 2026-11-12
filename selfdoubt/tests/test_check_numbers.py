"""Tests for selfdoubt.check_numbers: every tagged number in a draft must match its committed source."""
import json
import subprocess

import pytest

from selfdoubt import check_numbers as cn


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for k, v in {"user.name": "t", "user.email": "t@example.invalid", "commit.gpgsign": "false"}.items():
        subprocess.run(["git", "-C", str(tmp_path), "config", k, v], check=True)
    res = tmp_path / "res.json"
    res.write_text(json.dumps({"tables": {"ans|128": {"stale": {"m_base": -11.1703125}}},
                               "rate": 0.5834, "ci95": [7.57, 11.03]}))
    subprocess.run(["git", "-C", str(tmp_path), "add", "res.json"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "r"], check=True)
    return tmp_path


def tag(path, ptr):
    return f"<!-- src: {path}#{ptr} -->"


def test_pointer_resolves_keys_with_pipes_and_list_indices():
    doc = {"tables": {"ans|128": {"stale": {"m_base": -11.17}}}, "ci": [1.5, 2.5], "a/b": {"~x": 3}}
    assert cn.resolve_pointer(doc, "/tables/ans|128/stale/m_base") == -11.17
    assert cn.resolve_pointer(doc, "/ci/1") == 2.5
    assert cn.resolve_pointer(doc, "/a~1b/~0x") == 3
    with pytest.raises(KeyError):
        cn.resolve_pointer(doc, "/tables/missing")


@pytest.mark.parametrize("text,value,pct,ok", [
    ("-11.17", -11.1703125, False, True),
    ("−11.17", -11.1703125, False, True),   # unicode minus as written in our docs
    ("-11.2", -11.1703125, False, True),
    ("-11.18", -11.1703125, False, False),
    ("58.3", 0.5834, True, True),          # percent: file holds a fraction
    ("58", 0.5834, True, True),
    ("59", 0.5834, True, False),
    ("1,234", 1234.0, False, True),
])
def test_matches_respects_shown_precision(text, value, pct, ok):
    assert cn.matches(text, value, pct) is ok


def test_check_passes_on_correct_tags(repo):
    draft = repo / "draft.md"
    draft.write_text(f"Answer margin −11.17{tag('res.json', '/tables/ans|128/stale/m_base')} "
                     f"and stale rate 58.3%{tag('res.json', '/rate')}; CI [+7.57{tag('res.json', '/ci95/0')}].\n")
    report = cn.check(draft, repo_root=repo)
    assert report.errors == []
    assert len(report.checked) == 3
    assert all(c.commit for c in report.checked)


def test_check_flags_mismatch_missing_file_bad_pointer(repo):
    draft = repo / "draft.md"
    draft.write_text(f"a 0.60{tag('res.json', '/rate')} b 1.0{tag('nope.json', '/x')} "
                     f"c 2.0{tag('res.json', '/tables/zzz')}\n")
    report = cn.check(draft, repo_root=repo)
    kinds = sorted(e.kind for e in report.errors)
    assert kinds == ["bad-pointer", "mismatch", "missing-file"]


def test_check_flags_uncommitted_source(repo):
    (repo / "res.json").write_text(json.dumps({"rate": 0.5834}))  # modified after commit
    (repo / "new.json").write_text(json.dumps({"x": 1.5}))           # never committed
    draft = repo / "draft.md"
    draft.write_text(f"r 0.58{tag('res.json', '/rate')} x 1.5{tag('new.json', '/x')}\n")
    kinds = sorted(e.kind for e in cn.check(draft, repo_root=repo).errors)
    assert kinds == ["dirty-source", "untracked-source"]


def test_untagged_numbers_are_warnings_unless_strict(repo):
    draft = repo / "draft.md"
    draft.write_text("Qwen3-1.7B on arXiv 2510.11713 had a rate of 0.42 and 16% stale.\n"
                     "`code 3.14` is ignored. It ended at 0.97.\n")
    report = cn.check(draft, repo_root=repo)
    assert [u.text for u in report.untagged] == ["0.42", "16%", "0.97"]
    assert report.errors == []
    assert cn.main([str(draft), "--repo", str(repo)]) == 0
    assert cn.main([str(draft), "--repo", str(repo), "--strict"]) == 1
