"""§53 build tests (no model forward; the head-edit test reads o_proj weights from the local snapshot)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_stage4 as S4  # noqa: E402
import exp53_run as R  # noqa: E402
import exp51_common as CM  # noqa: E402


def test_rank_p_and_bar():
    d = list(np.linspace(0, 0.5, 20))
    assert R.rank_p(0.9, d) == pytest.approx(1 / 21)
    assert R.rank_p(0.5, d) == pytest.approx(2 / 21)          # a tie counts against the target
    assert R.rank_p(-1.0, d) == pytest.approx(1.0)
    assert R.z_desc(1.0, [0.0, 0.0]) is None


def test_pairs_exclude_targets_and_are_distinct():
    p = R.draw_pairs()
    assert len(p["A_rand"]) == 20 and len({tuple(map(tuple, x)) for x in p["A_rand"]}) == 20
    for (l1, h1), (l2, h2) in p["A_rand"]:
        assert (l1, l2) == (32, 33) and h1 != 2 and h2 != 10
    assert len(p["A_late"]) == 10 and len({tuple(map(tuple, x)) for x in p["A_late"]}) == 10
    for pair in p["A_late"]:
        for l, h in pair:
            assert 30 <= l <= 35 and (l, h) not in R.TARGETS
    assert R.draw_pairs() == p                                    # seeded


def test_head_edit_reproduces_submission_1_and_diagnostic():
    from memctl.analysis import edit51 as ED
    W = R.o_proj_weights([32, 33])
    for eid, sign in ((R.SUB1, -1.0), (R.DIAG, 1.0)):
        _, comps = ED.load_edit(R.EDITS51, eid)
        mine = R.head_edit(W, R.TARGETS, sign)
        for a, b in zip(comps, mine):
            assert np.array_equal(np.asarray(a["idx"]), b["idx"]) and np.array_equal(np.asarray(a["D"]), b["D"])


def _fake_recs(tmp, rows, labels):
    for r in rows:
        CM.write_json(S4.b0_path(tmp, R.SPLIT, r["case_id"]), {"populations": labels(r)})


def test_draw_sample_quota_strata_and_determinism(tmp_path):
    rows = [{"case_id": f"N-{t}{i:02d}_{s}_H{h}", "task": t, "set": s}
            for t in ("DE", "RC", "TU") for i in range(60) for s in ("ANS", "UNA-rel", "UNA-swap") for h in (0, 1)]
    lab = lambda r: (["FA"] if (r["set"] == "ANS" and r["task"] != "TU" and int(r["case_id"][4:6]) % 2) else  # noqa: E731
                     ["C"] if r["set"] == "ANS" else ["UNA"])
    _fake_recs(tmp_path, rows, lab)
    s1 = R.draw_sample(tmp_path, rows)
    assert len(s1["ids"]["C"]) == 150 and len(s1["ids"]["UNA"]) == 150
    n_fa = sum("FA" in lab(r) for r in rows)
    assert len(s1["ids"]["FA"]) == min(150, n_fa)
    (tmp_path / "sample.json").unlink()
    assert R.draw_sample(tmp_path, rows) == s1
    una = [c for c in s1["ids"]["UNA"]]
    assert sum("UNA-swap" in c for c in una) == 75                 # N's proportion (half here) by largest remainder


def test_guard_refuses_without_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUT", tmp_path)
    with pytest.raises(PermissionError):
        R.require_manifest(tmp_path)


def test_cases53_disjoint_and_complete():
    c = json.loads((ROOT / "runs/exp53/cases53_counts.json").read_text())
    assert c["pass"] and c["n_rows"] == 1950 and c["excluded_key_splits"] == ["E", "S"]
    assert c["key_overlap"]["E"]["rows_with_key_in_split"] == 0 and c["key_overlap"]["S"]["rows_with_key_in_split"] == 0


def test_randread_sign_rule_rev1b():
    """Rev 1b: every D-randread draw has a non-negative median N-dev FA gate, and its edit's read vector is sign*s*r."""
    from memctl.analysis import edit51 as ED
    c = json.loads((ROOT / "runs/exp53/calibration.json").read_text())["D_randread"]
    rr = np.load(R.VECS)["randread_unit"]
    assert len(c["edits"]) == 20 and all(x["median_scaled_gate"] >= 0 for x in c["scales"])
    for x, eid in zip(c["scales"], c["edits"]):
        assert x["sign"] == (-1.0 if x["median_scaled_gate_preflip"] < 0 else 1.0)
        _, (comp,) = ED.load_edit(R.EDITS, eid)
        assert np.allclose(np.asarray(comp["V"])[:, 0], x["sign"] * x["scale"] * rr[x["draw"]])
