"""§44 rev 8 device check (laptop vs GPU 7B rows): equality, R differences, eligibility / gap flips (no model)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import exp44_device_check as D  # noqa: E402


def row(q, fold=2, lp_c=-0.5, lp_x=-4.0, rc=0, rx=20, R=None, d_c=None):
    r = {"question_id": q, "fold": fold, "n_layers": 4, "lp_clean": lp_c, "lp_corrupt": lp_x, "rank_clean": rc,
         "rank_corrupt": rx, "patched": R is not None}
    if R is not None:
        r.update({"R": R, "d_c": d_c, "sanity": {"r_zero": R[0]}})
    return r


def test_device_check(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    rows_a = [row("q1", R=[0.0, 0.2, 0.6, 1.0], d_c=0.5), row("q2", R=[0.0, 0.6, 0.9, 1.0], d_c=0.25),
              row("q3", lp_x=-1.6), row("q4", fold=0), row("q5"),
              row("q6", lp_x=-0.8)]
    rows_b = [row("q1", R=[0.0, 0.21, 0.61, 1.0], d_c=0.5), row("q2", R=[0.0, 0.4, 0.9, 1.0], d_c=0.5),
              row("q3", lp_x=-1.4), row("q4", fold=0),
              row("q6", lp_x=-0.1)]
    for d, rows in ((a, rows_a), (b, rows_b)):
        for r in rows:
            (d / f"{r['question_id']}.json").write_text(json.dumps(r))
        (d / "_timing.json").write_text("{}")
    D.main(["--laptop", str(a), "--gpu", str(b), "--out", str(tmp_path / "dc")])
    s = json.loads((tmp_path / "dc.json").read_text())["summary"]
    assert s["n_common"] == 4 and s["n_laptop_rows"] == 5 and s["n_gpu_rows"] == 4        # fold 0 never compared
    assert s["n_both_patched"] == 2 and s["d_c_equal"] == 1
    assert s["d_c_unequal"] == [{"question_id": "q2", "diff_layers": 1, "one_none": False}]
    assert abs(s["max_abs_R_diff_overall"] - 0.2) < 1e-9
    assert s["gap_1nat_flips"] == ["q3"] and s["gap_sign_flips"] == ["q6"] and s["eligibility_flips"] == ["q3"]
