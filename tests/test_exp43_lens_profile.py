"""§43 logit-lens profile arithmetic (scripts/exp43_lens_profile.py) on stub rows; the dev reader's fold refusal.
No model is loaded."""
import numpy as np
import pytest

from scripts import exp43_lens_dev as dev
from scripts import exp43_lens_profile as prof

L = 4
UNK = (900, 901)


def row(qid, ranks, top1):
    return {"qid": qid, "gold_ranks": np.array(ranks), "top1": np.array(top1)}


ROWS = [
    row("a", [500, 50, 5, 1], [7, 7, 7, 3]),        # success, C, emerges at layer 2
    row("b", [400, 300, 200, 2], [7, 7, 7, 7]),     # success, C, emerges at layer 3
    row("c", [9000, 8000, 7000, 6000], [7, 900, 901, 900]),  # refusal, A
    row("d", [900, 8, 30, 40], [7, 7, 7, 7]),       # answered wrong, B, emerges at layer 1
]
CLS = {"a": "success", "b": "success", "c": "refusal", "d": "answered wrong"}


def test_layer_profile():
    out = prof.layer_profile(ROWS, CLS, L, UNK, k=10)
    assert [r["depth"] for r in out] == [0.0, 0.25, 0.5, 0.75]           # §34: layer / n_layers
    assert [r["gold_topk"] for r in out] == [0.0, 0.25, 0.25, 0.5]
    assert [r["gold_top1"] for r in out] == [0.0, 0.0, 0.0, 0.25]
    assert [r["median_rank_successes"] for r in out] == [450.0, 175.0, 102.5, 1.5]
    assert [r["refusal_unknown_top1"] for r in out] == [0.0, 1.0, 1.0, 1.0]  # either refusal id counts
    assert all(r["n_refusals"] == 1 and r["n"] == 4 for r in out)


def test_abc_and_emergence():
    t = prof.abc_split(ROWS, CLS, k=10)
    assert t["all"] == {"n": 4, "A": 0.25, "B": 0.25, "C": 0.5}
    assert t["success"]["C"] == 1.0 and t["refusal"]["A"] == 1.0 and t["answered wrong"]["B"] == 1.0
    assert t["all failures"] == {"n": 2, "A": 0.5, "B": 0.5, "C": 0.0}
    e = prof.emergence(ROWS, L, k=10)
    assert e["n"] == 3 and e["n_read"] == 4 and e["median"] == pytest.approx(0.5)


def test_no_successes_or_refusals_gives_none():
    out = prof.layer_profile([ROWS[3]], {"d": "answered wrong"}, L, UNK)
    assert out[0]["median_rank_successes"] is None and out[0]["refusal_unknown_top1"] is None


def test_dev_reader_refuses_confirmation_folds(capsys):
    assert dev.main(["llama3b", "--folds", "0,3"]) == dev.REFUSED
    assert "REFUSED" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        dev.check_folds((2,))
