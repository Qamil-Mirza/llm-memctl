"""§57 stage 2: half-sample check of the K50 bootstrap lean (Advisor request, 2026-11-11).
K50 on the full FA mean vs K50 on random halves of the FA bases (seed 56); also the effective number of distinct
rows per bootstrap draw. If halves give larger K50, noise from fewer distinct rows pushes K50 up."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/qamil-mirza/Code/llm-memctl")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp57_stage2 as S2
import exp57_stage0 as S0

rows = [x for x in S0.dev_a() if x["cls"] == "FA"]
out = {}
for reg in ("P", "N"):
    M, base = [], []
    for x in rows:
        z = np.load(S2.RUN / f"{x['case_id']}.npz")
        M.append(-z[f"dhat_{reg}"].reshape(-1)); base.append(json.loads(str(z["meta"]))["base_key"])
    M = np.stack(M)
    full = S2.k50(M.mean(0, dtype=np.float64))[0]
    ub = sorted(set(base)); rng = np.random.default_rng(56)
    halves, distinct = [], []
    for _ in range(50):
        keep = set(rng.choice(ub, len(ub) // 2, replace=False))
        idx = [i for i, b in enumerate(base) if b in keep]
        halves.append(S2.k50(M[idx].mean(0, dtype=np.float64))[0])
    for _ in range(200):
        distinct.append(len(set(rng.choice(len(ub), len(ub)))) / len(ub))
    out[reg] = {"n_FA": len(rows), "n_bases": len(ub), "K50_full": int(full),
                "K50_half_q05_q50_q95": np.percentile(halves, [5, 50, 95]).tolist(),
                "share_halves_above_full": float(np.mean(np.array(halves) > full)),
                "boot_mean_distinct_base_share": float(np.mean(distinct))}
    print(reg, out[reg], flush=True)
    del M
(S2.OUT / "k50_halfsample.json").write_text(json.dumps(out, indent=1))
