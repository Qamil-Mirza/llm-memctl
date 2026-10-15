"""Development folds 0-1 only; refuses nothing else because it loads only {0, 1}. Free dev-fold check for the §36 draft (folds 0-1 only). Fixed candidate = §35 arm-B answer (frozen note, unknown
prohibited). Natural policy C0 = cached refusal flag. Baseline C4' = final-layer rank of ' unknown'/'unknown' at the
3B's own answer position (§34b logit rows; gold-free). Matched-coverage risk and AUROC."""
import json, sys, random
from pathlib import Path
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
from exp35_report import load
data, qids, _ = load({0, 1}, Path('runs/exp35'), 'runs', Path('runs/exp34_jlens_own'))
def unk_rank(R, q):
    p = Path(f'runs/exp34_jlens_own/{R}/{q}.json')
    if not p.exists(): return None
    m = json.load(open(p))['measures']; m = eval(m) if isinstance(m, str) else m
    x = m['logit']['P1']; x = eval(x) if isinstance(x, str) else x
    return min(x['unknown_sp']['final_rank'], x['unknown']['final_rank'])
def auroc(pos, neg):
    if not pos or not neg: return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return s / (len(pos) * len(neg))
for R in ('qwen3b', 'qwen7b'):
    d = data[R]
    rows = []
    for q in d.frozen:
        y = d.correct('B', q); r = unk_rank(R, q)
        if y is None or r is None: continue
        rows.append((q, y, d.cached_refusal(q), r, d.rows['B'][q]['refusal']))
    n = len(rows); k = sum(y for _, y, *_ in rows)
    print(f'{R}: fixed-candidate rows {n}; B candidate judged correct {k} ({k/n:.2f}); B still unknown {sum(r[4] for r in rows)}')
    # C0 point
    ans = [r for r in rows if not r[2]]
    cov0 = len(ans) / n; risk0 = sum(not r[1] for r in ans) / len(ans)
    print(f'  C0 natural: coverage {cov0:.2f}, risk {risk0:.3f}, answered-correct {sum(r[1] for r in ans)}')
    # C4' at matched coverage: answer the len(ans) rows with highest unknown rank (unknown least likely)
    srt = sorted(rows, key=lambda r: -r[3]); top = srt[:len(ans)]
    print(f"  C4' unknown-rank gate at matched coverage: risk {sum(not r[1] for r in top)/len(top):.3f}")
    print(f"  AUROC of unknown-rank for B-candidate correctness: {auroc([r[3] for r in rows if r[1]], [r[3] for r in rows if not r[1]]):.3f}")
    # AUROC of the natural refusal flag as a 0/1 score
    print(f"  AUROC of natural answer/refuse flag: {auroc([0 if r[2] else 1 for r in rows if r[1]], [0 if r[2] else 1 for r in rows if not r[1]]):.3f}")
    print(f"  cached refusals among fixed rows: {sum(r[2] for r in rows)}; of which B correct {sum(r[1] for r in rows if r[2])}")
