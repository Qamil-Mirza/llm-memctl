# §44 stage 0 (laptop, $0, 2026-10-29)

Spec: research/proposals/2026-10-29-exp44-patching.md rev 0 (6e21476). Code: memctl/analysis/patching.py,
scripts/exp44_patch.py, scripts/exp44_report.py. Numbers: stage0.json; rows: qwen3b/, qwen7b/; report_stage0.md.
All rows are dev (fold 0), the first rows of the §34 population in (fold, qid) order, run with --patch-all.

## (a) Qwen 3B, fp32 CPU, 5 dev rows

| qid | tokens clean/cut | rank clean/corrupt | gap (nats) | R(0) | R(L-1) | d_c | in strict / relaxed filter | sanity |
|---|---|---|---|---|---|---|---|---|
| 00ca467f | 908/645 | 3/4 | 2.69 | 0.0008 | 1.0000 | 0.583 | no / yes | PASS (asserted) |
| 06db6396 | 912/749 | 1/2 | 14.57 | 0.0002 | 1.0000 | 0.722 | no / no | PASS (recorded) |
| 0bb5a684 | 1127/937 | 2/3 | -0.12 | 0.180 | 1.0000 | 0.500 | no / no | FAIL R(0) (recorded; gap ~0, R meaningless) |
| 0db4c65d | 1003/847 | 6/1 | -1.63 | 0.0005 | 1.0000 | 0.750 | no / no | PASS (recorded) |
| 15745da0 | 836/772 | 701/4308 | 9.85 | -0.0004 | 1.0000 | 0.667 | no / no | PASS (recorded) |

R curves (every row): ~0 through about 40% depth, then rising; R >= 0.5 at 0.58-0.75 depth on the four rows with a
real gap; R(L-1) = 1 exactly (to float precision) on every row. Ranks reproduce the §34 / §34c pod reads exactly on
4/5 rows (15745da0: 701 vs 585 clean, far outside the top-10; fp32 laptop vs bf16 pod).
Timing: 52.5 s/row all-patched (two prefills ~27 s, 36 single-token patched passes ~25 s); peak RSS 16.8 GB.

## (b) Qwen 7B, bf16 CPU: feasible

Loads (weights stay mmapped: 0.5 GB RSS after load), peak RSS 14.3 GB (/usr/bin/time max RSS 14.65 GB) under
GUARD_VM_GB=22. 2 dev rows: 149.8 s and 161.1 s (155.5 s/row): the two bf16 prefills take ~139 s (this CPU, an
i7-11800H, has AVX-512 but no AVX512-BF16), the 28 patched passes only ~16.5 s. Sanity PASS on both (R(0) = -0.003,
0.008; R(L-1) = 1). Ranks match §34/§34c on both. Weights downloaded (free) into cache/exp44_hf (inside the repo).

| row | rank clean/corrupt | gap | d_c |
|---|---|---|---|
| 00ca467f | 2/1 | -0.91 | 0.893 (not in filter) |
| 06db6396 | 1/2 | 16.50 | 0.643 (not in filter) |

## Projections (330 rows; default run patches only rows in the widest pre-declared filter, clean <= 10 and corrupt > 3)

| model | s/row base | s/row patches | 330 all patched | 330 eligible-only | 126 dev eligible-only |
|---|---|---|---|---|---|
| qwen3b fp32 | 27.2 | 25.3 | 4.8 h | 3.6 h | 1.4 h |
| qwen7b bf16 | 139.0 | 16.5 | 14.3 h | 13.5 h | 5.2 h |

The 7B dev run is one long laptop afternoon / evening; all 330 rows is an overnight-plus (~13.5 h). The 3B and 7B
cannot run together (RAM 23 GB). A pod is not needed.

## Filter preview (dev, from the §34 / §34c reads already on disk, no new run)

The §34c ablated reads use byte-identical cut prompts (sha256 match on 126/126 dev rows), so their final P1 ranks
preview the §3 filter: strict (clean <= 10, corrupt > 10) survives 31/126 (3B) and 39/126 (7B), both below 40, so
the pre-declared relaxation (corrupt > 3) would apply to both: 58/126 (3B), 65/126 (7B). The laptop dev run decides
(the 7B at 39 is one row from the line, and fp32/bf16-CPU ranks can differ from the pod's).
