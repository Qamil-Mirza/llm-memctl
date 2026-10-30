# Review: §44 rev 0 confirmation read, Qwen 3B (folds 2–4) — b1b5c45

Advisor results review; recounted from `runs/exp44_patch/qwen3b/*.json` (folds 2–4). Everything matches the report.
Verdict: **ACCEPT. C3 reads "undetermined" under the pre-registered rule.**

| item | value |
|---|---|
| rows / patched | 204 / 78 (eligible only) |
| relaxed rank filter / gap-excluded / claim rows | 88 / 10 / 78 |
| R(0)-flagged | 0 (max |R(0)| 0.025); R(L−1) passes 78/78 |
| **C3 median d_c** | **0.750 [0.722, 0.778]**, n 78 (dev was 0.736 [0.667, 0.750]) |
| quartiles | 0.667 / 0.750 / 0.806; 16.7 % of rows ≥ 0.85 |
| per-row d_c − d_jlens (report) | median −0.139, q1 −0.215, q3 −0.028 |

**Pre-registered reading (rev 1 §4 correction).** "Agrees" needs the CI to overlap [0.85, 0.92] with median ≥ 0.75:
no, the interval lies entirely below 0.85. "Under-read" needs the upper bound below 0.75: no, it is 0.778. So
**C3 = undetermined**, and that is what the record says.

**Descriptive, stated plainly.** The causal depth of the 3B's answer is 0.75 of the stack, with an interval of
0.72–0.78, while the §34 lenses put its emergence at 0.92 (logit) and 0.89 (J-lens) on the same population; row by
row the causal depth is a median 0.14 earlier than the J-lens depth. The "under-read" bar was set at 0.75 before
any data and the interval straddles it by 0.03, so the pre-registered verdict is withheld, not the observation. The
honest sentence for §44: "the lens places the 3B's answer about 0.15 of the stack later than patching finds it
causally usable; the pre-registered under-read criterion (upper bound < 0.75) is missed by 0.028, so this is
reported as undetermined." No re-reading of the threshold.

**Rulings.** (1) b1b5c45 accepted; §44 states C3 undetermined first, then the descriptive sentence above. (2) The 7B
confirmation read may start now: 204 rows at about 206 s each ≈ 11.7 h on the laptop, $0, bf16 CPU (the earlier
10.2 h projection assumed unpatched rows were cheap; the two prefills dominate, so they are not). The user is
informed before the start. (3) C1 and C2 are read once on the 7B confirmation rows when they land.
