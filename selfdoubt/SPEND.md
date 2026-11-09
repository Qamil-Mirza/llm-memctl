# selfdoubt spend log

Budget: the ≈ $1.66 left under the programme's $36 cap (user decision, 2026-11-08). The second RunPod account
($25) is not used yet. Every pod needs the user's own approval line in the Executor pane.

| # | Date (UTC) | What | Pod | Approved cap | Lifetime | Billed | Running total | Left of ≈ $1.66 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-09 15:58:52–16:17:16 | Stage 0 timed stub (sizing only, 10 dev items × Qwen3-8B + Qwen3-1.7B, max_tokens 16384) | n5vlwx4k2olf2l (A40 SECURE, $0.59/h, disk 80) | $0.35 / 35 min | 1,104 s (18.4 min) | $0.185 (balance $5.0544 → $4.8695; per-pod billing record not yet posted) | $0.185 | ≈ $1.475 |
| 2 | 2026-11-09 16:36:44– (running) | Stage 0, user's 1.7B-first plan of 2026-11-09 (held-out: 1.7B GSM8K 250, 1.7B MATH-500 first 100 in seed-56 shuffle order, 8B GSM8K 250; selfdoubt/stage0_full_pod.sh) | 7j0yqkyb346ymt (A40 SECURE, $0.59/h, disk 80) | quote ≈ $1.26 (~2.1 h); hard DELETE at create + 2 h 15 min ≈ $1.35 max | — | — | ≤ $1.535 | ≥ ≈ $0.125 |

Notes
- Row 1: the per-pod billing record had not posted by 2026-11-09 ~16:35 UTC; the balance drop ($0.185, no other pods
  ran) is used. Replace with the posted figure when it appears.
- Row 1 approval: the user's "Approved" in the Executor pane, 2026-11-09, after the quote above (Advisor-pane approval
  2026-11-08: "a timed test on one A40, hard cap $0.35 (35 min)").
- Row 2 approval: the user's line in the Executor pane, 2026-11-09: "approved: self-doubt Stage 0, 1.7B-first plan,
  one A40, hard cap $1.35 / 2 h 15 min". First create at ~16:35 UTC failed (no A40 capacity, no pod, $0); the
  identical create succeeded on retry at 16:36:44. Hard DELETE timer at create + 8,100 s (18:51:44 UTC).
  Fill in lifetime and billed $ after the run.
