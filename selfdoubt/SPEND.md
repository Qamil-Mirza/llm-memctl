# selfdoubt spend log

Budget: the ≈ $1.66 left under the programme's $36 cap (user decision, 2026-11-08). The second RunPod account
($25) is not used yet. Times are Pacific (PST = UTC−8 on these dates). Every pod needs the user's own approval line in the Executor pane.

| # | Date (PT; UTC in brackets) | What | Pod | Approved cap | Lifetime | Billed | Running total | Left of ≈ $1.66 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-09 07:58:52–08:17:16 PT (15:58:52–16:17:16 UTC) | Stage 0 timed stub (sizing only, 10 dev items × Qwen3-8B + Qwen3-1.7B, max_tokens 16384) | n5vlwx4k2olf2l (A40 SECURE, $0.59/h, disk 80) | $0.35 / 35 min | 1,104 s (18.4 min) | $0.185 (billing record $0.18488) | $0.185 | ≈ $1.475 |
| 2 | 2026-11-09 08:36:44–10:45:51 PT (16:36:44–18:45:51 UTC) | Stage 0, user's 1.7B-first plan of 2026-11-09 (held-out: 1.7B GSM8K 250, 1.7B MATH-500 first 100 in seed-56 shuffle order, 8B GSM8K 250; selfdoubt/stage0_full_pod.sh) | 7j0yqkyb346ymt (A40 SECURE, $0.59/h, disk 80) | quote ≈ $1.26 (~2.1 h); hard DELETE at create + 2 h 15 min ≈ $1.35 max | 7,747 s (2 h 9.1 min) | $1.278 (balance $4.8695 → $3.5915) | $1.463 | ≈ $0.197 |

Notes
- Row 1: the billing record posted by 10:46 PT ($0.18488), matching the balance drop.
- Row 1 approval: the user's "Approved" in the Executor pane, 2026-11-09, after the quote above (Advisor-pane approval
  2026-11-08: "a timed test on one A40, hard cap $0.35 (35 min)").
- Row 2 approval: the user's line in the Executor pane, 2026-11-09: "approved: self-doubt Stage 0, 1.7B-first plan,
  one A40, hard cap $1.35 / 2 h 15 min". First create at ~08:35 PT (16:35 UTC) failed (no A40 capacity, no pod, $0); the
  identical create succeeded on retry at 08:36:44 PT. Hard DELETE timer at create + 8,100 s: 10:51:44 PT (18:51:44 UTC).
  Outcome: step 1 (1.7B GSM8K 250) done 09:08 PT. Step 2 (1.7B MATH-500 100): first answers done 09:22 PT; continuations
  stopped at the run deadline 10:45 PT with 945 of 1,100 finished. Step 3 (8B GSM8K) never started. The waiter deleted
  the pod at 10:45:51 PT; 0 pods after. Billed = the balance drop (no other pods ran); the per-pod record was only part-posted
  at 10:46 PT. Replace with the full posted figure when it appears.
