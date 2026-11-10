# selfdoubt spend log

Two accounts, two budgets (user decisions; per-pod approval lines in the Executor pane are required on both):
- **berkeley** (`RUNPOD_API_KEY`, qamilmirza@berkeley.edu): the ≈ $1.66 left under the programme's $36 cap (2026-11-08).
- **gmail** (`RUNPOD_API_KEY_2`, qamilmirza@gmail.com): $25, all of it for this project, separate from the $36 cap
  (2026-11-09). Pre-flight 2026-11-09 (read-only, $0): balance $25.0000, 0 pods.

Launch with `KEY_VAR=RUNPOD_API_KEY_2 EXPECT_EMAIL=qamilmirza@gmail.com` for the gmail account. The launcher aborts
unless the email matches, the balance covers the hard cap and the account has 0 pods. Times are Pacific (PST = UTC−8
on these dates).

| # | Account | Date (PT; UTC in brackets) | What | Pod | Approved cap | Lifetime | Billed | Running total (this account) | Left (this account) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | berkeley | 2026-11-09 07:58:52–08:17:16 PT (15:58:52–16:17:16 UTC) | Stage 0 timed stub (sizing only, 10 dev items × Qwen3-8B + Qwen3-1.7B, max_tokens 16384) | n5vlwx4k2olf2l (A40 SECURE, $0.59/h, disk 80) | $0.35 / 35 min | 1,104 s (18.4 min) | $0.185 (billing record $0.18488) | $0.185 | ≈ $1.475 of ≈ $1.66 |
| 2 | berkeley | 2026-11-09 08:36:44–10:45:51 PT (16:36:44–18:45:51 UTC) | Stage 0, user's 1.7B-first plan of 2026-11-09 (held-out: 1.7B GSM8K 250, 1.7B MATH-500 first 100 in seed-56 shuffle order, 8B GSM8K 250; selfdoubt/stage0_full_pod.sh) | 7j0yqkyb346ymt (A40 SECURE, $0.59/h, disk 80) | quote ≈ $1.26 (~2.1 h); hard DELETE at create + 2 h 15 min ≈ $1.35 max | 7,747 s (2 h 9.1 min) | $1.300 (billing record $1.30029, 7,747,545 ms) | $1.485 | ≈ $0.175 of ≈ $1.66 |
| 3 | gmail | 2026-11-09 16:04:43–19:51:56 PT (2026-11-10 00:04:43–03:51:56 UTC) | Stage 0 completion pod 1 (PROPOSAL decision 7): Qwen3-1.7B GSM8K dev 250; MATH-500 held-out re-run of the 19 incomplete items + items 101–250 (plan selfdoubt/plans/s0c_pod1.env) | oep20eldnrgkcg (A40 SECURE, $0.59/h, disk 80) | quote central $2.79 (4.7 h), ceiling $3.62; hard DELETE at create + 27,700 s (7 h 42 min) = $4.54 max | 13,633 s (3 h 47.2 min) | posting: ≈ $2.23 at $0.59/h ($1.701 posted at 19:52 PT) | pending (row 4 running) | pending |
| 4 | gmail | 2026-11-09 from 16:15:19 PT (2026-11-10 00:15:19 UTC) | Stage 0 completion pod 2 (PROPOSAL decision 7): Qwen3-8B GSM8K held-out 250; MATH-500 held-out first 100 (plan selfdoubt/plans/s0c_pod2.env) | x3akypcjbh6x0l (A40 SECURE, $0.59/h, disk 80) | quote central $2.84 (4.8 h), ceiling $3.49; hard DELETE at create + 26,700 s (7 h 25 min) = $4.38 max | running | pending | pending | pending |
| 5 | gmail | 2026-11-09 17:16:41–18:36:50 PT (2026-11-10 01:16:41–02:36:50 UTC) | Combined DEV pod (Advisor item 9): Stage 2 pilot (Qwen3-1.7B, GSM8K dev rows ec37f18e, 80 stale + 80 new) then Stage 1 dev (250 items × 15 continuations, cut 0.9, payloads plain + doc); plan selfdoubt/plans/s1s2_dev.env | m8hbjos8105hu9 (A40 SECURE, $0.59/h, disk 80) | quote central $1.00 (1 h 42 min), ceiling $1.52; hard DELETE at create + 11,700 s (3 h 15 min) = $1.92 max | 4,809 s (1 h 20.2 min) | $0.807 (billing record $0.80708, 4,808,758 ms) | pending (rows 3–4 running) | pending |

Running totals: berkeley $1.485 of ≈ $1.66 (≈ $0.175 left; programme ≈ $35.83 of $36; balance $3.5692). gmail $0.000 of $25.

Notes
- Row 1: the billing record posted by 10:46 PT ($0.18488), matching the balance drop.
- Row 1 approval: the user's "Approved" in the Executor pane, 2026-11-09, after the quote above (Advisor-pane approval
  2026-11-08: "a timed test on one A40, hard cap $0.35 (35 min)").
- Row 2 approval: the user's line in the Executor pane, 2026-11-09: "approved: self-doubt Stage 0, 1.7B-first plan,
  one A40, hard cap $1.35 / 2 h 15 min". First create at ~08:35 PT (16:35 UTC) failed (no A40 capacity, no pod, $0); the
  identical create succeeded on retry at 08:36:44 PT. Hard DELETE timer at create + 8,100 s: 10:51:44 PT (18:51:44 UTC).
  Outcome: step 1 (1.7B GSM8K 250) done 09:08 PT. Step 2 (1.7B MATH-500 100): first answers done 09:22 PT; continuations
  stopped at the run deadline 10:45 PT with 945 of 1,100 finished. Step 3 (8B GSM8K) never started. The waiter deleted
  the pod at 10:45:51 PT; 0 pods after. Billed: the per-pod record, fully posted when checked at 13:49 PT, is $1.30029. The
  balance drop of $1.278 measured at 10:46 PT was $0.022 short because billing was still posting.
- Rows 3–4 approval: the user's line in the Executor pane, 2026-11-09: "approved: self-doubt Stage 0 completion, two A40
  pods on the gmail account, hard caps $4.54 (pod 1) and $4.38 (pod 2)". Both together: central $5.63, max $8.92 of $25.
  Pre-flight before pod 1: gmail, balance $25.0000, 0 pods. Before pod 2: balance $24.9092, 1 pod (pod 1, allowed
  sibling). Both created at $0.59/h. Hard DELETE timers: pod 1 at 23:46:23 PT, pod 2 at 23:40:19 PT (07:46:23 and
  07:40:19 UTC on 11-10). Steps started: pod 1 GSM8K dev at 16:10:19 PT; pod 2 8B GSM8K held-out at 16:16:59 PT.
- Row 5 approval: the user's "Approved" in the Executor pane, 2026-11-09, after the quote "approved: self-doubt
  combined dev pod (Stage 2 pilot + Stage 1 dev), one A40 on the gmail account, hard cap $1.92 / 3 h 15 min". Advisor
  accepted the build (93b9f9a) and the quote; Stage 1 readings frozen at fd00fcc.
  Pre-flight: gmail, balance $23.7187, 2 pods (pods 1 and 2, allowed siblings; their remaining worst case $7.605
  added to the cap). Created at $0.59/h; hard DELETE timer at 20:31:41 PT (04:31:41 UTC on 11-10). Pilot started
  17:17:50 PT (setup 20 s); first rows ~16 s each on the A40.
  Outcome: pilot done 18:00:30 PT (160 rows, 2,580 s from pod start, rc 0); Stage 1 dev 250 items done 18:36:41 PT
  (2,171 s, rc 0). The waiter copied back at 18:36:49 PT and deleted the pod at 18:36:50 PT; 2 pods after (pods 1, 2).
  Gmail balance $21.3316 at 18:40 PT with pods 1 and 2 still billing.
- Row 5 billed: the per-pod record had fully posted by 19:52 PT ($0.80708, 4,808,758 ms).
- Row 3 outcome: GSM8K dev done 16:45:32 PT (rc 0). MATH-500 held-out (items 1–250; the earlier berkeley store's done
  rows reused): FULL 150 new, CONT 1,859 new, done 19:51:26 PT (11,153 s, rc 0, not stopped at the deadline). Store
  complete: 250 items × 12 rows, 0 partial, 74 rows at max tokens. The waiter copied back and deleted the pod at
  19:51:56 PT; 1 pod after (pod 2). Gmail balance $19.7915 at 19:52 PT with pod 2 still billing.
