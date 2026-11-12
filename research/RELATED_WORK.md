# Related-work ledger: self-doubt

One row per paper close to our claim. A row records what the paper does, a quoted passage from it, how we differ,
and an identifier that `python -m selfdoubt.check_refs` resolves (OpenReview IDs are checked by hand).
The weekly literature scan proposes rows; nothing is added without someone reading the paper.
Verdict: **threat** (could pre-empt our claim), **support** (we build on it), **context** (cite only).

| Paper | ID | What it does | Quoted passage | How we differ | Verdict |
|---|---|---|---|---|---|
| Wu, Miroyan et al., "Are Large Reasoning Models Interruptible?" (ICML 2026) | arXiv:2510.11713 | Names self-doubt: a valid mid-thinking update is ignored. Behavioural only. | to fill | We look inside the model for where the update is lost. | support |
| "Hypothesis Anchoring" (ICLR 2027, desk-rejected) | https://openreview.net/forum?id=qXmWvPFkZP | Injects the correct answer mid-reasoning on GSM8K; every mechanistic test null. | to fill (abstract read 2026-11-11) | Valid update rather than an injected answer; stale-answer scoring; named models. | threat |
| "Who Owns the Answer?" (ARR, May 2026) | https://openreview.net/forum?id=XAYzryI95u | Scratchpad versus prompt rules on a synthetic task in Qwen3-4B; three layer-25 heads. | to fill | Real benchmark updates mid-thinking; our models. | threat |
| "Learning to Revise When Updates Arrive Mid-Reasoning" (ARR, Oct 2026) | https://openreview.net/forum?id=kKHmd2Q7ra | Trains models on the Wu et al. setting; promises analysis protocols. | to fill | Mechanism, not training. | threat |
| "The Termination Circuit" (LessWrong, unreviewed) | https://www.lesswrong.com/posts/ajhzc6ktEKyFeJFBS/the-termination-circuit-how-reasoning-models-stop-thinking | Stop-thinking decision on Qwen3-1.7B; a late MLP carries about 40%. | to fill | A different decision (using an update, not stopping). | context |
