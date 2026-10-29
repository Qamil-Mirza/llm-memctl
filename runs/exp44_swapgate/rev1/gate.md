# §44 rev 1 §A: swap-subset feasibility gate

Data only (tokenizers, no forward pass). Spec: research/proposals/2026-10-29-exp44-patching.md rev 1 §A; rules in gate.json `rules` and in scripts/exp44_swap_gate.py. Counts only.

## qwen3b

| | dev (0-1) | confirmation (2-4) | all |
|---|---|---|---|
| population | 126 | 204 | 330 |
| eligible | 36 | 54 | 90 |

Eligible by LongMemEval question type:

| type | dev | confirmation | all |
|---|---|---|---|
| knowledge-update | 10 | 13 | 23 |
| multi-session | 0 | 5 | 5 |
| single-session-assistant | 8 | 16 | 24 |
| single-session-preference | 0 | 0 | 0 |
| single-session-user | 17 | 16 | 33 |
| temporal-reasoning | 1 | 4 | 5 |

Eligible by §23 label:

| label | confirmation | all |
|---|---|---|
| success | 27 | 47 |
| refusal | 15 | 21 |
| wrong | 12 | 22 |

Exclusions in order of application (of 330):

| exclusion | removed | remaining |
|---|---|---|
| gold_zero_evidence | 180 | 150 |
| gold_several_evidence | 5 | 145 |
| gold_in_question | 11 | 134 |
| gold_elsewhere | 39 | 95 |
| no_substitute | 1 | 94 |
| first_token_equal | 4 | 90 |
| alignment_failure | 0 | 90 |

Eligible by surface type (all): {'number': 37, 'name': 28, 'other': 23, 'date': 2}; population surface types: {'number': 133, 'other': 141, 'name': 48, 'date': 8}; distinct substitutes used: 19.
Regex vs normalised cross-check: {'checks': 1218, 'disagreements': 0}; continuation split unstable: 0.

Gate arithmetic: eligible folds 2-4 = 54 (need >= 50); question types with >= 15 = {'single-session-assistant': 16, 'single-session-user': 16} (need >= 2); in-view failures folds 2-4 = 27 (>= 30 proceed, 15-29 pool, < 15 stop); all 330: 43. **Verdict: pool (claims read on all 330)**

## llama3b

| | dev (0-1) | confirmation (2-4) | all |
|---|---|---|---|
| population | 126 | 204 | 330 |
| eligible | 39 | 56 | 95 |

Eligible by LongMemEval question type:

| type | dev | confirmation | all |
|---|---|---|---|
| knowledge-update | 12 | 14 | 26 |
| multi-session | 0 | 5 | 5 |
| single-session-assistant | 9 | 16 | 25 |
| single-session-preference | 0 | 0 | 0 |
| single-session-user | 17 | 17 | 34 |
| temporal-reasoning | 1 | 4 | 5 |

Eligible by §23 label:

| label | confirmation | all |
|---|---|---|
| success | 44 | 71 |
| refusal | 6 | 12 |
| wrong | 6 | 12 |

Exclusions in order of application (of 330):

| exclusion | removed | remaining |
|---|---|---|
| gold_zero_evidence | 180 | 150 |
| gold_several_evidence | 5 | 145 |
| gold_in_question | 11 | 134 |
| gold_elsewhere | 39 | 95 |
| no_substitute | 0 | 95 |
| first_token_equal | 0 | 95 |
| alignment_failure | 0 | 95 |

Eligible by surface type (all): {'number': 39, 'name': 28, 'other': 24, 'date': 4}; population surface types: {'number': 133, 'other': 141, 'name': 48, 'date': 8}; distinct substitutes used: 20.
Regex vs normalised cross-check: {'checks': 1121, 'disagreements': 0}; continuation split unstable: 0.

Gate arithmetic: eligible folds 2-4 = 56 (need >= 50); question types with >= 15 = {'single-session-assistant': 16, 'single-session-user': 17} (need >= 2); in-view failures folds 2-4 = 12 (>= 30 proceed, 15-29 pool, < 15 stop); all 330: 24. **Verdict: stop** (descriptive; the gate binds on Qwen 3B only)
