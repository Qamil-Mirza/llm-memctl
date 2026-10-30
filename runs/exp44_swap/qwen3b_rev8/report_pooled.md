# §44 swap conditions S / U: qwen3b rev 8, folds [0, 1, 2, 3, 4]

**Rev 8 (83fe60a, D-47): a repaired design tested on the same rows after the fault was seen.** No claim here is confirmatory. The rows whose substitute is unchanged give no new information; the fresh evidence is the changed-substitute rows.

S0 on the changed-substitute rows (successes): 0.700 [0.400, 1.000] (n=10, k=7) -> swap not valid; on the unchanged rows (already known from rev 1): 0.857 [0.714, 0.964] (n=28, k=24). Changed rows on disk 23 (labels {'success': 10, 'refusal': 4, 'wrong': 9}; missing 0).

Power, stated in advance (rev 8): S1(c) (half-width <= 0.15) needs about 40 failures; with fewer, S1(c) is expected to fail on power even if the repair is sound: 'not reliably distinguishable on these rows, underpowered', documented, no further rev. S0 on the repaired design is the question rev 8 can answer. Failures here: 32 (underpowered for S1(c)).

Spec: research/proposals/2026-10-29-exp44-patching.md rev 1 amendment §C, rev 2. Status per claim: {"S0": "read (rev 8: post-hoc-repaired design, not confirmatory)", "S1": "not read: S0 failed", "S2": "descriptive", "S3": "descriptive / not read", "U1": "read (rev 8: post-hoc-repaired design, not confirmatory)"}

Swap rows 70 (labels {'success': 38, 'refusal': 14, 'wrong': 18}); patched 57; U rows 97.
Classes at 1 nat by §23 label: {'success': {'answer_production': 31, 'evidence_use': 3, 'indeterminate': 4}, 'refusal': {'answer_production': 11, 'evidence_use': 0, 'indeterminate': 3}, 'wrong': {'answer_production': 15, 'evidence_use': 0, 'indeterminate': 3}}

**S0** following rate among successes: 0.816 [0.684, 0.921] (n=38, k=31); by margin {'0.5': 0.8157894736842105, '1': 0.8157894736842105, '2': 0.8157894736842105}; need >= 0.80 and lower >= 0.70 -> swap not valid

**S1** failures n=32; shares by margin {"0.5": {"answer_production": 0.8125, "evidence_use": 0.0, "indeterminate": 0.1875}, "1": {"answer_production": 0.8125, "evidence_use": 0.0, "indeterminate": 0.1875}, "2": {"answer_production": 0.8125, "evidence_use": 0.03125, "indeterminate": 0.15625}}; answer-production (1 nat) 0.812 [0.656, 0.938], half-width 0.141; (a) True (b) True (move 0.000) (c) True -> established on a post-hoc-repaired design: answer-production failures are the majority

**S2** refusal rows: AP n=9 p=0.556, EU n=0 p=—; difference — [—, —] -> descriptive (fewer than 8 refusal rows in a class)

**S3** d_T forward median 0.889 [0.889, 0.889] (n=57), reverse 0.889 [0.889, 0.889]; asymmetry median 0.000; beside {"rev0_d_c": {"n": 48, "median": 0.7777777777777778}, "lens_logit": {"n": 50, "median": 0.9027777777777777}, "lens_jlens": {"n": 49, "median": 0.8888888888888888}}

S3 validity -> fallback (specificity ratio <= 0.25 from the median d_T onward). Shift: shift(l) = Delta_clean - Delta_patched(l), positive = toward S; mean over the patched (following) rows; same quantity for donor, random and forward patch.
Absolute check (primary): mismatched-donor (difference patch) max |mean shift| 1.106 (bound 0.5) -> within False
Fallback (specificity ratio 0.25, layers >= l0 = 32): pass True, failing layers [] -- FALLBACK USED

Mismatched-donor mean shift per layer: [-0.002, -0.002, -0.002, -0.003, -0.003, 0.0, 0.002, -0.006, -0.011, -0.01, -0.011, -0.011, -0.002, -0.015, 0.012, -0.008, -0.005, -0.012, -0.043, 0.017, 0.025, 0.044, -0.06, 0.051, 0.103, 0.126, 0.204, 0.866, 0.58, 0.455, 0.66, 0.763, 1.076, 1.106, 1.06, 0.857]
Forward-patch mean shift per layer: [0.001, 0.0, 0.0, -0.003, -0.004, -0.005, 0.0, 0.002, -0.001, -0.006, -0.011, -0.01, -0.008, -0.006, -0.005, 0.001, 0.0, 0.011, 0.014, 0.037, 0.874, 0.981, 0.995, 1.049, 1.11, 1.169, 1.412, 3.809, 4.161, 4.747, 5.239, 10.538, 17.637, 27.015, 27.323, 28.065]
Random (difference-norm perturbation) mean shift per layer: [0.002, 0.002, -0.003, -0.002, 0.001, 0.002, -0.001, 0.004, -0.014, 0.009, 0.001, 0.016, -0.009, -0.006, 0.008, 0.018, -0.001, -0.004, 0.021, -0.029, 0.102, -0.01, 0.066, -0.016, 0.133, 0.234, 0.181, 0.092, 0.31, 0.044, 0.136, 0.112, 0.986, 1.777, 1.562, 1.811]

**U1** (full population refusals in scope) gold under U 0.228 [0.152, 0.315] (n=92, no marker 5); U_steer share 0.380; Steer-under-U - U 0.152 [0.065, 0.239]; U still refuses 0.424 -> not the format
U1 on the swap subset: 0.389 [0.167, 0.611] (n=18) -> partial
