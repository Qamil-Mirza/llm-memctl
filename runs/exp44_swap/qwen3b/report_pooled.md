# §44 swap conditions S / U: qwen3b, folds [0, 1, 2, 3, 4]

Spec: research/proposals/2026-10-29-exp44-patching.md rev 1 amendment §C, rev 2. Status per claim: {"S0": "read", "S1": "not read: S0 failed", "S2": "descriptive", "S3": "descriptive / not read", "U1": "read"}

Swap rows 89 (labels {'success': 46, 'refusal': 21, 'wrong': 22}); patched 63; U rows 97.
Classes at 1 nat by §23 label: {'success': {'answer_production': 34, 'evidence_use': 4, 'indeterminate': 8}, 'refusal': {'answer_production': 13, 'evidence_use': 1, 'indeterminate': 7}, 'wrong': {'answer_production': 16, 'evidence_use': 2, 'indeterminate': 4}}

**S0** following rate among successes: 0.739 [0.609, 0.870] (n=46, k=34); by margin {'0.5': 0.7391304347826086, '1': 0.7391304347826086, '2': 0.7391304347826086}; need >= 0.80 and lower >= 0.70 -> swap not valid

**S1** failures n=43; shares by margin {"0.5": {"answer_production": 0.6744186046511628, "evidence_use": 0.023255813953488372, "indeterminate": 0.3023255813953488}, "1": {"answer_production": 0.6744186046511628, "evidence_use": 0.06976744186046512, "indeterminate": 0.2558139534883721}, "2": {"answer_production": 0.6744186046511628, "evidence_use": 0.06976744186046512, "indeterminate": 0.2558139534883721}}; answer-production (1 nat) 0.674 [0.535, 0.814], half-width 0.140; (a) False (b) True (move 0.000) (c) True -> not reliably distinguishable on these rows

**S2** refusal rows: AP n=11 p=0.455, EU n=1 p=0.000; difference 0.455 [0.182, 0.727] -> descriptive (fewer than 8 refusal rows in a class)

**S3** d_T forward median 0.889 [0.889, 0.889] (n=63), reverse 0.889 [0.889, 0.889]; asymmetry median 0.000; beside {"rev0_d_c": {"n": 26, "median": 0.7777777777777778}, "lens_logit": {"n": 53, "median": 0.9166666666666666}, "lens_jlens": {"n": 52, "median": 0.8888888888888888}}

S3 validity -> fallback (specificity ratio <= 0.25 from the median d_T onward). Shift: shift(l) = Delta_clean - Delta_patched(l), positive = toward S; mean over the patched (following) rows; same quantity for donor, random and forward patch.
Absolute check (primary): mismatched-donor (difference patch) max |mean shift| 1.218 (bound 0.5) -> within False
Fallback (specificity ratio 0.25, layers >= l0 = 32): pass True, failing layers [] -- FALLBACK USED

Mismatched-donor mean shift per layer: [-0.0, -0.001, -0.0, -0.0, 0.001, 0.002, 0.005, 0.001, -0.001, -0.001, -0.001, 0.004, -0.001, -0.012, -0.005, -0.026, -0.025, -0.026, -0.038, 0.004, 0.012, 0.077, 0.054, 0.074, 0.282, 0.31, 0.146, 0.608, 0.449, 0.345, 0.476, 0.698, 1.218, 1.209, 1.149, 1.058]
Forward-patch mean shift per layer: [0.001, 0.0, 0.0, -0.002, -0.001, -0.003, 0.003, 0.005, -0.002, -0.007, -0.007, -0.006, -0.01, -0.007, -0.008, -0.004, -0.007, 0.008, 0.019, 0.008, 0.833, 0.896, 0.917, 0.961, 1.127, 1.207, 1.494, 3.904, 4.32, 4.968, 5.643, 11.148, 17.741, 26.71, 26.932, 27.302]
Random (difference-norm perturbation) mean shift per layer: [0.002, 0.001, 0.001, -0.0, 0.004, 0.005, 0.004, -0.006, -0.008, -0.004, 0.0, 0.011, 0.006, 0.003, -0.002, -0.011, 0.003, 0.012, 0.01, -0.019, 0.022, 0.012, 0.007, 0.123, 0.051, 0.079, 0.088, 0.242, 0.181, 0.045, 0.047, 0.331, 1.23, 1.733, 1.624, 1.883]

**U1** (full population refusals in scope) gold under U 0.228 [0.152, 0.315] (n=92, no marker 5); U_steer share 0.380; Steer-under-U - U 0.152 [0.065, 0.239]; U still refuses 0.424 -> not the format
U1 on the swap subset: 0.389 [0.167, 0.611] (n=18) -> partial
