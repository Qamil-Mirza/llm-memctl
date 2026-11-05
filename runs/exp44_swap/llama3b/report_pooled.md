# §44 swap conditions S / U: llama3b, folds [0, 1, 2, 3, 4]

Spec: research/proposals/2026-10-29-exp44-patching.md rev 1 amendment §C, rev 2. Status per claim: {"S0": "descriptive (Llama, rev 2)", "S1": "descriptive (Llama, rev 2)", "S2": "descriptive (Llama, rev 2)", "S3": "descriptive (Llama, rev 2)", "U1": "descriptive (Llama, rev 2)"}

Swap rows 90 (labels {'success': 66, 'refusal': 12, 'wrong': 12}); patched 74; U rows 45.
Classes at 1 nat by §23 label: {'success': {'answer_production': 55, 'evidence_use': 0, 'indeterminate': 11}, 'refusal': {'answer_production': 10, 'evidence_use': 1, 'indeterminate': 1}, 'wrong': {'answer_production': 9, 'evidence_use': 0, 'indeterminate': 3}}

**S0** following rate among successes: 0.833 [0.742, 0.924] (n=66, k=55); by margin {'0.5': 0.8333333333333334, '1': 0.8333333333333334, '2': 0.8333333333333334}; need >= 0.80 and lower >= 0.70 -> valid

**S1** failures n=24; shares by margin {"0.5": {"answer_production": 0.7916666666666666, "evidence_use": 0.041666666666666664, "indeterminate": 0.16666666666666666}, "1": {"answer_production": 0.7916666666666666, "evidence_use": 0.041666666666666664, "indeterminate": 0.16666666666666666}, "2": {"answer_production": 0.75, "evidence_use": 0.125, "indeterminate": 0.125}}; answer-production (1 nat) 0.792 [0.625, 0.958], half-width 0.167; (a) True (b) True (move 0.042) (c) False -> not reliably distinguishable on these rows

**S2** refusal rows: AP n=10 p=0.300, EU n=1 p=0.000; difference 0.300 [0.000, 0.600] -> descriptive (fewer than 8 refusal rows in a class)

**S3** d_T forward median 0.768 [0.750, 0.786] (n=74), reverse 0.786 [0.750, 0.786]; asymmetry median 0.000; beside {"rev0_d_c": {"n": 0, "median": null}, "lens_logit": {"n": 0, "median": null}, "lens_jlens": {"n": 0, "median": null}}

S3 validity -> fallback (specificity ratio <= 0.25 from the median d_T onward). Shift: shift(l) = Delta_clean - Delta_patched(l), positive = toward S; mean over the patched (following) rows; same quantity for donor, random and forward patch.
Absolute check (primary): mismatched-donor (difference patch) max |mean shift| 1.002 (bound 0.5) -> within False
Fallback (specificity ratio 0.25, layers >= l0 = 22): pass True, failing layers [] -- FALLBACK USED

Mismatched-donor mean shift per layer: [0.0, 0.0, -0.0, 0.002, 0.003, 0.032, -0.004, 0.007, 0.014, -0.027, -0.049, -0.005, 0.07, 0.205, 0.219, 0.296, 0.411, 0.291, 0.391, 0.303, 0.424, 0.505, 0.553, 0.506, 0.644, 0.564, 0.855, 1.002]
Forward-patch mean shift per layer: [-0.0, 0.0, 0.002, 0.002, 0.002, 0.002, 0.004, 0.007, 0.008, 0.012, 0.023, 0.039, 0.165, 0.365, 0.825, 2.154, 3.232, 2.829, 3.517, 3.895, 6.186, 9.06, 11.448, 11.843, 14.628, 15.157, 18.855, 19.76]
Random (difference-norm perturbation) mean shift per layer: [-0.0, 0.0, 0.0, -0.001, -0.003, -0.003, 0.003, 0.003, 0.006, 0.008, 0.01, -0.004, -0.003, 0.041, 0.045, 0.068, 0.119, 0.041, 0.198, 0.149, 0.135, 0.045, 0.143, 0.065, 0.413, 0.567, 1.032, 1.499]

**U1** (full population refusals in scope) gold under U 0.089 [0.022, 0.178] (n=45, no marker 0); U_steer share 0.222; Steer-under-U - U 0.133 [0.044, 0.244]; U still refuses 0.689 -> not the format
U1 on the swap subset: 0.083 [0.000, 0.250] (n=12) -> not the format
