# §44 swap conditions S / U: qwen3b, folds [0, 1]

Spec: research/proposals/2026-10-29-exp44-patching.md rev 1 amendment §C, rev 2. Status per claim: {"S0": "descriptive (dev scope)", "S1": "descriptive", "S2": "descriptive", "S3": "descriptive / not read", "U1": "descriptive"}

Swap rows 10 (labels {'success': 6, 'refusal': 2, 'wrong': 2}); patched 7; U rows 0.
Classes at 1 nat by §23 label: {'success': {'answer_production': 4, 'evidence_use': 0, 'indeterminate': 2}, 'refusal': {'answer_production': 1, 'evidence_use': 0, 'indeterminate': 1}, 'wrong': {'answer_production': 2, 'evidence_use': 0, 'indeterminate': 0}}

**S0** following rate among successes: 0.667 [0.333, 1.000] (n=6, k=4); by margin {'0.5': 0.6666666666666666, '1': 0.6666666666666666, '2': 0.6666666666666666}; need >= 0.80 and lower >= 0.70 -> swap not valid

**S1** failures n=4; shares by margin {"0.5": {"answer_production": 0.75, "evidence_use": 0.0, "indeterminate": 0.25}, "1": {"answer_production": 0.75, "evidence_use": 0.0, "indeterminate": 0.25}, "2": {"answer_production": 0.75, "evidence_use": 0.0, "indeterminate": 0.25}}; answer-production (1 nat) 0.750 [0.250, 1.000], half-width 0.375; (a) False (b) True (move 0.000) (c) False -> not reliably distinguishable on these rows

**S2** refusal rows: AP n=0 p=—, EU n=0 p=—; difference — [—, —] -> descriptive (fewer than 8 refusal rows in a class)

**S3** d_T forward median 0.889 [0.750, 0.917] (n=7), reverse 0.889 [0.861, 0.917]; asymmetry median 0.000; mismatched-donor max |mean shift| 12.691 (bound 0.5) -> within False; beside {"rev0_d_c": {"n": 7, "median": 0.75}, "lens_logit": {"n": 5, "median": 0.8888888888888888}, "lens_jlens": {"n": 5, "median": 0.8888888888888888}}

Mismatched-donor mean shift per layer: [5.849, 4.276, 4.485, 4.813, 6.029, 6.599, 6.903, 6.212, 5.926, 5.793, 5.905, 6.06, 5.484, 5.179, 5.252, 5.503, 4.905, 4.968, 5.695, 5.32, 5.799, 7.192, 6.722, 6.894, 8.376, 8.424, 9.888, 10.661, 9.717, 10.673, 10.745, 11.275, 11.654, 12.563, 12.572, 12.691]
Random-overwrite mean shift per layer: [10.6, 12.06, 9.866, 11.684, 10.768, 11.726, 13.122, 13.096, 11.835, 10.868, 10.644, 11.506, 9.103, 11.016, 10.352, 9.69, 14.028, 10.439, 12.59, 11.717, 14.014, 13.385, 13.63, 11.714, 11.892, 12.652, 12.632, 10.63, 13.693, 12.36, 12.763, 15.183, 12.323, 14.034, 13.341, 15.869]

**U1** (full population refusals in scope) gold under U — [—, —] (n=0, no marker 0); U_steer share —; Steer-under-U - U — [—, —]; U still refuses — -> None
U1 on the swap subset: — [—, —] (n=0) -> None
