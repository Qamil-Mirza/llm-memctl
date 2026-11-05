# §50 why false abstention: held read (computed)

Design sha256 `13e991b4f0512ce7cbd586e8a27e8aba51748c38108932e973561180f7541db2`.

## Stage 1 per task (joint population: answer-bearing note, gold or abstain, P_ans)
- **DE**: FA 180 (48 bases), answered 94 (39 bases); S gate INVALID; Δ_S 0.0017 [-0.13466022694252985, 0.1643309601392958] -> invalid / underpowered; Δ_M -1.2093 [-2.766445870460153, -0.0518477219960918] -> lower; R_S 0.7785590349463383, r_S -0.0022089272707085847; AUROC_M 0.5500591016548463; **profile: Other (S invalid / underpowered, M lower)**
- **RC**: FA 57 (30 bases), answered 223 (48 bases); S gate INVALID; Δ_S -0.0017 [-0.0051915816503451135, -1.1279027635672498e-05] -> invalid / underpowered; Δ_M -3.1288 [-3.871919466034577, -2.2164249814488937] -> lower; R_S 0.0048859212059874394, r_S undefined; AUROC_M 0.9000865392180002; **profile: Other (S invalid / underpowered, M lower)**
- **TU**: FA 14 (9 bases), answered 258 (48 bases); S gate INVALID; Δ_S -0.1392 [-0.3383375321355732, 0.026163606631769094] -> invalid / underpowered; Δ_M -4.4261 [-7.291698728221306, -1.7125043302108784] -> invalid / underpowered; R_S 0.2856442916526082, r_S undefined; AUROC_M 0.8909191583610189; **profile: underpowered**

## Q-2 flip pairs (selected, conditional)
- pairs 47 from 43 bases
- paired M_city (abstained − answered): -2.500 [-3.960, -1.279] (n 47, bases 43)
- paired s: -0.011 [-0.033, +0.004] (n 47, bases 43)

## Q-4 / Q-5
- {"spearman_s_m_answered": -0.5640837246881786, "spearman_kv_m_answered": -0.5423132227440287, "n": 575, "n_kv": 575, "kv_mean": {"fa": 1.629740563052038, "ans": 1.6312071107620463}, "per_task": {"label": "added after the read, descriptive; the pooled value is a composition effect across tasks", "DE": {"n": 94, "spearman_s_m": 0.016479427807679803, "n_kv": 94, "spearman_kv_m": -0.17835494707943506, "s_mean": 0.7092409726322326, "m_mean": 14.442658251904426, "kv_mean": 2.1145887939835695}, "RC": {"n": 223, "spearman_s_m": -0.11328929826687674, "n_kv": 223, "spearman_kv_m": 0.2170386273525287, "s_mean": 0.004164228541028743, "m_mean": 20.177313471054283, "kv_mean": 0.05311591694009851}, "TU": {"n": 258, "spearman_s_m": -0.14874839678024912, "n_kv": 258, "spearman_kv_m": 0.190037485033413, "s_mean": 0.2408808727721913, "m_mean": 14.983173558878343, "kv_mean": 2.8191003588220114}}}
- {"spearman_producing_vs_p_ans": 0.11358396363540545, "n": 826, "note": "exploratory read-site dependence; no caching claim"}

## Post-hoc sensitivity (not pre-registered)
- post-hoc sensitivity (Advisor recount 2026-11-02), not pre-registered; answered rows restricted to city-first answer lines; the pre-registered states above stand. delta_m applies the §4.4 cell rule (fallback to task x hardness when fine cells cover < 50% of FA); fine_cells_forced beside it
- DE: {"fine_cells_forced": {"delta_m": -4.363834516595779, "n_cells": 19, "covered_fa": 76, "lo95": -5.942056114545718, "hi95": -3.079412246403871, "note": "fine cells kept although they cover < 50% of FA when covered/n_fa < 0.5"}, "answered_name_first_excluded": 49, "n_ans_city_first": 45, "n_fa": 180, "delta_m": -5.028123158738504, "coarse": true, "n_cells": 3, "covered_fa": 180, "lo95": -6.273295813419131, "hi95": -3.8967181719708552}
- RC: {"fine_cells_forced": {"delta_m": -3.128779007697364, "n_cells": 19, "covered_fa": 57, "lo95": -3.871919466034577, "hi95": -2.2164249814488937, "note": "fine cells kept although they cover < 50% of FA when covered/n_fa < 0.5"}, "answered_name_first_excluded": 0, "n_ans_city_first": 223, "n_fa": 57, "delta_m": -3.128779007697364, "coarse": false, "n_cells": 19, "covered_fa": 57, "lo95": -3.871919466034577, "hi95": -2.2164249814488937}
- TU: {"fine_cells_forced": {"delta_m": -4.716104061749517, "n_cells": 10, "covered_fa": 14, "lo95": -7.343205694256407, "hi95": -2.2037906916260677, "note": "fine cells kept although they cover < 50% of FA when covered/n_fa < 0.5"}, "answered_name_first_excluded": 15, "n_ans_city_first": 243, "n_fa": 14, "delta_m": -4.716104061749517, "coarse": false, "n_cells": 10, "covered_fa": 14, "lo95": -7.343205694256407, "hi95": -2.2037906916260677}

## Stage 2: does not open: no eligible task
- stage 2 not run: opening rule not met
