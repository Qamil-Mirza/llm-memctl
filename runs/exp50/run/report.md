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
- {"spearman_s_m_answered": -0.5640837246881786, "spearman_kv_m_answered": -0.5423132227440287, "n": 575, "n_kv": 575, "kv_mean": {"fa": 1.629740563052038, "ans": 1.6312071107620463}}
- {"spearman_producing_vs_p_ans": 0.11358396363540545, "n": 826, "note": "exploratory read-site dependence; no caching claim"}

## Stage 2: does not open: no eligible task
- stage 2 not run: opening rule not met
