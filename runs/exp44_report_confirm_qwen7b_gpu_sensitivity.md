# §44 7B GPU confirmation: device-flip sensitivity (review 28fb69f hold rule)

(a) excluding the 5 device-flip rows (in the claim set: ['16c90bf4']): C1 median d_c 0.679 [0.643, 0.714] n=82 -> both partly right; C2 mean R(13) 0.003 [-0.011, 0.018] -> carries little
(b) median and CI shifted -0.036 (one layer): 0.643 [0.607, 0.679] -> both partly right
(b) median and CI shifted +0.036 (one layer): 0.714 [0.679, 0.750] -> both partly right

Hold rule: the C1 verdict is the same under (a) and (b), so it stands; the laptop 7B run was stopped (rows kept).
