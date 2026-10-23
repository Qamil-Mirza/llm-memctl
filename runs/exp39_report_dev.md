# §39 report: the §38 layer-29 direction at the ':' position only (DEVELOPMENT ONLY, folds 0-1)

Run runs/exp39_dev/qwen3b (engine hf-transformers-5.17.0-cpu-float32); source runs/exp38_dev/qwen3b. Layer 29, a2 vnorm, Steer-k k in [1, 2, 4]. Gold: D1 string rule, PROVISIONAL; ans is the primary quantity.
directions.npz sha256 be1e3a09bdb5157d24126ebe16b14c035d429ebd51409e9b615b347c5b29f921; v_29 sha256 a7f967ecdf2f59a9e1f64d99e366d33011f27daa8b09b18f5ff33d1e1ba20c6c; vector added sha256 e0000366813f3c94cf4875c5ba948fc7f4bac220099cc8848edb79196045d71c.
Random-0: seeded layer-29 random vector (rng [38, 29], norm ||2 v_29||), rebuilt deterministically; NOT the pod's layer-29 random vector (bf16 re-estimate, different norm); sha256 633bc895c2af0dc7f5c88d235c557eaaa4fa7955d2e3c4ba7efdfe9fa2269513 (vector added in this run: 633bc895c2af0dc7f5c88d235c557eaaa4fa7955d2e3c4ba7efdfe9fa2269513).

## Reproduction
Gate (flag rule against the §38 dev rows, bound 0.8): 0.889 -> pass (R 1.000; dropped ['07741c44', '0bb5a684', '19b5f2b3', '22d2cb42']). vs §35 cache 0.889.
Byte-identity with the §38 dev rows: 64/64 (1.000; stop below 0.9) -> pass.
Flags match, bytes differ (kept, marked): none.
Rows whose bytes differ: none.
Same-run repeats (descriptive): unsteered A in the arms stage = repro A on 28/28; unsteered cut = §38 A-cut on 28/28.

## Reading (§3, pre-registered)

M1 supported (development only, gold PROVISIONAL): a0 = 26/28, aA = 26/28, g0 = 7/28, gA = 8/28, ans(Random-0) = 0/28. M1 needs a0 >= 21/28, g0 >= 7/28, Random-0 <= 5/28; M2 needs a0 <= 12/28; split needs a0 >= 21/28 and g0 <= 3/28. Meaning: the pushed ':' state carries the answer path (the ':' K/V above layer 29 stay pushed for every later token).

## Arms on the 28 reproduced R rows (pairs 13, unseen 15)

| arm | n | ans | degenerate | refusal | D1 gold (PROVISIONAL) | ans pairs | gold pairs | ans unseen | gold unseen |
|---|---|---|---|---|---|---|---|---|---|
| A (unsteered) | 28 | 0/28 (0.000) | 0/28 (0.000) | 28/28 (1.000) | 0/28 (0.000) | 0/13 (0.000) | 0/13 (0.000) | 0/15 (0.000) | 0/15 (0.000) |
| B (unsteered) | 28 | 13/28 (0.464) | 0/28 (0.000) | 15/28 (0.536) | 6/28 (0.214) | 13/13 (1.000) | 6/13 (0.462) | 0/15 (0.000) | 0/15 (0.000) |
| Steer-all (§38, reused) | 28 | 26/28 (0.929) | 0/28 (0.000) | 2/28 (0.071) | 8/28 (0.286) | 13/13 (1.000) | 6/13 (0.462) | 13/15 (0.867) | 2/15 (0.133) |
| Steer-0 | 28 | 26/28 (0.929) | 0/28 (0.000) | 2/28 (0.071) | 7/28 (0.250) | 13/13 (1.000) | 6/13 (0.462) | 13/15 (0.867) | 1/15 (0.067) |
| Steer-0..1 | 28 | 26/28 (0.929) | 0/28 (0.000) | 2/28 (0.071) | 7/28 (0.250) | 13/13 (1.000) | 6/13 (0.462) | 13/15 (0.867) | 1/15 (0.067) |
| Steer-0..2 | 28 | 26/28 (0.929) | 0/28 (0.000) | 2/28 (0.071) | 7/28 (0.250) | 13/13 (1.000) | 6/13 (0.462) | 13/15 (0.867) | 1/15 (0.067) |
| Steer-0..4 | 28 | 26/28 (0.929) | 0/28 (0.000) | 2/28 (0.071) | 7/28 (0.250) | 13/13 (1.000) | 6/13 (0.462) | 13/15 (0.867) | 1/15 (0.067) |
| Random-0 | 28 | 0/28 (0.000) | 0/28 (0.000) | 28/28 (1.000) | 0/28 (0.000) | 0/13 (0.000) | 0/13 (0.000) | 0/15 (0.000) | 0/15 (0.000) |
| A-cut (unsteered) | 28 | 4/28 (0.143) | 0/28 (0.000) | 24/28 (0.857) | 2/28 (0.071) | 2/13 (0.154) | 1/13 (0.077) | 2/15 (0.133) | 1/15 (0.067) |
| Steer-0-cut | 28 | 17/28 (0.607) | 0/28 (0.000) | 11/28 (0.393) | 2/28 (0.071) | 8/13 (0.615) | 1/13 (0.077) | 9/15 (0.600) | 1/15 (0.067) |
| Steer-all-cut | 28 | 17/28 (0.607) | 0/28 (0.000) | 11/28 (0.393) | 3/28 (0.107) | 8/13 (0.615) | 2/13 (0.154) | 9/15 (0.600) | 1/15 (0.067) |

## Dose curve (descriptive)

| steps steered | ans | D1 gold (PROVISIONAL) | refusal | degenerate |
|---|---|---|---|---|
| steps 0 | 26/28 (0.929) | 7/28 (0.250) | 2/28 (0.071) | 0/28 (0.000) |
| steps 0-1 | 26/28 (0.929) | 7/28 (0.250) | 2/28 (0.071) | 0/28 (0.000) |
| steps 0-2 | 26/28 (0.929) | 7/28 (0.250) | 2/28 (0.071) | 0/28 (0.000) |
| steps 0-4 | 26/28 (0.929) | 7/28 (0.250) | 2/28 (0.071) | 0/28 (0.000) |
| all steps | 26/28 (0.929) | 8/28 (0.286) | 2/28 (0.071) | 0/28 (0.000) |

## Cut shares (descriptive; plausibility share = gold(cut) / gold(intact), same rows)

- Steer-0-cut: gold 2/28 vs intact 7/28 -> plausibility share 0.286; ans 17 vs 26 (share 0.654)
- Steer-all-cut: gold 3/28 vs intact 8/28 -> plausibility share 0.375; ans 17 vs 26 (share 0.654)
- A-cut (unsteered): gold 2/28 vs intact 0/28 -> plausibility share -; ans 4 vs 0 (share -)
- cut prompts without a marker (no note): 8

## Damage-0 (descriptive)

- reproduced damage rows (4): A ans 4/4 (1.000), gold 4/4 (1.000); Damage-0 ans 4/4 (1.000), gold 4/4 (1.000), refusal 0/4 (0.000)
- all damage rows (8): A ans 8/8 (1.000), gold 4/8 (0.500); Damage-0 ans 8/8 (1.000), gold 4/8 (0.500), refusal 0/8 (0.000)

## Steer-0 against Steer-all per row (descriptive, amendment 1.6)

First token the same on 28/28; full text identical on 11/28.

| question | first token same | text same | Steer-0 | Steer-all |
|---|---|---|---|---|
| 37f165cf | yes | yes | 'unknown' (refusal) | 'unknown' (refusal) |
| 4388e9dd | yes | no | 'Andy is described as wearing an untidy, stained white shirt.' (answer) | 'Andy is described as wearing an untidy, stained white shirt in the script.' (answer) |
| 4adc0475 | yes | yes | '5' (answer) | '5' (answer) |
| 4bc144e2 | yes | yes | '$55' (answer) | '$55' (answer) |
| 51b23612 | yes | no | 'Political propaganda examples using humor and satire, such as the Soviet cartoon mocking Western culture' (answer) | 'Political Propaganda Cartoon Mentioned Was Not Explicitly Named' (answer) |
| 60d45044 | yes | no | 'Favorite rice type not specified' (answer) | 'Favorite rice not specified' (answer) |
| 66f24dbb | yes | yes | 'Yellow dress and a pair of earrings' (answer) | 'Yellow dress and a pair of earrings' (answer) |
| 6ade9755 | yes | yes | 'Serenity Yoga' (answer) | 'Serenity Yoga' (answer) |
| 75832dbd | yes | no | 'Romance' (answer) | 'Romance Genre Book Recommendations Needed' (answer) |
| 7e00a6cb | yes | yes | 'Red Light District Hostel' (answer) | 'Red Light District Hostel' (answer) |
| 89527b6b | yes | no | 'The Plesiosaur has a blue scaly body in the image.' (answer) | 'The Plesiosaur had a blue scaly body in the image.' (answer) |
| 89941a94 | yes | no | 'The memory does not contain information about the user having other bikes besides the mountain bike' (answer) | 'The user has a road bike and has been using it for long rides, including' (answer) |
| 8ebdbe50 | yes | no | 'Data Science certification' (answer) | 'Data Science Certification' (answer) |
| b320f3f8 | yes | yes | 'Blue Snaggletooth action figure' (answer) | 'Blue Snaggletooth action figure' (answer) |
| b6025781 | yes | no | 'Next week, user could try recipes that incorporate quinoa and roasted vegetables as protein' (answer) | 'Next Protein Sources for Quinoa and Roasted Vegetables: Beans, Lentils' (answer) |
| b86304ba | yes | yes | 'unknown' (refusal) | 'unknown' (refusal) |
| caf03d32 | yes | yes | 'user' (answer) | 'user' (answer) |
| cc539528 | yes | no | 'Back-end programming languages like Ruby, Python, or PHP.' (answer) | 'Back-end programming languages like Ruby, Python, or PHP were recommended for learning,' (answer) |
| d6062bb9 | yes | no | '1998' (answer) | '1998 Views' (answer) |
| d905b33f | yes | yes | '20%' (answer) | '20%' (answer) |
| fca70973 | yes | no | 'Theme park recommendations not provided in the given conversation.' (answer) | 'Theme Park Events: Consider checking the websites of popular parks like Disneyland, Knott' (answer) |
| gpt4_1d80365e | yes | no | '1' (answer) | '1 Day' (answer) |
| gpt4_213fd887 | yes | no | 'Volleyball league' (answer) | 'Volleyball League (1st Event)' (answer) |
| gpt4_4929293a | yes | no | "Cousin's wedding" (answer) | "Cousin's Wedding" (answer) |
| gpt4_4ef30696 | yes | no | '35' (answer) | '25' (answer) |
| gpt4_74aed68e | yes | no | '21' (answer) | '21 days' (answer) |
| gpt4_88806d6e | yes | yes | 'Tom' (answer) | 'Tom' (answer) |
| gpt4_98f46fc6 | yes | no | 'First event not mentioned' (answer) | 'First event participation cannot be determined from the given information.' (answer) |

Development only: no claim on folds 2-4.