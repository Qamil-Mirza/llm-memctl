# §39b report: first-token-only clean-cache control (DEVELOPMENT ONLY, folds 0-1)

Run runs/exp39b_dev/qwen3b (engine hf-transformers-5.17.0-cpu-float32); §38 source runs/exp38_dev/qwen3b; §39 rows runs/exp39_dev/qwen3b (arms.jsonl sha256 03d5ac981812cb240d7c5439a3c1b49ec1d816dea9c41c332ef605581b48d212). Gold: D1 string rule, PROVISIONAL; ans is the primary quantity.
Forcing: unpushed prefix cache through the token before ':'; ':' unpushed (argmax recorded, unused); forced id = gen_ids[0]; greedy, no hook at any step; EOS / newline stop (newline token kept); max new tokens counts the forced one.
Tokenisation of a forced text after 'Answer:': first id of ids(continuation + text) after the ids(continuation) prefix (prefix asserted stable).

## Reproduction
Gate (flag rule against the §38 dev rows, bound 0.8): 0.889 -> pass (R 1.000; dropped ['07741c44', '0bb5a684', '19b5f2b3', '22d2cb42']).
Byte-identity with the §38 dev rows: 64/64 (1.000; stop below 0.9) -> pass.
Flags match, bytes differ (kept, marked): none.

## Checks
- Unknown-first equals the A row byte-exact on 28/28 (must be all; the unpushed ':' argmax equals the forced ' unknown' on 28/28).
- Step-0 first token (Steer-0, one new token) equals the §39 Steer-0 first token on 28/28 (must be all).

## Reading (amendment 2, pre-registered)

a0 = 26/28, g0 = 7/28 read from the §39 Steer-0 rows (equal to the pre-registered a0 = 26, g0 = 7).

token carries it (development only, gold PROVISIONAL): aF = 26/28, gF = 7/28 against a0 = 26/28, g0 = 7/28. State needed needs aF <= 12/28 (2 aF < a0); token carries it needs aF >= 21/28 and gF >= 6/28; split needs aF >= 21/28 and gF <= 3/28. Meaning: the first generated token alone keeps the model answering and right; the pushed ':' state is not needed.

## Arms on the 28 reproduced R rows (pairs 13, unseen 15)

| arm | n | ans | degenerate | refusal | D1 gold (PROVISIONAL) | ans pairs | gold pairs | ans unseen | gold unseen |
|---|---|---|---|---|---|---|---|---|---|
| A (unsteered) | 28 | 0/28 (0.000) | 0/28 (0.000) | 28/28 (1.000) | 0/28 (0.000) | 0/13 (0.000) | 0/13 (0.000) | 0/15 (0.000) | 0/15 (0.000) |
| Steer-0 (§39) | 28 | 26/28 (0.929) | 0/28 (0.000) | 2/28 (0.071) | 7/28 (0.250) | 13/13 (1.000) | 6/13 (0.462) | 13/15 (0.867) | 1/15 (0.067) |
| Steer-first | 28 | 26/28 (0.929) | 0/28 (0.000) | 2/28 (0.071) | 7/28 (0.250) | 13/13 (1.000) | 6/13 (0.462) | 13/15 (0.867) | 1/15 (0.067) |
| Steer-0-cut (§39) | 28 | 17/28 (0.607) | 0/28 (0.000) | 11/28 (0.393) | 2/28 (0.071) | 8/13 (0.615) | 1/13 (0.077) | 9/15 (0.600) | 1/15 (0.067) |
| Steer-first-cut | 28 | 17/28 (0.607) | 0/28 (0.000) | 11/28 (0.393) | 2/28 (0.071) | 8/13 (0.615) | 1/13 (0.077) | 9/15 (0.600) | 1/15 (0.067) |
| Gold-first | 28 | 28/28 (1.000) | 0/28 (0.000) | 0/28 (0.000) | 13/28 (0.464) | 13/13 (1.000) | 8/13 (0.615) | 15/15 (1.000) | 5/15 (0.333) |
| Unknown-first | 28 | 0/28 (0.000) | 0/28 (0.000) | 28/28 (1.000) | 0/28 (0.000) | 0/13 (0.000) | 0/13 (0.000) | 0/15 (0.000) | 0/15 (0.000) |

## Soft refusals (descriptive; answer-class phrases matching /not (specified|mentioned|provided|determined|explicitly|contain|named|available|stated|given)|does not contain|cannot be determined|no (information|mention)/, case-insensitive; §41 rev 1)

- Steer-first: 4 of 26 answers (28 rows)
- Steer-0 (§39): 4 of 26 answers (28 rows)
- Steer-first-cut: 1 of 17 answers (28 rows)
- Gold-first: 2 of 28 answers (28 rows)

## Steer-first against Steer-0 per row (descriptive)

Full text identical on 26/28; divergence index (first differing generated token) counts: {'1': 1, '2': 1}. Steer-first-cut = Steer-0-cut text on 28/28. Gold-first in-context and alone first ids disagree on: no rows.

| question | forced | Steer-first | Steer-0 | same text | divergence | Gold-first | Unknown-first = A |
|---|---|---|---|---|---|---|---|
| 37f165cf | ' unknown' | 'unknown' (refusal, gold no) | 'unknown' (refusal, gold no) | yes | - | ' ' -> '416' (answer, gold no) | yes |
| 4388e9dd | ' Andy' | 'Andy is described as wearing an untidy, stained white shirt.' (answer, gold no) | 'Andy is described as wearing an untidy, stained white shirt.' (answer, gold no) | yes | - | ' Andy' -> 'Andy is described as wearing an untidy, stained white shirt.' (answer, gold no) | yes |
| 4adc0475 | ' ' | '5' (answer, gold yes) | '5' (answer, gold yes) | yes | - | ' ' -> '5' (answer, gold yes) | yes |
| 4bc144e2 | ' $' | '$55' (answer, gold no) | '$55' (answer, gold no) | yes | - | ' $' -> '$55' (answer, gold no) | yes |
| 51b23612 | ' Political' | 'Political propaganda using humor and satire, such as the examples mentioned, did not include' (answer, gold no) | 'Political propaganda examples using humor and satire, such as the Soviet cartoon mocking Western culture' (answer, gold no) | no | 2 | ' Nu' -> 'Nu, pogodi!' (answer, gold yes) | yes |
| 60d45044 | ' Favorite' | 'Favorite rice type not specified' (answer, gold no) | 'Favorite rice type not specified' (answer, gold no) | yes | - | ' Japanese' -> 'Japanese short-grain rice' (answer, gold yes) | yes |
| 66f24dbb | ' Yellow' | 'Yellow dress and a pair of earrings' (answer, gold yes) | 'Yellow dress and a pair of earrings' (answer, gold yes) | yes | - | ' a' -> 'a yellow dress and a pair of earrings' (answer, gold yes) | yes |
| 6ade9755 | ' Ser' | 'Serenity Yoga' (answer, gold yes) | 'Serenity Yoga' (answer, gold yes) | yes | - | ' Ser' -> 'Serenity Yoga' (answer, gold yes) | yes |
| 75832dbd | ' Romance' | 'Romance' (answer, gold no) | 'Romance' (answer, gold no) | yes | - | ' The' -> 'The memory does not contain the specific question about recent publications or conferences.' (answer, gold no) | yes |
| 7e00a6cb | ' Red' | 'Red Light District Hostel' (answer, gold no) | 'Red Light District Hostel' (answer, gold no) | yes | - | ' International' -> 'International Budget Hostel' (answer, gold yes) | yes |
| 89527b6b | ' The' | 'The Plesiosaur has a blue scaly body in the image.' (answer, gold no) | 'The Plesiosaur has a blue scaly body in the image.' (answer, gold no) | yes | - | ' The' -> 'The Plesiosaur has a blue scaly body in the image.' (answer, gold no) | yes |
| 89941a94 | ' The' | 'The memory does not contain information about the user having other bikes besides the mountain bike' (answer, gold no) | 'The memory does not contain information about the user having other bikes besides the mountain bike' (answer, gold no) | yes | - | ' Yes' -> 'Yes' (answer, gold no) | yes |
| 8ebdbe50 | ' Data' | 'Data Science certification' (answer, gold yes) | 'Data Science certification' (answer, gold yes) | yes | - | ' Data' -> 'Data Science certification' (answer, gold yes) | yes |
| b320f3f8 | ' Blue' | 'Blue Snaggletooth action figure' (answer, gold no) | 'Blue Snaggletooth action figure' (answer, gold no) | yes | - | ' a' -> 'a rare blue Snaggletooth action figure' (answer, gold yes) | yes |
| b6025781 | ' Next' | 'Next steps could involve researching recipes that incorporate quinoa and roasted vegetables as protein sources' (answer, gold no) | 'Next week, user could try recipes that incorporate quinoa and roasted vegetables as protein' (answer, gold no) | no | 1 | ' The' -> 'The user needs new meal prep ideas and protein sources for quinoa and roasted vegetables' (answer, gold no) | yes |
| b86304ba | ' unknown' | 'unknown' (refusal, gold no) | 'unknown' (refusal, gold no) | yes | - | ' The' -> 'The value of the painting of a sunset is not specified in the given memory.' (answer, gold no) | yes |
| caf03d32 | ' user' | 'user' (answer, gold no) | 'user' (answer, gold no) | yes | - | ' The' -> 'The user has been struggling with slow cooker recipes and is seeking advice on improving results' (answer, gold no) | yes |
| cc539528 | ' Back' | 'Back-end programming languages like Ruby, Python, or PHP.' (answer, gold no) | 'Back-end programming languages like Ruby, Python, or PHP.' (answer, gold no) | yes | - | ' I' -> 'I recommended learning a back-end programming language such as Ruby, Python, or PHP' (answer, gold yes) | yes |
| d6062bb9 | ' ' | '1998' (answer, gold no) | '1998' (answer, gold no) | yes | - | ' ' -> '1998' (answer, gold no) | yes |
| d905b33f | ' ' | '20%' (answer, gold yes) | '20%' (answer, gold yes) | yes | - | ' ' -> '20%' (answer, gold yes) | yes |
| fca70973 | ' Theme' | 'Theme park recommendations not provided in the given conversation.' (answer, gold no) | 'Theme park recommendations not provided in the given conversation.' (answer, gold no) | yes | - | ' The' -> 'The user has some recent experience with theme parks and may have some ideas based on' (answer, gold no) | yes |
| gpt4_1d80365e | ' ' | '1' (answer, gold no) | '1' (answer, gold no) | yes | - | ' ' -> '1' (answer, gold no) | yes |
| gpt4_213fd887 | ' Volley' | 'Volleyball league' (answer, gold yes) | 'Volleyball league' (answer, gold yes) | yes | - | ' volleyball' -> 'volleyball league' (answer, gold yes) | yes |
| gpt4_4929293a | ' Cous' | "Cousin's wedding" (answer, gold no) | "Cousin's wedding" (answer, gold no) | yes | - | ' Michael' -> "Michael's engagement party" (answer, gold yes) | yes |
| gpt4_4ef30696 | ' ' | '35' (answer, gold no) | '35' (answer, gold no) | yes | - | ' ' -> '35' (answer, gold no) | yes |
| gpt4_74aed68e | ' ' | '21' (answer, gold no) | '21' (answer, gold no) | yes | - | ' ' -> '21' (answer, gold no) | yes |
| gpt4_88806d6e | ' Tom' | 'Tom' (answer, gold yes) | 'Tom' (answer, gold yes) | yes | - | ' Tom' -> 'Tom' (answer, gold yes) | yes |
| gpt4_98f46fc6 | ' First' | 'First event not mentioned' (answer, gold no) | 'First event not mentioned' (answer, gold no) | yes | - | ' I' -> "I don't have the specific dates or order of events for the charity gala and" (answer, gold no) | yes |

Development only: no claim on folds 2-4.