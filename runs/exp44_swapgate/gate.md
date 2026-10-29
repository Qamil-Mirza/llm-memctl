# §44 rev 3: swap-subset gate recomputed with the shared-prefix rule

Data only (tokenizers, no forward pass). Spec: research/proposals/2026-10-29-exp44-patching.md rev 1 §A with the rev 2 rulings 2 (per-row draw, cap 3) and 3 (number subtypes, same-type 'other'); rules in gate.json `rules` and in scripts/exp44_swap_gate.py. Counts only.

## qwen3b

| | dev (0-1) | confirmation (2-4) | all |
|---|---|---|---|
| population | 126 | 204 | 330 |
| eligible | 35 | 54 | 89 |

Eligible by LongMemEval question type:

| type | dev | confirmation | all |
|---|---|---|---|
| knowledge-update | 11 | 14 | 25 |
| multi-session | 0 | 5 | 5 |
| single-session-assistant | 7 | 15 | 22 |
| single-session-preference | 0 | 0 | 0 |
| single-session-user | 16 | 16 | 32 |
| temporal-reasoning | 1 | 4 | 5 |

Eligible by §23 label:

| label | confirmation | all |
|---|---|---|
| success | 25 | 46 |
| refusal | 16 | 21 |
| wrong | 13 | 22 |

Exclusions in order of application (of 330):

| exclusion | removed | remaining |
|---|---|---|
| gold_zero_evidence | 180 | 150 |
| gold_several_evidence | 5 | 145 |
| gold_in_question | 11 | 134 |
| gold_elsewhere | 39 | 95 |
| no_substitute | 6 | 89 |
| first_token_equal | 0 | 89 |
| alignment_failure | 0 | 89 |

Eligible by surface type (all): {'number:duration': 7, 'name': 28, 'other': 18, 'date': 4, 'number:bare': 27, 'number:currency': 4, 'number:percent': 1}; population surface types: {'number:bare': 73, 'other': 141, 'number:duration': 35, 'name': 48, 'number:currency': 20, 'date': 8, 'number:percent': 5}; distinct substitute rows used: 71; distinct substitute strings: 71, maximum multiplicity 3 (most used: [('emma', 3), ('harvard university', 2), ('a mix of irish and italian', 2), ('3', 2), ('ikea', 2), ('jaipur rugs', 2), ('techcorp', 2), ('over a year', 2)]); cap stats: {}.
Eligible by surface type x §23 label (all): {'date': {'success': 0, 'refusal': 2, 'wrong': 2}, 'name': {'success': 14, 'refusal': 7, 'wrong': 7}, 'number:bare': {'success': 15, 'refusal': 4, 'wrong': 8}, 'number:currency': {'success': 2, 'refusal': 1, 'wrong': 1}, 'number:duration': {'success': 7, 'refusal': 0, 'wrong': 0}, 'number:percent': {'success': 0, 'refusal': 1, 'wrong': 0}, 'other': {'success': 8, 'refusal': 6, 'wrong': 4}}.
Regex vs normalised cross-check: {'checks': 1055, 'disagreements': 0}; continuation split unstable: 0.

Gate arithmetic: eligible folds 2-4 = 54 (need >= 50); question types with >= 15 = {'single-session-assistant': 15, 'single-session-user': 16} (need >= 2); in-view failures folds 2-4 = 29 (>= 30 proceed, 15-29 pool, < 15 stop); all 330: 43. **Verdict: pool (claims read on all 330)**

Rev 2 check (5146f5b: the POOL verdict stands if >= 85 eligible and >= 40 failures on all 330): eligible 89, failures 43 -> **recorded for rev 4 (rev 3: POOL regardless; the 85 / 40 bar was a re-read trigger)**

Rows with a non-empty shared prefix: 12 (prefixes {'6:': 2, '1': 2, ' $': 4, ' The': 1, ' a': 2, ' the': 1}):

- 59524333 f0 wrong '6:00 pm' -> '6:45 AM': prefix '6:', read '0' vs '4'
- 0e4e4c46 f1 wrong '132 points' -> '17': prefix '1', read '3' vs '7'
- 0f05491a f1 wrong '120' -> '15': prefix '1', read '2' vs '5'
- 7527f7e2 f1 success '$800' -> '$0.75': prefix ' $', read '8' vs '0'
- 7a8d0b71 f1 success '$2,000' -> '$400,000': prefix ' $', read '2' vs '4'
- 86f00804 f1 success 'The Seven Husbands of Evelyn Hugo' -> 'The Sugar Factory at Icon Park.': prefix ' The', read ' Seven' vs ' Sugar'
- 3d86fd0a f2 success 'a coffee shop in the city' -> 'a blue Snaggletooth': prefix ' a', read ' coffee' vs ' blue'
- bc8a6e93 f2 wrong 'a lemon blueberry cake' -> 'a coffee shop in the city': prefix ' a', read ' lemon' vs ' coffee'
- cc06de0d f3 refusal '$6' -> '$50': prefix ' $', read '6' vs '5'
- 852ce960 f4 wrong '$400,000' -> '$25,000': prefix ' $', read '4' vs '2'
- b46e15ee f4 wrong "the 'Walk for Hunger' charity event" -> "the woman selling jam at the farmer's market": prefix ' the', read " '" vs ' woman'
- c19f7a0b f4 refusal '6:30 pm' -> '6:00 pm': prefix '6:', read '3' vs '0'

Rows excluded at the substitute step (no_substitute / first_token_equal):

- 71a3fd6b f0 wrong [other] '+49 (0) 62 32 / 14 23 - 0': no_substitute
- e66b632c f0 wrong [other] '27 minutes and 45 seconds': no_substitute
- 118b2229 f1 wrong [other] '45 minutes each way': no_substitute
- 51b23612 f1 refusal [other] 'Nu, pogodi!': no_substitute
- 37d43f65 f4 success [other] '16GB': no_substitute
- b759caee f4 success [other] '@jessica_poole_jewellery': no_substitute

Eligible rows (gold -> substitute, type):

- 15745da0 f0 [number:duration] 'three months' -> 'Two weeks'
- 1b9b7252 f0 [name] 'Mindful.org.' -> 'Harvard University'
- 2698e78f f0 [other] 'every week' -> 'under my bed'
- 59524333 f0 [date] '6:00 pm' -> '6:45 AM'
- 5d3d2817 f0 [other] 'Marketing specialist at a small startup' -> 'Japanese short-grain rice'
- 60d45044 f0 [other] 'Japanese short-grain rice' -> 'A mix of Irish and Italian'
- 76d63226 f0 [number:bare] '55-inch' -> '38'
- 7e00a6cb f0 [name] 'International Budget Hostel' -> 'Harvard University'
- 95bcc1c8 f0 [number:bare] '10' -> '3'
- ba61f0b9 f0 [number:bare] '6' -> '100'
- c14c00dd f0 [name] "Trader Joe's" -> 'Paris'
- c5e8278d f0 [name] 'Johnson' -> 'IKEA'
- c7cf7dfd f0 [name] 'Nostalgia' -> 'Jaipur Rugs'
- fca762bc f0 [name] 'Memrise' -> 'Jaipur Rugs'
- fea54f57 f0 [name] 'Evolution' -> 'Johnson'
- 07741c44 f1 [other] 'under my bed' -> 'in my bedroom'
- 0862e8bf f1 [name] 'Luna' -> 'TechCorp'
- 0e4e4c46 f1 [number:bare] '132 points' -> '17'
- 0f05491a f1 [number:bare] '120' -> '15'
- 19b5f2b3 f1 [number:duration] 'two weeks' -> 'three months'
- 29f2956b f1 [number:duration] '30 minutes' -> '4 days.'
- 2bf43736 f1 [other] "Chapter 4 of Book 1, titled 'Vocal Prayer and Meditation'." -> 'I recommended learning Ruby, Python, or PHP as a back-end programming language.'
- 36580ce8 f1 [other] 'bronchitis' -> 'over a year'
- 3ba21379 f1 [other] 'Ford F-150 pickup truck' -> 'The music shop on Main St.'
- 5831f84d f1 [number:bare] '15' -> '5'
- 618f13b2 f1 [number:bare] 'six' -> 'Two'
- 66f24dbb f1 [other] 'a yellow dress' -> 'over a year'
- 7401057b f1 [number:bare] 'Two' -> 'five'
- 7527f7e2 f1 [number:currency] '$800' -> '$0.75'
- 7a8d0b71 f1 [number:currency] '$2,000' -> '$400,000'
- 86f00804 f1 [name] 'The Seven Husbands of Evelyn Hugo' -> 'The Sugar Factory at Icon Park.'
- b6019101 f1 [number:bare] '5' -> '25'
- d52b4f67 f1 [name] 'The Grand Ballroom' -> 'Nostalgia'
- e01b8e2f f1 [name] 'Hawaii' -> 'Memrise'
- gpt4_468eb064 f1 [name] 'Emma' -> 'Absinthe'
- 08e075c7 f2 [number:duration] '9 months' -> '10 minutes'
- 1de5cff2 f2 [name] 'Veja' -> 'Pennsylvania'
- 1faac195 f2 [name] 'Denver' -> 'Emma'
- 3d86fd0a f2 [other] 'a coffee shop in the city' -> 'a blue Snaggletooth'
- 3e321797 f2 [number:duration] '10 minutes' -> '3 weeks ago'
- 45dc21b6 f2 [number:bare] '3' -> '2'
- 4b24c848 f2 [number:bare] 'five' -> 'six'
- 6aeb4375 f2 [number:bare] 'four' -> 'five'
- 70b3e69b f2 [name] 'Manolo García' -> 'IKEA'
- 86b68151 f2 [name] 'IKEA' -> 'Denver'
- ad7109d1 f2 [number:bare] '500 Mbps' -> '18'
- bc8a6e93 f2 [other] 'a lemon blueberry cake' -> 'a coffee shop in the city'
- cf22b7bf f2 [number:bare] '10 pounds' -> '2'
- ed4ddc30 f2 [number:bare] '20' -> '856'
- gpt4_fa19884d f2 [other] 'a bluegrass band that features a banjo player' -> "the woman selling jam at the farmer's market"
- 099778bb f3 [number:percent] '20%' -> '10%'
- 18dcd5a5 f3 [number:bare] '4' -> '5-2'
- 577d4d32 f3 [date] '7 pm' -> '2 AM'
- 58bf7951 f3 [name] 'The Glass Menagerie' -> 'Mindful.org.'
- 5a4f22c0 f3 [name] 'TechCorp' -> 'Target'
- 60472f9c f3 [number:bare] '2' -> '18'
- 8464fc84 f3 [name] 'Roscioli' -> 'TechCorp'
- 8b9d4367 f3 [name] 'Jaipur Rugs' -> 'Kansas City Masterpiece'
- a82c026e f3 [other] 'Dark Souls 3 DLC' -> 'A mix of Irish and Italian'
- a96c20ee f3 [name] 'Harvard University' -> 'Summer Vibes'
- affe2881 f3 [number:bare] '32' -> '220'
- cc06de0d f3 [number:currency] '$6' -> '$50'
- ccb36322 f3 [name] 'Spotify' -> 'Emma'
- e8a79c70 f3 [number:bare] '2-3 eggs' -> '500'
- e982271f f3 [name] 'Revolution Hall' -> 'Emma'
- f523d9fe f3 [name] 'Doc Martin' -> "Trader Joe's"
- f9e8c073 f3 [number:bare] 'five' -> 'six'
- faba32e5 f3 [number:duration] '24 hours' -> '3 days.'
- 001be529 f4 [other] 'over a year' -> 'the sports store downtown'
- 01493427 f4 [number:bare] '25' -> '132 points'
- 0e5e2d1a f4 [number:bare] '38 subjects' -> '856'
- 1e043500 f4 [name] 'Summer Vibes' -> 'Tom'
- 26bdc477 f4 [number:bare] 'five' -> 'four'
- 3249768e f4 [name] 'Absinthe' -> 'The Glass Menagerie'
- 5809eb10 f4 [number:bare] '2014.' -> '3,000 miles'
- 5e1b23de f4 [number:bare] '3' -> '25'
- 778164c6 f4 [other] 'Grilled Snapper with Mango Salsa' -> 'The GR-90 trail.'
- 7a87bd0c f4 [number:duration] '4 weeks' -> '5.5 weeks'
- 852ce960 f4 [number:currency] '$400,000' -> '$25,000'
- 8aef76bc f4 [other] 'Mod Podge or another sealant' -> 'The GR-90 trail.'
- b46e15ee f4 [other] "the 'Walk for Hunger' charity event" -> "the woman selling jam at the farmer's market"
- c19f7a0b f4 [date] '6:30 pm' -> '6:00 pm'
- c7dc5443 f4 [number:bare] '5-2' -> '8 meals'
- d596882b f4 [name] 'By Chloe' -> 'Data Science'
- dccbc061 f4 [other] 'A staunch atheist' -> 'Philips LED bulb'
- dd2973ad f4 [date] '2 AM' -> '7 pm'
- ec81a493 f4 [number:bare] '500' -> '3'
- f4f1d8a4 f4 [other] 'my sister' -> 'bronchitis'
- gpt4_b5700ca0 f4 [name] 'the Episcopal Church' -> 'Kansas City Masterpiece'

## llama3b

| | dev (0-1) | confirmation (2-4) | all |
|---|---|---|---|
| population | 126 | 204 | 330 |
| eligible | 36 | 54 | 90 |

Eligible by LongMemEval question type:

| type | dev | confirmation | all |
|---|---|---|---|
| knowledge-update | 11 | 14 | 25 |
| multi-session | 0 | 5 | 5 |
| single-session-assistant | 8 | 15 | 23 |
| single-session-preference | 0 | 0 | 0 |
| single-session-user | 16 | 16 | 32 |
| temporal-reasoning | 1 | 4 | 5 |

Eligible by §23 label:

| label | confirmation | all |
|---|---|---|
| success | 42 | 66 |
| refusal | 6 | 12 |
| wrong | 6 | 12 |

Exclusions in order of application (of 330):

| exclusion | removed | remaining |
|---|---|---|
| gold_zero_evidence | 180 | 150 |
| gold_several_evidence | 5 | 145 |
| gold_in_question | 11 | 134 |
| gold_elsewhere | 39 | 95 |
| no_substitute | 5 | 90 |
| first_token_equal | 0 | 90 |
| alignment_failure | 0 | 90 |

Eligible by surface type (all): {'number:duration': 7, 'name': 28, 'other': 19, 'date': 4, 'number:bare': 27, 'number:currency': 4, 'number:percent': 1}; population surface types: {'number:bare': 73, 'other': 141, 'number:duration': 35, 'name': 48, 'number:currency': 20, 'date': 8, 'number:percent': 5}; distinct substitute rows used: 71; distinct substitute strings: 70, maximum multiplicity 3 (most used: [('emma', 3), ('harvard university', 2), ('a mix of irish and italian', 2), ('3', 2), ('ikea', 2), ('jaipur rugs', 2), ('techcorp', 2), ('over a year', 2)]); cap stats: {}.
Eligible by surface type x §23 label (all): {'date': {'success': 4, 'refusal': 0, 'wrong': 0}, 'name': {'success': 22, 'refusal': 4, 'wrong': 2}, 'number:bare': {'success': 18, 'refusal': 4, 'wrong': 5}, 'number:currency': {'success': 3, 'refusal': 0, 'wrong': 1}, 'number:duration': {'success': 5, 'refusal': 0, 'wrong': 2}, 'number:percent': {'success': 0, 'refusal': 1, 'wrong': 0}, 'other': {'success': 14, 'refusal': 3, 'wrong': 2}}.
Regex vs normalised cross-check: {'checks': 1107, 'disagreements': 0}; continuation split unstable: 0.

Gate arithmetic: eligible folds 2-4 = 54 (need >= 50); question types with >= 15 = {'single-session-assistant': 15, 'single-session-user': 16} (need >= 2); in-view failures folds 2-4 = 12 (>= 30 proceed, 15-29 pool, < 15 stop); all 330: 24. **Verdict: stop** (descriptive; the gate binds on Qwen 3B only)

Rows with a non-empty shared prefix: 9 (prefixes {'6:': 1, ' $': 4, ' The': 1, ' a': 2, ' the': 1}):

- 59524333 f0 success '6:00 pm' -> '6:45 AM': prefix '6:', read '00' vs '45'
- 7527f7e2 f1 success '$800' -> '$0.75': prefix ' $', read '800' vs '0'
- 7a8d0b71 f1 success '$2,000' -> '$400,000': prefix ' $', read '2' vs '400'
- 86f00804 f1 success 'The Seven Husbands of Evelyn Hugo' -> 'The Sugar Factory at Icon Park.': prefix ' The', read ' Seven' vs ' Sugar'
- 3d86fd0a f2 success 'a coffee shop in the city' -> 'a blue Snaggletooth': prefix ' a', read ' coffee' vs ' blue'
- bc8a6e93 f2 success 'a lemon blueberry cake' -> 'a coffee shop in the city': prefix ' a', read ' lemon' vs ' coffee'
- cc06de0d f3 success '$6' -> '$50': prefix ' $', read '6' vs '50'
- 852ce960 f4 wrong '$400,000' -> '$0.75': prefix ' $', read '400' vs '0'
- b46e15ee f4 refusal "the 'Walk for Hunger' charity event" -> "the woman selling jam at the farmer's market": prefix ' the', read " '" vs ' woman'

Rows excluded at the substitute step (no_substitute / first_token_equal):

- 71a3fd6b f0 success [other] '+49 (0) 62 32 / 14 23 - 0': no_substitute
- e66b632c f0 success [other] '27 minutes and 45 seconds': no_substitute
- 118b2229 f1 success [other] '45 minutes each way': no_substitute
- 37d43f65 f4 success [other] '16GB': no_substitute
- b759caee f4 success [other] '@jessica_poole_jewellery': no_substitute

Eligible rows (gold -> substitute, type):

- 15745da0 f0 [number:duration] 'three months' -> 'Two weeks'
- 1b9b7252 f0 [name] 'Mindful.org.' -> 'Harvard University'
- 2698e78f f0 [other] 'every week' -> 'under my bed'
- 59524333 f0 [date] '6:00 pm' -> '6:45 AM'
- 5d3d2817 f0 [other] 'Marketing specialist at a small startup' -> 'Japanese short-grain rice'
- 60d45044 f0 [other] 'Japanese short-grain rice' -> 'A mix of Irish and Italian'
- 76d63226 f0 [number:bare] '55-inch' -> '38'
- 7e00a6cb f0 [name] 'International Budget Hostel' -> 'Harvard University'
- 95bcc1c8 f0 [number:bare] '10' -> '3'
- ba61f0b9 f0 [number:bare] '6' -> '100'
- c14c00dd f0 [name] "Trader Joe's" -> 'Paris'
- c5e8278d f0 [name] 'Johnson' -> 'IKEA'
- c7cf7dfd f0 [name] 'Nostalgia' -> 'Jaipur Rugs'
- fca762bc f0 [name] 'Memrise' -> 'Jaipur Rugs'
- fea54f57 f0 [name] 'Evolution' -> 'Johnson'
- 07741c44 f1 [other] 'under my bed' -> 'in my bedroom'
- 0862e8bf f1 [name] 'Luna' -> 'TechCorp'
- 0e4e4c46 f1 [number:bare] '132 points' -> '17'
- 0f05491a f1 [number:bare] '120' -> '15'
- 19b5f2b3 f1 [number:duration] 'two weeks' -> 'three months'
- 29f2956b f1 [number:duration] '30 minutes' -> '4 days.'
- 2bf43736 f1 [other] "Chapter 4 of Book 1, titled 'Vocal Prayer and Meditation'." -> 'I recommended learning Ruby, Python, or PHP as a back-end programming language.'
- 36580ce8 f1 [other] 'bronchitis' -> 'over a year'
- 3ba21379 f1 [other] 'Ford F-150 pickup truck' -> 'The music shop on Main St.'
- 51b23612 f1 [other] 'Nu, pogodi!' -> 'The GR-90 trail.'
- 5831f84d f1 [number:bare] '15' -> '5'
- 618f13b2 f1 [number:bare] 'six' -> 'Two'
- 66f24dbb f1 [other] 'a yellow dress' -> 'over a year'
- 7401057b f1 [number:bare] 'Two' -> 'five'
- 7527f7e2 f1 [number:currency] '$800' -> '$0.75'
- 7a8d0b71 f1 [number:currency] '$2,000' -> '$400,000'
- 86f00804 f1 [name] 'The Seven Husbands of Evelyn Hugo' -> 'The Sugar Factory at Icon Park.'
- b6019101 f1 [number:bare] '5' -> '25'
- d52b4f67 f1 [name] 'The Grand Ballroom' -> 'Nostalgia'
- e01b8e2f f1 [name] 'Hawaii' -> 'Memrise'
- gpt4_468eb064 f1 [name] 'Emma' -> 'Absinthe'
- 08e075c7 f2 [number:duration] '9 months' -> '10 minutes'
- 1de5cff2 f2 [name] 'Veja' -> 'Pennsylvania'
- 1faac195 f2 [name] 'Denver' -> 'Emma'
- 3d86fd0a f2 [other] 'a coffee shop in the city' -> 'a blue Snaggletooth'
- 3e321797 f2 [number:duration] '10 minutes' -> '3 weeks ago'
- 45dc21b6 f2 [number:bare] '3' -> '2'
- 4b24c848 f2 [number:bare] 'five' -> 'six'
- 6aeb4375 f2 [number:bare] 'four' -> 'five'
- 70b3e69b f2 [name] 'Manolo García' -> 'IKEA'
- 86b68151 f2 [name] 'IKEA' -> 'Denver'
- ad7109d1 f2 [number:bare] '500 Mbps' -> '18'
- bc8a6e93 f2 [other] 'a lemon blueberry cake' -> 'a coffee shop in the city'
- cf22b7bf f2 [number:bare] '10 pounds' -> '2'
- ed4ddc30 f2 [number:bare] '20' -> '856'
- gpt4_fa19884d f2 [other] 'a bluegrass band that features a banjo player' -> "the woman selling jam at the farmer's market"
- 099778bb f3 [number:percent] '20%' -> '10%'
- 18dcd5a5 f3 [number:bare] '4' -> '5-2'
- 577d4d32 f3 [date] '7 pm' -> '6:30 pm'
- 58bf7951 f3 [name] 'The Glass Menagerie' -> 'Mindful.org.'
- 5a4f22c0 f3 [name] 'TechCorp' -> 'Target'
- 60472f9c f3 [number:bare] '2' -> '18'
- 8464fc84 f3 [name] 'Roscioli' -> 'TechCorp'
- 8b9d4367 f3 [name] 'Jaipur Rugs' -> 'Kansas City Masterpiece'
- a82c026e f3 [other] 'Dark Souls 3 DLC' -> 'A mix of Irish and Italian'
- a96c20ee f3 [name] 'Harvard University' -> 'Summer Vibes'
- affe2881 f3 [number:bare] '32' -> '220'
- cc06de0d f3 [number:currency] '$6' -> '$50'
- ccb36322 f3 [name] 'Spotify' -> 'Emma'
- e8a79c70 f3 [number:bare] '2-3 eggs' -> '2014.'
- e982271f f3 [name] 'Revolution Hall' -> 'Emma'
- f523d9fe f3 [name] 'Doc Martin' -> "Trader Joe's"
- f9e8c073 f3 [number:bare] 'five' -> 'six'
- faba32e5 f3 [number:duration] '24 hours' -> '3 days.'
- 001be529 f4 [other] 'over a year' -> 'the sports store downtown'
- 01493427 f4 [number:bare] '25' -> '132 points'
- 0e5e2d1a f4 [number:bare] '38 subjects' -> '856'
- 1e043500 f4 [name] 'Summer Vibes' -> 'Tom'
- 26bdc477 f4 [number:bare] 'five' -> 'four'
- 3249768e f4 [name] 'Absinthe' -> 'The Glass Menagerie'
- 5809eb10 f4 [number:bare] '2014.' -> '5'
- 5e1b23de f4 [number:bare] '3' -> '25'
- 778164c6 f4 [other] 'Grilled Snapper with Mango Salsa' -> 'Mod Podge or another sealant'
- 7a87bd0c f4 [number:duration] '4 weeks' -> '5.5 weeks'
- 852ce960 f4 [number:currency] '$400,000' -> '$0.75'
- 8aef76bc f4 [other] 'Mod Podge or another sealant' -> 'The GR-90 trail.'
- b46e15ee f4 [other] "the 'Walk for Hunger' charity event" -> "the woman selling jam at the farmer's market"
- c19f7a0b f4 [date] '6:30 pm' -> '2 AM'
- c7dc5443 f4 [number:bare] '5-2' -> '8 meals'
- d596882b f4 [name] 'By Chloe' -> 'Data Science'
- dccbc061 f4 [other] 'A staunch atheist' -> 'Philips LED bulb'
- dd2973ad f4 [date] '2 AM' -> '6:00 pm'
- ec81a493 f4 [number:bare] '500' -> '3'
- f4f1d8a4 f4 [other] 'my sister' -> 'bronchitis'
- gpt4_b5700ca0 f4 [name] 'the Episcopal Church' -> 'Kansas City Masterpiece'
