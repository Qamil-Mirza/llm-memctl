# §44 rev 2: swap-subset gate recomputed under rulings 2-3

Data only (tokenizers, no forward pass). Spec: research/proposals/2026-10-29-exp44-patching.md rev 1 §A with the rev 2 rulings 2 (per-row draw, cap 3) and 3 (number subtypes, same-type 'other'); rules in gate.json `rules` and in scripts/exp44_swap_gate.py. Counts only.

## qwen3b

| | dev (0-1) | confirmation (2-4) | all |
|---|---|---|---|
| population | 126 | 204 | 330 |
| eligible | 32 | 50 | 82 |

Eligible by LongMemEval question type:

| type | dev | confirmation | all |
|---|---|---|---|
| knowledge-update | 10 | 13 | 23 |
| multi-session | 0 | 4 | 4 |
| single-session-assistant | 6 | 15 | 21 |
| single-session-preference | 0 | 0 | 0 |
| single-session-user | 15 | 14 | 29 |
| temporal-reasoning | 1 | 4 | 5 |

Eligible by §23 label:

| label | confirmation | all |
|---|---|---|
| success | 24 | 43 |
| refusal | 14 | 19 |
| wrong | 12 | 20 |

Exclusions in order of application (of 330):

| exclusion | removed | remaining |
|---|---|---|
| gold_zero_evidence | 180 | 150 |
| gold_several_evidence | 5 | 145 |
| gold_in_question | 11 | 134 |
| gold_elsewhere | 39 | 95 |
| no_substitute | 6 | 89 |
| first_token_equal | 7 | 82 |
| alignment_failure | 0 | 82 |

Eligible by surface type (all): {'number:duration': 7, 'name': 28, 'other': 17, 'number:bare': 27, 'number:percent': 1, 'date': 2}; population surface types: {'number:bare': 73, 'other': 141, 'number:duration': 35, 'name': 48, 'number:currency': 20, 'date': 8, 'number:percent': 5}; distinct substitute rows used: 63; distinct substitute strings: 63, maximum multiplicity 3 (most used: [('3', 3), ('emma', 3), ('harvard university', 2), ('a mix of irish and italian', 2), ('ikea', 2), ('jaipur rugs', 2), ('techcorp', 2), ('over a year', 2)]); cap stats: {}.
Eligible by surface type x §23 label (all): {'date': {'success': 0, 'refusal': 1, 'wrong': 1}, 'name': {'success': 14, 'refusal': 7, 'wrong': 7}, 'number:bare': {'success': 15, 'refusal': 4, 'wrong': 8}, 'number:duration': {'success': 7, 'refusal': 0, 'wrong': 0}, 'number:percent': {'success': 0, 'refusal': 1, 'wrong': 0}, 'other': {'success': 7, 'refusal': 6, 'wrong': 4}}.
Regex vs normalised cross-check: {'checks': 1169, 'disagreements': 0}; continuation split unstable: 0.

Gate arithmetic: eligible folds 2-4 = 50 (need >= 50); question types with >= 15 = {'single-session-assistant': 15} (need >= 2); in-view failures folds 2-4 = 26 (>= 30 proceed, 15-29 pool, < 15 stop); all 330: 39. **Verdict: pool (claims read on all 330) + type-diversity caveat (fewer than two question types of >= 15 on folds 2-4; rev 2 ruling 7)**

Rev 2 check (5146f5b: the POOL verdict stands if >= 85 eligible and >= 40 failures on all 330): eligible 82, failures 39 -> **FLAG: Advisor re-reads the gate (rev 3)**

Rows excluded at the substitute step (no_substitute / first_token_equal):

- 59524333 f0 wrong [date] '6:00 pm': first_token_equal
- 71a3fd6b f0 wrong [other] '+49 (0) 62 32 / 14 23 - 0': no_substitute
- e66b632c f0 wrong [other] '27 minutes and 45 seconds': no_substitute
- 118b2229 f1 wrong [other] '45 minutes each way': no_substitute
- 51b23612 f1 refusal [other] 'Nu, pogodi!': no_substitute
- 7527f7e2 f1 success [number:currency] '$800': first_token_equal
- 7a8d0b71 f1 success [number:currency] '$2,000': first_token_equal
- 3d86fd0a f2 success [other] 'a coffee shop in the city': first_token_equal
- cc06de0d f3 refusal [number:currency] '$6': first_token_equal
- 37d43f65 f4 success [other] '16GB': no_substitute
- 852ce960 f4 wrong [number:currency] '$400,000': first_token_equal
- b759caee f4 success [other] '@jessica_poole_jewellery': no_substitute
- c19f7a0b f4 refusal [date] '6:30 pm': first_token_equal

Eligible rows (gold -> substitute, type):

- 15745da0 f0 [number:duration] 'three months' -> 'Two weeks'
- 1b9b7252 f0 [name] 'Mindful.org.' -> 'Harvard University'
- 2698e78f f0 [other] 'every week' -> 'under my bed'
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
- 0e4e4c46 f1 [number:bare] '132 points' -> '20'
- 0f05491a f1 [number:bare] '120' -> '3'
- 19b5f2b3 f1 [number:duration] 'two weeks' -> 'three months'
- 29f2956b f1 [number:duration] '30 minutes' -> '4 days.'
- 2bf43736 f1 [other] "Chapter 4 of Book 1, titled 'Vocal Prayer and Meditation'." -> 'I recommended learning Ruby, Python, or PHP as a back-end programming language.'
- 36580ce8 f1 [other] 'bronchitis' -> 'over a year'
- 3ba21379 f1 [other] 'Ford F-150 pickup truck' -> 'The music shop on Main St.'
- 5831f84d f1 [number:bare] '15' -> '5'
- 618f13b2 f1 [number:bare] 'six' -> 'Two'
- 66f24dbb f1 [other] 'a yellow dress' -> 'over a year'
- 7401057b f1 [number:bare] 'Two' -> 'five'
- 86f00804 f1 [name] 'The Seven Husbands of Evelyn Hugo' -> 'Patagonia and Southwest Airlines.'
- b6019101 f1 [number:bare] '5' -> '25'
- d52b4f67 f1 [name] 'The Grand Ballroom' -> 'Nostalgia'
- e01b8e2f f1 [name] 'Hawaii' -> 'Memrise'
- gpt4_468eb064 f1 [name] 'Emma' -> 'Absinthe'
- 08e075c7 f2 [number:duration] '9 months' -> '10 minutes'
- 1de5cff2 f2 [name] 'Veja' -> 'Pennsylvania'
- 1faac195 f2 [name] 'Denver' -> 'Emma'
- 3e321797 f2 [number:duration] '10 minutes' -> '3 weeks ago'
- 45dc21b6 f2 [number:bare] '3' -> '2'
- 4b24c848 f2 [number:bare] 'five' -> 'six'
- 6aeb4375 f2 [number:bare] 'four' -> 'five'
- 70b3e69b f2 [name] 'Manolo García' -> 'IKEA'
- 86b68151 f2 [name] 'IKEA' -> 'Denver'
- ad7109d1 f2 [number:bare] '500 Mbps' -> '18'
- bc8a6e93 f2 [other] 'a lemon blueberry cake' -> 'the sports store downtown'
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
- 8aef76bc f4 [other] 'Mod Podge or another sealant' -> 'The GR-90 trail.'
- b46e15ee f4 [other] "the 'Walk for Hunger' charity event" -> "'Data Analysis using Python' webinar"
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
| eligible | 34 | 51 | 85 |

Eligible by LongMemEval question type:

| type | dev | confirmation | all |
|---|---|---|---|
| knowledge-update | 11 | 13 | 24 |
| multi-session | 0 | 4 | 4 |
| single-session-assistant | 7 | 15 | 22 |
| single-session-preference | 0 | 0 | 0 |
| single-session-user | 15 | 15 | 30 |
| temporal-reasoning | 1 | 4 | 5 |

Eligible by §23 label:

| label | confirmation | all |
|---|---|---|
| success | 40 | 62 |
| refusal | 6 | 12 |
| wrong | 5 | 11 |

Exclusions in order of application (of 330):

| exclusion | removed | remaining |
|---|---|---|
| gold_zero_evidence | 180 | 150 |
| gold_several_evidence | 5 | 145 |
| gold_in_question | 11 | 134 |
| gold_elsewhere | 39 | 95 |
| no_substitute | 5 | 90 |
| first_token_equal | 5 | 85 |
| alignment_failure | 0 | 85 |

Eligible by surface type (all): {'number:duration': 7, 'name': 28, 'other': 18, 'date': 4, 'number:bare': 27, 'number:percent': 1}; population surface types: {'number:bare': 73, 'other': 141, 'number:duration': 35, 'name': 48, 'number:currency': 20, 'date': 8, 'number:percent': 5}; distinct substitute rows used: 66; distinct substitute strings: 65, maximum multiplicity 3 (most used: [('emma', 3), ('harvard university', 2), ('2 am', 2), ('a mix of irish and italian', 2), ('3', 2), ('ikea', 2), ('jaipur rugs', 2), ('techcorp', 2)]); cap stats: {}.
Eligible by surface type x §23 label (all): {'date': {'success': 4, 'refusal': 0, 'wrong': 0}, 'name': {'success': 22, 'refusal': 4, 'wrong': 2}, 'number:bare': {'success': 18, 'refusal': 4, 'wrong': 5}, 'number:duration': {'success': 5, 'refusal': 0, 'wrong': 2}, 'number:percent': {'success': 0, 'refusal': 1, 'wrong': 0}, 'other': {'success': 13, 'refusal': 3, 'wrong': 2}}.
Regex vs normalised cross-check: {'checks': 1262, 'disagreements': 0}; continuation split unstable: 0.

Gate arithmetic: eligible folds 2-4 = 51 (need >= 50); question types with >= 15 = {'single-session-assistant': 15, 'single-session-user': 15} (need >= 2); in-view failures folds 2-4 = 11 (>= 30 proceed, 15-29 pool, < 15 stop); all 330: 23. **Verdict: stop** (descriptive; the gate binds on Qwen 3B only)

Rows excluded at the substitute step (no_substitute / first_token_equal):

- 71a3fd6b f0 success [other] '+49 (0) 62 32 / 14 23 - 0': no_substitute
- e66b632c f0 success [other] '27 minutes and 45 seconds': no_substitute
- 118b2229 f1 success [other] '45 minutes each way': no_substitute
- 7527f7e2 f1 success [number:currency] '$800': first_token_equal
- 7a8d0b71 f1 success [number:currency] '$2,000': first_token_equal
- 3d86fd0a f2 success [other] 'a coffee shop in the city': first_token_equal
- cc06de0d f3 success [number:currency] '$6': first_token_equal
- 37d43f65 f4 success [other] '16GB': no_substitute
- 852ce960 f4 wrong [number:currency] '$400,000': first_token_equal
- b759caee f4 success [other] '@jessica_poole_jewellery': no_substitute

Eligible rows (gold -> substitute, type):

- 15745da0 f0 [number:duration] 'three months' -> 'Two weeks'
- 1b9b7252 f0 [name] 'Mindful.org.' -> 'Harvard University'
- 2698e78f f0 [other] 'every week' -> 'under my bed'
- 59524333 f0 [date] '6:00 pm' -> '2 AM'
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
- 86f00804 f1 [name] 'The Seven Husbands of Evelyn Hugo' -> 'Patagonia and Southwest Airlines.'
- b6019101 f1 [number:bare] '5' -> '25'
- d52b4f67 f1 [name] 'The Grand Ballroom' -> 'Nostalgia'
- e01b8e2f f1 [name] 'Hawaii' -> 'Memrise'
- gpt4_468eb064 f1 [name] 'Emma' -> 'Absinthe'
- 08e075c7 f2 [number:duration] '9 months' -> '10 minutes'
- 1de5cff2 f2 [name] 'Veja' -> 'Pennsylvania'
- 1faac195 f2 [name] 'Denver' -> 'Emma'
- 3e321797 f2 [number:duration] '10 minutes' -> '3 weeks ago'
- 45dc21b6 f2 [number:bare] '3' -> '2'
- 4b24c848 f2 [number:bare] 'five' -> 'six'
- 6aeb4375 f2 [number:bare] 'four' -> 'five'
- 70b3e69b f2 [name] 'Manolo García' -> 'IKEA'
- 86b68151 f2 [name] 'IKEA' -> 'Denver'
- ad7109d1 f2 [number:bare] '500 Mbps' -> '18'
- bc8a6e93 f2 [other] 'a lemon blueberry cake' -> 'the sports store downtown'
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
- 8aef76bc f4 [other] 'Mod Podge or another sealant' -> 'The GR-90 trail.'
- b46e15ee f4 [other] "the 'Walk for Hunger' charity event" -> "'Data Analysis using Python' webinar"
- c19f7a0b f4 [date] '6:30 pm' -> '2 AM'
- c7dc5443 f4 [number:bare] '5-2' -> '8 meals'
- d596882b f4 [name] 'By Chloe' -> 'Data Science'
- dccbc061 f4 [other] 'A staunch atheist' -> 'Philips LED bulb'
- dd2973ad f4 [date] '2 AM' -> '6:00 pm'
- ec81a493 f4 [number:bare] '500' -> '3'
- f4f1d8a4 f4 [other] 'my sister' -> 'bronchitis'
- gpt4_b5700ca0 f4 [name] 'the Episcopal Church' -> 'Kansas City Masterpiece'
