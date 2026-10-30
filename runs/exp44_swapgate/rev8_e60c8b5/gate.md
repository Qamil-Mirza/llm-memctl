# §44 rev 8: repaired swap gate (no 'other'; bare numbers form-matched; determiner matched), Qwen 3B

Data only (tokenizers, no forward pass). Spec: research/proposals/2026-10-29-exp44-patching.md rev 1 §A with the rev 2 rulings 2 (per-row draw, cap 3) and 3 (number subtypes, same-type 'other') and rev 8 (83fe60a, D-47: no 'other', bare form-matched, determiner matched); rules in gate.json `rules` and in scripts/exp44_swap_gate.py. Counts only.

## qwen3b

| | dev (0-1) | confirmation (2-4) | all |
|---|---|---|---|
| population | 126 | 204 | 330 |
| eligible | 27 | 43 | 70 |

Eligible by LongMemEval question type:

| type | dev | confirmation | all |
|---|---|---|---|
| knowledge-update | 8 | 14 | 22 |
| multi-session | 0 | 5 | 5 |
| single-session-assistant | 6 | 13 | 19 |
| single-session-preference | 0 | 0 | 0 |
| single-session-user | 12 | 10 | 22 |
| temporal-reasoning | 1 | 1 | 2 |

Eligible by §23 label:

| label | confirmation | all |
|---|---|---|
| success | 22 | 38 |
| refusal | 12 | 14 |
| wrong | 9 | 18 |

Exclusions in order of application (of 330):

| exclusion | removed | remaining |
|---|---|---|
| gold_zero_evidence | 180 | 150 |
| gold_several_evidence | 5 | 145 |
| gold_in_question | 11 | 134 |
| gold_elsewhere | 39 | 95 |
| type_other_rev8 | 24 | 71 |
| no_substitute | 1 | 70 |
| first_token_equal | 0 | 70 |
| alignment_failure | 0 | 70 |

Eligible by surface type (all): {'number:duration': 7, 'name': 27, 'date': 4, 'number:bare': 27, 'number:currency': 4, 'number:percent': 1}; population surface types: {'number:bare': 73, 'other': 141, 'number:duration': 35, 'name': 48, 'number:currency': 20, 'date': 8, 'number:percent': 5}; distinct substitute rows used: 58; distinct substitute strings: 56, maximum multiplicity 3 (most used: [('emma', 3), ('harvard university', 2), ('10 pounds', 2), ('ikea', 2), ('jaipur rugs', 2), ('techcorp', 2), ('100', 2), ('five', 2)]); cap stats: {'rev8_reject_bare_unit': 29, 'rev8_reject_bare_digits': 159, 'rev8_reject_bare_form': 163, 'rev8_reject_determiner': 95}.
Eligible by surface type x §23 label (all): {'date': {'success': 0, 'refusal': 2, 'wrong': 2}, 'name': {'success': 14, 'refusal': 6, 'wrong': 7}, 'number:bare': {'success': 15, 'refusal': 4, 'wrong': 8}, 'number:currency': {'success': 2, 'refusal': 1, 'wrong': 1}, 'number:duration': {'success': 7, 'refusal': 0, 'wrong': 0}, 'number:percent': {'success': 0, 'refusal': 1, 'wrong': 0}}.
Regex vs normalised cross-check: {'checks': 875, 'disagreements': 0}; continuation split unstable: 0.

Gate arithmetic: eligible folds 2-4 = 43 (need >= 50); question types with >= 15 = {} (need >= 2); in-view failures folds 2-4 = 21 (>= 30 proceed, 15-29 pool, < 15 stop); all 330: 32. **Verdict: n/a under rev 8 (POOL fixed by rev 4; rev 1 arithmetic recorded beside)**

Rev 8 against the recorded rev 3 / rev 4 set (89 rows): 'other' excluded 18; unchanged substitute 47 (record identity asserted); changed substitute 23; lost (no passing candidate) 1.

Changed-substitute rows (the only fresh evidence, honesty clause):

- 76d63226 f0 success [number:bare] '55-inch': '38' (fails bare_unit) -> '10 pounds', read '5' vs '1'
- 95bcc1c8 f0 success [number:bare] '10': '3' (fails bare_digits) -> '15', read '0' vs '5'
- ba61f0b9 f0 wrong [number:bare] '6': '100' (fails bare_digits) -> '8', read '6' vs '8'
- 0e4e4c46 f1 wrong [number:bare] '132 points': '17' (fails bare_digits) -> '500 Mbps', read '1' vs '5'
- 0f05491a f1 wrong [number:bare] '120': '15' (fails bare_digits) -> '100', read '2' vs '0'
- 5831f84d f1 wrong [number:bare] '15': '5' (fails bare_digits) -> '5-2', read '1' vs '5'
- b6019101 f1 success [number:bare] '5': '25' (fails bare_digits) -> '2', read '5' vs '2'
- d52b4f67 f1 wrong [name] 'The Grand Ballroom': 'Nostalgia' (fails determiner) -> 'The Glass Menagerie', read ' Grand' vs ' Glass'
- ad7109d1 f2 refusal [number:bare] '500 Mbps': '18' (fails bare_digits) -> '132 points', read '5' vs '1'
- cf22b7bf f2 wrong [number:bare] '10 pounds': '2' (fails bare_digits) -> '55-inch', read '1' vs '5'
- ed4ddc30 f2 refusal [number:bare] '20': '856' (fails bare_digits) -> '32', read '2' vs '3'
- 18dcd5a5 f3 wrong [number:bare] '4': '5-2' (fails bare_digits) -> '3', read '4' vs '3'
- 58bf7951 f3 success [name] 'The Glass Menagerie': 'Mindful.org.' (fails determiner) -> 'The Grand Ballroom', read ' Glass' vs ' Grand'
- 60472f9c f3 success [number:bare] '2': '18' (fails bare_digits) -> '6', read '2' vs '6'
- affe2881 f3 success [number:bare] '32': '220' (fails bare_digits) -> '25', read '3' vs '2'
- e8a79c70 f3 success [number:bare] '2-3 eggs': '500' (fails bare_digits) -> '10 pounds', read '2' vs '1'
- 01493427 f4 refusal [number:bare] '25': '132 points' (fails bare_digits) -> '33', read '2' vs '3'
- 0e5e2d1a f4 wrong [number:bare] '38 subjects': '856' (fails bare_digits) -> '55-inch', read '3' vs '5'
- 3249768e f4 wrong [name] 'Absinthe': 'The Glass Menagerie' (fails determiner) -> 'Tom', read ' Abs' vs ' Tom'
- 5809eb10 f4 success [number:bare] '2014.': '3,000 miles' (fails bare_unit) -> '1300', read '2' vs '1'
- 5e1b23de f4 success [number:bare] '3': '25' (fails bare_digits) -> '5', read '3' vs '5'
- c7dc5443 f4 refusal [number:bare] '5-2': '8 meals' (fails bare_digits) -> '50', read '-' vs '0'
- ec81a493 f4 success [number:bare] '500': '3' (fails bare_digits) -> '100', read '5' vs '1'

Lost rows (rev 3 substitute fails rev 8, no passing candidate):

- gpt4_b5700ca0 f4 [name] 'the Episcopal Church': 'Kansas City Masterpiece' (fails determiner) -> no_substitute

Rev 2 check (5146f5b: the POOL verdict stands if >= 85 eligible and >= 40 failures on all 330): eligible 70, failures 32 -> **recorded for rev 4 (rev 3: POOL regardless; the 85 / 40 bar was a re-read trigger)**

Rows with a non-empty shared prefix: 12 (prefixes {'6:': 2, '1': 2, ' $': 4, ' The': 3, '5': 1}):

- 59524333 f0 wrong '6:00 pm' -> '6:45 AM': prefix '6:', read '0' vs '4'
- 95bcc1c8 f0 success '10' -> '15': prefix '1', read '0' vs '5'
- 0f05491a f1 wrong '120' -> '100': prefix '1', read '2' vs '0'
- 7527f7e2 f1 success '$800' -> '$0.75': prefix ' $', read '8' vs '0'
- 7a8d0b71 f1 success '$2,000' -> '$400,000': prefix ' $', read '2' vs '4'
- 86f00804 f1 success 'The Seven Husbands of Evelyn Hugo' -> 'The Sugar Factory at Icon Park.': prefix ' The', read ' Seven' vs ' Sugar'
- d52b4f67 f1 wrong 'The Grand Ballroom' -> 'The Glass Menagerie': prefix ' The', read ' Grand' vs ' Glass'
- 58bf7951 f3 success 'The Glass Menagerie' -> 'The Grand Ballroom': prefix ' The', read ' Glass' vs ' Grand'
- cc06de0d f3 refusal '$6' -> '$50': prefix ' $', read '6' vs '5'
- 852ce960 f4 wrong '$400,000' -> '$25,000': prefix ' $', read '4' vs '2'
- c19f7a0b f4 refusal '6:30 pm' -> '6:00 pm': prefix '6:', read '3' vs '0'
- c7dc5443 f4 refusal '5-2' -> '50': prefix '5', read '-' vs '0'

Rows excluded at the substitute step (no_substitute / first_token_equal / type_other_rev8):

- 2698e78f f0 success [other] 'every week': type_other_rev8
- 5d3d2817 f0 success [other] 'Marketing specialist at a small startup': type_other_rev8
- 60d45044 f0 refusal [other] 'Japanese short-grain rice': type_other_rev8
- 71a3fd6b f0 wrong [other] '+49 (0) 62 32 / 14 23 - 0': type_other_rev8
- e66b632c f0 wrong [other] '27 minutes and 45 seconds': type_other_rev8
- 07741c44 f1 success [other] 'under my bed': type_other_rev8
- 118b2229 f1 wrong [other] '45 minutes each way': type_other_rev8
- 2bf43736 f1 success [other] "Chapter 4 of Book 1, titled 'Vocal Prayer and Meditation'.": type_other_rev8
- 36580ce8 f1 success [other] 'bronchitis': type_other_rev8
- 3ba21379 f1 refusal [other] 'Ford F-150 pickup truck': type_other_rev8
- 51b23612 f1 refusal [other] 'Nu, pogodi!': type_other_rev8
- 66f24dbb f1 refusal [other] 'a yellow dress': type_other_rev8
- 3d86fd0a f2 success [other] 'a coffee shop in the city': type_other_rev8
- bc8a6e93 f2 wrong [other] 'a lemon blueberry cake': type_other_rev8
- gpt4_fa19884d f2 refusal [other] 'a bluegrass band that features a banjo player': type_other_rev8
- a82c026e f3 refusal [other] 'Dark Souls 3 DLC': type_other_rev8
- 001be529 f4 wrong [other] 'over a year': type_other_rev8
- 37d43f65 f4 success [other] '16GB': type_other_rev8
- 778164c6 f4 success [other] 'Grilled Snapper with Mango Salsa': type_other_rev8
- 8aef76bc f4 success [other] 'Mod Podge or another sealant': type_other_rev8
- b46e15ee f4 wrong [other] "the 'Walk for Hunger' charity event": type_other_rev8
- b759caee f4 success [other] '@jessica_poole_jewellery': type_other_rev8
- dccbc061 f4 refusal [other] 'A staunch atheist': type_other_rev8
- f4f1d8a4 f4 wrong [other] 'my sister': type_other_rev8
- gpt4_b5700ca0 f4 refusal [name] 'the Episcopal Church': no_substitute

Eligible rows (gold -> substitute, type):

- 15745da0 f0 [number:duration] 'three months' -> 'Two weeks'
- 1b9b7252 f0 [name] 'Mindful.org.' -> 'Harvard University'
- 59524333 f0 [date] '6:00 pm' -> '6:45 AM'
- 76d63226 f0 [number:bare] '55-inch' -> '10 pounds'
- 7e00a6cb f0 [name] 'International Budget Hostel' -> 'Harvard University'
- 95bcc1c8 f0 [number:bare] '10' -> '15'
- ba61f0b9 f0 [number:bare] '6' -> '8'
- c14c00dd f0 [name] "Trader Joe's" -> 'Paris'
- c5e8278d f0 [name] 'Johnson' -> 'IKEA'
- c7cf7dfd f0 [name] 'Nostalgia' -> 'Jaipur Rugs'
- fca762bc f0 [name] 'Memrise' -> 'Jaipur Rugs'
- fea54f57 f0 [name] 'Evolution' -> 'Johnson'
- 0862e8bf f1 [name] 'Luna' -> 'TechCorp'
- 0e4e4c46 f1 [number:bare] '132 points' -> '500 Mbps'
- 0f05491a f1 [number:bare] '120' -> '100'
- 19b5f2b3 f1 [number:duration] 'two weeks' -> 'three months'
- 29f2956b f1 [number:duration] '30 minutes' -> '4 days.'
- 5831f84d f1 [number:bare] '15' -> '5-2'
- 618f13b2 f1 [number:bare] 'six' -> 'Two'
- 7401057b f1 [number:bare] 'Two' -> 'five'
- 7527f7e2 f1 [number:currency] '$800' -> '$0.75'
- 7a8d0b71 f1 [number:currency] '$2,000' -> '$400,000'
- 86f00804 f1 [name] 'The Seven Husbands of Evelyn Hugo' -> 'The Sugar Factory at Icon Park.'
- b6019101 f1 [number:bare] '5' -> '2'
- d52b4f67 f1 [name] 'The Grand Ballroom' -> 'The Glass Menagerie'
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
- ad7109d1 f2 [number:bare] '500 Mbps' -> '132 points'
- cf22b7bf f2 [number:bare] '10 pounds' -> '55-inch'
- ed4ddc30 f2 [number:bare] '20' -> '32'
- 099778bb f3 [number:percent] '20%' -> '10%'
- 18dcd5a5 f3 [number:bare] '4' -> '3'
- 577d4d32 f3 [date] '7 pm' -> '2 AM'
- 58bf7951 f3 [name] 'The Glass Menagerie' -> 'The Grand Ballroom'
- 5a4f22c0 f3 [name] 'TechCorp' -> 'Target'
- 60472f9c f3 [number:bare] '2' -> '6'
- 8464fc84 f3 [name] 'Roscioli' -> 'TechCorp'
- 8b9d4367 f3 [name] 'Jaipur Rugs' -> 'Kansas City Masterpiece'
- a96c20ee f3 [name] 'Harvard University' -> 'Summer Vibes'
- affe2881 f3 [number:bare] '32' -> '25'
- cc06de0d f3 [number:currency] '$6' -> '$50'
- ccb36322 f3 [name] 'Spotify' -> 'Emma'
- e8a79c70 f3 [number:bare] '2-3 eggs' -> '10 pounds'
- e982271f f3 [name] 'Revolution Hall' -> 'Emma'
- f523d9fe f3 [name] 'Doc Martin' -> "Trader Joe's"
- f9e8c073 f3 [number:bare] 'five' -> 'six'
- faba32e5 f3 [number:duration] '24 hours' -> '3 days.'
- 01493427 f4 [number:bare] '25' -> '33'
- 0e5e2d1a f4 [number:bare] '38 subjects' -> '55-inch'
- 1e043500 f4 [name] 'Summer Vibes' -> 'Tom'
- 26bdc477 f4 [number:bare] 'five' -> 'four'
- 3249768e f4 [name] 'Absinthe' -> 'Tom'
- 5809eb10 f4 [number:bare] '2014.' -> '1300'
- 5e1b23de f4 [number:bare] '3' -> '5'
- 7a87bd0c f4 [number:duration] '4 weeks' -> '5.5 weeks'
- 852ce960 f4 [number:currency] '$400,000' -> '$25,000'
- c19f7a0b f4 [date] '6:30 pm' -> '6:00 pm'
- c7dc5443 f4 [number:bare] '5-2' -> '50'
- d596882b f4 [name] 'By Chloe' -> 'Data Science'
- dd2973ad f4 [date] '2 AM' -> '7 pm'
- ec81a493 f4 [number:bare] '500' -> '100'
