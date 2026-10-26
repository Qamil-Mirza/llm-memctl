# §43 rev 1 build evidence (B1 BOS fix, B3 FamilyConfig), laptop, $0

2026-10-26, Executor. Spec: research/proposals/2026-10-26-exp43-cross-family.md rev 1 (rulings B1, B3).

## B1: the lens encode of a chat prompt adds no special tokens

- Code: `memctl/analysis/jlens_readout.encode_chat_ids` (tokenizer(text, add_special_tokens=False, truncation)). Used by
  the `read_record` guard in `scripts/jlens_read.py` (was `lens_model.encode`, i.e. jlens's tokenizer call with special
  tokens on, plus `add_bos_token` forced on by `from_hf(force_bos=True)`), and by `scripts/jlens_eval.logits_at`
  (`eval_ids`: a chat-template text keeps its own BOS; a raw text gets the tokenizer's BOS only when the family fits
  with one, i.e. as the lens was fitted; Qwen has no BOS token, so nothing changes for it).
- (a) Qwen 3B, one dev row (fold 0, `00ca467f`, the first §34-population record; 908 tokens; the §34 lens
  `runs/exp34_jlens/lenses/qwen3b.pt`), CPU fp32: `b1_qwen_inprocess.py` reads it with `read_record` as committed at
  9490d07 (verbatim copy `_jlens_read_9490d07.py`) and with the fixed one, twice each, alternating, on one loaded model.
  **Every array (ranks and top-k ids and logits, J-lens and logit lens, 36 layers x 7 positions x 3 tracked tokens) and
  every measure is identical before vs after and run vs run** (`b1_qwen_inprocess.json`: before_after_identical true,
  run_to_run_identical true; the forwarded ids are the same sha on all four reads).
- The first attempt, one process per code state (`b1_qwen_read.py`, `b1_qwen_{before,after}.*`, `b1_qwen_compare.json`),
  is kept as found: the positions and forwarded ids agree, but logits differ by up to 1.6e-3 between two processes on
  this CPU, so up to 234 of 756 tracked ranks (mostly far down the vocabulary) and 1-2 top-10 ids differ. That is
  process-to-process CPU nondeterminism, not the fix (the in-process test shows the same code reproduces itself
  bit for bit and the two codes agree bit for bit). It is a caveat for any laptop-vs-laptop comparison across processes.
- (b) `tests/test_exp43_families.py::test_lens_encode_of_a_chat_prompt_has_the_right_bos`: exactly one BOS, first, for
  Llama-3.2-3B (128000) and gemma-3-4b (2); none for Phi-4-mini (its "BOS" 199999 = <|endoftext|> never appears); the
  readout's chat ids equal the guard's; and the old route (add_bos_token on, special tokens added) gives a second BOS
  for Llama and Gemma and a leading <|endoftext|> for Phi. Qwen: `encode_chat_ids` equals the default encode.

## B3: FamilyConfig, Qwen unchanged

- `memctl/analysis/families.py`: LLAMA3B, GEMMA4B, PHI4MINI (+ QWEN3B as a reference no §38-§42 path reads).
- `steering.HFEngine(model, threads, family=None, weights=None)`: family None is the old code path, line for line.
- `jacobian_data.load_family_records(cfg, fold)` + `family_population`: identical output to stage 0's
  `runs/exp43_stage0/data.py load_records24` on folds 0-1 for all three families (checked on the laptop).
- Qwen fingerprints before any B3 code (`qwen_baseline.py` -> `qwen_baseline_before.json`) and after (`..._after.json`):
  identical (Qwen chat text and ids of a fixed prompt, HFEngine stop ids, and a full §38 stub design run on the real
  §35 cache, every jsonl row minus timing plus directions/gate/sanity json). The test file re-checks both against
  the before file.
