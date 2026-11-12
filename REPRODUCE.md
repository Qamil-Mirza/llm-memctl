# Reproducing the numbers

## 1. Self-doubt (live work): conventions

- **Code** lives in `selfdoubt/`, one module per stage, with tests in `selfdoubt/tests`. Nothing imports archived code.
- **Frozen inputs** are small and committed under `selfdoubt/frozen/`: `split.json` and `s2_heldout/rows.jsonl`. Every
  summary records their sha256.
- **Per-row outputs** go to `runs/selfdoubt/<stage>/<run>/`, which is gitignored. After a reading is recorded they are
  sealed into `archive/selfdoubt/<stage>.tar.zst` with a per-file sha256 manifest, and the tarball sha goes into the
  summary.
- **Summaries** are `selfdoubt/results/<reading>.json`, committed. Each carries:
  - the input shas;
  - the script path and git sha;
  - an engine block: package versions, model revision, dataset revision, seeds.
- **One command per reported number.** It is listed in the table below and reads runs or tarballs only, with no model
  call. A number without a row here is not reported.
- **Pins:**
  - each pod stores `pip freeze` as `env.txt`, committed beside its results;
  - model revisions are in `selfdoubt/common.py` (Qwen3-8B b968826d, Qwen3-1.7B 70d244cc);
  - the dataset is `dynamic-lm/update-interrupt-benchmark@6ac4ea4b`;
  - never use a floating `main`.
- **What is committed:** code, frozen inputs under 10 MB, summaries, env freezes and manifests. Per-row outputs are
  tarred with sha256. Venvs and model weights are never stored; record their revision instead.

| Reading | Command | Reads |
|---|---|---|
| (filled in as self-doubt stages are recorded) | | |

## 2. Archived eras

Code and record live at git tags; data lives in `archive/` tarballs. `archive/README.md` lists every tarball.

```bash
git worktree add --detach .worktrees/fa false-abstention-era-final      # or memctl-era-final
for f in archive/false-abstention/kept/*.tar.zst; do zstd -dc "$f" | tar -xf - -C .worktrees/fa; done
for f in archive/false-abstention/bulk/*.tar.zst; do zstd -dc "$f" | tar -xf - -C .worktrees/fa; done  # tests and §51/§53 dry runs read bulk files
ln -s "$PWD/.venv" .worktrees/fa/.venv          # exp51_harness.py binds ROOT/.venv site-packages into its sandbox
cd .worktrees/fa && export PYTHONPATH=$PWD:$PWD/external/jacobian-lens  # memctl + jlens from the worktree
```

Memory-controller era (F1–F10): the same pattern with `git worktree add --detach .worktrees/mc memctl-era-final`, then
`for f in archive/memctl-era/*.tar.zst; do zstd -dc "$f" | tar -xf - -C .worktrees/mc; done` (paths inside are
repo-relative: `runs/…`, `external/…`). These two tarballs have not been test-restored since they were made on
2026-11-08; their manifests list paths only, and their sha256 are in `archive/memctl-era/SHA256SUMS`.

**Environment.** Use the laptop `.venv` (Python 3.13.9). Its `pip freeze` is in K21; the pins are in `constraints.txt`
at the tag, plus scipy 1.18.1 and scikit-learn 1.9.1. The `.venv` has editable installs of `memctl` and `jlens` that
point at main-checkout paths which no longer exist. `PYTHONPATH` as above takes precedence, so no reinstall is needed.
Use `pip install -e .` inside the worktree if you prefer.

**Model snapshots** (HF revisions):
- Qwen2.5-3B-Instruct aa8e7253;
- Qwen2.5-7B-Instruct a09a3545 (deleted locally; the shard LFS shas are in K21);
- the §43/§52 readers as listed in `research/audits/2026-11-11-repo-inventory.md` §4 (in K21).

### Finding → command (run inside the restored worktree)

| Finding | Untar | Command(s) |
|---|---|---|
| F1–F10 | `memctl-era/*` at `memctl-era-final`, plus K01 for F2/F6 | see `EXPERIMENTS.md` at that tag |
| F11 §43 | K05, K01, K02, K03 | `scripts/exp43_lens_report.py --family {llama,gemma,phi} [--confirm]`, `exp43_llama_report.py`, `exp43_lens_profile.py` |
| F12 §44 | K06, K02 | `scripts/exp44_report.py [--confirm]`, `exp44_swap_report.py` |
| F13 §45 | K07, K06, K21 | `scripts/exp45_report.py --half all`, `exp45_render.py`; audit sentence: K21 `recompute-from-scratchpads/831d3c8f/audit45*.py` |
| F14 §46 | K08 | `scripts/exp46_report.py`; probe numbers from `exp46_probe.py` over the capture npz |
| F15 §47 | K09, K06 | `scripts/exp47_report.py --root runs/exp47_bridge/cuda-float32` |
| F16 §49 | K10 | `scripts/exp49_report.py` (needs sklearn 1.9.1 for the pickles) |
| F17 §50 | K11 | `scripts/exp50_report.py` (stage 1) |
| F18 §51 | K12, K04 | `scripts/exp51_stage4.py --stage h-report --root <root> --subs <subs>` (caveat 2) |
| F19 §53 | K14, K12 | `scripts/exp53_run.py --stage report` |
| F20 §54 | K15, K21 | `scripts/exp54_run.py --stage pairs` / `regrade`; note audit: K21 `noteaudit/` (RULES.md, labels.json) |
| F21 §52 | K13 | `scripts/exp52_report.py` (**verified byte-equal 2026-11-11**) |
| F22 §55 | K16, K15, K01 | `scripts/exp55_report.py --root runs/exp55/pod/run` (**verified byte-equal 2026-11-11**) |
| F23 §56 | K17, K12, K04 | `scripts/exp56_stage1_report.py` |
| F24 §57 | K18, K17, K12, K04 | `scripts/exp57_stage1.py`; `exp57_stage2.py report`; `exp57_stage3.py report` (**verified byte-equal 2026-11-11**) |

Arguments not marked as verified come from the 2026-11-10 inventory; check each script's `--help`.

**Test-suite check (2026-11-11, laptop CPU, `nice -n 19`).** The suite ran in a worktree at
`false-abstention-era-final`, restored from the tarballs:
- **Kept tier only:** 720 passed, 6 failed (894 s). Two tests read bulk files (B03, B04). Three exp51 sandbox tests
  need `ROOT/.venv`. One test, `test_exp43_lens_pod::test_pod_chain_skips_stages_past_the_deadline`, failed once under
  load at `HARD_STOP_S=1`; it passed alone and in the next full run.
- **Kept + bulk + the `.venv` link:** **726 passed, 0 failed, 0 skipped (947 s)**. This matches the 726 that passed on
  the full tree before archiving.
- Logs are in K21 `verification/`.

**Caveats (not fixed; fixing them would mean new code):**
1. `scripts/exp57_k50_halfsample.py:7` hard-codes `ROOT = Path("/home/qamil-mirza/Code/llm-memctl")`. Edit that line
   in the worktree before running it.
2. `exp51_stage4.py` loads the engine even for the report, and `require_design` matches `gpu/design.json`, so
   `--root gpu_podHT` fails its guard as laid out.
3. §51 pod S read an E+S extract of `cases51.jsonl` (K21 `transfer-provenance/exp51_podS`, sha 55e89193…). It is a
   byte-identical prefix of the K12 file (sha 5eeb7c40…), which the freeze pins.
4. §43 per-family J-lens `.pt` files were never copied back from the pods, so §43 J-lens readings need a refit.
5. The laptop runs of §37–§44 record the engine but not the model revision. One Qwen2.5-3B snapshot (aa8e7253) was on
   the laptop.
