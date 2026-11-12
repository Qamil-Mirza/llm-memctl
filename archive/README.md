# archive/

This folder holds the closed work, packed on 2026-11-11 under the consolidation plan. It is
gitignored. Only this README and the `SHA256SUMS` files are tracked.

## Code and record: git tags, not folders

| Era | Tag | Commit | What is at the tag |
|---|---|---|---|
| Memory controllers (Experiments 1–32, STATE F1–F10) | `memctl-era-final` | 0c016d4 | memctl-era code, configs, record |
| False abstention (§33–§57, STATE F11–F24) | `false-abstention-era-final` | ef91067 | `memctl/`, `scripts/`, `tests/`, `docs/`, full `EXPERIMENTS.md`, reviews, proposals, briefings, audits, roles, ~12.2k tracked run files |

The tags exist only on this laptop. The origin repo is public and nothing was pushed. The single-file copy of the
whole history is the bundle. It is on this laptop only until the user copies it elsewhere; the Mac route was blocked
on 2026-11-11 (ssh: permission denied).
- `archive/llm-memctl-all-2026-11-11.bundle` (291 MB; sha256 `8cd070f3e9c60f7115b89da4dd8540ddbfe97f81578d3f8f739ced6d0ace3edd`);
- restore with `git clone llm-memctl-all-2026-11-11.bundle`.

## Data

**`memctl-era/`** holds the data of Experiments 1–32. Its two tarballs date from 2026-11-08; their manifests list paths
only. The sha256 of each tarball is in `memctl-era/SHA256SUMS`.

**`false-abstention/`** holds every file that was under `runs/`, `cache/`, `data/` and `external/jacobian-lens` on
2026-11-11. The only exceptions are the deletions listed below. The tarballs come in two tiers:
- **kept** (K01–K21): per-row outputs, frozen inputs, the activations some findings need, populations, data and
  provenance;
- **bulk** (B01–B05): stubs, smokes, scratch, activations no report reads, and lens checkpoints.

Every member was checked after packing. The tarball was streamed back and each file's sha256, size and link target
compared with a manifest written from the original. All 233,366 members (kept 213,294, bulk 20,072) matched before any original was removed.

| Tarball | Holds | Needed by |
|---|---|---|
| K01 | §23/§24 populations (`runs/exp2{3,4}_*`), verdicts, `_sweeps_exp24`, `cache/generations_*`, `cache/judge_exp23` | F1, F2, F6, F11, F22 (LME line), §44/§45/§47 rows |
| K02 | `exp33_jacobian`, `exp34_jlens` per-row (not the lenses), `exp34_jlens_{ablated,own}`, §34 reports and pod logs | F11, F12 |
| K03 | §35–§37 per-row, `cache/judge_exp35` | F6, Q15 |
| K04 | §38–§42 (including `exp38_dev/qwen3b/directions.npz`) | F6, Q14/Q16; F18, F23, F24 |
| K05 | §43 | F11 |
| K06 | §44 (json; no resid) | F12, F13, F15 |
| K07 | §45 maps including the stage-1/3 npz | F13 |
| K08 | §46 per-triple json, `exp46_pod`, `exp46_probe` npz | F14 |
| K09 | §47 | F15 |
| K10 | §49 including `run/resid` npy and probe pickles (sklearn 1.9.1) | F16 |
| K11 | §50 rows, cases, design | F17 |
| K12 | §51 per-row, `cases51.jsonl` (sha 5eeb7c40…, the frozen input pinned by `freeze/F{0,1}_draft.json`), editor provenance | F18, F19, F23, F24 |
| K13 | §52 | F21 |
| K14 | §53 per-row, cases, manifest | F19 |
| K15 | §54 | F20, F22 |
| K16 | §55 per-row (`pod/`), tracked files | F22 |
| K17 | §56 including `run/stage1/*.npz` and `devb_una` | F23, F24 |
| K18 | §57 including `run/stage2/*.npz` and `run/stage3` | F24 |
| K20 | `data/` (LongMemEval, LoCoMo, J-lens corpus), `external/jacobian-lens@581d3986` | populations, `jlens` |
| K21 | provenance (see below) | F13 audit, F20 note audit, recounts |
| B01 | stubs, smokes, scratch, old pipeline drivers and logs, stub judges | none |
| B02 | J-lens files: `exp34_jlens/lenses/*` (checkpoints, 7B, 0.5B, 3B) and `exp45_maps/lenses/qwen3b.pt` | regenerating §45 maps only |
| B03 | unread activations (`exp50` resid and s2dev, `exp46_patch` and `exp44_swap` resid, stub safetensors, `exp53/edits`) | none |
| B04 | §51 editor workspace, `run/b0/E/*.npy`, dry runs and dummy models | none |
| B05 | §55 bundles, stub runs, stopped `run/` | none |

No reported number needs a bulk tarball. The test suite does read two bulk groups: B03 (`runs/exp53/edits/*.json`)
and B04 (`runs/exp51/dryrun_api/api_usage.json`). So "kept only" restores the numbers, and "kept + bulk"
restores the suite.

Tarball shas are in `false-abstention/SHA256SUMS`. Per-file manifests (path, type, bytes, sha256 or link target) are in
`false-abstention/manifests/` and inside K21. `manifests/_assignment.tsv` maps every original path to its tarball.

**K21 (provenance)** holds:
- the Advisor's recompute scripts copied from /tmp scratchpads (§45 audit, §54 note-audit rules and labels, the
  §46–§53/§56 recounts), plus `recompute36.py`;
- `pip freeze` of the laptop `.venv` and of the deleted `runs/exp55/d30venv`;
- the checks behind each deletion;
- `transfer-provenance/`: 19 files that existed only inside the deleted pod upload bundles (`bundle_info.json`,
  `COMMIT`, `head_commit`, and the E+S extract of `cases51.jsonl` shipped to §51 pod S, sha 55e89193…, which is a
  byte-identical prefix of the K12 file);
- the builder and checker scripts, the repo inventory, the consolidation plan and its review, the step-9
  verification logs (`verification/`), and all per-file manifests;
- `research/briefings/2026-11-07-codex-next-directions-for-advisor.md`, archived on the user's word (2026-11-11).

## Deleted on 2026-11-11 (user's yes, D1–D8)

| Item | Size | Why it is safe |
|---|---|---|
| `cache/exp44_hf` (Qwen2.5-7B-Instruct @ a09a3545) | 14.5 GB | all 4 weight shards' sha256 equal their LFS sha (K21) |
| `runs/_transfer` (pod upload bundles) | 3.4 GB | every member is in git, on disk (now in a K/B tarball), or saved in K21 |
| `runs/exp55/d30venv` | 1.2 GB | `pip freeze` in K21 |
| `runs/exp34_jlens/lenses/community_qwen7b.pt` | 0.65 GB | sha256 equals HF `andyx10/jacobian-lens-qwen2.5-7b-instruct@f70dfaca` `qwen7b_lens.pt` |
| `.superpowers/`, the stray file `\` | 1.2 MB | none needed |

## Restoring the false-abstention tree

```bash
git worktree add --detach .worktrees/fa false-abstention-era-final
for f in archive/false-abstention/{kept,bulk}/*.tar.zst; do zstd -dc "$f" | tar -xf - -C .worktrees/fa; done
ln -s "$PWD/.venv" .worktrees/fa/.venv     # the §51 sandbox binds ROOT/.venv
cd .worktrees/fa && export PYTHONPATH=$PWD:$PWD/external/jacobian-lens
```

This was verified on 2026-11-11:
- `git status` was clean after untarring, so every tracked run file was byte-identical;
- `exp52_report.py`, `exp55_report.py --root runs/exp55/pod/run` and `exp57_stage3.py report` rewrote their tracked
  reports byte-identically;
- the full suite results are in REPRODUCE.md.

See REPRODUCE.md for the finding → command table and caveats.
