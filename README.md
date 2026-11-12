# Data Science Honors Thesis

**Author(s):** Qamil Mirza

**Currently investigating:** what inside a reasoning model makes it ignore a valid update that arrives in the middle
of its thinking ("self-doubt").

## Directory layout

| Path | What it holds |
|---|---|
| `selfdoubt/` | The current work: code, tests, the pre-registration (`PROPOSAL.md`), spend log (`SPEND.md`), frozen inputs and result summaries |
| `research/` | The research record: `PAPER_CONTRACT.md` (the claim, dates, gates and evidence rules every experiment is checked against), `STATE.md` (what is supported, and how strongly), `DECISIONS.md` (every decision by number), `OPEN_QUESTIONS.md`, `RELATED_WORK.md` (closest papers) |
| `EXPERIMENTS.md` | Every experiment: what it asked, what it found, what it cost |
| `REPRODUCE.md` | How to rerun each reported number, and the conventions for new work |
| `archive/` | Earlier, closed work. Only its `README.md` and checksum files are in git; the data itself stays local |
| `runs/`, `external/` | Run outputs and the original authors' code. Local only, not in git |

Two earlier phases of the thesis are closed: learned memory controllers (§1–§32) and false abstention in small models
(§33–§57). Their code and full record are at the git tags `memctl-era-final` and `false-abstention-era-final`, and
[archive/README.md](archive/README.md) explains how to restore them.

## Current goal

Wu, Miroyan et al. ("Are Large Reasoning Models Interruptible?", arXiv 2510.11713) showed that when a reasoning model
is given a corrected problem partway through its reasoning, it often finishes the old problem anyway.

This project asks where, inside the model, the update gets lost. Our first results, on Qwen3 1.7B, suggest the model
does read the update, then lets it fade before it writes the answer. The aim is to find which layers, attention heads
or neurons cause that, and to test the explanation causally on Qwen3 1.7B and 8B. Every claim is checked against
simple baselines: random parts of the model of the same size, a plain bias on the output, and the original paper's
prompt fix.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```
