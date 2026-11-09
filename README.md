# memctl

**Honors thesis: A Mechanistic Investigation of Evidence-Use Failures In Language Models.**

When a small language model has the evidence it needs in front of it, why does it still answer "unknown" or answer
wrong? This repository holds the code, runs and research record for that question. Most of the work is on Qwen2.5-3B,
with checks on Llama, Gemma and Phi.

> **Start here:** [research/STATE.md](research/STATE.md) (what is currently supported, with strength and caveats),
> then [EXPERIMENTS.md](EXPERIMENTS.md) (the record; it wins any disagreement) and
> [research/EXPERIMENT_INDEX.md](research/EXPERIMENT_INDEX.md) (where each experiment's configs, runs, commits and
> costs live).

## How the project got here

1. **Memory controllers (Experiments 1–32).** The project began by asking whether a learned memory controller helps
   a model answer questions over long histories (LongMemEval, LoCoMo). The only clear win was query-aware selection at
   question time, and a 31-parameter logistic regression did as well as the learned head (§32). RL never beat
   imitation. The ceiling was the reader itself: with perfect evidence in view, a 3B reader still trailed a 7B one.
2. **Why the reader fails (Experiments 34–50).** That ceiling became the thesis. The main findings so far:
   - The answer is usually present inside the model when it fails. On most failures the gold answer's first token is
     in the final layer's top 10, in all four families (§34, §43).
   - Many failures are a refusal, not a gap in knowledge. One instruction line ("do not reply unknown") lifts 3B
     accuracy by 0.176 (§35). A single content-free steering direction at layer 29 opens the refusal, and the evidence
     then decides whether the answer is right (§38–§40).
   - The fixes are not selective. Every push tried so far also makes the model answer questions it should refuse
     (§41, §42, §49).
   - On controlled tasks, most false "unknown"s are "note, then refuse": the model writes the right answer in its own
     note and then says "unknown" (§49). Asking the model whether the evidence is sufficient was not a usable measure
     (§50).
3. **Experiments 51–55.** Agent-proposed weight edits and their random-edit controls (§51, §53), a screen of readers
   below 3B (§52), twin-pair patching for a native "unknown" switch (§54), and a clean-prompt check (§55): with the
   evidence and model fixed, a consistent abstention rule does not reduce false "unknown"s, and removing the option
   removes them only by making the model answer questions that have no answer. Results and caveats are in
   [EXPERIMENTS.md](EXPERIMENTS.md); what they support is in [research/STATE.md](research/STATE.md).

## Research questions

- **RQ1 (behaviour):** can a push make the model answer, without making it guess, when the evidence is in view?
- **RQ2 (inside the model):** is a failure one of not using the evidence, or of using it and still not producing
  the answer?
- **RQ3 (exploratory):** if the evidence is used, what makes the model refuse anyway, and where would a fix go?

## Where things are

| Path | What |
|---|---|
| [research/](research/) | State, decisions, open questions, proposals, reviews, audits and briefings |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Full record of §33–§55, with numbers and caveats; §1–§32 condensed |
| `memctl/` | Library: data loaders for the §23/§24 prompts, answer selection and grading, the lenses, steering, patching and editing tools, LLM backends and the judge |
| `scripts/` | Experiment scripts (`expNN_*.py`, the J-lens fit / read / eval scripts, pod chains) |
| `runs/`, `cache/` | Run outputs and cached model outputs, not in git (never delete the §23/§24 generation caches: every prompt is rebuilt from them) |
| [docs/research/](docs/research/) | Per-experiment report copies (§23, §24, §34, §35) and the J-lens design notes under `docs/superpowers/` |

**The memory-controller era (§1–§32)** was removed from the working tree on 2026-11-09. Its code, configs, tests and
full experiment text are at the git tag `memctl-era-final`; its run data is in the local archive
`archive/memctl-era-data.tar.zst` (manifest beside it). To look at or restore a file:

```bash
git show memctl-era-final:memctl/controllers/heuristics.py
git checkout memctl-era-final -- memctl/rl                 # bring a folder back
tar -I zstd -xf archive/memctl-era-data.tar.zst runs/exp19_gate_f0   # unpack one run
```

The earlier Phase 2 work (eviction policies on LoCoMo) is on branch `phase2-locomo-baseline`, tag `phase2-complete`.

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install -c constraints.txt -e ".[dev,models,stats]"
.venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest -q
```

Secrets (`HF_TOKEN`, `RUNPOD_API_KEY`, `ANTHROPIC_API_KEY`) go in `.env`, copied from `.env.example`. GPU runs use
rented RunPod pods (image-based A40, HF fp32); each experiment's proposal records its cost and hard stop, and each
pod chain (`scripts/expNN_pod.sh`) has a laptop stub mode.
