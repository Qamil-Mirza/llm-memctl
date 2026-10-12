"""The Jacobian lens (§33): how much a reader's output depends on each region of its prompt.

Pure functions over strings, token offsets and numpy arrays, plus one gradient pass that needs torch
and transformers (imported inside the functions that use them). Regions: the chat template, the
instruction, the evidence memory lines, the distractor memory lines and the question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGIONS = ("template", "instruction", "evidence", "distractor", "question")
MEMORY_HEADER = "## Memory\n"
INPUT_HEADER = "\n\n## Current input"
TARGETS = ("gold", "margin", "own")
ANSWER_PREFIX = "Answer:"


def segment_prompt(prompt: str, memory_lines: tuple[str, ...], evidence_lines: tuple[str, ...]) -> list[tuple[str, int, int]]:
    """Char spans (region, start, end) tiling the prompt. Memory lines are found by exact search in order,
    so a line whose content contains a newline stays one line. Gaps (separators) join the span before them."""
    header = prompt.index(MEMORY_HEADER) + len(MEMORY_HEADER)
    spans: list[tuple[str, int, int]] = [("instruction", 0, header)]
    cursor = header
    evidence = set(evidence_lines)
    for line in memory_lines:
        start = prompt.find(line, cursor)
        if start < 0:
            raise ValueError(f"memory line not found in prompt: {line[:60]!r}")
        if start > cursor:  # separator before this line belongs to the previous span
            region, s, _ = spans[-1]
            spans[-1] = (region, s, start)
        end = start + len(line)
        spans.append(("evidence" if line in evidence else "distractor", start, end))
        cursor = end
    question = prompt.index(INPUT_HEADER, cursor)
    if question > cursor:
        region, s, _ = spans[-1]
        spans[-1] = (region, s, question)
    spans.append(("question", question, len(prompt)))
    return spans


def token_regions(offsets: list[tuple[int, int]], prompt_start: int, prompt_len: int,
                  segments: list[tuple[str, int, int]]) -> list[str]:
    """A region per token. A token is placed by its start char; tokens starting outside the prompt are template."""
    regions = []
    for start, end in offsets:
        if end <= prompt_start or start >= prompt_start + prompt_len:
            regions.append("template")
            continue
        char = max(start, prompt_start) - prompt_start
        region = next((name for name, s, e in segments if s <= char < e), None)
        regions.append(region or "template")
    return regions


def wrap_chat(tokenizer, prompt: str) -> str:
    """The chat text the vLLM server built: one user message, generation prompt, no thinking (as llm.py)."""
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def unknown_id(tokenizer) -> int:
    """The first token of " unknown", the competitor in the margin target."""
    return _ids(tokenizer, " unknown")[0]


def continuation_ids(tokenizer, target: str, gold: str, output: str) -> tuple[list[int], int]:
    """Ids appended after the chat text, and the index into them where scoring starts. The gold answer is
    tokenised on its own (" " + gold) so the scored ids are exactly the gold's, whatever "Answer:" merges into."""
    if target == "own":
        return _ids(tokenizer, output), 0
    prefix = _ids(tokenizer, ANSWER_PREFIX)
    answer = _ids(tokenizer, " " + gold)
    if target == "gold":
        return prefix + answer, len(prefix)
    if target == "margin":
        return prefix + answer[:1], len(prefix)
    raise ValueError(f"unknown target {target!r} (known: {TARGETS})")


@dataclass
class JacobianResult:
    regions: list[str]  # per prompt-side token (chat text only, continuation excluded)
    g_l2: np.ndarray  # per-token L2 norm of d target / d embedding
    g_gxi: np.ndarray  # per-token |gradient . embedding|
    target_value: float
    n_tokens: int


def _scalar(logits, n_prompt: int, cont: list[int], start: int, target: str, unk: int):
    """The target from the logits of chat text + continuation. gold/own: summed teacher-forced log-probs of
    cont[start:]. margin: logit of the first gold token minus the logit of " unknown", at the position predicting it."""
    import torch

    if target == "margin":
        row = logits[n_prompt + start - 1].float()
        return row[cont[start]] - row[unk]
    logp = torch.log_softmax(logits[n_prompt + start - 1: n_prompt + len(cont) - 1].float(), -1)
    return logp.gather(-1, torch.tensor(cont[start:], device=logp.device)[:, None]).sum()


def jacobian_pass(model, tokenizer, prompt: str, memory_lines, evidence_lines, gold: str, output: str,
                  targets: tuple[str, ...] = ("gold", "margin")) -> dict[str, JacobianResult]:
    """Gradients of each target with respect to the prompt-side input embeddings. gold and margin share one
    forward (the margin continuation is a prefix of the gold one); own needs a second."""
    import torch

    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"unknown targets {sorted(unknown)} (known: {TARGETS})")
    text = wrap_chat(tokenizer, prompt)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = list(encoded["input_ids"])
    n_prompt = len(prompt_ids)
    regions = token_regions(encoded["offset_mapping"], text.index(prompt), len(prompt),
                            segment_prompt(prompt, memory_lines, evidence_lines))
    unk = unknown_id(tokenizer)
    embedding = model.get_input_embeddings()
    groups = [(forward, [t for t in group if t in targets])
              for forward, group in (("gold", ("gold", "margin")), ("own", ("own",)))]
    results: dict[str, JacobianResult] = {}
    for forward, group in groups:
        if not group:
            continue
        cont, start = continuation_ids(tokenizer, forward, gold, output)
        full = torch.tensor([prompt_ids + cont], device=embedding.weight.device)
        embeds = embedding(full).detach().requires_grad_(True)
        logits = model(inputs_embeds=embeds).logits[0]
        x = embeds[0, :n_prompt].detach().float()
        for number, target in enumerate(group):
            t_cont, t_start = continuation_ids(tokenizer, target, gold, output)
            value = _scalar(logits, n_prompt, t_cont, t_start, target, unk)
            (grad,) = torch.autograd.grad(value, embeds, retain_graph=number < len(group) - 1)
            g = grad[0, :n_prompt].float()
            results[target] = JacobianResult(regions, g.norm(dim=-1).cpu().numpy(),
                                             (g * x).sum(-1).abs().cpu().numpy(), float(value.detach()), n_prompt)
        del logits, embeds
    return results


def target_value(model, tokenizer, prompt: str, gold: str, output: str, target: str) -> float:
    """The scalar target without gradients (the ablation check)."""
    import torch

    prompt_ids = _ids(tokenizer, wrap_chat(tokenizer, prompt))
    cont, start = continuation_ids(tokenizer, target, gold, output)
    device = model.get_input_embeddings().weight.device
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([prompt_ids + cont], device=device)).logits[0]
        return float(_scalar(logits, len(prompt_ids), cont, start, target, unknown_id(tokenizer)))
