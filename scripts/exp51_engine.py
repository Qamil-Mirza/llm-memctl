#!/usr/bin/env python
"""§51 engine: Qwen2.5-3B (or the dummy tiny Qwen2) in float32 on the CPU, with the weight-edit layer, the online
triggers (B1 logit bias, B4 steering) and the component hooks (M2 transplant / clamp, captures). Batch size 1, greedy.

Positions are absolute token positions in the sequence (prompt chat ids first). Generation: KV over ids[:-1]; ids[-1]
is the first stepped position (step 0); every later step feeds the previous argmax. With a controlled prefix the
last stored id is B0's marker ':' token, so step 0 is P_ans.

Spec: research/proposals/2026-11-02-exp51-agent-weight-editing.md rev 1a §2, §5.1-5.2, §6.2, §13.3 (D-72).

Fixed choices of this build:
- Marker (total effect): the first time the decoded generated text contains "\\nAnswer:", the token that completed it
  is the marker's last token; its forward (the next step, where it is the input) is the trigger position. B1's bias
  acts on the logits of that forward and every later one; B4 adds at layer 29 at that forward and every later one.
  A later marker changes nothing; no marker, no trigger.
- Stop (total effect): EOS / <|im_end|>, max_new_tokens 160, or once the text after the last "\\nAnswer:" holds a
  non-empty line ended by a newline (strict marker; §49's stop matched "answer:" anywhere).
- Controlled prefix: <= 16 new tokens (MAX_NEW_RELEASE), stop at EOS or a newline after a non-empty answer.
- Hybrid answer-site: the KV over positions <= P_ans − 1 is computed with every edit reverted, then the arm's edit is
  re-applied for P_ans and later.
- Teacher-forced reads: one forward with no cache; residuals = decoder-layer outputs; component outputs from forward
  hooks on the module (o_proj / down_proj output, RMSNorm output); logits only at the requested positions
  (logits_to_keep = an index tensor), full vocabulary or the requested token ids.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from memctl.analysis import edit51 as ED  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
DUMMY_DIR = ROOT / "runs/exp51/dummy_model"
DUMMY36_DIR = ROOT / "runs/exp51/dummy_model_36"   # the real layer count at tiny width (second dry run)
MAX_NEW_TOTAL, MAX_NEW_RELEASE = 160, 16
MARKER = "\nAnswer:"
STEER_LAYER = 29
COMPONENT_MODULE = {"o_proj": "self_attn.o_proj", "down_proj": "mlp.down_proj", "input_layernorm": "input_layernorm",
                    "post_attention_layernorm": "post_attention_layernorm", "resid": ""}   # resid: the decoder layer


def peak_rss_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024 ** 2
    return float("nan")


def rss_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024 ** 2
    return float("nan")


# ---------------------------------------------------------------------------------------------- arm settings

@dataclass
class Arm:
    """What acts during one generation (all optional). bias: {"ids", "value", "mode": online|whole|prefix};
    steer: {"layer", "vector", "mode": online|prefix}; comp: [{"layer", "tensor", "op": add|set, "values": {pos: vec}}]
    (M2, prefix positions only); edit: (edit_id, dose) applied by the caller through the edit layer; hybrid: bool."""
    bias: dict | None = None
    steer: dict | None = None
    comp: list = field(default_factory=list)
    hybrid: bool = False


@dataclass
class GenResult:
    gen_ids: list
    text: str
    trigger_step: int | None
    n_prompt: int
    wall_s: float
    stop: str
    step_resid: dict | None = None     # absolute position -> [n_layers, hidden] (only with capture_resid)


# ---------------------------------------------------------------------------------------------- engine

class Engine51:
    def __init__(self, snapshot: Path = SNAPSHOT, threads: int = 16, edit_dir: Path | None = None,
                 verify_disk: bool = True, device: str = "cpu"):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.device = device
        if device != "cpu":
            # the GPU route: strict float32 (no TF32), every new tensor on the device
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_default_device(device)
        self.torch, self.snapshot, self.threads = torch, Path(snapshot), threads
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(str(snapshot))
        self.net = AutoModelForCausalLM.from_pretrained(str(snapshot), dtype=torch.float32).to(device).eval()
        self.load_s = time.perf_counter() - t0
        self.layers = self.net.model.layers
        self.n_layers, self.hidden = len(self.layers), self.net.config.hidden_size
        self.stop_ids = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id}
        self.edits = ED.EditLayer(self.net, snapshot if verify_disk else None,
                                  edit_dir or (ROOT / "runs/exp51/edits"))
        self.name = (f"hf-{transformers.__version__}-torch-{torch.__version__}-cpu-float32-t{threads}" if device == "cpu"
                     else f"hf-{transformers.__version__}-torch-{torch.__version__}-{device}-float32-notf32")
        self._offset = 0
        self._step_no = None
        self._hooks: list = []

    # -------------------------------------------------------------- text
    def chat_ids(self, prompt: str) -> list[int]:
        from memctl.analysis.jacobian import wrap_chat
        return list(self.tok(wrap_chat(self.tok, prompt), add_special_tokens=False).input_ids)

    def encode(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False).input_ids)

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)

    def fingerprint(self) -> dict:
        return {"engine": self.name, "snapshot": str(self.snapshot), "revision": self.snapshot.name,
                "n_layers": self.n_layers, "hidden": self.hidden}

    # -------------------------------------------------------------- hooks
    def _module(self, layer: int, tensor: str):
        path = COMPONENT_MODULE[tensor]
        return self.layers[layer] if not path else self.layers[layer].get_submodule(path)

    def _install(self, arm: Arm, state: dict, captures: dict | None = None):
        """Forward hooks for steering (decoder-layer output) and component ops / captures (module output)."""
        torch = self.torch
        handles = []
        if arm.steer is not None:
            vec = torch.tensor(np.asarray(arm.steer["vector"], np.float32))
            layer = int(arm.steer.get("layer", STEER_LAYER))

            def steer_fn(module, inputs, output):
                if not state.get("steer_on"):
                    return None
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                h[0, -1] += vec
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
            handles.append(self.layers[layer].register_forward_hook(steer_fn))
        ops: dict = {}
        for c in arm.comp:
            ops.setdefault((int(c["layer"]), c["tensor"]), []).append(c)
        for key in set(ops) | set((captures or {}).get("comp_keys", [])):
            layer, tensor = key

            def fn(module, inputs, output, key=key):
                tup = isinstance(output, tuple)
                h = output[0] if tup else output
                T = h.shape[1]
                lo = self._offset
                out = None
                for c in ops.get(key, []):
                    for p, v in c["values"].items():
                        p = int(p)
                        if lo <= p < lo + T:
                            if out is None:
                                out = h.clone()
                            v = torch.tensor(np.asarray(v, np.float32))
                            if c["op"] == "add":
                                out[0, p - lo] += v
                            else:
                                out[0, p - lo] = v
                if captures is not None and key in captures.get("comp_keys", []):
                    src = h if out is None else out          # captures see the output after any op
                    for p in captures["positions"]:
                        if lo <= p < lo + T:
                            captures["comp"][(key, p)] = src[0, p - lo].detach().float().cpu().numpy().copy()
                if out is None:
                    return None
                return (out,) + tuple(output[1:]) if tup else out
            handles.append(self._module(layer, tensor).register_forward_hook(fn))
        return handles

    def _remove(self, handles):
        for h in handles:
            h.remove()

    # -------------------------------------------------------------- forward
    def _prefill(self, ids: list[int]):
        torch = self.torch
        self._offset = 0
        with torch.no_grad():
            out = self.net(input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=1)
        return out.past_key_values

    def _step(self, token: int, cache, pos: int):
        torch = self.torch
        self._offset = pos
        with torch.no_grad():
            out = self.net(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1]

    def generate(self, ids: list[int], arm: Arm | None = None, mode: str = "total", max_new: int | None = None,
                 edit: tuple | None = None, capture_resid: bool = False, text_before: str = "") -> GenResult:
        """mode 'total' (prompt ids; triggers online), 'prefix' (stored ids through P_ans; triggers on from step 0) or
        'resume' (the B1 / B4 shortcut, build review 8c: B0's ids through its FIRST marker, triggers on from step 0,
        the total-effect stop rule applied to `text_before` + the new text, max_new = 160 − tokens already generated).
        `edit` = (edit_id, dose) is applied for the call and reverted after (None: whatever is applied stays)."""
        arm = arm or Arm()
        torch = self.torch
        max_new = max_new or (MAX_NEW_TOTAL if mode == "total" else MAX_NEW_RELEASE)
        if mode == "resume" and max_new <= 0:
            raise ValueError("resume needs max_new > 0")
        t0 = time.perf_counter()
        if edit is not None:
            self.edits.apply(*edit)
        state = {"steer_on": False}
        handles = self._install(arm, state)
        step_resid: dict = {}
        if capture_resid:
            cur: dict = {}

            def mk(l):
                def fn(module, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if h.shape[1] == 1:
                        cur[l] = h[0, -1].detach().float().cpu().numpy().copy()
                return fn
            handles += [layer.register_forward_hook(mk(l)) for l, layer in enumerate(self.layers)]
        try:
            if arm.hybrid:
                saved = self.edits.state()
                self.edits.revert()
                cache = self._prefill(ids[:-1])
                for name, s in saved.items():
                    self.edits.apply(s["edit_id"], s["dose"], only={s["i"]})
            else:
                cache = self._prefill(ids[:-1])
            gen: list[int] = []
            trigger = 0 if mode in ("prefix", "resume") else None
            pos, tok = len(ids) - 1, ids[-1]
            stop = "max_new"
            bias = arm.bias
            bias_ids = torch.tensor(sorted(bias["ids"]), dtype=torch.long) if bias else None
            while True:
                on = trigger is not None and len(gen) >= trigger
                state["steer_on"] = arm.steer is not None and (on if arm.steer.get("mode") != "whole" else True)
                logits = self._step(tok, cache, pos)
                if capture_resid:
                    step_resid[pos] = np.stack([cur[l] for l in range(self.n_layers)])
                if bias is not None and (bias["mode"] == "whole" or on):
                    logits = logits.clone()
                    logits[bias_ids] += float(bias["value"])
                nxt = int(logits.argmax())
                if nxt in self.stop_ids:
                    stop = "eos"
                    break
                gen.append(nxt)
                text = self.decode(gen)
                if trigger is None and MARKER in text:
                    trigger = len(gen)          # the marker's last token is the input of step len(gen)
                if mode == "total" and _answer_line_done(text):
                    stop = "answer_line"
                    break
                if mode == "resume" and _answer_line_done(text_before + text):
                    stop = "answer_line"
                    break
                if mode == "prefix" and "\n" in text and text.strip():
                    stop = "newline"
                    break
                if len(gen) >= max_new:
                    break
                pos, tok = pos + 1, nxt
        finally:
            self._remove(handles)
            if edit is not None:
                self.edits.revert()
        return GenResult(gen, self.decode(gen), trigger, len(ids), time.perf_counter() - t0, stop,
                         step_resid if capture_resid else None)

    def read(self, ids: list[int], positions: list[int], layers: list[int] = (), comps: list[tuple] = (),
             logit_positions: list[int] = (), logit_ids: list[int] | None = None, arm: Arm | None = None,
             edit: tuple | None = None) -> dict:
        """One teacher-forced pass: resid[p][l] (decoder-layer l output at p), comp[(layer, tensor)][p], logits[p]."""
        torch = self.torch
        arm = arm or Arm()
        captures = {"positions": list(positions), "comp_keys": [tuple(c) for c in comps], "comp": {}}
        saved = {}

        def mk(l):
            def fn(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                saved[l] = h[0, list(positions)].detach().float().cpu().numpy().copy()
            return fn
        t0 = time.perf_counter()
        if edit is not None:
            self.edits.apply(*edit)
        handles = self._install(arm, {"steer_on": False}, captures)
        handles += [self.layers[l].register_forward_hook(mk(l)) for l in layers]
        try:
            self._offset = 0
            keep = torch.tensor(sorted(set(logit_positions)), dtype=torch.long) if logit_positions else 1
            with torch.no_grad():
                out = self.net(input_ids=torch.tensor([ids]), use_cache=False, logits_to_keep=keep)
            lg = None
            if logit_positions:
                order = sorted(set(logit_positions))
                full = out.logits[0].float()
                rows = [full[order.index(p)] for p in logit_positions]
                lg = torch.stack(rows)
                lg = (lg if logit_ids is None else lg[:, list(logit_ids)]).cpu().numpy().copy()
        finally:
            self._remove(handles)
            if edit is not None:
                self.edits.revert()
        resid = {l: saved[l] for l in layers}
        comp = {}
        for (key, p), v in captures["comp"].items():
            comp.setdefault(key, {})[p] = v
        return {"resid": resid, "comp": comp, "logits": lg, "wall_s": time.perf_counter() - t0, "n_tokens": len(ids)}


    def generate_batched(self, batch: list[list[int]], arm: Arm | None = None, mode: str = "total",
                         max_new: int | list[int] | None = None, edit: tuple | None = None,
                         text_before: list[str] | None = None) -> list[GenResult]:
        """The batched route: one fixed batch (the caller fixes the layout: rows sorted by case id, consecutive
        chunks), left padding with an attention mask and explicit position ids, greedy, with the same per-sequence
        semantics as `generate`: B1's bias and B4's add switch on per sequence at its own first-marker forward
        (total) or from step 0 (prefix / resume), stop rules per sequence, finished sequences masked. Component
        hooks (M2) are not supported here (they stay at batch 1)."""
        arm = arm or Arm()
        if arm.comp or arm.hybrid:
            raise ValueError("component hooks and the hybrid estimand run at batch 1")
        torch = self.torch
        B = len(batch)
        cap = max_new if isinstance(max_new, list) else [max_new or (MAX_NEW_TOTAL if mode == "total"
                                                                     else MAX_NEW_RELEASE)] * B
        tb = text_before or [""] * B
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        L = max(len(x) for x in batch)
        padn = [L - len(x) for x in batch]
        t0 = time.perf_counter()
        if edit is not None:
            self.edits.apply(*edit)
        on = torch.zeros(B, dtype=torch.bool)
        handles = []
        if arm.steer is not None:
            vec = torch.tensor(np.asarray(arm.steer["vector"], np.float32))
            whole = arm.steer.get("mode") == "whole"

            def steer_fn(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.shape[1] != 1:
                    return None                                   # never during the prefill
                m = torch.ones(B, dtype=torch.bool) if whole else on
                if not bool(m.any()):
                    return None
                h = h.clone()
                h[m, -1] += vec
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
            handles.append(self.layers[int(arm.steer.get("layer", STEER_LAYER))].register_forward_hook(steer_fn))
        try:
            ids = torch.tensor([[pad] * p + x for p, x in zip(padn, batch)])
            att = torch.tensor([[0] * p + [1] * len(x) for p, x in zip(padn, batch)])
            pos = (att.cumsum(1) - 1).clamp(min=0)
            with torch.no_grad():
                out = self.net(input_ids=ids[:, :-1], attention_mask=att[:, :-1], position_ids=pos[:, :-1],
                               use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            cur, cur_pos = ids[:, -1:], pos[:, -1:]
            gens = [[] for _ in range(B)]
            trig = [0 if mode in ("prefix", "resume") else None for _ in range(B)]
            done, stops = [False] * B, ["max_new"] * B
            bias = arm.bias
            bias_ids = torch.tensor(sorted(bias["ids"]), dtype=torch.long) if bias else None
            while not all(done):
                for b in range(B):
                    on[b] = trig[b] is not None and len(gens[b]) >= trig[b] and not done[b]
                with torch.no_grad():
                    o = self.net(input_ids=cur, attention_mask=att, position_ids=cur_pos, past_key_values=cache,
                                 use_cache=True, logits_to_keep=1)
                logits = o.logits[:, -1].clone()
                if bias is not None:
                    m = torch.ones(B, dtype=torch.bool) if bias["mode"] == "whole" else on
                    if bool(m.any()):
                        logits[m.nonzero().flatten()[:, None], bias_ids[None, :]] += float(bias["value"])
                nxt = logits.argmax(-1).tolist()
                for b in range(B):
                    if done[b]:
                        continue
                    t = nxt[b]
                    if t in self.stop_ids:
                        done[b], stops[b] = True, "eos"
                        continue
                    gens[b].append(t)
                    text = self.decode(gens[b])
                    if trig[b] is None and MARKER in text:
                        trig[b] = len(gens[b])
                    if mode == "total" and _answer_line_done(text):
                        done[b], stops[b] = True, "answer_line"
                    elif mode == "resume" and _answer_line_done(tb[b] + text):
                        done[b], stops[b] = True, "answer_line"
                    elif mode == "prefix" and "\n" in text and text.strip():
                        done[b], stops[b] = True, "newline"
                    elif len(gens[b]) >= cap[b]:
                        done[b] = True
                feed = [nxt[b] if not done[b] else pad for b in range(B)]
                cur = torch.tensor(feed)[:, None]
                att = torch.cat([att, torch.tensor([[0 if done[b] else 1] for b in range(B)])], 1)
                cur_pos = cur_pos + 1
            wall = time.perf_counter() - t0
        finally:
            self._remove(handles)
            if edit is not None:
                self.edits.revert()
        return [GenResult(g, self.decode(g), trig[b] if mode == "total" else 0, len(batch[b]), wall / B, stops[b])
                for b, g in enumerate(gens)]

    def generate_batch(self, batch: list[list[int]], max_new: int = MAX_NEW_TOTAL) -> list[list[int]]:
        """(d) measurement only: left-padded greedy batch generation (HF generate), every sequence run to EOS or
        max_new, then each cut post hoc where the batch-1 total-effect stop rule would have stopped (greedy decoding
        is causal, so the cut equals stopping early)."""
        torch = self.torch
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        L = max(len(x) for x in batch)
        inp = torch.tensor([[pad] * (L - len(x)) + x for x in batch])
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in batch])
        with torch.no_grad():
            out = self.net.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                                    eos_token_id=list(self.stop_ids), pad_token_id=pad)
        res = []
        for row in out[:, L:].tolist():
            gen = []
            for t in row:
                if t in self.stop_ids:
                    break
                gen.append(t)
                if _answer_line_done(self.decode(gen)):
                    break
            res.append(gen)
        return res


def _answer_line_done(text: str) -> bool:
    i = text.rfind(MARKER)
    if i < 0:
        return False
    lines = text[i + len(MARKER):].split("\n")
    return any(l.strip() for l in lines[:-1])


# ---------------------------------------------------------------------------------------------- dummy model

def make_dummy(out: Path = DUMMY_DIR, snapshot: Path = SNAPSHOT, layers: int = 4, hidden: int = 64) -> Path:
    """A tiny random Qwen2 with the real tokenizer and vocabulary (bf16 on disk, as the real snapshot): the dry-run
    and test model. Its outputs mean nothing."""
    import shutil
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    out = Path(out)
    if (out / "model.safetensors").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(snapshot) / "config.json").read_text())
    cfg.update({"num_hidden_layers": layers, "hidden_size": hidden, "intermediate_size": 2 * hidden,
                "num_attention_heads": 4, "num_key_value_heads": 2, "max_window_layers": layers})
    torch.manual_seed(5107)
    net = Qwen2ForCausalLM(Qwen2Config(**{k: v for k, v in cfg.items() if k not in ("architectures",
                                                                                     "transformers_version")}))
    net.to(torch.bfloat16).save_pretrained(str(out), safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
        if (Path(snapshot) / f).exists():
            shutil.copy(Path(snapshot) / f, out / f)
    return out


def load(backend: str = "hf", threads: int = 16, edit_dir: Path | None = None, device: str = "cpu",
         snapshot: Path | None = None) -> Engine51:
    if backend == "dummy":
        return Engine51(make_dummy(), threads=threads, edit_dir=edit_dir, device=device)
    if backend == "dummy36":
        return Engine51(make_dummy(DUMMY36_DIR, layers=36), threads=threads, edit_dir=edit_dir, device=device)
    return Engine51(Path(snapshot or SNAPSHOT), threads=threads, edit_dir=edit_dir, device=device)
