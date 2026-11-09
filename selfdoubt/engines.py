"""Generation engines for Stage 0. Each takes a list of Requests and returns one result dict per request:
{token_ids, text, n_prompt_tokens, n_gen_tokens, finish_reason ("stop"|"length"), hit_max, max_tokens}.

text is tokenizer.decode(generated ids) with special tokens kept, as in the authors' inference_loop.
Prompts are tokenised once here with add_special_tokens=False; Qwen3 has no BOS token, so this equals the
authors' raw-text path, and there is never a double BOS (asserted).
"""
from __future__ import annotations

import os
import subprocess
import threading
import time

from selfdoubt.common import MODELS, SAMPLING


def encode_prompt(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    if bos is not None and len(ids) > 1:
        assert not (ids[0] == bos and ids[1] == bos), "double BOS"
    return ids


def load_tokenizer(model: str, path: str | None = None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path or model, revision=None if path else MODELS[model])


class GpuMemPoller:
    """Max nvidia-smi memory.used (MiB) while active; silent no-op without nvidia-smi."""

    def __init__(self, interval: float = 1.0):
        self.interval, self.max_mib, self._stop, self._t = interval, None, threading.Event(), None

    def _q(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            return max(int(x) for x in out.split())
        except Exception:
            return None

    def _run(self):
        while not self._stop.is_set():
            v = self._q()
            if v is not None:
                self.max_mib = v if self.max_mib is None else max(self.max_mib, v)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=15)


def peak_rss_mib() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class HFEngine:
    """CPU (or single-GPU) transformers engine for the local smoke test. One request at a time with
    torch.manual_seed(request seed), so each sample is reproducible and independent of batching."""
    name = "hf"

    def __init__(self, model: str, path: str | None = None, device: str = "cpu", dtype: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.model_name = model
        self.tok = load_tokenizer(model, path)
        self.model = AutoModelForCausalLM.from_pretrained(path or model, revision=None if path else MODELS[model],
                                                          dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device
        gc = self.model.generation_config
        self.eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
        self.pad = gc.pad_token_id if gc.pad_token_id is not None else self.eos[0]

    def generate(self, reqs, max_tokens: int, max_model_len: int = 40960):
        torch = self.torch
        out = []
        for r in reqs:
            ids = encode_prompt(self.tok, r.prompt)
            mt = max(1, min(max_tokens, max_model_len - len(ids)))
            torch.manual_seed(r.seed)
            x = torch.tensor([ids], device=self.device)
            with torch.no_grad():
                g = self.model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, max_new_tokens=mt,
                                        eos_token_id=self.eos, pad_token_id=self.pad, **SAMPLING)
            new = g[0, len(ids):].tolist()
            stopped = len(new) > 0 and new[-1] in self.eos
            hit = (not stopped) and len(new) >= mt
            out.append({"token_ids": new, "text": self.tok.decode(new), "n_prompt_tokens": len(ids),
                        "n_gen_tokens": len(new), "finish_reason": "length" if hit else "stop", "hit_max": hit,
                        "max_tokens": mt})
        return out

    def memory(self):
        return {"peak_rss_mib": round(peak_rss_mib(), 1)}


class VLLMEngine:
    """vLLM engine for the pod: one LLM.generate per chunk, a SamplingParams per request (own seed)."""
    name = "vllm"

    def __init__(self, model: str, path: str | None = None, max_model_len: int = 40960,
                 gpu_memory_utilization: float = 0.90, dtype: str = "bfloat16"):
        # In-process engine core so torch.cuda.max_memory_allocated() sees the model and KV cache.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM, SamplingParams
        self.SP = SamplingParams
        self.model_name = model
        self.tok = load_tokenizer(model, path)
        self.max_model_len = max_model_len
        self.llm = LLM(model=path or model, revision=None if path else MODELS[model], tokenizer=path or model,
                       dtype=dtype, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       enable_prefix_caching=True, seed=0)

    def generate(self, reqs, max_tokens: int, max_model_len: int | None = None):
        mml = max_model_len or self.max_model_len
        prompts, params, mts = [], [], []
        for r in reqs:
            ids = encode_prompt(self.tok, r.prompt)
            mt = max(1, min(max_tokens, mml - len(ids)))
            prompts.append({"prompt_token_ids": ids})
            params.append(self.SP(n=1, max_tokens=mt, seed=r.seed, skip_special_tokens=False, **SAMPLING))
            mts.append(mt)
        res = self.llm.generate(prompts, params, use_tqdm=False)
        out = []
        for o, mt in zip(res, mts):
            c = o.outputs[0]
            new = list(c.token_ids)
            hit = c.finish_reason == "length"
            out.append({"token_ids": new, "text": self.tok.decode(new), "n_prompt_tokens": len(o.prompt_token_ids),
                        "n_gen_tokens": len(new), "finish_reason": c.finish_reason, "hit_max": hit, "max_tokens": mt})
        return out

    def memory(self):
        import torch
        return {"torch_max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                "torch_max_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)}


class SimulatedInterrupt(Exception):
    pass


class FakeEngine:
    """Deterministic stand-in for tests: FULL answers 'think ... \\boxed{7}'; continuations echo a fixed answer."""
    name = "fake"

    def __init__(self, model: str, tokenizer, answers: dict | None = None, fail_after: int | None = None):
        self.model_name, self.tok, self.answers, self.calls, self.fail_after = model, tokenizer, answers or {}, 0, \
            fail_after

    def generate(self, reqs, max_tokens: int, max_model_len: int = 40960):
        out = []
        for r in reqs:
            if self.fail_after is not None and self.calls >= self.fail_after:
                raise SimulatedInterrupt("simulated interrupt")
            self.calls += 1
            ans = self.answers.get((r.arm, r.cut), "7")
            text = ("<think>\nLet me compute step by step. First 3 plus 4 is 7. Then check again.\n</think>\n\n"
                    f"The answer is \\boxed{{{ans}}}.<|im_end|>") if r.arm == "FULL" else \
                f" continuing.\n</think>\n\n\\boxed{{{ans}}}<|im_end|>"
            ids = self.tok.encode(text)
            out.append({"token_ids": ids, "text": text, "n_prompt_tokens": len(encode_prompt(self.tok, r.prompt)),
                        "n_gen_tokens": len(ids), "finish_reason": "stop", "hit_max": False, "max_tokens": max_tokens})
        return out

    def memory(self):
        return {}
