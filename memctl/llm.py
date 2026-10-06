"""Language-model backends behind one small interface, plus a disk cache.

Backends:
- `stub`     no model; answers with the best-matching memory line. For tests and smoke runs.
- `scripted` a Python function as the model. For tests.
- `hf`       a local Hugging Face transformers model (GPU, Apple MPS or CPU).
- `openai`   any OpenAI-compatible chat endpoint you run yourself (vLLM, Ollama, llama.cpp).

Every backend counts its calls and tokens, so compute can be charged to whoever
made the call (task model, controller, compressor).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Callable, Protocol

from memctl.embed import content_words
from memctl.memory.items import count_tokens


class LLM(Protocol):
    name: str

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str: ...


class Usage:
    """Running totals for one model."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0

    def add(self, prompt: str, output: str, seconds: float) -> None:
        self.calls += 1
        self.input_tokens += count_tokens(prompt)
        self.output_tokens += count_tokens(output)
        self.seconds += seconds

    def snapshot(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "seconds": self.seconds}


class StubLLM:
    """No model. Returns the line of the prompt's memory section that shares most words
    with the final question; says so when nothing matches. Not a result, only a pipeline check."""

    name = "stub"

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        head, _, tail = prompt.rpartition("\n## ")
        wanted = set(content_words(tail))
        best, best_score = "", 0
        for line in head.splitlines():
            if not line.startswith("["):
                continue
            score = len(wanted & set(content_words(line)))
            if score > best_score:
                best, best_score = line, score
        return best.split("] ", 1)[-1] if best else "unknown"


class ScriptedLLM:
    """A function from prompt to reply, standing in for a model in tests."""

    def __init__(self, function: Callable[[str], str], name: str = "scripted") -> None:
        self.function = function
        self.name = name

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        return self.function(prompt)


class HuggingFaceLLM:
    """A local transformers model with greedy decoding."""

    def __init__(
        self, model_name: str, device: str | None = None, load_in_4bit: bool = False, revision: str | None = None
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_name if revision is None else f"{model_name}@{revision}"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        options: dict = {"dtype": torch.float32 if device == "cpu" else torch.float16, "revision": revision}
        if load_in_4bit:  # weights in 4-bit NF4; computation in float16. Changes the generated text.
            from transformers import BitsAndBytesConfig

            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            options["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **options)
        if not load_in_4bit:
            self.model = self.model.to(device)
        self.model.eval()
        self.device = self.model.get_input_embeddings().weight.device

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:  # Qwen models think before answering unless told not to
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                input_ids=ids, attention_mask=self.torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False
            )
        return self.tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()


class OpenAICompatibleLLM:
    """Talks to a chat-completions endpoint: vLLM, Ollama (`/v1`), llama.cpp server."""

    def __init__(self, model_name: str, base_url: str, api_key: str | None = None, timeout_s: float = 300.0) -> None:
        self.name = model_name
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout_s = timeout_s

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        body = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "seed": 0,
        }
        # Some proxies (RunPod's, behind Cloudflare) reject urllib's default User-Agent with 403.
        headers = {"Content-Type": "application/json", "User-Agent": "memctl"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.load(response)["choices"][0]["message"]["content"].strip()


class TrackedLLM:
    """Wraps a backend: counts usage and, with `cache_dir`, saves every generation to disk."""

    def __init__(self, backend: LLM, cache_dir: str | None = None, settings: dict | None = None) -> None:
        self.backend = backend
        self.name = backend.name
        self.settings = settings or {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.usage = Usage()
        self.cache_hits = 0

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        started = time.perf_counter()
        path = None
        if self.cache_dir is not None:
            material = {"prompt": prompt, "max_new_tokens": max_new_tokens, **self.settings}
            key = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
            path = self.cache_dir / key[:2] / f"{key}.json"
            if path.exists():
                self.cache_hits += 1
                output = json.loads(path.read_text())["output"]
                self.usage.add(prompt, output, time.perf_counter() - started)
                return output
        output = self.backend.generate(prompt, max_new_tokens)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"prompt": prompt, "max_new_tokens": max_new_tokens, **self.settings, "output": output}))
        self.usage.add(prompt, output, time.perf_counter() - started)
        return output


class _Lazy:
    """Loads the real model on first use, so building an experiment stays cheap."""

    def __init__(self, name: str, load: Callable[[], LLM]) -> None:
        self.name = name
        self._load = load
        self._backend: LLM | None = None

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        if self._backend is None:
            self._backend = self._load()
        return self._backend.generate(prompt, max_new_tokens)


def build_llm(config: dict) -> TrackedLLM:
    """`config`: {backend, name, revision?, base_url?, api_key_env?, load_in_4bit?, cache_dir?}."""
    backend_name = config.get("backend", "stub")
    name = config.get("name", backend_name)
    if backend_name == "stub":
        backend: LLM = StubLLM()
    elif backend_name == "scripted":
        backend = ScriptedLLM(config["function"], name)
    elif backend_name == "hf":
        backend = _Lazy(
            name,
            lambda: HuggingFaceLLM(
                name, config.get("device"), bool(config.get("load_in_4bit", False)), config.get("revision")
            ),
        )
    elif backend_name == "openai":
        import os

        key = os.environ.get(config["api_key_env"]) if config.get("api_key_env") else None
        backend = OpenAICompatibleLLM(name, config["base_url"], key, float(config.get("timeout_s", 300.0)))
    else:
        raise KeyError(f"unknown model backend '{backend_name}' (known: stub, scripted, hf, openai)")
    # Everything that changes the generated text is part of the cache key.
    settings = {
        "backend": backend_name, "name": name, "revision": config.get("revision"),
        "load_in_4bit": bool(config.get("load_in_4bit", False)), "temperature": 0.0,
    }
    return TrackedLLM(backend, config.get("cache_dir"), settings)
