"""Language-model backends behind one small interface, plus the disk cache.

Backends: `stub` (no model, for tests), `hf` (Hugging Face transformers, runs
on GPU, Apple MPS or CPU) and `vllm` (a vLLM server you started yourself).
"""

from __future__ import annotations

import copy
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Protocol

from memctl.embed import content_words


class LLM(Protocol):
    name: str

    def generate(self, prompt: str, max_new_tokens: int, shared_prefix: str = "") -> str:
        """Return the model's reply. `shared_prefix` is the start of `prompt` that many
        prompts have in common; backends may compute it once and reuse it."""
        ...


class StubLLM:
    """A model-free stand-in: answers with the memory line that best matches the question.

    It asks for recall(id) when an archive index line matches better than anything it can read.
    """

    name = "stub"

    def generate(self, prompt: str, max_new_tokens: int, shared_prefix: str = "") -> str:
        sections = {}
        for block in prompt.split("## ")[1:]:
            title, _, body = block.partition("\n")
            sections[title.split(" ")[0]] = [line for line in body.splitlines() if line[:1] in ("(", "[")]
        wanted = set(content_words(prompt.rpartition("## Question")[2]))

        def best(lines: list[str]) -> tuple[int, str]:
            return max(((len(wanted & set(content_words(line))), line) for line in lines), default=(0, ""))

        readable = sections.get("Memory", []) + sections.get("Retrieved", []) + sections.get("Recalled", [])
        memory_score, memory_line = best(readable)
        index_score, index_line = best(sections.get("Archive", []))
        if index_score > memory_score:
            return f"recall({index_line[1:index_line.index(']')]})"
        if memory_score == 0:
            return "Not mentioned in the conversation"
        return memory_line.split(") ", 1)[-1]


class HuggingFaceLLM:
    """A local transformers model with greedy decoding.

    Speed-up: the shared prefix of the prompt (the memory, which is the same for
    every question asked at the same moment) is run through the model once and
    its internal state is reused. Outputs are the same as without the reuse.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        precision = torch.float32 if device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=precision).to(device).eval()
        self.prefix_text = ""
        self.prefix_ids = None
        self.prefix_state = None

    def chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:  # Qwen models think before answering unless told not to
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _ids(self, text: str):
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    def _remember_prefix(self, prefix_text: str) -> None:
        """Run the shared prefix through the model once and keep the result."""
        self.prefix_state = None  # free the old one first
        self.prefix_ids = self._ids(prefix_text)
        with self.torch.no_grad():
            self.prefix_state = self.model(input_ids=self.prefix_ids, use_cache=True).past_key_values
        self.prefix_text = prefix_text

    def generate(self, prompt: str, max_new_tokens: int, shared_prefix: str = "") -> str:
        text = self.chat_text(prompt)
        options = {"max_new_tokens": max_new_tokens, "do_sample": False}
        cut = text.find(shared_prefix) + len(shared_prefix) if shared_prefix else 0
        if shared_prefix and text.find(shared_prefix) >= 0 and len(shared_prefix) > 2000:
            if text[:cut] != self.prefix_text:
                self._remember_prefix(text[:cut])
            ids = self.torch.cat([self.prefix_ids, self._ids(text[cut:])], dim=1)
            options["past_key_values"] = copy.deepcopy(self.prefix_state)
        else:
            ids = self._ids(text)
        with self.torch.no_grad():
            output = self.model.generate(input_ids=ids, attention_mask=self.torch.ones_like(ids), **options)
        return self.tokenizer.decode(output[0][ids.shape[1] :], skip_special_tokens=True).strip()


class VllmLLM:
    """Talks to a running vLLM server (OpenAI-compatible API). Not tested on this machine (no GPU)."""

    def __init__(self, model_name: str, base_url: str) -> None:
        self.name = model_name
        self.url = base_url.rstrip("/") + "/v1/chat/completions"

    def generate(self, prompt: str, max_new_tokens: int, shared_prefix: str = "") -> str:
        body = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)["choices"][0]["message"]["content"].strip()


class CachedLLM:
    """Wraps a backend and saves every generation to disk, keyed by prompt + settings."""

    def __init__(self, backend: LLM, cache_dir: str, settings: dict) -> None:
        self.backend = backend
        self.name = backend.name
        self.settings = settings
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def generate(self, prompt: str, max_new_tokens: int, shared_prefix: str = "") -> str:
        key_material = {"prompt": prompt, "max_new_tokens": max_new_tokens, **self.settings}
        key = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())["output"]
        self.misses += 1
        output = self.backend.generate(prompt, max_new_tokens, shared_prefix)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({**key_material, "output": output}, indent=1))
        return output


class LazyLLM:
    """Loads the real model only when the disk cache misses, so cached reruns start instantly."""

    def __init__(self, name: str, load) -> None:
        self.name = name
        self.load = load
        self.backend: LLM | None = None

    def generate(self, prompt: str, max_new_tokens: int, shared_prefix: str = "") -> str:
        if self.backend is None:
            self.backend = self.load()
        return self.backend.generate(prompt, max_new_tokens, shared_prefix)


def build_llm(model_config: dict) -> CachedLLM:
    """Create the model named in the config, wrapped in the disk cache."""
    backend_name = model_config.get("backend", "stub")
    name = model_config.get("name", "stub")
    if backend_name == "stub":
        backend: LLM = StubLLM()
    elif backend_name == "hf":
        backend = LazyLLM(name, lambda: HuggingFaceLLM(name, model_config.get("device")))
    elif backend_name == "vllm":
        backend = VllmLLM(name, model_config["base_url"])
    else:
        raise KeyError(f"unknown model backend '{backend_name}' (known: stub, hf, vllm)")
    settings = {"backend": backend_name, "name": name, "temperature": 0.0}
    return CachedLLM(backend, model_config.get("cache_dir", "cache/generations"), settings)
