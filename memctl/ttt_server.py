"""Query-only test-time training (qTTT) in front of a frozen reader, served as an OpenAI-compatible endpoint (§31).

    python -m memctl.ttt_server serve --model Qwen/Qwen2.5-3B-Instruct --served-name qwen2.5-3b-instruct-hf
    python -m memctl.ttt_server bench --config-from Qwen/Qwen2.5-3B-Instruct --layers 2 --prompt-tokens 1400

Method: Bansal et al., "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs"
(arXiv 2512.13898, Algorithm 1 and Appendix C). For each request, separately:
  1. one prefill of the prompt with the original weights; its keys and values are kept and never recomputed;
  2. N steps of AdamW on the query projections W_Q of every attention layer (nothing else is trainable), each on
     the next-token loss of one random k-token span of the same prompt, conditioned on the frozen cached prefix;
  3. greedy decoding of the answer from the frozen cache with the adapted W_Q;
  4. W_Q is restored bit-for-bit and the optimizer is dropped. Nothing is learned across questions.
Paper defaults: N = 32, k = 128, AdamW with weight decay 0.01, gradient clipping at 1.0. The paper picks the
learning rate per dataset on held-out data; §31 fixes 3e-6 before any run (see EXPERIMENTS.md §31).

One loaded model serves two names: `NAME` (no adaptation: the paired baseline through the same code path) and
`NAME-qttt` (adaptation on). memctl's `openai` backend talks to either, and memctl.sweep caches and replays the
answers exactly as for vLLM.

Self-contained on purpose (torch, transformers and the standard library only): the pod fetches this one file
at a pinned commit, with no memctl install.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import resource
import sys
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

QTTT_SUFFIX = "-qttt"


@dataclass(frozen=True)
class TTTSettings:
    steps: int = 32  # N_TTT, paper default
    span: int = 128  # k, paper default
    lr: float = 3e-6  # fixed in §31 before any run: the paper's recommended range is 1e-6 to 1e-5
    weight_decay: float = 0.01  # paper
    clip: float = 1.0  # paper
    repetition_penalty: float = 1.05  # the Qwen2.5 generation_config value that vLLM applies by default


def _torch():
    import torch

    return torch


class FP32QueryProjection:
    """Replaces an attention layer's q_proj: the weight is held and applied in float32.

    Needed because at lr 3e-6 one AdamW step changes a weight by about 3e-6, while a bfloat16 weight of size
    0.02 has a rounding step near 1e-4: in bfloat16 the update would round away and nothing would adapt."""

    @staticmethod
    def wrap(linear):
        torch = _torch()

        class _Q(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(linear.weight.detach().float().clone())
                bias = None if linear.bias is None else linear.bias.detach().float().clone()
                # The paper adapts W_Q only; the bias stays frozen.
                self.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad=False)
                self.register_buffer("original", self.weight.detach().clone(), persistent=False)

            def forward(self, x):
                return torch.nn.functional.linear(x.float(), self.weight, self.bias).to(x.dtype)

        return _Q()


class QueryTTT:
    """The per-query loop on one loaded causal LM (any HF model whose attention layers have `self_attn.q_proj`)."""

    def __init__(self, model, tokenizer=None, settings: TTTSettings | None = None, eos_ids=None, pad_id=None) -> None:
        torch = _torch()
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings or TTTSettings()
        self.device = model.get_input_embeddings().weight.device
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.queries = []
        for layer in model.model.layers:
            wrapped = FP32QueryProjection.wrap(layer.self_attn.q_proj).to(self.device)
            layer.self_attn.q_proj = wrapped
            self.queries.append(wrapped)
        model.eval()  # no dropout anywhere, also during adaptation
        self.eos_ids = eos_ids
        self.pad_id = pad_id

    def trainable(self) -> list:
        return [q.weight for q in self.queries]

    def restore(self) -> None:
        with self.torch.no_grad():
            for q in self.queries:
                q.weight.copy_(q.original)
                q.weight.grad = None

    def chat_ids(self, prompt: str) -> list[int]:
        """The prompt as a single user turn, through the model's chat template (as vLLM's chat endpoint does)."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _span_loss(self, ids, cache, start: int, length: int):
        """Next-token loss of ids[start:start+length] predicting ids[start+1:start+length+1], attending to the
        frozen cached prefix ids[:start] (keys and values from the original weights)."""
        torch = self.torch
        past = None
        if start > 0:
            past = copy.deepcopy(cache)
            past.crop(start - past.get_seq_length())  # a negative crop: accepted by transformers 4.4x and 5.x
        hidden = self.model.model(input_ids=ids[:, start:start + length], past_key_values=past, use_cache=True)
        logits = self.model.lm_head(hidden.last_hidden_state).float()[0]
        return torch.nn.functional.cross_entropy(logits, ids[0, start + 1:start + length + 1])

    def answer_ids(self, prompt_ids: list[int], max_new_tokens: int, steps: int, seed: int,
                   min_new_tokens: int = 0) -> tuple[list[int], dict]:
        torch = self.torch
        s = self.settings
        ids = torch.tensor([prompt_ids], device=self.device)
        total = ids.shape[1]
        started = time.perf_counter()
        with torch.no_grad():  # one prefill of all but the last token: the frozen K, V
            cache = self.model.model(input_ids=ids[:, :-1], use_cache=True).past_key_values
        prefilled = time.perf_counter()
        losses: list[float] = []
        try:
            if steps > 0 and total >= 3:
                length = min(s.span, total - 2)
                generator = torch.Generator().manual_seed(seed)
                optimizer = torch.optim.AdamW(self.trainable(), lr=s.lr, weight_decay=s.weight_decay)
                for _ in range(steps):
                    # start in [0, total-2-length]: the span's last target is at most the second-to-last prompt token
                    start = int(torch.randint(0, total - 1 - length, (1,), generator=generator))
                    loss = self._span_loss(ids, cache, start, length)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.trainable(), s.clip)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                del optimizer
            adapted = time.perf_counter()
            from transformers import GenerationConfig

            config = GenerationConfig(
                do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                repetition_penalty=s.repetition_penalty, eos_token_id=self.eos_ids, pad_token_id=self.pad_id,
            )
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=cache,
                    generation_config=config,
                )
        finally:
            self.restore()  # nothing carries over to the next question
        new = output[0, total:].tolist()
        done = time.perf_counter()
        info = {"prompt_tokens": total, "output_tokens": len(new), "steps": steps if total >= 3 else 0,
                "prefill_s": prefilled - started, "adapt_s": adapted - prefilled, "generate_s": done - adapted,
                "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None}
        return new, info

    def answer(self, prompt: str, max_new_tokens: int, steps: int) -> tuple[str, dict]:
        prompt_ids = self.chat_ids(prompt)
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:15], 16)  # same prompt, same spans, any order
        new, info = self.answer_ids(prompt_ids, max_new_tokens, steps, seed)
        return self.tokenizer.decode(new, skip_special_tokens=True).strip(), info


def _trim_layer_types(config) -> None:
    """With fewer layers than the checkpoint, the per-layer type list must shrink too, or the cache keeps empty
    layers (and a crop fails on them)."""
    if getattr(config, "layer_types", None):
        config.layer_types = config.layer_types[:config.num_hidden_layers]


def _from(build, source, dtype, **kwargs):
    """`dtype=` in transformers 5.x, `torch_dtype=` in 4.x (the vLLM 0.8.5 image)."""
    try:
        return build(source, dtype=dtype, **kwargs)
    except TypeError:
        return build(source, torch_dtype=dtype, **kwargs)


def load_engine(model_id: str, revision: str | None = None, dtype: str = "bfloat16", device: str = "cuda",
                settings: TTTSettings | None = None, random_init: dict | None = None) -> QueryTTT:
    """Loads model and tokenizer. `random_init`: config overrides for a randomly initialised model of the same
    architecture (CPU stubs: no weights are downloaded, only config and tokenizer)."""
    torch = _torch()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    torch_dtype = getattr(torch, dtype)
    if random_init is not None:
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        for key, value in random_init.items():
            setattr(config, key, value)
        _trim_layer_types(config)
        torch.manual_seed(0)
        model = _from(AutoModelForCausalLM.from_config, config, torch_dtype)
        eos = config.eos_token_id
    else:
        model = _from(AutoModelForCausalLM.from_pretrained, model_id, torch_dtype, revision=revision)
        eos = None
    model = model.to(device)
    try:
        generation = GenerationConfig.from_pretrained(model_id, revision=revision)
        eos = generation.eos_token_id if generation.eos_token_id is not None else eos
    except OSError:
        pass
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos[0] if isinstance(eos, list) else eos)
    return QueryTTT(model, tokenizer, settings, eos_ids=eos, pad_id=pad)


# ---------------------------------------------------------------------------------------------------------------
# Serving: an OpenAI-compatible /v1/chat/completions, one or more model replicas (processes) behind one port.


def _worker(args: dict, tasks, results) -> None:
    engine = load_engine(args["model"], args.get("revision"), args["dtype"], args["device"],
                         TTTSettings(**args["settings"]), args.get("random_init"))
    results.put(("ready", None, None))
    while True:
        task = tasks.get()
        if task is None:
            return
        request_id, prompt, max_new_tokens, steps = task
        try:
            text, info = engine.answer(prompt, max_new_tokens, steps)
            results.put((request_id, text, info))
        except Exception as error:  # report, keep serving
            import traceback

            traceback.print_exc()
            results.put((request_id, None, {"error": f"{type(error).__name__}: {error}"}))


class Pool:
    """Replicas of the model in separate processes; requests wait in one shared queue (each is served alone)."""

    def __init__(self, args: dict, replicas: int) -> None:
        context = multiprocessing.get_context("spawn")
        self.tasks, self.results = context.Queue(), context.Queue()
        self.processes = [context.Process(target=_worker, args=(args, self.tasks, self.results), daemon=True)
                          for _ in range(replicas)]
        for process in self.processes:
            process.start()
        for _ in self.processes:
            kind, _, _ = self.results.get()
            assert kind == "ready"
        self.pending: dict[int, tuple[threading.Event, list]] = {}
        self.lock = threading.Lock()
        self.counter = 0
        threading.Thread(target=self._collect, daemon=True).start()

    def _collect(self) -> None:
        while True:
            request_id, text, info = self.results.get()
            with self.lock:
                event, slot = self.pending.pop(request_id)
            slot.extend([text, info])
            event.set()

    def submit(self, prompt: str, max_new_tokens: int, steps: int) -> tuple[str | None, dict]:
        with self.lock:
            self.counter += 1
            request_id = self.counter
            event, slot = threading.Event(), []
            self.pending[request_id] = (event, slot)
        self.tasks.put((request_id, prompt, max_new_tokens, steps))
        event.wait()
        return slot[0], slot[1]

    def close(self) -> None:
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            process.join(timeout=10)


def make_handler(pool, served_name: str, steps: int, log_path: str | None):
    names = {served_name: 0, served_name + QTTT_SUFFIX: steps}
    log_lock = threading.Lock()
    totals = {name: {"n": 0, "wall_s": 0.0, "prefill_s": 0.0, "adapt_s": 0.0, "generate_s": 0.0,
                     "prompt_tokens": 0, "output_tokens": 0} for name in names}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # quiet
            pass

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": name, "object": "model"} for name in names]})
            elif self.path.rstrip("/") == "/stats":  # running means per served name: the §31 time gate reads this
                with log_lock:
                    self._send(200, {name: {key: value / max(1, totals[name]["n"]) if key != "n" else value
                                            for key, value in totals[name].items()} for name in totals})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            name = body.get("model")
            if name not in names:  # a wrong name must never fill a cache under the wrong label
                self._send(404, {"error": f"model {name!r} is not served here (served: {sorted(names)})"})
                return
            if float(body.get("temperature", 0.0)) != 0.0:
                self._send(400, {"error": "only greedy decoding (temperature 0) is served"})
                return
            messages = body.get("messages", [])
            if len(messages) != 1 or messages[0].get("role") != "user":
                self._send(400, {"error": "exactly one user message is supported"})
                return
            started = time.time()
            text, info = pool.submit(messages[0]["content"], int(body.get("max_tokens", 64)), names[name])
            if text is None:
                self._send(500, {"error": info.get("error", "failed")})
                return
            record = {"time": started, "wall_s": time.time() - started, "model": name, **info}
            with log_lock:
                totals[name]["n"] += 1
                for key in totals[name]:
                    if key != "n" and isinstance(record.get(key), (int, float)):
                        totals[name][key] += record[key]
                if log_path:
                    with open(log_path, "a") as handle:
                        handle.write(json.dumps(record) + "\n")
            self._send(200, {"object": "chat.completion", "model": name,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": info["prompt_tokens"],
                                       "completion_tokens": info["output_tokens"]},
                             "ttt": info})

    return Handler


def serve(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="memctl.ttt_server serve")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--steps", type=int, default=TTTSettings.steps)
    parser.add_argument("--span", type=int, default=TTTSettings.span)
    parser.add_argument("--lr", type=float, default=TTTSettings.lr)
    parser.add_argument("--repetition-penalty", type=float, default=TTTSettings.repetition_penalty)
    parser.add_argument("--random-init", help="JSON config overrides: a random model of the same architecture")
    parser.add_argument("--log", help="append one JSON line of timings per request")
    args = parser.parse_args(argv)
    settings = TTTSettings(steps=args.steps, span=args.span, lr=args.lr, repetition_penalty=args.repetition_penalty)
    worker_args = {"model": args.model, "revision": args.revision, "dtype": args.dtype, "device": args.device,
                   "settings": asdict(settings),
                   "random_init": json.loads(args.random_init) if args.random_init else None}
    import signal

    pool = Pool(worker_args, args.replicas)

    def stop(*_) -> None:  # a plain kill also stops the replicas
        pool.close()
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(pool, args.served_name, args.steps, args.log))
    print(f"serving {args.served_name} and {args.served_name}{QTTT_SUFFIX} on :{args.port} "
          f"({args.replicas} replicas; {json.dumps(asdict(settings))})", flush=True)
    try:
        server.serve_forever()
    finally:
        pool.close()


# ---------------------------------------------------------------------------------------------------------------
# Timing bench: the real architecture with fewer layers and random weights, on random token ids.


def bench(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="memctl.ttt_server bench")
    parser.add_argument("--config-from", required=True, help="HF id or path whose config.json gives the shapes")
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[1400, 3800])
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=TTTSettings.steps)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args(argv)
    torch = _torch()
    if args.threads:
        torch.set_num_threads(args.threads)
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(args.config_from)
    config.num_hidden_layers = args.layers
    _trim_layer_types(config)
    torch.manual_seed(0)
    model = _from(AutoModelForCausalLM.from_config, config, getattr(torch, args.dtype))
    engine = QueryTTT(model, settings=TTTSettings(steps=args.steps), eos_ids=config.eos_token_id, pad_id=0)
    params = sum(p.numel() for p in model.parameters())
    rows = []
    for total in args.prompt_tokens:
        for repeat in range(args.repeats + 1):  # the first is a warm-up
            ids = torch.randint(0, config.vocab_size, (total,), generator=torch.Generator().manual_seed(repeat)).tolist()
            # min_new_tokens forces a fixed decode length on a random model
            _, info = engine.answer_ids(ids, args.new_tokens, args.steps, seed=repeat, min_new_tokens=args.new_tokens)
            if repeat:
                rows.append({"prompt_tokens": total, **info})
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(json.dumps({"layers": args.layers, "params": params, "dtype": args.dtype, "threads": torch.get_num_threads(),
                      "new_tokens": args.new_tokens, "steps": args.steps, "peak_rss_gb": peak_gb, "rows": rows}))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("serve", "bench"):
        sys.exit("usage: python -m memctl.ttt_server {serve,bench} ...")
    {"serve": serve, "bench": bench}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
