"""A fake vLLM 0.8.5 server for the §29 free rehearsal (no model; no network beyond localhost).
    python configs/sweeps/exp29/fake_vllm.py PORT
It answers the four routes the §29 pipeline calls, in vLLM 0.8.5's response shapes:
- GET  /v1/models                 the served model id
- POST /tokenize                  {"count", "tokens"} for {"prompt", "add_special_tokens"}
- POST /v1/completions            echo + logprobs: the prompt's tokens and log-probabilities (first null), then one
                                  generated token; usage.prompt_tokens. The values come from memctl.reader_utility's
                                  stub (word overlap), spread evenly over the gold's tokens.
- POST /v1/chat/completions       the reader (memctl.llm's StubLLM, then "Answer: ...") or the judge ("no")
Its tokenizer is a regex, consistent between /tokenize and /v1/completions: a newline is its own token."""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from memctl.llm import StubLLM
from memctl.reader_utility import MODEL, stub_logprob

PIECES = re.compile(r"\n|[^\S\n]*[^\s]+|[^\S\n]+")


def tokens(text: str) -> list[str]:
    return PIECES.findall(text)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"object": "list", "data": [{"id": MODEL, "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            pieces = tokens(body["prompt"])
            self.reply({"count": len(pieces), "tokens": list(range(len(pieces))), "max_model_len": 32768})
        elif self.path == "/v1/completions":
            text = body["prompt"]
            assert body.get("echo") and body.get("logprobs") is not None and body["max_tokens"] == 1
            user = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
            gold = text.split("<|im_start|>assistant\n", 1)[1]
            pieces = tokens(text)
            n = len(tokens(gold))
            total = stub_logprob(user, gold, n)["logprob"]
            values = [None] + [-1.0] * (len(pieces) - 1)
            for i in range(len(pieces) - n, len(pieces)):
                values[i] = total / n
            offsets, at = [], 0
            for piece in pieces + ["<|im_end|>"]:
                offsets.append(at)
                at += len(piece)
            self.reply({"choices": [{"index": 0, "text": text + "<|im_end|>", "logprobs": {
                "tokens": pieces + ["<|im_end|>"], "token_logprobs": values + [-0.1], "text_offset": offsets,
                "top_logprobs": [None] + [{}] * len(pieces)}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": len(pieces), "completion_tokens": 1, "total_tokens": len(pieces) + 1}})
        elif self.path == "/v1/chat/completions":
            prompt = body["messages"][-1]["content"]
            if "## Memory" in prompt:
                content = "Note: stub.\nAnswer: " + StubLLM().generate(prompt)[:200]
            else:
                content = "no"
            self.reply({"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]})
        else:
            self.send_error(404)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
