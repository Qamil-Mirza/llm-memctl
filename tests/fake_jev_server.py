"""A local stand-in for a Jev-compatible server (TypeSafe Jev or a self-hosted OpenJev), for tests and stub runs.

It speaks the wire API (`POST /v1/systemone`, `GET /v1/models`) in OpenJev's response shape, including its
`Server-Timing` header, and answers with memctl's hand-written FakeJevClient heuristic. It is NOT a model and says
nothing about Jev or OpenJev: it exists so the real HTTP client, the endpoint override and resume can be checked
with no network and no spend (EXPERIMENTS.md §26).

    python -m tests.fake_jev_server --port 8091 [--latency-ms 0] [--log requests.jsonl]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from memctl.controllers.jev import FakeJevClient

MODEL = "openjev-0.1-FAKE"


class FakeJevServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address=("127.0.0.1", 0), latency_s: float = 0.0, log_path: str | None = None) -> None:
        super().__init__(address, _Handler)
        self.latency_s = latency_s
        self.log_path = log_path  # optional JSONL: request size and question count, for cost estimates
        self.requests: list[dict] = []  # path, model, whether an Authorization header came, number of questions
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def start(self) -> "FakeJevServer":
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self


class _Handler(BaseHTTPRequestHandler):
    server: FakeJevServer

    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, status: int, payload: dict, headers: dict | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send(200, {"models": [{"name": MODEL}, {"name": "jev-latest"}]})
        else:
            self._send(404, {"detail": "Not Found"})

    def do_POST(self) -> None:
        if self.path != "/v1/systemone":
            self._send(404, {"detail": "Not Found"})
            return
        started = time.perf_counter()
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        request = json.loads(raw)
        with self.server._lock:
            self.server.requests.append({
                "path": self.path, "model": request.get("model"),
                "authorization": "Authorization" in self.headers, "questions": len(request.get("questions", {})),
            })
            if self.server.log_path:
                state_chars = len(json.dumps(request["state"], ensure_ascii=False))
                with open(self.server.log_path, "a") as log:
                    log.write(json.dumps({"bytes": len(raw), "state_chars": state_chars,
                                          "questions": len(request["questions"])}) + "\n")
        if self.server.latency_s:
            time.sleep(self.server.latency_s)
        response = FakeJevClient().ask(request["state"], request["questions"])
        answers = {
            key: {"type": "choice", "choice": a.choice, "probabilities": a.probabilities, "confidence": a.confidence}
            for key, a in response.answers.items()
        }
        model_ms = (time.perf_counter() - started) * 1000
        self._send(
            200, {"model": MODEL, "answers": answers, "usage": {"input_tokens": response.input_tokens, "output_tokens": 0}},
            {"server-timing": f"model;dur={model_ms:.1f}, server;dur=0.1, total;dur={model_ms + 0.1:.1f}"},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--latency-ms", type=float, default=0.0, help="added to every answer")
    parser.add_argument("--log", help="append one JSON line per request (sizes only, no content)")
    args = parser.parse_args()
    server = FakeJevServer(("127.0.0.1", args.port), latency_s=args.latency_ms / 1000, log_path=args.log)
    print(f"fake Jev server on {server.base_url} (FAKE answers)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
