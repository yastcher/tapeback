"""A stand-in for llama-server, started by tapeback in place of the real binary.

It takes the flags tapeback passes, holds the key from LLAMA_API_KEY, answers /health
(loading at first, then ready) and /v1/chat/completions, and appends every start and
every request to FAKE_LLAMA_LOG as a JSON line for the test to read.

- FAKE_LLAMA_REPLIES: the answers in order, separated by a record separator (\\x1e);
  the last one repeats.
- FAKE_LLAMA_FITS: the largest --ctx-size that fits in VRAM. Above it the server writes
  what llama.cpp writes when the weights do not fit (exit 1), or with
  FAKE_LLAMA_OOM=context what it writes when the KV cache does not (older builds
  then crash).
- FAKE_LLAMA_MODE: "die" exits while loading for another reason; "deaf" ignores SIGTERM;
  "slow" takes SLOW_ANSWER seconds over each answer.
"""

import argparse
import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Copied from llama.cpp's own messages, so the test exercises what users see.
OUT_OF_MEMORY = (
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2304.00 MiB on device 0: "
    "cudaMalloc failed: out of memory"
)
CONTEXT_OUT_OF_MEMORY = "llama_init_from_model: failed to initialize the context"
LOAD_ERROR = "llama_model_load: error loading model: invalid magic characters: 'GGUX'"

# /health answers "loading" this many times before the model is ready.
LOADING_POLLS = 2
SLOW_ANSWER = 3.0


def _record(entry: dict[str, object]) -> None:
    with open(os.environ["FAKE_LLAMA_LOG"], "a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctx-size", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args, _ = parser.parse_known_args()
    key = os.environ.get("LLAMA_API_KEY", "")
    start = {"argv": sys.argv[1:], "pid": os.getpid(), "key": key, "cwd": os.getcwd()}
    _record(start | {"time": time.time()})

    mode = os.environ.get("FAKE_LLAMA_MODE", "")
    fits = int(os.environ.get("FAKE_LLAMA_FITS") or 0)
    if mode == "die":
        print(LOAD_ERROR, flush=True)
        sys.exit(1)
    if fits and args.ctx_size > fits:
        if os.environ.get("FAKE_LLAMA_OOM") == "context":
            print(CONTEXT_OUT_OF_MEMORY, flush=True)
            os.kill(os.getpid(), signal.SIGSEGV)
        print(OUT_OF_MEMORY, flush=True)
        sys.exit(1)
    if mode == "deaf":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    replies = os.environ.get("FAKE_LLAMA_REPLIES", "").split("\x1e")
    polls = [0]
    answered = [0]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self._reply(404, {"error": {"message": "File Not Found"}})
                return
            polls[0] += 1
            if polls[0] <= LOADING_POLLS:
                self._reply(503, {"error": {"code": 503, "message": "Loading model"}})
            else:
                self._reply(200, {"status": "ok"})

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            auth = self.headers.get("Authorization")
            _record({"path": self.path, "auth": auth, "body": body})
            if auth != f"Bearer {key}":
                self._reply(401, {"error": {"code": 401, "message": "Invalid API Key"}})
                return
            if mode == "slow":
                time.sleep(SLOW_ANSWER)
            reply = replies[min(answered[0], len(replies) - 1)]
            answered[0] += 1
            message = {"role": "assistant", "content": reply}
            choice = {"index": 0, "message": message, "finish_reason": "stop"}
            self._reply(
                200,
                {"id": "fake", "object": "chat.completion", "created": 0, "model": "fake"}
                | {"choices": [choice]},
            )

        def _reply(self, status: int, payload: dict[str, object]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            pass  # the server's log is the test's business, not its output

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
