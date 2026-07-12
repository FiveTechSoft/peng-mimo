#!/usr/bin/env python3
"""Dependency-free OpenAI-compatible HTTP gateway for the colibri engine."""

import argparse
import codecs
import collections
import contextlib
import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


HERE = Path(__file__).resolve().parent
END = b"\x01\x01END\x01\x01\n"
READY = b"\x01\x01READY\x01\x01\n"
MAX_BODY = 4 << 20
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://tauri.localhost",
    "tauri://localhost",
)


class APIError(Exception):
    def __init__(self, status, message, param=None, code=None, error_type="invalid_request_error",
                 headers=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type
        self.headers = headers or {}


class ClientCancelled(Exception):
    pass


def error_object(error):
    return {"error": {"message": error.message, "type": error.error_type,
                      "param": error.param, "code": error.code}}


class GenerationScheduler:
    """Bounded FIFO admission for the engine's single mutable KV context."""

    def __init__(self, max_queue=8, queue_timeout=300):
        if max_queue < 0:
            raise ValueError("max_queue cannot be negative")
        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.condition = threading.Condition()
        self.queue = collections.deque()
        self.active = False
        self.closed = False
        self.admitted = 0
        self.completed = 0
        self.rejected = 0
        self.timed_out = 0
        self.cancelled = 0

    @contextlib.contextmanager
    def admit(self, cancelled=None):
        ticket = object()
        queued_at = time.monotonic()
        with self.condition:
            if self.closed:
                raise APIError(503, "The inference scheduler is shutting down.", None,
                               "scheduler_closed", "server_error")
            if (self.active or self.queue) and len(self.queue) >= self.max_queue:
                self.rejected += 1
                raise APIError(429, "The inference queue is full.", None, "queue_full",
                               "rate_limit_error", {"Retry-After": "1"})
            self.queue.append(ticket)
            deadline = queued_at + self.queue_timeout
            while True:
                if self.closed:
                    self.queue.remove(ticket)
                    self.condition.notify_all()
                    raise APIError(503, "The inference scheduler is shutting down.", None,
                                   "scheduler_closed", "server_error")
                if not self.active and self.queue[0] is ticket:
                    break
                if cancelled and cancelled():
                    self.queue.remove(ticket)
                    self.cancelled += 1
                    self.condition.notify_all()
                    raise ClientCancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.queue.remove(ticket)
                    self.timed_out += 1
                    self.condition.notify_all()
                    raise APIError(429, "Timed out waiting for the inference engine.", None,
                                   "queue_timeout", "rate_limit_error", {"Retry-After": "1"})
                self.condition.wait(min(remaining, 0.25))
            self.queue.popleft()
            self.active = True
            self.admitted += 1
            wait_seconds = time.monotonic() - queued_at
        try:
            yield wait_seconds
        finally:
            with self.condition:
                self.active = False
                self.completed += 1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"active": self.active, "queued": len(self.queue),
                    "max_queue": self.max_queue, "queue_timeout_seconds": self.queue_timeout,
                    "admitted": self.admitted, "completed": self.completed,
                    "rejected": self.rejected, "timed_out": self.timed_out,
                    "cancelled": self.cancelled}

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def content_text(content, param):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise APIError(400, "Message content must be a string or an array of text parts.", param)
    parts = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
            raise APIError(400, "Colibri currently supports text message content only.",
                           f"{param}.{index}", "unsupported_content_type")
        if not isinstance(part.get("text"), str):
            raise APIError(400, "Text content parts require a string `text` field.",
                           f"{param}.{index}.text")
        parts.append(part["text"])
    return "".join(parts)


DEFAULT_MIMO_SYSTEM = "You are MiMo, a helpful AI assistant engineered by Xiaomi."


def chat_arch(explicit=None):
    """peng default: mimo. GLM template kept for legacy/colibri clients (PENG_ARCH=glm)."""
    if explicit:
        return explicit
    return (os.environ.get("PENG_ARCH")
            or os.environ.get("COLI_ARCH")
            or "mimo").strip().lower()


def render_chat_glm(messages, enable_thinking=False, reasoning_effort=None):
    """Text-only subset of the official GLM-5.2 chat template (legacy)."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    prompt = ["[gMASK]<sop>"]
    if enable_thinking:
        effort = "High" if reasoning_effort == "high" else "Max"
        prompt.append(f"<|system|>Reasoning Effort: {effort}")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        text = content_text(message.get("content"), f"messages.{index}.content")
        if role in ("system", "developer"):
            prompt.append(f"<|system|>{text}")
        elif role == "user":
            prompt.append(f"<|user|>{text}")
        elif role == "assistant":
            prompt.append(f"<|assistant|><think></think>{text.strip()}")
        else:
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
    prompt.append("<|assistant|><think>" if enable_thinking else
                  "<|assistant|><think></think>")
    return "".join(prompt)


def render_chat_mimo(messages, enable_thinking=False, reasoning_effort=None):
    """ChatML MiMo-V2.5 template (matches c/mimo.c mimo_turn_render / HF apply_chat_template).

    No newline between <|im_end|> and the next <|im_start|>. THINK=0 style:
    generation prompt pre-fills empty <think></think> when enable_thinking is False.
    """
    del reasoning_effort  # MiMo path does not use GLM-style effort tags
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    parts = []
    has_system = any(
        isinstance(m, dict) and m.get("role") in ("system", "developer")
        for m in messages
    )
    if not has_system:
        parts.append(f"<|im_start|>system\n{DEFAULT_MIMO_SYSTEM}<|im_end|>")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        text = content_text(message.get("content"), f"messages.{index}.content")
        if role in ("system", "developer"):
            parts.append(f"<|im_start|>system\n{text}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{text}<|im_end|>")
        elif role == "assistant":
            # History stays as model output (often already includes <think>…); close turn.
            parts.append(f"<|im_start|>assistant\n{text}<|im_end|>")
        else:
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
    if enable_thinking:
        parts.append("<|im_start|>assistant\n")
    else:
        parts.append("<|im_start|>assistant\n<think></think>")
    return "".join(parts)


def render_chat(messages, enable_thinking=False, reasoning_effort=None, arch=None):
    """Dispatch chat template: default MiMo (peng); PENG_ARCH=glm for legacy GLM."""
    a = chat_arch(arch)
    if a in ("glm", "glm52", "colibri"):
        return render_chat_glm(messages, enable_thinking, reasoning_effort)
    if a in ("mimo", "mimo_v2", "peng"):
        return render_chat_mimo(messages, enable_thinking, reasoning_effort)
    raise APIError(400, f"Unknown chat architecture {a!r} (use mimo or glm).", "arch")


def generation_options(body, limit):
    if body.get("n", 1) != 1:
        raise APIError(400, "Colibri currently supports `n=1` only.", "n", "unsupported_value")
    for name in ("tools", "functions"):
        if body.get(name):
            raise APIError(400, f"`{name}` is not supported yet.", name, "unsupported_parameter")
    if body.get("stop") is not None:
        raise APIError(400, "Custom stop sequences are not supported yet.", "stop", "unsupported_parameter")
    if body.get("logprobs"):
        raise APIError(400, "Log probabilities are not supported yet.", "logprobs", "unsupported_parameter")
    if body.get("frequency_penalty", 0) or body.get("presence_penalty", 0):
        raise APIError(400, "Token penalties are not supported yet.", None, "unsupported_parameter")
    if body.get("seed") is not None:
        raise APIError(400, "Per-request seeds are not supported yet.", "seed", "unsupported_parameter")
    response_format = body.get("response_format")
    if response_format not in (None, {"type": "text"}):
        raise APIError(400, "Only the default text response format is supported.",
                       "response_format", "unsupported_parameter")

    maximum = body.get("max_completion_tokens")
    maximum_param = "max_completion_tokens"
    if maximum is None:
        maximum = body.get("max_tokens")
        maximum_param = "max_tokens"
    if maximum is None:
        maximum = min(256, limit)
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    temperature = 0.7 if temperature is None else temperature
    top_p = 0.9 if top_p is None else top_p
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= limit:
        raise APIError(400, f"`{maximum_param}` must be an integer between 1 and {limit}.", maximum_param)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise APIError(400, "`temperature` must be between 0 and 2.", "temperature")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise APIError(400, "`top_p` must be greater than 0 and at most 1.", "top_p")
    return maximum, float(temperature), float(top_p)


def read_engine_turn(stream, sentinel, on_bytes, cancel_check=None, poll_s=0.25):
    """Read until sentinel. Optional cancel_check() aborts with ClientCancelled (F-08)."""
    pending = b""
    fileno = None
    try:
        fileno = stream.fileno()
    except (AttributeError, OSError, ValueError):
        fileno = None
    while True:
        if cancel_check and cancel_check():
            raise ClientCancelled()
        if fileno is not None:
            readable, _, _ = select.select([fileno], [], [], poll_s)
            if not readable:
                continue
        byte = stream.read(1)
        if byte == b"":
            raise RuntimeError("engine exited unexpectedly")
        pending += byte
        if pending.endswith(sentinel):
            data = pending[:-len(sentinel)]
            if data:
                on_bytes(data)
            break
        if len(pending) > len(sentinel):
            on_bytes(pending[:-len(sentinel)])
            pending = pending[-len(sentinel):]

    # STAT line may arrive after a short delay
    while True:
        if cancel_check and cancel_check():
            raise ClientCancelled()
        if fileno is not None:
            readable, _, _ = select.select([fileno], [], [], poll_s)
            if not readable:
                continue
        line = stream.readline()
        break
    fields = line.decode("utf-8", "replace").strip().split()
    if len(fields) < 5 or fields[0] != "STAT":
        raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
    return {
        "completion_tokens": int(fields[1]),
        "tokens_per_second": float(fields[2]),
        "cache_hit_percent": float(fields[3]),
        "rss_gb": float(fields[4]),
        "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
        "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
    }


class Engine:
    """Single SERVE child. F-08: client cancel kills the child and respawns (load ~20–50s).
    F-09: alive() for readiness probes."""

    def __init__(self, executable, model, cap=8, max_tokens=1024, env=None,
                 ebits=4, dbits=8):
        self.executable = str(executable)
        self.model = str(model)
        self.cap = int(cap)
        self.max_tokens = int(max_tokens)
        self.ebits = int(ebits)
        self.dbits = int(dbits)
        self.base_env = dict(env or os.environ)
        self.process = None
        self.lock = threading.Lock()
        self.restarts = 0
        self.last_restart_reason = ""
        self._spawn("initial")

    def _child_env(self):
        env = dict(self.base_env)
        env.update({
            "SNAP": self.model,
            "SERVE": "1",
            "NGEN": str(self.max_tokens),
        })
        return env

    def _spawn(self, reason=""):
        if self.process is not None and self.process.poll() is None:
            self._kill_process()
        # peng defaults: cap ebits dbits (chat_peng uses 64 4 8)
        cmd = [self.executable, str(self.cap), str(self.ebits), str(self.dbits)]
        self.process = subprocess.Popen(
            cmd, env=self._child_env(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        self.last_restart_reason = reason or "spawn"
        try:
            read_engine_turn(self.process.stdout, READY, lambda _: None)
        except Exception:
            self._kill_process()
            raise RuntimeError(f"engine failed READY after {reason or 'spawn'}")
        if reason and reason != "initial":
            self.restarts += 1
            sys.stderr.write(f"[engine] restarted ({reason}) pid={self.process.pid} "
                             f"total_restarts={self.restarts}\n")

    def _kill_process(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.kill()
            except OSError:
                pass
            try:
                self.process.wait(timeout=15)
            except Exception:
                pass
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                if pipe:
                    pipe.close()
            except OSError:
                pass
        self.process = None

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def pid(self):
        return self.process.pid if self.process is not None else None

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, cancel_check=None):
        payload = prompt.encode("utf-8")
        if b"\0" in payload:
            raise APIError(400, "NUL bytes are not supported in prompts.", "messages")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def decode(data):
            text = decoder.decode(data)
            if text:
                on_text(text)

        with self.lock:
            if not self.alive():
                self._spawn("recover-dead")
            header = f"\x02PROMPT {len(payload)} {max_tokens} {temperature:.8g} {top_p:.8g}\n".encode()
            try:
                self.process.stdin.write(header + payload + b"\n")
                self.process.stdin.flush()
                stats = read_engine_turn(self.process.stdout, END, decode,
                                         cancel_check=cancel_check)
                tail = decoder.decode(b"", final=True)
                if tail:
                    on_text(tail)
                return stats
            except ClientCancelled:
                # F-08: hard-kill frees the scheduler; next request pays load time
                sys.stderr.write("[engine] client cancelled — killing SERVE child\n")
                self._spawn("client-cancel")
                raise
            except Exception:
                if not self.alive():
                    try:
                        self._spawn("recover-after-error")
                    except Exception as restart_err:
                        sys.stderr.write(f"[engine] restart failed: {restart_err}\n")
                raise

    def close(self):
        with self.lock:
            if self.process is not None:
                if self.process.poll() is None:
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=5)
                    except Exception:
                        self._kill_process()
                        return
                self._kill_process()


def model_object(model_id, created):
    return {"id": model_id, "object": "model", "created": created, "owned_by": "peng"}


class APIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine, model_id, api_key=None, max_tokens=1024,
                 cors_origins=DEFAULT_CORS_ORIGINS, max_queue=8, queue_timeout=300):
        super().__init__(address, APIHandler)
        self.engine = engine
        self.model_id = model_id
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.scheduler = GenerationScheduler(max_queue, queue_timeout)
        self.cors_origins = tuple(cors_origins)
        self.created = int(time.time())


class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "colibri"

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] %s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, status, body, request_id=None, headers=None):
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if request_id:
            self.send_header("x-request-id", request_id)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def send_cors_headers(self):
        origin = self.headers.get("Origin")
        if not origin or ("*" not in self.server.cors_origins and origin not in self.server.cors_origins):
            return
        self.send_header("Access-Control-Allow-Origin", "*" if "*" in self.server.cors_origins else origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Expose-Headers",
                         "x-request-id, x-colibri-queue-wait-ms, Retry-After")
        self.send_header("Access-Control-Max-Age", "600")
        if "*" not in self.server.cors_origins:
            self.send_header("Vary", "Origin")

    def require_auth(self):
        if self.server.api_key and self.headers.get("Authorization") != f"Bearer {self.server.api_key}":
            raise APIError(401, "Invalid or missing API key.", None, "invalid_api_key",
                           "authentication_error")

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError(400, "Invalid Content-Length header.")
        if length < 1 or length > MAX_BODY:
            raise APIError(400, f"Request body must be between 1 and {MAX_BODY} bytes.")
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise APIError(400, "Request body must be valid JSON.")
        if not isinstance(body, dict):
            raise APIError(400, "Request body must be a JSON object.")
        return body

    def check_model(self, body):
        model = body.get("model")
        if model != self.server.model_id:
            raise APIError(404, f"The model `{model}` does not exist.", "model", "model_not_found")

    def do_GET(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            path = urlsplit(self.path).path
            if path in ("/health", "/ready", "/live"):
                # F-09: liveness of the HTTP process vs readiness of the SERVE child
                eng = self.server.engine
                alive = eng.alive()
                sched = self.server.scheduler.snapshot()
                body = {
                    "status": "ok" if alive else "engine_down",
                    "ready": alive,
                    "live": True,
                    "engine": {
                        "alive": alive,
                        "pid": eng.pid(),
                        "restarts": getattr(eng, "restarts", 0),
                        "last_restart_reason": getattr(eng, "last_restart_reason", ""),
                    },
                    "scheduler": sched,
                }
                if path == "/live":
                    self.send_json(200, {"status": "ok", "live": True}, request_id)
                    return
                if path == "/ready" or (path == "/health" and not alive):
                    # /ready always 503 when engine dead; /health also 503 when dead
                    # so naive probes fail closed (still returns JSON body).
                    if not alive:
                        self.send_json(503, body, request_id)
                        return
                self.send_json(200, body, request_id)
                return
            self.require_auth()
            if path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [model_object(
                    self.server.model_id, self.server.created)]}, request_id)
            elif path.startswith("/v1/models/") and unquote(path[11:]) == self.server.model_id:
                self.send_json(200, model_object(self.server.model_id, self.server.created), request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_json(error.status, error_object(error), request_id, error.headers)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            self.require_auth()
            body = self.read_json()
            self.check_model(body)
            path = urlsplit(self.path).path
            if path == "/v1/chat/completions":
                self.chat_completion(body, request_id)
            elif path == "/v1/completions":
                self.completion(body, request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_json(error.status, error_object(error), request_id, error.headers)
        except ClientCancelled:
            pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.log_error("request failed: %s", error)
            api_error = APIError(500, "The colibri engine failed to process the request.",
                                 None, "engine_error", "server_error")
            try:
                self.send_json(500, error_object(api_error), request_id)
            except OSError:
                pass

    def generation(self, body, prompt, request_id, chat):
        maximum, temperature, top_p = generation_options(body, self.server.max_tokens)
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        stream_options = body.get("stream_options") if stream else None
        if stream and stream_options is not None and not isinstance(stream_options, dict):
            raise APIError(400, "`stream_options` must be an object.", "stream_options")
        include_usage = bool((stream_options or {}).get("include_usage"))
        object_name = "chat.completion" if chat else "text_completion"
        id_prefix = "chatcmpl-" if chat else "cmpl-"
        completion_id = id_prefix + uuid.uuid4().hex
        created = int(time.time())

        with self.server.scheduler.admit(self.client_disconnected) as queue_wait:
            queue_headers = {"x-colibri-queue-wait-ms": str(round(queue_wait * 1000))}
            if not stream:
                output = []
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, output.append,
                    cancel_check=self.client_disconnected)
                text = "".join(output)
                finish = "length" if stats["length_limited"] else "stop"
                choice = ({"index": 0, "message": {"role": "assistant", "content": text,
                           "refusal": None}, "logprobs": None, "finish_reason": finish} if chat else
                          {"index": 0, "text": text, "logprobs": None, "finish_reason": finish})
                self.send_json(200, {"id": completion_id, "object": object_name, "created": created,
                    "model": self.server.model_id, "choices": [choice], "usage": self.usage(stats)},
                    request_id, queue_headers)
                return

            stream_object = "chat.completion.chunk" if chat else object_name
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("x-request-id", request_id)
            for name, value in queue_headers.items(): self.send_header(name, value)
            self.send_cors_headers()
            self.end_headers()
            connected = True

            def event(choices, usage_marker=False):
                nonlocal connected
                if not connected:
                    return
                event_body = {"id": completion_id, "object": stream_object, "created": created,
                              "model": self.server.model_id, "choices": choices}
                if include_usage:
                    event_body["usage"] = None if not usage_marker else usage_marker
                try:
                    data = json.dumps(event_body, ensure_ascii=False, separators=(",", ":"))
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    connected = False

            if chat:
                event([{"index": 0, "delta": {"role": "assistant", "content": ""},
                        "logprobs": None, "finish_reason": None}])

            def emit(text):
                choice = ({"index": 0, "delta": {"content": text}, "logprobs": None,
                           "finish_reason": None} if chat else
                          {"index": 0, "text": text, "logprobs": None, "finish_reason": None})
                event([choice])

            def cancel_or_stream_dead():
                if not connected:
                    return True
                return self.client_disconnected()

            try:
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, emit,
                    cancel_check=cancel_or_stream_dead)
            except ClientCancelled:
                # stream already half-written; free scheduler via engine restart
                raise
            finish = "length" if stats["length_limited"] else "stop"
            final_choice = ({"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish}
                            if chat else {"index": 0, "text": "", "logprobs": None,
                                          "finish_reason": finish})
            event([final_choice])
            if include_usage:
                event([], self.usage(stats))
            if connected:
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
            self.close_connection = True

    def client_disconnected(self):
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
            return self.connection.recv(1, flags) == b""
        except (OSError, ValueError):
            return True

    @staticmethod
    def usage(stats):
        prompt = stats["prompt_tokens"]
        completion = stats["completion_tokens"]
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion}

    def chat_completion(self, body, request_id):
        reasoning_effort = body.get("reasoning_effort")
        efforts = (None, "none", "minimal", "low", "medium", "high", "xhigh")
        if reasoning_effort not in efforts:
            raise APIError(400, "`reasoning_effort` must be none, minimal, low, medium, high, or xhigh.",
                           "reasoning_effort")
        enable_thinking = body.get("enable_thinking", reasoning_effort not in (None, "none"))
        if not isinstance(enable_thinking, bool):
            raise APIError(400, "`enable_thinking` must be a boolean.", "enable_thinking")
        prompt = render_chat(body.get("messages"), enable_thinking, reasoning_effort)
        self.generation(body, prompt, request_id, True)

    def completion(self, body, request_id):
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise APIError(400, "Colibri currently requires `prompt` to be a string.", "prompt")
        self.generation(body, prompt, request_id, False)


def serve(model, host="127.0.0.1", port=8000, model_id="mimo-v2.5-peng", api_key=None,
          cap=8, max_tokens=1024, engine=None, env=None, cors_origins=None,
          max_queue=8, queue_timeout=300):
    if engine is None:
        engine = HERE / "mimo"
    if not 1 <= max_tokens:
        raise ValueError("max_tokens must be positive")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if max_queue < 0:
        raise ValueError("max_queue cannot be negative")
    if queue_timeout <= 0:
        raise ValueError("queue_timeout must be positive")
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        print("WARNING: API is listening beyond localhost without COLI_API_KEY", file=sys.stderr)
    runtime = Engine(engine, model, cap, max_tokens, env)
    origins = DEFAULT_CORS_ORIGINS if cors_origins is None else tuple(cors_origins)
    server = APIServer((host, port),runtime,model_id,api_key,max_tokens,origins,max_queue,queue_timeout)
    print(f"OpenAI-compatible API listening on http://{host}:{port}/v1", file=sys.stderr)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.scheduler.close()
        server.server_close()
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("COLI_MODEL"), required=not os.environ.get("COLI_MODEL"))
    parser.add_argument("--engine", default=os.environ.get("COLI_ENGINE", str(HERE / "mimo")),
                        help="engine binary (default: ./mimo; legacy GLM: path to glm)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id",
                        default=os.environ.get("COLI_MODEL_ID",
                                               os.environ.get("PENG_MODEL_ID", "mimo-v2.5-peng")))
    parser.add_argument("--api-key", default=os.environ.get("COLI_API_KEY"))
    parser.add_argument("--cors-origin", action="append", default=None,
                        help="allowed browser origin; repeat as needed (use '*' for any origin)")
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-queue", type=int, default=int(os.environ.get("COLI_MAX_QUEUE", "8")))
    parser.add_argument("--queue-timeout", type=float,
                        default=float(os.environ.get("COLI_QUEUE_TIMEOUT", "300")))
    args = parser.parse_args()
    serve(args.model, args.host, args.port, args.model_id, args.api_key,
          args.cap,args.max_tokens,args.engine,cors_origins=args.cors_origin,
          max_queue=args.max_queue,queue_timeout=args.queue_timeout)


if __name__ == "__main__":
    main()
