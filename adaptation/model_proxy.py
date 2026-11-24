"""Inference-only HTTP proxy between a sandboxed agent and the model server, with capture.

oMLX serves far more than inference: /admin, /v1/web/fetch, /v1/mcp/execute and others.
Any of those would let an agent route around its network lock, so this proxy forwards only
POST /v1/chat/completions and GET /v1/models to the configured upstream base URL.

Requests are parsed strictly: one request per connection, no bare CR or LF in the header
block, no folded or colon-less header lines, no Transfer-Encoding, one strict
Content-Length, and a request rebuilt from a small header allowlist. The forwarded
chat payload (after Pi's capture extension set effort and temperature) is appended to a
capture file that only the proxy can write. This file, not anything inside the agent's
reach, is the source of SFT rows.

Standalone: the Harbor runner mounts this file read-only into a sidecar container.

  python3 -B model_proxy.py --listen 0.0.0.0:17481 --upstream http://host.docker.internal:17481/v1 \
      --capture /capture/requests.jsonl
"""

import argparse
import json
import os
import re
import socket
import ssl
import threading
import time
from urllib.parse import urlparse

ROUTES = {("POST", "/v1/chat/completions"): "/chat/completions", ("GET", "/v1/models"): "/models"}
FORWARD = {"content-type", "content-length", "accept", "authorization", "user-agent"}
TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
MAX_HEAD, MAX_BODY = 64 << 10, 64 << 20
CLIENT_TIMEOUT_S, MAX_CONNECTIONS = 30, 8
# Tinker sampling prices, USD per million tokens (thinkingmachines.ai/tinker/pricing, as of 2026-10-19).
MODEL_PRICES = {
    "thinkingmachines/Inkling-2-Mini": {"prefill": 0.15, "cached_prefill": 0.015, "sample": 0.50},
    "thinkingmachines/Inkling-2-Small": {"prefill": 0.30, "cached_prefill": 0.03, "sample": 1.00},
    # OpenRouter, provider pinned to PrimeIntellect (openrouter.ai endpoints listing, 2026-11-22).
    "openrouter:PrimeIntellect:thinkingmachines/inkling-2-small": {"prefill": 0.15, "cached_prefill": 0.03, "sample": 0.50},
    "openrouter:PrimeIntellect:thinkingmachines/inkling-2-mini": {"prefill": 0.075, "cached_prefill": 0.015, "sample": 0.30},
}
PRICES = MODEL_PRICES["thinkingmachines/Inkling-2-Mini"]


def prices_for(model, price_model=None):
    """Price table for a served model. tinker:// adapter paths carry no base model, so they are
    priced as Inkling 2 Mini (every adapter in this project) unless price_model names the base."""
    key = price_model or model
    if key in MODEL_PRICES:
        return MODEL_PRICES[key]
    if not price_model and str(model).startswith("tinker://"):
        return PRICES
    raise ValueError(f"No Tinker price for {key!r}; add it to MODEL_PRICES before paid sampling")
PAID_GATE = ("INKLING_ALLOW_PAID_TINKER_SAMPLING", "token-capped")


def cost(usage, prices=PRICES):
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return ((prompt - cached) * prices["prefill"] + cached * prices["cached_prefill"]
            + (usage.get("completion_tokens") or 0) * prices["sample"]) / 1e6


def last_usage(data):
    """Usage from a JSON body or the last SSE chunk that carries it."""
    head, _, body = data.partition(b"\r\n\r\n")
    candidates = [body] + [line[5:].strip() for line in body.split(b"\n") if line.startswith(b"data:")]
    usage = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("usage"), dict):
            usage = value["usage"]
    return usage


class Refused(Exception):
    pass


def parse(client):
    """Read one request; return (method, path, headers, body) or raise Refused(status)."""
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = client.recv(65536)
        if not chunk:
            raise Refused("400 Bad Request")
        data += chunk
        if data.find(b"\r\n\r\n") < 0 and len(data) > MAX_HEAD:
            raise Refused("431 Request Header Fields Too Large")
    head, _, rest = bytes(data).partition(b"\r\n\r\n")
    body = bytearray(rest)
    if len(head) > MAX_HEAD or b"\r" in head.replace(b"\r\n", b"") or b"\n" in head.replace(b"\r\n", b""):
        raise Refused("400 Bad Request")
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or parts[2] != "HTTP/1.1" or (parts[0], parts[1]) not in ROUTES:
        raise Refused("403 Forbidden")
    headers, lengths = {}, []
    for line in lines[1:]:
        name, colon, value = line.partition(":")
        if not colon or not TOKEN.match(name):
            raise Refused("400 Bad Request")
        name, value = name.lower(), value.strip()
        if name == "transfer-encoding":
            raise Refused("411 Length Required")
        if name == "content-length":
            lengths.append(value)
        elif name in FORWARD:
            headers[name] = value
    if parts[0] == "POST":
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,9}", lengths[0]):
            raise Refused("411 Length Required")
        length = int(lengths[0])
    else:
        if lengths and lengths != ["0"]:
            raise Refused("400 Bad Request")
        length = 0
    if length > MAX_BODY:
        raise Refused("413 Payload Too Large")
    while len(body) < length:
        chunk = client.recv(65536)
        if not chunk:
            raise Refused("400 Bad Request")
        body += chunk
    if len(body) != length:
        raise Refused("400 Bad Request")
    return parts[0], parts[1], headers, bytes(body)


class Proxy:
    """Inference-only proxy. For a paid HTTPS upstream (Tinker) it also adds the API key from
    the proxy's own environment, pins the model, caps output tokens and enforces a USD budget
    kept in a ledger, so the agent never sees the key and cannot switch model or overspend."""

    def __init__(self, upstream, capture=None, listen=("127.0.0.1", 0), *, api_key_env=None,
                 model=None, max_completion_tokens=None, budget_usd=None, ledger=None, sampling=None,
                 price_model=None, extra=None):
        url = urlparse(upstream)
        if url.scheme not in ("http", "https") or not url.hostname or (url.scheme == "http" and not url.port):
            raise ValueError("Upstream must be http://host:port/prefix or https://host/prefix")
        self.tls = url.scheme == "https"
        self.host, self.port, self.prefix = url.hostname, url.port or 443, url.path.rstrip("/")
        if self.tls:
            if os.environ.get(PAID_GATE[0]) != PAID_GATE[1]:
                raise PermissionError(f"Paid sampling needs {PAID_GATE[0]}={PAID_GATE[1]}")
            if not (api_key_env and os.environ.get(api_key_env) and model and max_completion_tokens
                    and budget_usd and budget_usd > 0 and ledger):
                raise ValueError("A paid upstream needs an API key variable, model, output cap, budget and ledger")
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self.model, self.max_completion_tokens = model, max_completion_tokens
        self.prices = prices_for(model, price_model) if self.tls else PRICES
        # Protocol sampling enforced on every request, including Pi's context-compaction calls,
        # which bypass the capture extension and would otherwise get the upstream defaults.
        self.sampling = dict(sampling or {})
        self.extra = dict(extra or {})  # Upstream routing fields (OpenRouter provider pin, usage accounting).
        self.budget_usd, self.ledger = budget_usd, ledger
        self.spent, self.reserved = 0.0, 0.0
        if ledger and os.path.exists(ledger):
            with open(ledger, encoding="utf-8") as rows:
                self.spent = sum(json.loads(line)["usd"] for line in rows if line.strip())
        self.capture, self.lock = capture, threading.Lock()
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(listen)
        self.server.listen(16)
        self.address = self.server.getsockname()

    def shape(self, payload, body):
        """Pin the model and output cap; Tinker ignores max_tokens, so set max_completion_tokens."""
        if not (self.model or self.max_completion_tokens or self.sampling or self.extra):
            return payload, body
        payload = dict(payload)
        payload.update(self.sampling)
        payload.update(self.extra)
        for key in ("n", "best_of"):  # One completion per request; others would multiply cost.
            payload.pop(key, None)
        if self.model:
            payload["model"] = self.model
        if self.max_completion_tokens:
            asked = payload.pop("max_tokens", None) or payload.get("max_completion_tokens") or self.max_completion_tokens
            payload["max_completion_tokens"] = min(int(asked), self.max_completion_tokens)
        if payload.get("stream"):
            payload["stream_options"] = {"include_usage": True}
        return payload, json.dumps(payload, ensure_ascii=False).encode()

    def reserve(self, body):
        """Book the worst case before the upstream call; prompt tokens are bounded by body bytes."""
        if not self.budget_usd:
            return 0.0
        worst = (len(body) * self.prices["prefill"] + (self.max_completion_tokens or 0) * self.prices["sample"]) / 1e6
        with self.lock:
            if self.spent + self.reserved + worst > self.budget_usd:
                raise Refused("429 Too Many Requests")
            self.reserved += worst
        return worst

    def charge(self, data, worst, rejected=False, params=None):
        """Replace the reservation with the real cost; book the worst case when a call that may have
        run returns no usage (dropped connection, mid-stream error). A 3xx/4xx refusal costs nothing."""
        if not self.budget_usd and not self.ledger:
            return
        usage = last_usage(data) if data else None
        reported = usage.get("cost") if usage else None  # OpenRouter reports the real charge.
        usd = (max(float(reported), cost(usage, self.prices)) if isinstance(reported, (int, float))
               else cost(usage, self.prices) if usage else (0.0 if rejected else worst))
        with self.lock:
            self.reserved -= worst
            self.spent += usd
            if self.ledger:
                with open(self.ledger, "a", encoding="utf-8") as sink:
                    sink.write(json.dumps({"time": time.time(), "usd": usd, "usage": usage,
                                           "usage_missing": usage is None, "forwarded": params}) + "\n")

    def connect(self):
        sock = socket.create_connection((self.host, self.port), timeout=600)
        if self.tls:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)
        return sock

    @staticmethod
    def check(body):
        payload = json.loads(body)
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            raise Refused("400 Bad Request")
        return payload

    def record(self, payload):
        if self.capture:
            row = {"messages": payload["messages"], "tools": payload.get("tools") or [],
                   "effort": payload.get("reasoning_effort"), "temperature": payload.get("temperature"),
                   "model": payload.get("model")}
            with self.lock, open(self.capture, "a", encoding="utf-8") as sink:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    def handle(self, client):
        upstream, sent, worst, seen, forwarded = None, False, 0.0, bytearray(), None
        try:
            client.settimeout(CLIENT_TIMEOUT_S)
            try:
                method, path, headers, body = parse(client)
                payload = None
                if method == "POST":
                    try:
                        payload = self.check(body)
                        payload, body = self.shape(payload, body)
                        # The parameters the upstream actually receives, for protocol checks.
                        forwarded = {key: payload.get(key) for key in (
                            "model", "temperature", "top_p", "top_k", "reasoning_effort",
                            "max_completion_tokens", "max_tokens", "stream", "seed")}
                    except (ValueError, TypeError):
                        raise Refused("400 Bad Request")
                    worst = self.reserve(body)
            except Refused as refused:
                client.sendall(f"HTTP/1.1 {refused.args[0]}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode())
                return
            if method == "GET" and self.model:
                # A pinned (paid) upstream serves exactly one model; answer locally, nothing is billed.
                listing = json.dumps({"object": "list", "data": [{"id": self.model, "object": "model"}]}).encode()
                client.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                               + f"Content-Length: {len(listing)}\r\n\r\n".encode() + listing)
                return
            sent = True  # From here on the call may be billed, whatever happens next.
            upstream = self.connect()
            host = self.host if self.tls else f"{self.host}:{self.port}"
            lines = [f"{method} {self.prefix}{ROUTES[(method, path)]} HTTP/1.1", f"Host: {host}"]
            lines += [f"{name}: {value}" for name, value in headers.items()
                      if not (self.api_key and name == "authorization")]
            if self.api_key:
                lines.append(f"Authorization: Bearer {self.api_key}")
            if method == "POST":
                lines.append(f"Content-Length: {len(body)}")
            lines.append("Connection: close")
            upstream.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
            first = upstream.recv(65536)
            seen += first
            accepted = re.match(rb"HTTP/1\.[01] 2\d\d ", first) is not None
            # Record only requests the model server accepted.
            if payload is not None and accepted:
                self.record(payload)
            client.sendall(first)
            chunk = upstream.recv(65536)
            while chunk:
                seen += chunk
                if len(seen) > 4 << 20:
                    del seen[:len(seen) - (1 << 20)]
                client.sendall(chunk)
                chunk = upstream.recv(65536)
        except OSError:
            pass
        finally:
            if worst and sent:
                self.charge(bytes(seen), worst, params=forwarded,
                            rejected=re.match(rb"HTTP/1\.[01] [34]\d\d ", bytes(seen[:16])) is not None)
            elif worst:
                with self.lock:
                    self.reserved -= worst
            self.slots.release()
            for sock in (client, upstream):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    def serve_forever(self):
        while True:
            try:
                client, _ = self.server.accept()
            except OSError:
                return
            if not self.slots.acquire(blocking=False):
                client.close()
                continue
            threading.Thread(target=self.handle, args=(client,), daemon=True).start()

    def start(self):
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def close(self):
        self.server.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--listen", required=True, help="host:port")
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--capture")
    parser.add_argument("--api-key-env", help="Environment variable holding the upstream API key")
    parser.add_argument("--model", help="Model or tinker:// sampler checkpoint to pin")
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--ledger")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--reasoning-effort", type=float)
    args = parser.parse_args()
    sampling = {key: value for key, value in (("temperature", args.temperature),
                                               ("reasoning_effort", args.reasoning_effort)) if value is not None}
    host, _, port = args.listen.rpartition(":")
    Proxy(args.upstream, args.capture, (host, int(port)), api_key_env=args.api_key_env, model=args.model,
          max_completion_tokens=args.max_completion_tokens, budget_usd=args.budget_usd,
          ledger=args.ledger, sampling=sampling).serve_forever()


if __name__ == "__main__":
    main()
