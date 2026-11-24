"""Serve native Inkling token completions on loopback with optional exact prefix reuse."""

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
from time import monotonic

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.apc import _clone_prompt_cache_for_apc
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.utils import load_model


CHUNK = 256
MAX_PREFIX = 8192
CONTEXT = 65536
PROJECTIONS = dict(wq_du="q_proj", wk_dv="k_proj", wv_dv="v_proj",
                   wr_du="r_proj", gate_attn="gate_proj", wo_ud="o_proj")
TENSOR = re.compile(r"language_model\.layers\.(\d+)\.attn\.(\w+)\.lora_([AB])\.weight")


class LoRA(nn.Module):
    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base = base
        self.a = a.T.astype(mx.float32)
        self.b = b.T.astype(mx.float32)
        self.scale = scale

    def __call__(self, x):
        result = self.base(x)
        delta = (x.astype(mx.float32) @ self.a @ self.b) * self.scale
        return result + delta.astype(result.dtype)


def load_adapter(language, directory):
    config = json.loads((directory/"adapter_config.json").read_text())
    if config["peft_type"] != "LORA" or config.get("use_rslora") or config.get("use_dora") or config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("Expected a uniform-rank Inkling attention LoRA")
    pairs = {}
    for name, value in mx.load(str(directory/"adapter_model.safetensors")).items():
        match = TENSOR.fullmatch(name)
        if match is None or match[2] not in PROJECTIONS:
            raise ValueError(f"Unsupported adapter tensor: {name}")
        layer, projection, side = int(match[1]), PROJECTIONS[match[2]], match[3]
        pairs.setdefault((layer, projection), {})[side] = value
    if not pairs:
        raise ValueError("The adapter contains no attention projections")
    rank = config["r"]
    scale = config["lora_alpha"] / rank
    for (layer, projection), weights in pairs.items():
        if layer >= len(language.model.layers) or set(weights) != {"A", "B"}:
            raise ValueError(f"Invalid LoRA pair for layer {layer}, {projection}")
        attention = language.model.layers[layer].self_attn
        base = attention[projection]
        output, width = base.weight.shape
        if weights["A"].shape != (rank, width) or weights["B"].shape != (output, rank):
            raise ValueError(f"Invalid LoRA shape for layer {layer}, {projection}")
        attention[projection] = LoRA(base, weights["A"], weights["B"], scale)


@dataclass
class Checkpoint:
    tokens: tuple[int, ...]
    cache: list


class NativeModel:
    def __init__(self, snapshot, adapter, prefix_cache):
        self.model = load_model(snapshot)
        self.language = self.model.language_model
        modes = [tuple(layer.mlp.switch_mlp[name].mode for name in ("gate_proj", "up_proj", "down_proj"))
                 for layer in self.language.model.layers]
        if modes != [("nvfp4",)*3]*20 + [("mxfp8",)*3]*4:
            raise ValueError("Expected the official Inkling 2 Mini mixed-precision checkpoint")
        if adapter is not None:
            load_adapter(self.language, adapter)
        self.model.eval()
        mx.eval(self.language.parameters())
        self.prefix_cache = prefix_cache
        self.checkpoint = None

    def complete(self, body):
        prompt = body["prompt"]
        limit = int(body["max_tokens"])
        vocab = self.language.config.unpadded_vocab_size
        if (not isinstance(prompt, list) or not prompt or not 1 <= limit <= 8192
                or len(prompt) + limit > CONTEXT or any(type(x) is not int or not 0 <= x < vocab for x in prompt)):
            raise ValueError("Expected valid token IDs within the 65,536-token context")
        if body.get("stop"):
            raise ValueError("Use stop_token_ids for stop sequences")
        stops = set(body.get("stop_token_ids", []))
        start = monotonic()
        use_cache = self.prefix_cache and body.get("prefix_cache", True) is not False
        prefix = tuple(prompt)
        checkpoint = self.checkpoint if use_cache else None
        reused = 0
        options = {}
        if checkpoint and len(checkpoint.tokens) < len(prompt) and prefix[:len(checkpoint.tokens)] == checkpoint.tokens:
            cache = _clone_prompt_cache_for_apc(checkpoint.cache)
            if cache is not None:
                reused = len(checkpoint.tokens)
                options["prompt_cache"] = cache

        saved = None

        def capture(index, state):
            nonlocal saved
            cache = _clone_prompt_cache_for_apc(state)
            if cache is not None:
                saved = Checkpoint(prefix[:reused+index], cache)

        # Keep the same block boundaries as a fresh prefill. Leave at least
        # one token to compute the next-token distribution.
        boundary = min(MAX_PREFIX, (len(prompt)-1)//CHUNK*CHUNK)
        if use_cache and boundary > reused:
            options.update(prompt_cache_checkpoint_len=boundary-reused, prompt_cache_checkpoint=capture)

        tokens, scores = [], []
        reason, ttft = "length", None
        iterator = generate_step(mx.array([prompt[reused:]]), self.model, None, None,
                                 max_tokens=limit, temperature=float(body.get("temperature", 1.0)),
                                 top_p=float(body.get("top_p", 1.0)), top_k=int(body.get("top_k", 0)),
                                 prefill_step_size=CHUNK, seed=body.get("seed", 23), **options)
        try:
            for token, logprobs in iterator:
                token = int(token.item()) if hasattr(token, "item") else int(token)
                tokens.append(token)
                scores.append(float(logprobs.reshape(-1)[token].item()))
                if ttft is None:
                    ttft = monotonic()-start
                if token in stops:
                    reason = "stop"
                    break
        finally:
            iterator.close()
        if saved is not None or (use_cache and reused == 0):
            self.checkpoint = saved
        return {"choices": [{"token_ids": tokens, "logprobs": {"token_logprobs": scores}, "finish_reason": reason}],
                "cache_metrics": {"reused_tokens": reused, "request_tokens": len(prompt),
                                  "ttft_s": ttft, "elapsed_s": monotonic()-start}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--adapter", type=Path, help="Directory containing the compatible PEFT attention LoRA")
    parser.add_argument("--model", default="inkling-native")
    parser.add_argument("--port", default=17681, type=int)
    parser.add_argument("--no-prefix-cache", action="store_true")
    args = parser.parse_args()
    model = NativeModel(args.snapshot, args.adapter, not args.no_prefix_cache)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"ready": True})
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/v1/completions":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                size = int(self.headers["Content-Length"])
                if not 0 < size <= 2**20:
                    raise ValueError("Invalid request length")
                body = json.loads(self.rfile.read(size))
                if body["model"] != args.model:
                    raise ValueError("Unknown model ID")
                response = model.complete(body)
            except (KeyError, TypeError, ValueError) as exc:
                self.reply(400, {"error": str(exc)})
                return
            self.reply(200, response)

    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Ready on 127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
