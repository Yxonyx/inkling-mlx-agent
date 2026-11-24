"""Measure a local OpenAI-compatible server with streamed responses."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen


def measure(url, model, prompt, max_tokens, effort):
    request = Request(
        f"{url.rstrip('/')}/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "reasoning_effort": effort,
            "stream": True,
            "stream_options": {"include_usage": True},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    first_token = last_token = None
    usage, chunks = {}, []
    finish_reason = None
    started = perf_counter()
    with urlopen(request, timeout=180) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                for key in ("reasoning", "reasoning_content", "content"):
                    if not delta.get(key):
                        continue
                    last_token = perf_counter()
                    if first_token is None:
                        first_token = last_token
                    chunks.append(delta[key])
                finish_reason = choice.get("finish_reason") or finish_reason
    if first_token is None:
        raise RuntimeError("The server streamed no text or reasoning")
    return {
        "ttft_s": round(first_token - started, 3),
        "decode_window_s": round(last_token - first_token, 3),
        "elapsed_s": round(perf_counter() - started, 3),
        "usage": usage,
        "finish_reason": finish_reason,
        "output_sha256": hashlib.sha256("".join(chunks).encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("prompt_file", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--effort", type=float, default=0.9)
    args = parser.parse_args()
    if min(args.repeats, args.max_tokens) < 1:
        parser.error("repeats and max-tokens must be positive")
    if not 0 <= args.effort <= 0.99:
        parser.error("effort must be in [0, 0.99]")
    prompt = args.prompt_file.read_text()
    for run in range(1, args.repeats + 1):
        row = measure(args.base_url, args.model, prompt, args.max_tokens, args.effort)
        print(json.dumps({"run": run, "effort": args.effort, **row}), flush=True)


if __name__ == "__main__":
    main()
