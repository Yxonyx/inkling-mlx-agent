"""Measure text inference on the pinned MLX-LM backend."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache


def measure(model, tokenizer, prompt, max_tokens):
    cache = make_prompt_cache(model)
    mx.reset_peak_memory()
    started = perf_counter()
    first_token = None
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, prompt_cache=cache):
        if first_token is None:
            first_token = perf_counter()
    return {
        "prompt_tokens": response.prompt_tokens,
        "generated_tokens": response.generation_tokens,
        "finish_reason": response.finish_reason,
        "prefill_tok_s": round(response.prompt_tps, 2),
        "decode_tok_s": round(response.generation_tps, 2),
        "ttft_s": round(first_token - started, 3),
        "elapsed_s": round(perf_counter() - started, 3),
        "mlx_peak_gib": round(mx.get_peak_memory() / 2**30, 3),
        "prompt_cache_gib": round(sum(layer.nbytes for layer in cache) / 2**30, 3),
        "cache_entries": len(cache),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Pinned local model snapshot")
    parser.add_argument("prompt_file", type=Path, help="Text or code fixture")
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 16384, 32768, 65536])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(*args.contexts, args.max_tokens, args.repeats) < 1:
        parser.error("contexts, max-tokens and repeats must be positive")
    model, tokenizer = load(str(args.model.resolve()))
    source = tokenizer.encode(args.prompt_file.read_text(), add_special_tokens=False)
    if not source:
        parser.error("prompt_file must yield at least one token")
    measure(model, tokenizer, source[:128], 2)
    for length in args.contexts:
        prompt = (source * ((length + len(source) - 1) // len(source)))[:length]
        rows = []
        for repeat in range(args.repeats):
            row = measure(model, tokenizer, prompt, args.max_tokens)
            rows.append(row)
            print(json.dumps({"run": repeat + 1, **row}), flush=True)
        metrics = ["prefill_tok_s", "decode_tok_s", "ttft_s", "mlx_peak_gib", "prompt_cache_gib"]
        print(json.dumps({"median_context": length, **{
            metric: median(row[metric] for row in rows) for metric in metrics
        }}), flush=True)


if __name__ == "__main__":
    main()
