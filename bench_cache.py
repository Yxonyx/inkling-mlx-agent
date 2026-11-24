"""Compare uncached and repeated-prefix requests on the same loaded model."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from bench_api import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("source", type=Path)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("pairs must be positive")

    source = args.source.read_text()
    warmup = "Respond with only the word ready."
    measure(args.base_url, args.model, warmup, max_tokens=8, effort=0)
    for pair in range(1, args.pairs + 1):
        prompt = f"Run identifier: {uuid4()}\n"
        prompt += ("\nPrevious file read result:\n```python\n" + source + "\n```\n") * 5
        prompt += "\n" + warmup
        for stage in ("cold", "warm"):
            row = measure(args.base_url, args.model, prompt, max_tokens=64, effort=0)
            print(json.dumps({"pair": pair, "stage": stage, **row}), flush=True)


if __name__ == "__main__":
    main()
