"""Check cache hits and output parity against a local native MLX server."""

import argparse
import json
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=17681)
    parser.add_argument("--model", default="inkling-native")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/completions"
    prefix = [100, 200, 300, 400] * 200
    extended = prefix + [500, 600, 700, 800] * 24

    def request(prompt, cache):
        data = json.dumps({"model": args.model, "prompt": prompt, "max_tokens": 8,
                           "seed": 23, "temperature": 1.0, "prefix_cache": cache}).encode()
        query = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(query, timeout=120) as response:
            return json.load(response)

    fresh = request(extended, False)
    request(prefix, True)
    reused = request(extended, True)
    if reused["cache_metrics"]["reused_tokens"] != 768 or fresh["choices"] != reused["choices"]:
        raise SystemExit("Cache hit or output parity check failed")
    print(json.dumps({"parity": True, "cache_off": fresh["cache_metrics"],
                      "cache_on": reused["cache_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
