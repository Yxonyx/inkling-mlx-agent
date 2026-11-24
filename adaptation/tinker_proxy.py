"""Fixed-effort local Tinker proxy with token budgeting and exact rollout capture."""

import argparse
import json
import os
from pathlib import Path

from adaptation.sft import MODEL


class FixedEffort:
    def __init__(self, renderer, effort):
        self.renderer = renderer
        self.effort = effort

    def __getattr__(self, key):
        return getattr(self.renderer, key)

    def build_generation_prompt(self, messages):
        return self.renderer.build_generation_prompt(messages, effort=self.effort)


class BoundedSampler:
    def __init__(self, sampler, output, token_cap, model_label, effort):
        self.sampler = sampler
        self.output = output
        self.remaining = token_cap
        self.model_label = model_label
        self.effort = effort

    async def sample(self, prompt, num_samples, sampling_params):
        from aiohttp import web
        from tinker_cookbook.capture.scope import current_scope

        if num_samples != 1:
            raise ValueError("Only single-completion requests are supported")
        reservation = prompt.length + sampling_params.max_tokens
        if reservation > self.remaining:
            raise web.HTTPTooManyRequests(text="Local Tinker token budget exhausted")
        self.remaining -= reservation
        coordinates = dict(current_scope())
        response = await self.sampler.sample(
            prompt=prompt, num_samples=num_samples, sampling_params=sampling_params,
        )
        for sequence in response.sequences:
            self.output.write(json.dumps({
                "scope": coordinates, "policy": self.model_label, "effort": self.effort,
                "prompt_tokens": prompt.to_ints(),
                "sampled_tokens": list(sequence.tokens), "sampled_logprobs": sequence.logprobs,
                "sequence_id": getattr(sequence, "sequence_id", None),
                "stop_reason": sequence.stop_reason,
            }) + "\n")
        self.output.flush()
        return response


def make_pi_app(sampler, output, *, model_label, effort, token_cap, auth_token=None):
    from tinker_cookbook.capture.proxy.app import ProxyDeps, make_app
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer = FixedEffort(get_renderer("tml_v0", get_tokenizer(MODEL)), effort)
    return make_app(ProxyDeps(
        renderer, BoundedSampler(sampler, output, token_cap, model_label, effort),
        model_label, default_max_tokens=4096,
    ), auth_token=auth_token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Sampler checkpoint URI. Omit to use the exact base.")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--token-cap", type=int, required=True)
    parser.add_argument("--effort", type=float, default=0.2)
    parser.add_argument("--port", type=int, default=7462)
    args = parser.parse_args()
    if not 0 <= args.effort <= 0.99 or not 1 <= args.port <= 65535 or args.token_cap < 1:
        parser.error("Use a valid loopback port, positive token cap and effort in [0, 0.99]")
    if os.environ.get("INKLING_ALLOW_PAID_TINKER_SAMPLING") != "token-capped":
        parser.error("Paid sampling requires separate approval")
    if not os.environ.get("TINKER_API_KEY"):
        parser.error("Missing TINKER_API_KEY")
    if args.model_path and not args.model_path.startswith("tinker://"):
        parser.error("Expected a Tinker sampler checkpoint URI")

    from aiohttp import web
    import tinker

    select = {"model_path": args.model_path} if args.model_path else {"base_model": MODEL}
    with open(args.capture, "x", opener=private_file) as output:
        client = tinker.ServiceClient().create_sampling_client(**select).result()
        app = make_pi_app(client, output, model_label=args.model_path or MODEL,
                          effort=args.effort, token_cap=args.token_cap)
        web.run_app(app, host="127.0.0.1", port=args.port)


def private_file(path, flags):
    return os.open(path, flags, 0o600)


if __name__ == "__main__":
    main()
