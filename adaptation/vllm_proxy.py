"""Local Pi proxy whose sampling runs on a vLLM server (Modal), with Tinker's exact prompt tokens.

The cookbook capture proxy renders every request with the `tml_v0` renderer at a fixed effort, exactly as in
the Tinker runs, and parses the sampled tokens back. Only `sample()` changes: it sends the token ids to vLLM's
/v1/completions (no chat template on the server) and returns tokens, log-probs and the stop reason.

  INKLING_ALLOW_MODAL_SAMPLING=token-capped VLLM_API_KEY=... python -m adaptation.vllm_proxy \
      --base-url https://<modal-endpoint>/v1 --model sft3 --capture runs/x/requests.jsonl --token-cap 3000000
`--model` is the served base name or a LoRA name registered on the server.
"""
import argparse, json, os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sequence:
    tokens: list
    logprobs: list
    stop_reason: str
    sequence_id: object = None


@dataclass
class Response:
    sequences: list = field(default_factory=list)


def completion_tokens(choice):
    """Sampled token ids from a vLLM completion choice (`token_ids`, or `token_id:N` log-prob tokens)."""
    if choice.get("token_ids") is not None:
        return list(choice["token_ids"])
    tokens = (choice.get("logprobs") or {}).get("tokens") or []
    if tokens and all(isinstance(t, str) and t.startswith("token_id:") for t in tokens):
        return [int(t.split(":", 1)[1]) for t in tokens]
    raise ValueError("vLLM response carries no token ids (needs return_token_ids or return_tokens_as_token_ids)")


class VLLMSampler:
    """The slice of tinker.SamplingClient the cookbook proxy uses, backed by vLLM /v1/completions."""

    def __init__(self, base_url, model, api_key=None, timeout_s=1800, allow_redirects=True):
        self.url = base_url.rstrip("/") + "/completions"
        self.model, self.api_key, self.timeout_s = model, api_key, timeout_s
        self.allow_redirects = allow_redirects

    def payload(self, prompt_ids, params):
        stop = list(params.stop or [])
        body = {"model": self.model, "prompt": prompt_ids, "max_tokens": params.max_tokens,
                "temperature": params.temperature, "top_p": params.top_p,
                "logprobs": 0, "return_tokens_as_token_ids": True, "return_token_ids": True,
                "skip_special_tokens": False, "include_stop_str_in_output": True}
        if params.top_k and params.top_k > 0:
            body["top_k"] = params.top_k
        if params.seed is not None:
            body["seed"] = params.seed
        if stop and isinstance(stop[0], int):
            body["stop_token_ids"] = stop
        elif stop:
            body["stop"] = stop
        return body

    async def sample(self, prompt, num_samples, sampling_params):
        import aiohttp

        if num_samples != 1:
            raise ValueError("Only single-completion requests are supported")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = self.payload(prompt.to_ints(), sampling_params)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as session:
            async with session.post(self.url, json=body, headers=headers, allow_redirects=self.allow_redirects) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"vLLM {resp.status}: {text[:500]}")
        choice = json.loads(text)["choices"][0]
        tokens = completion_tokens(choice)
        logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or [None] * len(tokens)
        stop_reason = "length" if choice.get("finish_reason") == "length" else "stop"
        return Response([Sequence(tokens, list(logprobs), stop_reason)])


def main():
    from adaptation.tinker_proxy import make_pi_app, private_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--token-cap", type=int, required=True)
    parser.add_argument("--effort", type=float, default=0.2)
    parser.add_argument("--port", type=int, default=7463)
    args = parser.parse_args()
    if not 0 <= args.effort <= 0.99 or args.token_cap < 1:
        parser.error("Use a positive token cap and effort in [0, 0.99]")
    if os.environ.get("INKLING_ALLOW_MODAL_SAMPLING") != "token-capped":
        parser.error("Paid GPU sampling requires separate approval (INKLING_ALLOW_MODAL_SAMPLING=token-capped)")

    from aiohttp import web

    sampler = VLLMSampler(args.base_url, args.model, os.environ.get("VLLM_API_KEY"))
    with open(args.capture, "x", opener=private_file) as output:
        app = make_pi_app(sampler, output, model_label=f"modal-vllm:{args.model}", effort=args.effort,
                          token_cap=args.token_cap)
        web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
