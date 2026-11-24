"""OpenAI-compatible proxy for Pi in front of vLLM, with Tinker's protocol including reasoning round-trips.

Why not the cookbook capture proxy: it returns only the text to the client and drops `reasoning_content` from
incoming assistant messages, so Pi never sees the model's thinking and later prompts lack it. Tinker (and the
official chat template) keep every earlier turn's reasoning in the prompt. Verified 2026-11-23.

Rendering is exactly the training path (harbor_sft.render_whole):
  tml_v0.create_conversation_prefix_with_tools(tools, system) + openai_messages_to_tinker(history)
  -> build_generation_prompt(..., effort=EFFORT)
Sampling goes to vLLM /v1/completions with those token ids (adaptation.vllm_proxy.VLLMSampler); the sampled
tokens are parsed by the renderer and returned with `content`, `reasoning_content` and `tool_calls`
(non-streaming JSON or a one-chunk SSE stream). Every call is captured (prompt tokens, sampled tokens,
log-probs) and counted against a token cap.

  INKLING_ALLOW_MODAL_SAMPLING=token-capped VLLM_API_KEY=... python -m adaptation.pi_proxy \
      --base-url <endpoint>/v1 --model rft1 --capture runs/x/requests.jsonl --token-cap 400000000 --port 7463

The same proxy can sample on Tinker instead (GRPO rollouts: sampler and trainer on one system, so the
importance ratios need no vLLM/trainer correction). Spend is capped in USD at Tinker list prices (prefill billed
in full, as if no prefix were cached):
  INKLING_ALLOW_PAID_TINKER_SAMPLING=token-capped TINKER_API_KEY=... python -m adaptation.pi_proxy --backend tinker \
      [--tinker-model-path tinker://.../sampler_weights/...] --max-usd 2 --capture runs/x/requests.jsonl \
      --token-cap 400000000 --port 7463
"""
import argparse, json, os, threading, time, uuid
from pathlib import Path

from adaptation.sft import MODEL

TINKER_PRICES = (0.15, 0.50)  # Inkling 2 Mini on Tinker, USD per million prefill / sampled tokens (list prices)


class TinkerSampler:
    """tinker.SamplingClient behind the `sample()` slice that VLLMSampler offers."""

    def __init__(self, client):
        self.client = client

    async def sample(self, prompt, num_samples, sampling_params):
        return await self.client.sample(prompt=prompt, num_samples=num_samples, sampling_params=sampling_params)


class PiProtocol:
    """Render OpenAI chat bodies to Inkling tokens and parse sampled tokens back (no network)."""

    def __init__(self, effort, system_suffix=""):
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.tokenizer = get_tokenizer(MODEL)
        self.renderer = get_renderer("tml_v0", self.tokenizer)
        self.effort = effort
        self.system_suffix = system_suffix  # appended to Pi's system prompt (prompt A/B tests, context distillation)

    def prompt(self, body):
        from tinker_cookbook.third_party.openai_compat import openai_messages_to_tinker

        messages = body["messages"]
        if not messages or messages[0].get("role") != "system":
            raise ValueError("Pi requests start with a system message")
        tools = [tool["function"] for tool in body.get("tools") or []]
        history, names = [], {}
        for message in messages[1:]:
            # Tool results carry the called function's name in their header (as in the training rows, where
            # validate_conversation fills it, and in the official chat template); Pi does not send it.
            if message.get("role") == "assistant":
                names.update({c["id"]: c["function"]["name"] for c in message.get("tool_calls") or []})
            elif message.get("role") == "tool" and not message.get("name") and message.get("tool_call_id") in names:
                message = {**message, "name": names[message["tool_call_id"]]}
            history.append(message)
        full = [*self.renderer.create_conversation_prefix_with_tools(tools, messages[0]["content"] + self.system_suffix),
                *openai_messages_to_tinker(history)]
        return self.renderer.build_generation_prompt(full, effort=self.effort)

    def stop(self):
        return self.renderer.get_stop_sequences()

    def parse(self, tokens):
        message, termination = self.renderer.parse_response(list(tokens))
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        thinking = "".join(p.get("thinking", "") for p in parts if p.get("type") == "thinking")
        calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.function if hasattr(call, "function") else call["function"]
            name = function.name if hasattr(function, "name") else function["name"]
            arguments = function.arguments if hasattr(function, "arguments") else function["arguments"]
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                          "function": {"name": name, "arguments": arguments}})
        return text, thinking, calls, termination


def make_app(protocol, sampler, capture, token_cap, model_label, prices=None, max_usd=None,
             budget_file=None):
    """`prices` (prefill, sampled) in USD per million tokens with `max_usd`: each call reserves its worst case
    (prompt + max_tokens) before sampling and settles at the real sampled length afterwards."""
    import tinker
    from aiohttp import web

    lock, state = threading.Lock(), {"remaining": token_cap, "usd": 0.0}
    cost = (lambda n_in, n_out: (n_in * prices[0] + n_out * prices[1]) / 1e6) if prices else (lambda n_in, n_out: 0.0)

    def publish_budget():
        if budget_file is not None:
            temporary = budget_file.with_suffix(".tmp")
            temporary.write_text(json.dumps({"upper_usd": state["usd"]}) + "\n")
            temporary.replace(budget_file)

    publish_budget()

    async def chat(request):
        body = await request.json()
        try:
            model_input = protocol.prompt(body)
        except (ValueError, KeyError, TypeError) as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=400)
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 4096)
        with lock:
            reservation = model_input.length + max_tokens
            worst = cost(model_input.length, max_tokens)
            if reservation > state["remaining"]:
                return web.json_response({"error": {"message": "token budget exhausted"}}, status=429)
            if max_usd is not None and state["usd"] + worst > max_usd:
                return web.json_response({"error": {"message": "USD budget exhausted"}}, status=429)
            state["remaining"] -= reservation
            state["usd"] += worst
            publish_budget()
        params = tinker.SamplingParams(max_tokens=max_tokens, stop=protocol.stop(),
                                       temperature=body.get("temperature", 1.0), top_p=body.get("top_p", 1.0),
                                       seed=body.get("seed"))
        # An interrupted or failed request may still be billed at the provider.
        response = await sampler.sample(model_input, 1, params)
        seq = response.sequences[0]
        text, thinking, calls, _ = protocol.parse(seq.tokens)
        with lock:
            state["usd"] += cost(model_input.length, len(seq.tokens)) - worst
            publish_budget()
            capture.write(json.dumps({"policy": model_label, "effort": protocol.effort,
                                      "prompt_tokens": model_input.to_ints(), "sampled_tokens": list(seq.tokens),
                                      "sampled_logprobs": seq.logprobs, "stop_reason": seq.stop_reason,
                                      "usd_total": round(state["usd"], 6)}) + "\n")
            capture.flush()
        finish = "tool_calls" if calls else ("length" if seq.stop_reason == "length" else "stop")
        message = {"role": "assistant", "content": text if (text or not calls) else None}
        if thinking:
            message["reasoning_content"] = thinking
        if calls:
            message["tool_calls"] = calls
        usage = {"prompt_tokens": model_input.length, "completion_tokens": len(seq.tokens),
                 "total_tokens": model_input.length + len(seq.tokens)}
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        if not body.get("stream"):
            return web.json_response({"id": cid, "object": "chat.completion", "created": created,
                                      "model": model_label, "usage": usage,
                                      "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_label}
        deltas = [{"role": "assistant", "content": ""}]
        if thinking:
            deltas.append({"reasoning_content": thinking})
        if text:
            deltas.append({"content": text})
        if calls:
            deltas.append({"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]})
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        for delta in deltas + [{}]:
            choice = {"index": 0, "delta": delta, "finish_reason": None if delta else finish}
            payload = {**base, "choices": [choice], **({"usage": None} if include_usage else {})}
            await resp.write(f"data: {json.dumps(payload)}\n\n".encode())
        if include_usage:
            await resp.write(f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def models(request):
        return web.json_response({"object": "list", "data": [{"id": model_label, "object": "model"}]})

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    return app


def main():
    from adaptation.tinker_proxy import private_file
    from adaptation.vllm_proxy import VLLMSampler

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["vllm", "tinker", "local"], default="vllm")
    p.add_argument("--base-url", help="token-ID completion endpoint (backend vllm or local)")
    p.add_argument("--model", help="served base or LoRA name (backend vllm or local)")
    p.add_argument("--tinker-model-path", help="Tinker sampler checkpoint (backend tinker; omit for the exact base)")
    p.add_argument("--max-usd", type=float, help="hard USD cap at Tinker list prices (required for backend tinker)")
    p.add_argument("--budget-file", type=Path, help="publish reserved and settled USD together for the runner")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--token-cap", type=int, required=True)
    p.add_argument("--effort", type=float, default=0.2)
    p.add_argument("--port", type=int, default=7463)
    p.add_argument("--system-suffix", type=Path, help="text file appended to Pi's system prompt")
    args = p.parse_args()
    if args.backend == "local":
        from urllib.parse import urlsplit

        url = urlsplit(args.base_url or "")
        if url.scheme != "http" or url.hostname != "127.0.0.1" or url.username is not None or not args.model:
            p.error("backend local needs --model and an http://127.0.0.1 completion endpoint")
    elif args.backend == "vllm":
        if not (args.base_url and args.model):
            p.error("backend vllm needs --base-url and --model")
        if os.environ.get("INKLING_ALLOW_MODAL_SAMPLING") != "token-capped":
            p.error("Paid GPU sampling requires INKLING_ALLOW_MODAL_SAMPLING=token-capped")
    else:
        if os.environ.get("INKLING_ALLOW_PAID_TINKER_SAMPLING") != "token-capped" or not os.environ.get("TINKER_API_KEY"):
            p.error("Tinker sampling needs TINKER_API_KEY and INKLING_ALLOW_PAID_TINKER_SAMPLING=token-capped")
        if not args.max_usd or args.max_usd <= 0:
            p.error("backend tinker needs a positive --max-usd")
        if args.tinker_model_path and not args.tinker_model_path.startswith("tinker://"):
            p.error("expected a tinker:// sampler checkpoint")
    from aiohttp import web

    with open(args.capture, "x", opener=private_file) as capture:
        suffix = args.system_suffix.read_text() if args.system_suffix else ""
        if args.backend == "local":
            sampler, label, prices = (VLLMSampler(args.base_url, args.model, allow_redirects=False),
                                      f"local-mlx:{args.model}", None)
        elif args.backend == "vllm":
            sampler, label, prices = (VLLMSampler(args.base_url, args.model, os.environ.get("VLLM_API_KEY")),
                                      f"modal-vllm:{args.model}", None)
        else:
            import tinker
            select = {"model_path": args.tinker_model_path} if args.tinker_model_path else {"base_model": MODEL}
            sampler = TinkerSampler(tinker.ServiceClient().create_sampling_client(**select).result())
            label, prices = f"tinker:{args.tinker_model_path or MODEL}", TINKER_PRICES
        app = make_app(PiProtocol(args.effort, suffix), sampler, capture, args.token_cap, label,
                       prices=prices, max_usd=args.max_usd if args.backend == "tinker" else None,
                       budget_file=args.budget_file)
        web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
