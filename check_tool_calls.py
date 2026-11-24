"""Verify an OpenAI-compatible local endpoint can complete a two-turn tool exchange."""

import argparse
import json
from urllib.request import Request, urlopen


def chat(base_url, model, messages, tools):
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps({
            "model": model, "messages": messages, "tools": tools,
            "temperature": 0, "max_tokens": 1024,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    return result["choices"][0], result.get("usage", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    args = parser.parse_args()
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup_color",
            "description": "Get the assigned color of a widget by its ID.",
            "parameters": {
                "type": "object",
                "properties": {"widget_id": {"type": "string"}},
                "required": ["widget_id"],
                "additionalProperties": False,
            },
        },
    }]
    messages = [{
        "role": "user",
        "content": "Use lookup_color for widget w17, then reply with only its exact assigned color.",
    }]
    first, first_usage = chat(args.base_url, args.model, messages, tools)
    assistant = first["message"]
    calls = assistant.get("tool_calls", [])
    if first["finish_reason"] != "tool_calls" or len(calls) != 1:
        raise SystemExit(f"Expected exactly one structured tool call, got {first!r}")
    call = calls[0]
    if (
        not call.get("id")
        or call["function"]["name"] != "lookup_color"
        or json.loads(call["function"]["arguments"]) != {"widget_id": "w17"}
    ):
        raise SystemExit(f"Incorrect tool call: {call!r}")
    messages.extend([
        {key: assistant[key] for key in ("role", "content", "tool_calls") if key in assistant},
        {
            "role": "tool", "tool_call_id": call["id"], "name": "lookup_color",
            "content": '{"color":"ultramarine"}',
        },
    ])
    second, second_usage = chat(args.base_url, args.model, messages, tools)
    actual = (second["message"].get("content") or "").strip().casefold()
    if second["finish_reason"] != "stop" or actual != "ultramarine":
        raise SystemExit(f"Tool result was not reflected in the answer: {second!r}")
    print(json.dumps({"tool_round_trip": "passed", "usage": [first_usage, second_usage]}))


if __name__ == "__main__":
    main()
