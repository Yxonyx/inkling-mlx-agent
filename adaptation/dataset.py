"""Convert verified train-only Pi trajectories to structured SFT JSONL."""

import argparse
import json
from pathlib import Path
import re

from adaptation.bench import BY_ID, ROOT, SCHEMA, suite_digest


PRIVATE = re.compile(
    r"""(?:/Users/|/Volumes/|/home/|/private/var/folders/|"""
    r"""\b(?:sk-|ghp_|github_pat_|hf_)[A-Za-z0-9_-]{12,}|"""
    r"""-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----)"""
)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assistant(message, *, preserve_reasoning=False):
    parts = message["content"]
    text = "".join(part["text"] for part in parts if part["type"] == "text")
    calls = [{
        "id": part["id"], "type": "function",
        "function": {"name": part["name"], "arguments": json.dumps(part["arguments"], ensure_ascii=False)},
    } for part in parts if part["type"] == "toolCall"]
    output = {"role": "assistant", "content": text}
    if preserve_reasoning:
        reasoning = "".join(part.get("thinking", "") for part in parts if part["type"] == "thinking")
        if reasoning:
            output["reasoning_content"] = reasoning
    if calls:
        output["tool_calls"] = calls
    return output


def canonical_reasoning(message):
    """One reasoning field for captured assistant turns: Tinker returns reasoning_content, OpenRouter
    returns reasoning / reasoning_details, and Pi echoes back whichever field it received."""
    if message.get("role") != "assistant":
        return message
    message = dict(message)
    details = message.pop("reasoning_details", None) or []
    text = message.pop("reasoning", None)
    if not message.get("reasoning_content"):
        joined = "".join(d.get("text") or "" for d in details if d.get("type") == "reasoning.text") or text
        if joined:
            message["reasoning_content"] = joined
    return message


def normalize(value, trial):
    work = str(trial / "work")
    if isinstance(value, str):
        pi = ROOT / "node_modules" / "@earendil-works" / "pi-coding-agent"
        return (value.replace(work, "/workspace").replace(str(trial), "/trial")
                .replace(str(pi), "/opt/pi"))
    if isinstance(value, dict):
        return {key: normalize(part, trial) for key, part in value.items()
                if key not in ("reasoning_content", "reasoning", "reasoning_details", "thinkingSignature")}
    if isinstance(value, list):
        return [normalize(part, trial) for part in value]
    return value


def validate_conversation(messages, tools):
    tool_names = {tool["function"]["name"] for tool in tools if tool["type"] == "function"}
    pending = {}
    for message in messages:
        role = message["role"]
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Unsupported role {role}")
        if role == "tool":
            call_id = message["tool_call_id"]
            if call_id not in pending:
                raise ValueError("Tool result is missing its assistant call")
            message.setdefault("name", pending.pop(call_id))
            continue
        if pending:
            raise ValueError("Missing tool result before the next message")
        for call in message.get("tool_calls", []):
            function = call["function"]
            if (role != "assistant" or call["id"] in pending or function["name"] not in tool_names
                    or not isinstance(json.loads(function["arguments"]), dict)):
                raise ValueError("Invalid tool call")
            pending[call["id"]] = function["name"]


def text_of(content):
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def check_capture(requests, events, answers, *, preserve_reasoning=False):
    """The capture must be exactly the conversation Pi observed, turn by turn.

    Every request uses the same tools, model, effort and temperature. Each later request is
    the previous one plus that turn's assistant message and its tool results, with the tool
    calls and result texts recorded in events.jsonl. A doctored or injected request fails.
    """
    for key in ("tools", "model", "effort", "temperature"):
        if len({json.dumps(request.get(key), sort_keys=True) for request in requests}) != 1:
            raise ValueError(f"Captured {key} changed within a trial")
    # Tool results in event order. Tool-call ids need not be unique across turns (the cookbook proxy's
    # tml_v0 parse numbers them per turn, e.g. "functions.bash:0"), so results are consumed in order.
    results = [(row["message"].get("toolCallId"), text_of(row["message"].get("content"))) for row in events
               if row.get("type") == "message_end" and row.get("message", {}).get("role") == "toolResult"]
    consumed = 0
    for index, (before, after) in enumerate(zip(requests, requests[1:])):
        prefix, added = after["messages"][:len(before["messages"])], after["messages"][len(before["messages"]):]
        if prefix != before["messages"] or not added or added[0].get("role") != "assistant":
            raise ValueError("Captured requests are not one growing conversation")
        expected = assistant(answers[index], preserve_reasoning=preserve_reasoning)
        calls = [(call["id"], call["function"]["name"], json.loads(call["function"]["arguments"] or "{}"))
                 for call in expected.get("tool_calls", [])]
        seen = [(call.get("id"), call.get("function", {}).get("name"),
                 json.loads(call.get("function", {}).get("arguments") or "{}"))
                for call in added[0].get("tool_calls") or []]
        if seen != calls or text_of(added[0].get("content")).strip() != expected["content"].strip():
            raise ValueError(f"Captured assistant turn {index} differs from Pi's events")
        if preserve_reasoning and (canonical_reasoning(added[0]).get("reasoning_content") or "") != expected.get("reasoning_content", ""):
            raise ValueError(f"Captured reasoning on assistant turn {index} differs from Pi's events")
        tools = added[1:]
        expected_results = results[consumed:consumed + len(calls)]
        consumed += len(calls)
        if [m.get("role") for m in tools] != ["tool"] * len(calls) \
                or [m.get("tool_call_id") for m in tools] != [call[0] for call in calls] \
                or [(m.get("tool_call_id"), text_of(m.get("content"))) for m in tools] != expected_results:
            raise ValueError(f"Captured tool results after turn {index} differ from Pi's events")


def rows_for_trial(trial, result):
    # Written by the inference-only proxy, outside anything the agent can write.
    requests = read_rows(trial / "capture" / "requests.jsonl")
    events = read_rows(trial / "events.jsonl")
    answers = [row["message"] for row in events if row.get("type") == "message_end"
               and row["message"].get("role") == "assistant"]
    if len(requests) != len(answers) or not requests:
        raise ValueError("Pi requests and assistant turns do not align")
    check_capture(requests, events, answers)
    tool_results = {row["message"].get("toolCallId") for row in events
                    if row.get("type") == "message_end" and row["message"].get("role") == "toolResult"}
    for index, (request, answer) in enumerate(zip(requests, answers, strict=True)):
        if request["effort"] != result["effort"]:
            raise ValueError("Effort changed within a trial")
        target = assistant(answer)
        calls = target.get("tool_calls", [])
        if answer.get("stopReason") not in ("toolUse", "stop") or not (target["content"] or calls):
            raise ValueError("No complete supervised target")
        if calls and not all(call["id"] in tool_results for call in calls):
            raise ValueError("Missing observed tool result")
        history = request["messages"]
        messages = [*history, target]
        tools = request["tools"]
        validate_conversation(messages, tools)
        row = normalize({
            "schema": SCHEMA, "suite_sha256": result["suite_sha256"], "split": "train",
            "task": result["task"], "turn": index, "effort": request["effort"],
            "provenance": {"model": result["model"], "label": result["label"], "verified": True},
            "messages": messages, "tools": tools,
        }, trial)
        if PRIVATE.search(json.dumps(row, ensure_ascii=False)):
            raise ValueError("A private path or credential pattern remains after normalization")
        yield row


def export(root, output):
    rows, rejected, tasks_seen = [], [], set()
    for path in sorted(root.glob("*/result.json")):
        result = json.loads(path.read_text())
        task = BY_ID.get(result["task"])
        if (task is None or task.split != "train" or result.get("split") != "train"
                or result.get("suite_sha256") != suite_digest() or result.get("schema") != SCHEMA):
            raise ValueError(f"Ineligible task or changed suite: {path.parent.name}")
        if task.id in tasks_seen:
            raise ValueError(f"Duplicate task: {task.id}")
        tasks_seen.add(task.id)
        if not result["passed"] or result["agent_exit"] or result.get("terminal_stop") != "stop":
            rejected.append({"task": task.id, "reason": "Verifier or terminal-state gate"})
            continue
        try:
            rows.extend(list(rows_for_trial(path.parent, result)))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            rejected.append({"task": task.id, "reason": str(exc)})
    if not rows:
        raise ValueError(f"No train rows exported: {rejected}")
    with output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "tasks": len({row["task"] for row in rows}), "rejected": rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.training_runs.resolve(), args.output)))


if __name__ == "__main__":
    main()
