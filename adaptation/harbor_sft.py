"""SFT-1: whole-trajectory supervised fine-tuning on verified Harbor Pi sessions.

  rows      Convert verified, cross-checked Harbor trials of the frozen SFT-1 pool into one
            SFT row per task: the shortest successful session (fewest total tokens).
            Use --preserve-reasoning for verified native Inkling rollouts at effort 0.2.
  plan      Render every row as ONE sequence with all assistant turns supervised, verify the
            loss mask turn by turn, build Tinker datums, and print a hashed plan with the cost.
  execute   Run an approved plan: needs --execute-approved-plan <hash>,
            INKLING_ALLOW_PAID_TINKER=sft-multi-step and TINKER_API_KEY.

  .venv-tinker/bin/python -m adaptation.harbor_sft rows runs/sft1-collect --output runs/sft1.jsonl
  .venv-tinker/bin/python -m adaptation.harbor_sft plan runs/sft1.jsonl
"""

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path

from adaptation.dataset import PRIVATE, assistant, canonical_reasoning, check_capture, read_rows, validate_conversation
from adaptation.harbor_data import RESULTS, SOURCES
from adaptation.sft import CHECKPOINT_TTL_S, LORA_TARGETS, MODEL, RANK, TRAIN_PER_MILLION

SCHEMA = "inkling-pi-harbor-sft-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def trial_row(trial, pool, *, preserve_reasoning=False, outcome="pass"):
    """One whole-session row from a Harbor trial, or raise with the reason it is unusable.

    outcome="pass" (default, SFT): a clean verified pass ending with a final answer.
    outcome="fail" (preference data only): a graded failure (reward 0) of a valid trial; the session may end
    without a final answer (timeout, output cap). The row carries provenance outcome="fail" and reward 0.0.
    """
    result = json.loads((trial / "result.json").read_text())
    if result.get("task") not in pool:
        raise ValueError("task is not in a frozen collection pool")
    if outcome == "pass" and not (result.get("passed") and result.get("reward") == 1.0
                                  and result.get("agent_exit") == 0 and result.get("terminal_stop") == "stop"
                                  and not result.get("toolchain_changes")):
        raise ValueError("not a clean verified pass")
    if outcome == "fail" and not (result.get("reward") == 0.0 and result.get("verifier_exit") is not None
                                  and not result.get("toolchain_changes")):
        raise ValueError("not a graded failure")
    if result.get("revision") != SOURCES["scale800"]["revision"]:
        raise ValueError("dataset revision differs from the pinned one")
    if preserve_reasoning and (
        result.get("temperature") != 1.0 or result.get("effort") != 0.2
        or not result.get("protocol") or result.get("protocol_violations") != []
        or result.get("compaction_calls") != 0
    ):
        raise ValueError("Native rows need a validated, uncompacted temperature=1.0, effort=0.2 rollout")
    if preserve_reasoning and outcome == "pass":
        diff = trial / "changes.diff"
        if not result.get("tool_calls") or not diff.is_file() or not diff.read_text().strip():
            raise ValueError("Native row must make a verified workspace change with Pi tools")
    requests = read_rows(trial / "capture" / "requests.jsonl")
    events = read_rows(trial / "events.jsonl")
    answers = [row["message"] for row in events if row.get("type") == "message_end"
               and row["message"].get("role") == "assistant"]
    if outcome == "fail" and len(requests) == len(answers) + 1:
        requests = requests[:-1]  # the last request was cut off (timeout) before an answer arrived
    if len(requests) != len(answers) or not requests:
        raise ValueError("captured requests and assistant turns do not align")
    check_capture(requests, events, answers, preserve_reasoning=preserve_reasoning)
    final = assistant(answers[-1], preserve_reasoning=preserve_reasoning)
    if outcome == "pass" and (final.get("tool_calls") or not final["content"].strip()):
        raise ValueError("the session does not end with a final answer")
    messages = [*map(canonical_reasoning, requests[-1]["messages"]), final]
    if preserve_reasoning and not any(m.get("reasoning_content") for m in messages):
        raise ValueError("Native row has no recorded reasoning")
    tools = requests[-1]["tools"]
    validate_conversation(messages, tools)
    row = {
        "schema": SCHEMA, "dataset": result["dataset"], "revision": result["revision"], "task": result["task"],
        "rollout": result["rollout"], "effort": requests[-1]["effort"], "temperature": result["temperature"],
        "provenance": {"model": result["model"], "label": result["label"], "backend": result.get("backend", "omlx-local"),
                       "image": result["image"], "reward": result["reward"], "verified": True,
                       "outcome": outcome,
                       "pi_version": result["pi_version"]},
        "metrics": {key: result[key] for key in ("input_tokens", "output_tokens", "tool_calls", "assistant_turns", "elapsed_s")},
        "messages": messages, "tools": tools,
    }
    if preserve_reasoning:
        row["provenance"].update(protocol=result["protocol"], served_model=result["served_model"],
                                 reasoning_targets="native")
    if PRIVATE.search(json.dumps(row, ensure_ascii=False)):
        raise ValueError("private path or credential pattern")
    return row


def build_rows(roots, output, *, preserve_reasoning=False, pools=None):
    pool = set()
    for path in pools or [RESULTS / "sft1-pool.json"]:
        pool |= set(json.loads(Path(path).read_text())["tasks"])
    if pool & set(json.loads((RESULTS / "dev-set.json").read_text())["tasks"]):
        raise ValueError("A collection pool overlaps the dev set")
    best, rejected = {}, []
    for root in roots:
        for trial in sorted(Path(root).glob("*--r*")):
            if not (trial / "result.json").is_file():
                continue
            try:
                row = trial_row(trial, pool, preserve_reasoning=preserve_reasoning)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                rejected.append({"trial": trial.name, "reason": str(exc)[:120]})
                continue
            size = row["metrics"]["input_tokens"] + row["metrics"]["output_tokens"]
            if row["task"] not in best or size < best[row["task"]][0]:
                best[row["task"]] = (size, row)
    rows = [row for _, row in sorted(best.values(), key=lambda item: item[1]["task"])]
    if len({row["effort"] for row in rows}) > 1:
        raise ValueError("Use one fixed reasoning effort")
    with Path(output).open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    reasons = {}
    for item in rejected:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    return {"rows": len(rows), "rejected_trials": len(rejected), "rejection_reasons": reasons}


def render_whole(rows, max_sequence):
    """Render each session as one sequence and verify the loss mask against every turn.

    For every assistant message k: the generation prompt of turn k must be an exact prefix of
    the sequence, and the single-turn rendering through message k must be a prefix as well.
    Supervised tokens may only fall in [len(gen_k) - 1, end_k) (the shared assistant header
    plus the assistant's own tokens), and every assistant span must carry loss.
    """
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.common import datum_from_model_input_weights
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.third_party.openai_compat import openai_messages_to_tinker
    import tinker_cookbook.model_info as model_info

    name = model_info.get_recommended_renderer_name(MODEL)
    if name != "tml_v0":
        raise ValueError(f"Unexpected Inkling renderer {name}")
    tokenizer = get_tokenizer(MODEL)
    renderer = get_renderer(name, tokenizer)
    datums, stats, skipped = [], [], []
    for row in rows:
        if row.get("schema") != SCHEMA or row.get("provenance", {}).get("verified") is not True:
            raise ValueError("Rows must come from adaptation.harbor_sft rows")
        effort, messages = row["effort"], row["messages"]
        if messages[0]["role"] != "system":
            raise ValueError("Missing captured Pi system prompt")
        tools = [tool["function"] for tool in row["tools"]]
        history = messages[1:]
        native_reasoning = row["provenance"].get("reasoning_targets") == "native"
        if native_reasoning:
            expected_reasoning_turns = sum(bool(message.get("reasoning_content")) for message in history)
            if not expected_reasoning_turns:
                raise ValueError(f"{row['task']}: native row has no reasoning to render")
            # The Cookbook renderer expects ThinkingPart in content. Passing raw
            # OpenAI dictionaries silently drops the reasoning_content field.
            history = openai_messages_to_tinker(history)
            specials = {label: tokenizer.tml_tokenizer.encode_special(token) for label, token in {
                "thinking": "content_thinking",
                "action": "content_invoke_tool_json",
                "stop": "content_model_end_sampling",
            }.items()}
        full = [*renderer.create_conversation_prefix_with_tools(tools, messages[0]["content"]), *history]
        model_input, weights = renderer.build_supervised_example(
            full, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES, effort=effort)
        ids, mask = model_input.to_ints(), weights.tolist()
        if len(ids) > max_sequence:
            skipped.append({"task": row["task"], "tokens": len(ids)})
            continue
        allowed = [False] * len(ids)
        turns, reasoning_turns = 0, 0
        for index, message in enumerate(full):
            if message["role"] != "assistant":
                continue
            turns += 1
            gen = renderer.build_generation_prompt(full[:index], effort=effort).to_ints()
            single = renderer.build_supervised_example(
                full[:index + 1], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE, effort=effort)[0].to_ints()
            if ids[:len(gen)] != gen or ids[:len(single)] != single:
                raise ValueError(f"{row['task']}: turn {turns} is not a prefix of the whole sequence")
            span = range(max(len(gen) - 1, 0), len(single))
            if not any(mask[position] for position in span[1:]):
                raise ValueError(f"{row['task']}: assistant turn {turns} carries no loss")
            if native_reasoning:
                supervised = {ids[position] for position in span if mask[position]}
                required = {"stop"}
                if message.get("tool_calls"):
                    required.add("action")
                content = message["content"]
                has_thinking = isinstance(content, list) and any(part["type"] == "thinking" for part in content)
                if has_thinking:
                    required.add("thinking")
                    reasoning_turns += 1
                missing = sorted(label for label in required if specials[label] not in supervised)
                if missing:
                    raise ValueError(f"{row['task']}: assistant turn {turns} lacks supervision for {missing}")
            for position in span:
                allowed[position] = True
        leaks = [position for position, weight in enumerate(mask) if weight and not allowed[position]]
        if leaks:
            raise ValueError(f"{row['task']}: {len(leaks)} supervised tokens outside assistant turns")
        if native_reasoning and reasoning_turns != expected_reasoning_turns:
            raise ValueError(f"{row['task']}: reasoning was dropped during conversion")
        datums.append(datum_from_model_input_weights(model_input, weights, None))
        stats.append({"task": row["task"], "turns": turns, "tokens": len(ids),
                      "supervised_tokens": int(sum(weight > 0 for weight in mask)),
                      "token_sha256": digest(ids), "mask_sha256": digest(mask)})
        if native_reasoning:
            stats[-1]["reasoning_turns"] = reasoning_turns
    return datums, stats, skipped


def plan(args):
    from tinker_cookbook.hyperparam_utils import get_lr

    payload = args.dataset.read_bytes()
    rows = read_rows(args.dataset)
    datums, stats, skipped = render_whole(rows, args.sequence_cap)
    if not datums:
        raise ValueError("No SFT rows")
    tokens = sum(item["tokens"] for item in stats)
    steps_per_epoch = math.ceil(len(datums) / args.batch_size)
    cost = TRAIN_PER_MILLION * Decimal(tokens * args.epochs) / Decimal(1_000_000)
    if cost > args.train_budget_usd:
        raise ValueError(f"Training estimate ${cost} exceeds the ${args.train_budget_usd} cap")
    lora_targets = LORA_TARGETS | ({"train_mlp": True} if args.train_mlp else {})
    manifest = {
        "model": MODEL, "renderer": "tml_v0", "train_on": "ALL_ASSISTANT_MESSAGES", "rank": RANK,
        "lora_targets": lora_targets, "learning_rate": args.learning_rate or get_lr(MODEL),
        "epochs": args.epochs, "batch_size": args.batch_size, "steps": steps_per_epoch * args.epochs,
        "seed": 0, "checkpoint_ttl_s": CHECKPOINT_TTL_S,
        "checkpoints": ["training_state", "sampler_weights", "external_weights"],
        "dataset_sha256": hashlib.sha256(payload).hexdigest(), "rows": len(datums), "skipped_over_cap": skipped,
        "sequence_cap": args.sequence_cap, "tokens_per_epoch": tokens,
        "estimated_training_usd": str(cost), "train_budget_usd": str(args.train_budget_usd),
        "rate_usd_per_million": str(TRAIN_PER_MILLION), "row_stats": stats,
    }
    manifest["plan_sha256"] = digest(manifest)
    return datums, manifest


def execute(datums, manifest, approved):
    import random

    if approved != manifest["plan_sha256"]:
        raise ValueError("The approved plan hash must match this exact batch")
    if os.environ.get("INKLING_ALLOW_PAID_TINKER") != "sft-multi-step":
        raise PermissionError("Paid multi-step SFT requires separate approval")
    if not os.environ.get("TINKER_API_KEY"):
        raise PermissionError("TINKER_API_KEY is required")
    import tinker

    client = tinker.ServiceClient().create_lora_training_client(
        base_model=MODEL, rank=RANK, seed=0, **manifest["lora_targets"]).result()
    order, history = list(range(len(datums))), []
    rng = random.Random(0)
    for epoch in range(manifest["epochs"]):
        rng.shuffle(order)
        for start in range(0, len(order), manifest["batch_size"]):
            batch = [datums[i] for i in order[start:start + manifest["batch_size"]]]
            backward = client.forward_backward(batch, loss_fn="cross_entropy").result()
            client.optim_step(tinker.AdamParams(learning_rate=manifest["learning_rate"])).result()
            history.append({"epoch": epoch, "batch": start // manifest["batch_size"], "metrics": backward.metrics})
            print(json.dumps(history[-1]), flush=True)
    name = f"inkling-2-mini-pi-sft1-{manifest['plan_sha256'][:8]}"
    state = client.save_state(name=name, ttl_seconds=CHECKPOINT_TTL_S).result()
    sampler = client.save_weights_for_sampler(name=name, ttl_seconds=CHECKPOINT_TTL_S).result()
    external = client.save_weights_external(name=name, ttl_seconds=CHECKPOINT_TTL_S).result()
    return {"plan_sha256": manifest["plan_sha256"], "training_checkpoint": state.path,
            "sampler_checkpoint": sampler.path, "external_weights": external.path, "steps": len(history)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rows = sub.add_parser("rows")
    rows.add_argument("roots", nargs="+", type=Path)
    rows.add_argument("--output", type=Path, required=True)
    rows.add_argument("--preserve-reasoning", action="store_true",
                      help="Require native, protocol-validated captures and keep their reasoning targets")
    rows.add_argument("--pool", type=Path, nargs="+", dest="pools",
                      help="Frozen collection pool manifest(s); default results/adaptation/sft1-pool.json")
    for name in ("plan", "execute"):
        command = sub.add_parser(name)
        command.add_argument("dataset", type=Path)
        command.add_argument("--epochs", type=int, default=2)
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--learning-rate", type=float, help="Default: the cookbook's get_lr for Inkling 2 Mini")
        command.add_argument("--sequence-cap", type=int, default=32768)
        command.add_argument("--train-budget-usd", type=Decimal, default=Decimal("3.00"))
        command.add_argument("--train-mlp", action="store_true",
                             help="Also put LoRA on the MLP/expert layers (ablation; attention stays on)")
        if name == "execute":
            command.add_argument("--execute-approved-plan", required=True, metavar="SHA256")
    args = parser.parse_args()
    if args.command == "rows":
        print(json.dumps(build_rows(args.roots, args.output, preserve_reasoning=args.preserve_reasoning, pools=args.pools), indent=2))
        return
    datums, manifest = plan(args)
    summary = {key: value for key, value in manifest.items() if key != "row_stats"}
    print(json.dumps(summary, indent=2), flush=True)
    if args.command == "execute":
        print(json.dumps(execute(datums, manifest, args.execute_approved_plan)), flush=True)


if __name__ == "__main__":
    main()
