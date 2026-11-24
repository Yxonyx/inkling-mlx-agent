"""Render Pi SFT rows and prepare an explicitly approved, single-step Tinker run."""

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

from adaptation.bench import BY_ID, SCHEMA, digest, suite_digest
from adaptation.dataset import PRIVATE, read_rows, validate_conversation


MODEL = "thinkingmachines/Inkling-2-Mini"
TRAIN_PER_MILLION = Decimal("0.75")
RANK = 16
LEARNING_RATE = 2e-5
CHECKPOINT_TTL_S = 7 * 24 * 60 * 60
# oMLX serves LoRA only on dense decoder projections, so the pilot adapts attention.
# Expert, router and unembedding adapters would export but could not load locally.
LORA_TARGETS = {"train_attn": True, "train_mlp": False, "train_unembed": False}


def render(rows, max_sequence):
    import tinker_cookbook.model_info as model_info
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.common import datum_from_model_input_weights
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    name = model_info.get_recommended_renderer_name(MODEL)
    if name != "tml_v0":
        raise ValueError(f"Unexpected Inkling renderer {name}")
    renderer = get_renderer(name, get_tokenizer(MODEL))
    datums, stats = [], []
    for row in rows:
        task = BY_ID.get(row.get("task"))
        if (task is None or task.split != "train" or row.get("split") != "train"
                or row.get("schema") != SCHEMA or row.get("suite_sha256") != suite_digest()
                or row.get("provenance", {}).get("verified") is not True):
            raise ValueError("SFT input must come from the verified train split")
        if PRIVATE.search(json.dumps(row, ensure_ascii=False)):
            raise ValueError("Private path or credential pattern in SFT input")
        history, target = row["messages"][:-1], row["messages"][-1]
        if target["role"] != "assistant" or not (target.get("content") or target.get("tool_calls")):
            raise ValueError("A supervised row needs a complete assistant target")
        if not history or history[0]["role"] != "system" or not isinstance(history[0]["content"], str):
            raise ValueError("Missing captured Pi system prompt")
        effort = row["effort"]
        if not isinstance(effort, (int, float)) or not 0 <= effort <= 0.99:
            raise ValueError("Invalid reasoning effort")
        validate_conversation([*history, target], row["tools"])
        tools = [tool["function"] for tool in row["tools"]]
        prefix = renderer.create_conversation_prefix_with_tools(tools, history[0]["content"])
        history = [*prefix, *history[1:]]
        generation = renderer.build_generation_prompt(history, effort=effort)
        model_input, weights = renderer.build_supervised_example(
            [*history, target], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE, effort=effort,
        )
        ids, prefix_ids = model_input.to_ints(), generation.to_ints()
        if ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError(f"Generation/training prefix mismatch for {row['task']} turn {row['turn']}")
        if len(ids) > max_sequence:
            raise ValueError("Sequence exceeds the local pilot limit")
        mask = weights.tolist()
        # The shared assistant header can be supervised. Earlier turns must be context-only.
        if any(mask[:len(prefix_ids) - 1]) or not any(mask[len(prefix_ids):]):
            raise ValueError("Assistant loss mask is not aligned with the generation prompt")
        datum = datum_from_model_input_weights(model_input, weights, None)
        datums.append(datum)
        stats.append({
            "task": row["task"], "turn": row["turn"], "effort": effort,
            "rendered_tokens": len(ids), "supervised_tokens": sum(weight > 0 for weight in mask),
            "datum_input_tokens": datum.model_input.length,
            "token_sha256": digest(ids), "mask_sha256": digest(mask),
        })
    if not datums:
        raise ValueError("No SFT rows")
    return datums, stats


def plan(args):
    if args.rows < 1 or not 2 <= args.sequence_cap <= 65536 or args.token_cap < 1 or args.budget_usd <= 0:
        raise ValueError("Positive row, token, and USD caps are required. Sequence cap is at most 65,536.")
    payload = args.dataset.read_bytes()
    all_rows = read_rows(args.dataset)
    efforts = {row["effort"] for row in all_rows}
    if len(efforts) != 1:
        raise ValueError("Use one fixed reasoning effort for a pilot")
    datums, rows = render(all_rows[:args.rows], args.sequence_cap)
    tokens = sum(row["rendered_tokens"] for row in rows)
    cost = TRAIN_PER_MILLION * Decimal(tokens) / Decimal(1_000_000)
    if tokens > args.token_cap or cost > args.budget_usd:
        raise ValueError("The selected batch exceeds the requested token or USD bound")
    manifest = {
        "model": MODEL, "renderer": "tml_v0", "rank": RANK, "lora_targets": LORA_TARGETS,
        "learning_rate": LEARNING_RATE,
        "gradient_steps": 1, "seed": 0, "checkpoint_ttl_s": CHECKPOINT_TTL_S,
        "checkpoints": ["training_state", "sampler_weights", "external_weights"],
        "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        "estimated_model_training_usd": str(cost), "rate_usd_per_million": str(TRAIN_PER_MILLION),
        "rendered_token_upper_bound": tokens, "token_cap": args.token_cap,
        "budget_usd": str(args.budget_usd), "rows": rows,
    }
    manifest["plan_sha256"] = digest(manifest)
    return datums, manifest


def execute(datums, manifest, approved_plan):
    if approved_plan != manifest["plan_sha256"]:
        raise ValueError("The approved plan hash must match this exact batch")
    if os.environ.get("INKLING_ALLOW_PAID_TINKER") != "one-step":
        raise PermissionError("Paid Tinker calls require separate approval")
    if not os.environ.get("TINKER_API_KEY"):
        raise PermissionError("TINKER_API_KEY is required for a paid run")
    import tinker

    service = tinker.ServiceClient()
    client = service.create_lora_training_client(base_model=MODEL, rank=RANK, seed=0, **LORA_TARGETS).result()
    backward = client.forward_backward(datums, loss_fn="cross_entropy").result()
    optimizer = client.optim_step(tinker.AdamParams(learning_rate=LEARNING_RATE)).result()
    state = client.save_state(name="inkling-2-mini-pi-pilot", ttl_seconds=CHECKPOINT_TTL_S).result()
    checkpoint = client.save_weights_for_sampler(name="inkling-2-mini-pi-pilot", ttl_seconds=CHECKPOINT_TTL_S).result()
    # Sampler checkpoints cannot be downloaded; external weights feed adaptation.export_adapter.
    external = client.save_weights_external(name="inkling-2-mini-pi-pilot", ttl_seconds=CHECKPOINT_TTL_S).result()
    return {
        "sampler_checkpoint": checkpoint.path, "training_checkpoint": state.path,
        "external_weights": external.path,
        "plan_sha256": manifest["plan_sha256"],
        "forward_backward_metrics": backward.metrics, "optimizer_metrics": optimizer.metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--sequence-cap", type=int, default=8192)
    parser.add_argument("--token-cap", type=int, default=25000)
    parser.add_argument("--budget-usd", type=Decimal, default=Decimal("0.02"))
    parser.add_argument("--execute-approved-plan", metavar="SHA256")
    args = parser.parse_args()
    datums, manifest = plan(args)
    print(json.dumps(manifest, indent=2), flush=True)
    if args.execute_approved_plan:
        print(json.dumps(execute(datums, manifest, args.execute_approved_plan)), flush=True)


if __name__ == "__main__":
    main()
