"""Render rows to plain token files for GPU jobs (runs locally, free).

The GPU jobs never tokenize: they read the exact token ids and loss weights that Tinker saw, produced by the
same `render_whole` (cookbook tml_v0 renderer, turn-by-turn mask check). Each record:
  {"task", "tokens": [n ints], "weights": [n-1 floats]}  where weights[j] belongs to predicting tokens[j+1]
NLL per supervised token = -sum_j weights[j] * log p(tokens[j+1] | tokens[:j+1]) / sum(weights).

  PYTHONPATH=. .venv-tinker/bin/python -m adaptation.gpu_export runs/gpu-phase1
writes g1.jsonl (6 reference rows with Tinker NLLs), sft3-train.jsonl (the 40 SFT-3 rows in render order)
and manifest.json (sha256 of every file).
"""
import json, sys
from hashlib import sha256
from pathlib import Path

from adaptation.dataset import read_rows
from adaptation.harbor_sft import render_whole

RUN = Path("runs/aggressive-sft-20261123")


def record(datum, stat):
    inp = datum.model_input.to_ints()
    tgt = datum.loss_fn_inputs["target_tokens"].tolist()
    w = datum.loss_fn_inputs["weights"].tolist()
    if inp[1:] != tgt[:-1] or len(w) != len(inp):
        raise ValueError(f"{stat['task']}: targets are not the inputs shifted by one")
    return {"task": stat["task"], "tokens": inp + [tgt[-1]], "weights": [float(x) for x in w]}


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return sha256(path.read_bytes()).hexdigest()


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    gate = json.loads((RUN / "gate-sft3.json").read_text())
    train_d, train_s, _ = render_whole(read_rows(RUN / "sft3-rows.jsonl"), 16384)
    held_d, held_s, _ = render_whole(read_rows(RUN / "mini-rows.jsonl"), 40000)
    train = [record(d, s) for d, s in zip(train_d, train_s)]
    g1 = []
    for group, D, S in (("trained", train_d, train_s), ("held_out", held_d, held_s)):
        by_task = {s["task"]: i for i, s in enumerate(S)}
        for task, ref in gate[group].items():
            r = record(D[by_task[task]], S[by_task[task]])
            r.update(group=group, tinker_base=ref["base"], tinker_sft3=ref["sft3"])
            g1.append(r)
    manifest = {"source_rows": {"sft3": sha256((RUN / "sft3-rows.jsonl").read_bytes()).hexdigest(),
                                "mini": sha256((RUN / "mini-rows.jsonl").read_bytes()).hexdigest()},
                "g1.jsonl": write(out / "g1.jsonl", g1),
                "sft3-train.jsonl": write(out / "sft3-train.jsonl", train),
                "g1_tokens": {r["task"]: len(r["tokens"]) for r in g1},
                "train_rows": len(train), "train_tokens": sum(len(r["tokens"]) for r in train),
                "max_train_tokens": max(len(r["tokens"]) for r in train),
                "tinker_sft3": {"lr": 5.4e-4, "epochs": 2, "batch_size": 4, "rank": 16, "lora_alpha": 32,
                                "adam": {"beta1": 0.9, "beta2": 0.95, "eps": 1e-12, "weight_decay": 0.0},
                                "shuffle": "random.Random(0).shuffle(order) once per epoch over render order",
                                "loss": "cross_entropy, sum over weighted tokens"}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "tinker_sft3"}, indent=2))


def rft(out, rows_file, val_n=7, cap=32768, seed=0):
    """Rejection-sampling fine-tuning set: verified on-policy passes, split by task.

    The three G1 held-out rows stay out of training (they are the gate's held-out reference); `val_n` more
    tasks are drawn at random for validation. gate.jsonl holds 3 mid-length training rows ("trained") and
    the held-out rows ("held_out") for the training-effect gate.
    """
    import random
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    gate_ref = json.loads((RUN / "gate-sft3.json").read_text())
    fixed_held = list(gate_ref["held_out"])
    datums, stats, _ = render_whole(read_rows(Path(rows_file)), cap)
    records = [record(d, s) for d, s in zip(datums, stats)]
    tasks = sorted(r["task"] for r in records if r["task"] not in fixed_held)
    val_tasks = set(random.Random(seed).sample(tasks, val_n)) | set(fixed_held)
    train = [r for r in records if r["task"] not in val_tasks]
    val = [r for r in records if r["task"] in val_tasks]
    picks = sorted(train, key=lambda r: len(r["tokens"]))[len(train) // 2 - 1:len(train) // 2 + 2]
    gate = [dict(r, group="trained") for r in picks] + [dict(r, group="held_out") for r in val
                                                       if r["task"] in fixed_held]
    manifest = {"source_rows": sha256(Path(rows_file).read_bytes()).hexdigest(), "cap": cap, "seed": seed,
                "train.jsonl": write(out / "train.jsonl", train), "val.jsonl": write(out / "val.jsonl", val),
                "gate.jsonl": write(out / "gate.jsonl", gate), "train_rows": len(train), "val_rows": len(val),
                "train_tokens": sum(len(r["tokens"]) for r in train),
                "max_tokens": max(len(r["tokens"]) for r in records), "val_tasks": sorted(val_tasks),
                "gate_trained": [r["task"] for r in picks]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "val_tasks"}, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "rft":
        rft(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
