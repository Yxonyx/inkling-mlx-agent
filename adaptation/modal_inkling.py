"""Phase 1 GPU jobs on Modal: G1 (logprobs equal Tinker) and G2 (SFT-3 reproduced), first downloadable adapter.

Each paid job needs a separate approval, passed as INKLING_ALLOW_MODAL=<job> (download | g1 | sft).
Hard caps: every function has a Modal timeout; the GPU stops when the function returns or times out.

  modal run adaptation/modal_inkling.py --job download                  # CPU, ~20 GB into a Volume
  modal run adaptation/modal_inkling.py --job g1                        # 1 GPU, ~10-20 min
  modal run adaptation/modal_inkling.py --job sft --targets qkvo        # 1 GPU, G2 reproduction
Inputs come from `python -m adaptation.gpu_export runs/gpu-phase1`; results land in runs/gpu-phase1/.
"""
import json, os, time
from pathlib import Path

import modal

REPO = "thinkingmachines/Inkling-2-Mini"
REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"  # weights unchanged in later commits (README only)
WEIGHTS_SHA256 = "65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1"  # Hub LFS metadata
WEIGHTS_SIZE = 19526019168
LOCAL = Path(__file__).resolve().parent
RUN = Path("runs/gpu-phase1")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0","transformers==5.24.0", "peft==0.22.0", "accelerate==1.16.0",
                 "safetensors", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf", "PYTHONPATH": "/root",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(LOCAL / "__init__.py", "/root/adaptation/__init__.py")
    .add_local_file(LOCAL / "inkling_attention.py", "/root/adaptation/inkling_attention.py")
    .add_local_file(LOCAL / "gpu_core.py", "/root/adaptation/gpu_core.py")
)
cache = modal.Volume.from_name("inkling-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("inkling-runs", create_if_missing=True)
app = modal.App("inkling-pi-phase1", image=image)


def model_dir():
    # The download takes only the files we need, so snapshot_download(local_files_only=True) would reject the
    # snapshot as incomplete (README, chat template, mtp are left out on purpose); use the pinned path directly.
    path = Path(os.environ["HF_HOME"]) / "hub" / f"models--{REPO.replace('/', '--')}" / "snapshots" / REVISION
    if not (path / "config.json").exists() or not (path / "model-00001-of-00001.safetensors").exists():
        raise FileNotFoundError(f"model not in the Volume: {path} (run --job download)")
    return str(path)


def load_model(block_size=1024):
    """bf16 text model; NVFP4/MXFP8 experts are dequantized on load by transformers 5.24."""
    import torch
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM
    from adaptation import inkling_attention

    inkling_attention.install(block_size=block_size)
    t0 = time.time()
    model, info = Inkling2LoopForCausalLM.from_pretrained(model_dir(), dtype=torch.bfloat16, device_map="cuda",
                                                          output_loading_info=True)
    if info["missing_keys"]:
        raise RuntimeError(f"missing weights (text model would be partly random): {sorted(info['missing_keys'])[:10]}")
    unexpected = sorted(k for k in info["unexpected_keys"] if "vision" not in k and "visual" not in k and "mtp" not in k)
    if unexpected:
        raise RuntimeError(f"unexpected text weights were not loaded: {unexpected[:10]}")
    load = {"seconds": round(time.time() - t0), "gpu": torch.cuda.get_device_name(),
            "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
            "memory_gb": round(torch.cuda.memory_allocated() / 1e9, 1)}
    print(json.dumps({"loaded": load}), flush=True)
    return model, load


@app.function(volumes={"/cache": cache}, timeout=3600, cpu=4, memory=16384)
def download():
    from huggingface_hub import snapshot_download

    path = snapshot_download(REPO, revision=REVISION, allow_patterns=["*.json", "model-*.safetensors", "*.txt",
                                                                       "*.model", "*.tiktoken"])
    cache.commit()
    return {"path": path, "files": sorted(os.listdir(path))}


@app.function(volumes={"/cache": cache}, timeout=1800, cpu=4, memory=8192)
def verify(expected_sha256: str, expected_size: int):
    """The weight file in the Volume must be the real file: size and sha256 equal the Hub's LFS metadata."""
    import hashlib, struct

    path = Path(model_dir()) / "model-00001-of-00001.safetensors"
    real = path.resolve()
    size = os.path.getsize(real)
    with open(real, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        tensors = len(json.loads(f.read(header_len))) - 1
    digest = hashlib.sha256()
    with open(real, "rb") as f:
        while chunk := f.read(64 << 20):
            digest.update(chunk)
    ok = size == expected_size and digest.hexdigest() == expected_sha256
    return {"link": str(path), "resolved": str(real), "is_symlink": path.is_symlink(), "size": size,
            "tensors": tensors, "sha256": digest.hexdigest(), "ok": ok}


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=45 * 60)
def g1(g1_jsonl: str):
    """Base NLL per supervised token on the 6 rows Tinker measured; pass if every row is within 2 %."""
    import torch
    from adaptation import gpu_core, inkling_attention

    model, load = load_model()
    rows = gpu_core.read_jsonl(g1_jsonl)
    out = {"load": load, "rows": {}}
    for row in rows:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        nll, w = gpu_core.weighted_nll(model, row)
        value = nll.item() / w
        out["rows"][row["task"]] = {"group": row["group"], "tokens": len(row["tokens"]), "gpu": round(value, 4),
                                    "tinker": row["tinker_base"],
                                    "relative_diff": round(value / row["tinker_base"] - 1, 4),
                                    "seconds": round(time.time() - t0, 1),
                                    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        print(json.dumps({row["task"]: out["rows"][row["task"]]}), flush=True)
    if inkling_attention.CALLS[0] == 0:
        raise RuntimeError("blockwise attention did not run")
    out["g1_passed"] = all(abs(r["relative_diff"]) <= 0.02 for r in out["rows"].values())
    return out


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=30 * 60)
def g1_tokens(g1_jsonl: str, reference_max_tokens: int = 7000):
    """Diagnostics: per-token log-probs of supervised targets. Variants: blockwise (1024) for every row;
    the unpatched transformers reference for rows short enough to afford its quadratic tensors."""
    import torch
    from adaptation import gpu_core, inkling_attention

    model, load = load_model()
    lm = gpu_core.causal_lm(model)
    rows = gpu_core.read_jsonl(g1_jsonl)

    def per_token(row):
        tokens = torch.tensor(row["tokens"], device="cuda")
        w = torch.tensor(row["weights"], device="cuda")
        with torch.no_grad():
            hidden = lm.model(input_ids=tokens[None, :-1], use_cache=False).last_hidden_state[0]
            pos = torch.nonzero(w > 0).squeeze(-1)
            out = []
            for chunk in pos.split(2048):
                logits = lm.logit_softcap(lm.lm_head(hidden[chunk]).float())
                out.append(torch.log_softmax(logits, -1).gather(-1, tokens[chunk + 1, None]).squeeze(-1))
        return pos.tolist(), torch.cat(out).tolist()

    result = {"load": load, "rows": {}}
    for row in rows:
        pos, block = per_token(row)
        entry = {"positions": pos, "blockwise": block}
        if len(row["tokens"]) <= reference_max_tokens:
            inkling_attention.uninstall()
            _, entry["reference"] = per_token(row)
            inkling_attention.install(block_size=1024)
        result["rows"][row["task"]] = entry
        print(row["task"], len(pos), "reference" in entry, flush=True)
    return result


@app.function(gpu="H100", volumes={"/cache": cache}, timeout=20 * 60)
def memprobe(train_jsonl: str, targets: str = "qkvo", lengths: str = "1024,2048,4096,8192,0",
             block_size: int = 1024, chunk: int = 2048):
    """Where does training memory go? Peak after forward and after backward for one row truncated to several
    lengths, with gradient checkpointing on; OOM is caught per length."""
    import torch
    from adaptation import gpu_core

    model, load = load_model(block_size=block_size)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model, lora = gpu_core.add_lora(model, targets)
    model.train()
    layers = gpu_core.causal_lm(model).model.layers
    out = {"load": load, "lora": lora, "checkpointing_layers": sum(bool(getattr(l, "gradient_checkpointing", False))
                                                                  for l in layers), "lengths": {}}
    row = max(gpu_core.read_jsonl(train_jsonl), key=lambda r: len(r["tokens"]))
    base_mem = torch.cuda.memory_allocated()
    for length in [int(x) or len(row["tokens"]) for x in lengths.split(",")]:
        # every position supervised: worst case for the loss chunks, and a prompt-only prefix still has a loss
        cut = {"task": row["task"], "tokens": row["tokens"][:length], "weights": [1.0] * (length - 1)}
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        entry = {}
        try:
            t0 = time.time()
            nll, _ = gpu_core.weighted_nll(model, cut, chunk=chunk, grad=True)
            entry["after_forward_gb"] = round((torch.cuda.memory_allocated() - base_mem) / 1e9, 2)
            entry["forward_peak_gb"] = round((torch.cuda.max_memory_allocated() - base_mem) / 1e9, 2)
            nll.backward()
            entry["backward_peak_gb"] = round((torch.cuda.max_memory_allocated() - base_mem) / 1e9, 2)
            entry["seconds"] = round(time.time() - t0, 1)
        except torch.cuda.OutOfMemoryError:
            entry["oom"] = True
            entry["peak_gb"] = round((torch.cuda.max_memory_allocated() - base_mem) / 1e9, 2)
        nll = None
        out["lengths"][length] = entry
        print(length, json.dumps(entry), flush=True)
    return out


# H200: training keeps ~2.2 MB of activations per token (memprobe); 11.8k-token rows need ~36 GB on top of 52 GB
@app.function(gpu="H200", volumes={"/cache": cache, "/runs": runs}, timeout=75 * 60)
def sft(train_jsonl: str, g1_jsonl: str, targets: str, lr: float, epochs: int, batch_size: int, rank: int,
        alpha: int, name: str, val_jsonl: str = "", evals_per_epoch: int = 0):
    """G2: train like Tinker SFT-3, then the training-effect gate on the G1 rows; save a PEFT adapter."""
    import torch
    from adaptation import gpu_core

    model, load = load_model(block_size=512)  # smaller blocks bound the transient memory on 30k-token rows
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    rows, g1_rows = gpu_core.read_jsonl(train_jsonl), gpu_core.read_jsonl(g1_jsonl)
    val_rows = gpu_core.read_jsonl(val_jsonl) if val_jsonl else []
    base = gpu_core.evaluate(model, g1_rows)
    base_val = gpu_core.evaluate(model, val_rows) if val_rows else {}
    model, lora = gpu_core.add_lora(model, targets, rank, alpha)
    print(json.dumps({"lora": lora, "base": base}), flush=True)

    def on_step(step, epoch, steps_per_epoch):
        if not val_rows or not evals_per_epoch:
            return None
        every = max(1, steps_per_epoch // evals_per_epoch)
        if step % every and step % steps_per_epoch:
            return None
        val = gpu_core.evaluate(model, val_rows)
        return {"val_nll": round(sum(val.values()) / len(val), 4)}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    history = gpu_core.train(model, rows, lr, epochs, batch_size, chunk=1024, on_step=on_step)
    tuned = gpu_core.evaluate(model, g1_rows)
    trained = [r["task"] for r in g1_rows if r["group"] == "trained"]
    held = [r["task"] for r in g1_rows if r["group"] == "held_out"]
    result = {"load": load, "lora": lora, "config": {"targets": targets, "lr": lr, "epochs": epochs,
              "batch_size": batch_size, "rank": rank, "alpha": alpha},
              "train_seconds": round(time.time() - t0), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
              "history": history, "base": base, "tuned": tuned,
              "gate": gpu_core.gate(base, tuned, trained, held),
              "tinker_sft3": {r["task"]: {"base": r.get("tinker_base"), "sft3": r.get("tinker_sft3")} for r in g1_rows},
              "tinker_gate": {"trained_relative_drop": 0.1276, "held_out_relative_rise": 0.0485}}
    if val_rows:
        result["val"] = {"base": base_val, "tuned": gpu_core.evaluate(model, val_rows)}
    out = Path("/runs") / name
    model.save_pretrained(out)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    runs.commit()
    files = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    return result, files


@app.function(gpu="B200", volumes={"/cache": cache, "/runs": runs}, timeout=90 * 60)
def distill(train_jsonl: str, heldout_jsonl: str, name: str, lr: float = 1e-4, epochs: int = 1,
            batch_size: int = 8, cap: int = 36000, rank: int = 16, alpha: int = 32):
    """On-policy distillation round: per-token reverse KL to precomputed teacher log-probs (Inkling 2 Small).
    Gate: the held-out reverse-KL estimate (student - teacher log-prob on the sampled tokens) must fall."""
    import torch
    from adaptation import gpu_core

    def load(text):
        rows = []
        for r in gpu_core.read_jsonl(text):
            keep = [i for i, p in enumerate(r["positions"]) if p + 1 < cap]
            rows.append({"task": r["task"], "tokens": r["tokens"][:cap], "positions": [r["positions"][i] for i in keep],
                         "teacher_logprobs": [r["teacher_logprobs"][i] for i in keep]})
        return [r for r in rows if r["positions"]]

    model, load_info = load_model(block_size=512)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    rows, held = load(train_jsonl), load(heldout_jsonl)
    model, lora = gpu_core.add_lora(model, "qkvo", rank, alpha)
    teacher_held = [torch.tensor(r["teacher_logprobs"], device="cuda") for r in held]
    teacher_train = [torch.tensor(r["teacher_logprobs"], device="cuda") for r in rows[:12]]
    kl_before = {"heldout": gpu_core.reverse_kl(model, held, teacher_held, chunk=1024),
                 "train_sample": gpu_core.reverse_kl(model, rows[:12], teacher_train, chunk=1024)}
    print(json.dumps({"lora": lora, "kl_before": kl_before}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    history = gpu_core.distill_train(model, rows, lr, epochs, batch_size, chunk=1024)
    kl_after = {"heldout": gpu_core.reverse_kl(model, held, teacher_held, chunk=1024),
                "train_sample": gpu_core.reverse_kl(model, rows[:12], teacher_train, chunk=1024)}
    result = {"load": load_info, "lora": lora, "rows": len(rows), "heldout_rows": len(held),
              "config": {"lr": lr, "epochs": epochs, "batch_size": batch_size, "cap": cap, "rank": rank},
              "kl_before": kl_before, "kl_after": kl_after, "history": history,
              "train_seconds": round(time.time() - t0), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    out = Path("/runs") / name
    model.save_pretrained(out)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    runs.commit()
    return result, {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}


@app.local_entrypoint()
def main(job: str, targets: str = "qkvo", lr: float = 5.4e-4, epochs: int = 2, batch_size: int = 4,
         rank: int = 16, alpha: int = 32, name: str = "", val: str = "", evals_per_epoch: int = 0,
         gpu: str = "H200", train: str = "", gate: str = ""):
    if os.environ.get("INKLING_ALLOW_MODAL") != job:
        raise PermissionError(f"paid Modal job '{job}' needs its own approval: INKLING_ALLOW_MODAL={job}")
    manifest = json.loads((RUN / "manifest.json").read_text())
    if job == "download":
        print(download.remote())
        job = "verify"
    if job == "verify":
        result = verify.remote(WEIGHTS_SHA256, WEIGHTS_SIZE)
        print(json.dumps(result))
        if not result["ok"]:
            raise RuntimeError("weights in the Volume do not match the Hub file")
    elif job == "g1":
        result = g1.remote((RUN / "g1.jsonl").read_text())
        (RUN / "g1-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}), "-> runs/gpu-phase1/g1-result.json")
    elif job == "distill":
        d = Path(train)  # directory with train-teacher.jsonl and heldout-teacher.jsonl
        result, files = distill.remote((d / "train-teacher.jsonl").read_text(), (d / "heldout-teacher.jsonl").read_text(),
                                       name or "distill-r1", lr=lr)
        out = RUN / (name or "distill-r1")
        out.mkdir(parents=True, exist_ok=True)
        for fname, data in files.items():
            (out / fname).write_bytes(data)
        print(json.dumps({"kl_before": result["kl_before"], "kl_after": result["kl_after"], "peak_gb": result["peak_gb"],
                          "train_seconds": result["train_seconds"]}), f"-> {out}")
    elif job == "memprobe":
        result = memprobe.with_options(gpu=gpu).remote(Path(train).read_text() if train else (RUN / "sft3-train.jsonl").read_text(),
                                 targets, lengths="16384,24576,0", block_size=512, chunk=1024)
        (RUN / "memprobe.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["lengths"]), "checkpointing layers:", result["checkpointing_layers"])
    elif job == "g1_tokens":
        result = g1_tokens.remote((RUN / "g1.jsonl").read_text())
        (RUN / "g1-tokens-gpu.json").write_text(json.dumps(result))
        print("-> runs/gpu-phase1/g1-tokens-gpu.json")
    elif job == "sft":
        if not (RUN / "g1-result.json").exists() or not json.loads((RUN / "g1-result.json").read_text())["g1_passed"]:
            raise RuntimeError("G1 must pass before any GPU training")
        name = name or f"sft3-repro-{targets}-lr{lr:g}-e{epochs}"
        train_path = Path(train) if train else RUN / "sft3-train.jsonl"
        gate_path = Path(gate) if gate else RUN / "g1.jsonl"
        result, files = sft.with_options(gpu=gpu).remote(train_path.read_text(), gate_path.read_text(), targets,
                                   lr, epochs, batch_size, rank, alpha, name,
                                   Path(val).read_text() if val else "", evals_per_epoch)
        out = RUN / name
        out.mkdir(parents=True, exist_ok=True)
        for fname, data in files.items():
            (out / fname).write_bytes(data)
        print(json.dumps({"gate": result["gate"], "tinker_gate": result["tinker_gate"], "peak_gb": result["peak_gb"],
                          "train_seconds": result["train_seconds"]}), f"-> {out}")
    else:
        raise ValueError(job)
    print("inputs:", {k: manifest[k] for k in ("g1.jsonl", "sft3-train.jsonl")})
