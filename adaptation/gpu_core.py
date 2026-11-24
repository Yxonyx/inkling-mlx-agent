"""Model-side logic shared by the GPU jobs (Modal) and the local tiny-model tests.

Everything here works on the exact token ids and loss weights exported by adaptation/gpu_export.py, so the
GPU sees the same sequences Tinker saw. Memory stays bounded on long rows: blockwise attention
(adaptation/inkling_attention.py), gradient checkpointing per decoder layer, and the loss is computed in
checkpointed vocabulary chunks that skip unsupervised positions.

Training mirrors Tinker's SFT-3 call sequence: one cross_entropy loss summed over weighted tokens per batch,
Adam(beta1 0.9, beta2 0.95, eps 1e-12, no weight decay, no clipping), random.Random(seed).shuffle(order) once
per epoch, LoRA rank 16 / alpha 32 on attention.
"""
import json, math, random, time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# transformers 5.24 module names (official checkpoint name in brackets)
LORA_TARGETS = {
    # query / key / value / output projections only (wq_du, wk_dv, wv_dv, wo_ud)
    "qkvo": r".*self_attn\.(q_proj|k_proj|v_proj|o_proj)\.linear",
    # every attention linear (adds gate_attn, wr_du, attn_sink_proj)
    "attn_all": r".*self_attn\.((q_proj|k_proj|v_proj|o_proj)\.linear|o_proj\.gate_proj|relative_position_bias\.linear|sink_proj)",
}


def read_jsonl(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def causal_lm(model):
    """The Inkling2LoopForCausalLM inside an optional PEFT wrapper."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(inner, "lm_head"):
        raise TypeError(f"expected Inkling2LoopForCausalLM, got {type(inner).__name__}")
    return inner


def _chunk_nll(hidden, targets, weights, lm_head, softcap):
    logits = softcap(lm_head(hidden).float())
    logp = torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
    return -(logp * weights).sum()


def weighted_nll(model, row, chunk=2048, grad=False):
    """Return (sum of weighted NLL, sum of weights) for one exported row."""
    lm = causal_lm(model)
    device = lm.lm_head.weight.device
    tokens = torch.tensor(row["tokens"], device=device)
    weights = torch.tensor(row["weights"], device=device, dtype=torch.float32)
    if len(weights) != len(tokens) - 1:
        raise ValueError(f"{row['task']}: weights must have len(tokens) - 1 entries")
    with torch.set_grad_enabled(grad):
        hidden = lm.model(input_ids=tokens[None, :-1], use_cache=False).last_hidden_state[0]
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, len(weights), chunk):
            w = weights[start:start + chunk]
            if not w.any():
                continue
            args = (hidden[start:start + chunk], tokens[start + 1:start + 1 + chunk], w, lm.lm_head, lm.logit_softcap)
            total = total + (checkpoint(_chunk_nll, *args, use_reentrant=False) if grad else _chunk_nll(*args))
    return total, float(weights.sum())


def evaluate(model, rows, chunk=2048):
    model.eval()
    out = {}
    for row in rows:
        nll, w = weighted_nll(model, row, chunk)
        out[row["task"]] = round(nll.item() / w, 4)
    return out


def add_lora(model, targets="qkvo", rank=16, alpha=32, seed=0):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=LORA_TARGETS[targets])
    model = get_peft_model(model, config)  # PEFT keeps adapter weights in fp32 (autocast_adapter_dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wrapped = sorted({n.rsplit(".lora_A", 1)[0] for n, _ in model.named_parameters() if ".lora_A" in n})
    if not wrapped:
        raise ValueError("LoRA wrapped no modules")
    return model, {"trainable_params": trainable, "wrapped_modules": len(wrapped), "example": wrapped[:3]}


def train(model, rows, lr, epochs, batch_size, seed=0, chunk=2048, on_step=None):
    """Tinker-equivalent SFT loop. on_step(step, epoch, record) may run evaluations and return extra fields."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-12, weight_decay=0.0)
    order, rng, history, step = list(range(len(rows))), random.Random(seed), [], 0
    steps_per_epoch = math.ceil(len(rows) / batch_size)
    for epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss_sum, weight_sum, t0 = 0.0, 0.0, time.time()
            for i in order[start:start + batch_size]:
                nll, w = weighted_nll(model, rows[i], chunk, grad=True)
                nll.backward()  # summed over the batch, as Tinker's cross_entropy loss:sum
                loss_sum, weight_sum = loss_sum + nll.item(), weight_sum + w
            grad_norm = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])).item()
            opt.step()
            step += 1
            record = {"step": step, "epoch": epoch, "batch": start // batch_size, "loss_sum": round(loss_sum, 3),
                      "nll_per_token": round(loss_sum / weight_sum, 4), "grad_norm": round(grad_norm, 4),
                      "seconds": round(time.time() - t0, 1)}
            if on_step:
                record.update(on_step(step, epoch, steps_per_epoch) or {})
            history.append(record)
            print(json.dumps(record), flush=True)
    return history


def gate(base, tuned, trained_tasks, held_tasks):
    """Training-effect gate (same rule as SFT-3): trained NLL down >= 10 %, held-out up <= 5 %."""
    mean = lambda d, ts: sum(d[t] for t in ts) / len(ts)
    drop = 1 - mean(tuned, trained_tasks) / mean(base, trained_tasks)
    rise = mean(tuned, held_tasks) / mean(base, held_tasks) - 1
    return {"trained_relative_drop": round(drop, 4), "held_out_relative_rise": round(rise, 4),
            "gate_passed": drop >= 0.10 and rise <= 0.05}


def sequence_logprob(model, row, chunk=2048, grad=False):
    """Sum of weighted log-probabilities of the supervised targets (= -weighted NLL)."""
    nll, _ = weighted_nll(model, row, chunk, grad=grad)
    return -nll


def reference_logprobs(model, rows, chunk=2048):
    """Log-probs under the frozen base: the PEFT adapter disabled (no copy of the model needed)."""
    model.eval()
    with model.disable_adapter(), torch.no_grad():
        return [sequence_logprob(model, r, chunk).item() for r in rows]


def dpo_train(model, pairs, beta, lr, epochs, batch_size, seed=0, chunk=2048, sft_alpha=0.0, on_step=None):
    """DPO on (chosen, rejected, ref_chosen, ref_rejected) pairs with bounded memory.

    L = -log sigmoid(beta * delta), delta = (lp_c - ref_c) - (lp_r - ref_r); dL/dtheta = s * (grad lp_c - grad lp_r)
    with s = -beta * sigmoid(-beta * delta). delta is computed without a graph, then each sequence is
    back-propagated on its own with the detached coefficient, so only one sequence is ever in the graph.
    Optional sft_alpha adds alpha * (-lp_c / tokens_c) (keeps the chosen likelihood from collapsing).
    Adam(0.9, 0.95, 1e-12) as in train(); the learning rate decays linearly to 0 (cookbook DPO schedule).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-12, weight_decay=0.0)
    order, rng, history, step = list(range(len(pairs))), random.Random(seed), [], 0
    total = math.ceil(len(pairs) / batch_size) * epochs
    for epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            for group in opt.param_groups:
                group["lr"] = lr * (1 - step / total)
            model.train()
            opt.zero_grad(set_to_none=True)
            batch = [pairs[i] for i in order[start:start + batch_size]]
            losses, accs, margins = [], [], []
            for chosen, rejected, ref_c, ref_r in batch:
                loss, delta = dpo_backward(model, chosen, rejected, ref_c, ref_r, beta, 1 / len(batch), chunk,
                                           sft_alpha)
                losses.append(loss); accs.append(float(delta > 0)); margins.append(beta * delta)
            grad_norm = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])).item()
            opt.step()
            step += 1
            record = {"step": step, "epoch": epoch, "dpo_loss": round(sum(losses) / len(losses), 4),
                      "accuracy": round(sum(accs) / len(accs), 3), "margin": round(sum(margins) / len(margins), 4),
                      "grad_norm": round(grad_norm, 4), "lr": round(lr * (1 - (step - 1) / total), 9)}
            if on_step:
                record.update(on_step(step, epoch, total // max(epochs, 1)) or {})
            history.append(record)
            print(json.dumps(record), flush=True)
    return history


def dpo_backward(model, chosen, rejected, ref_c, ref_r, beta, scale=1.0, chunk=2048, sft_alpha=0.0):
    """Accumulate scale * grad of the DPO loss for one pair, one sequence in the graph at a time."""
    with torch.no_grad():
        lp_c = sequence_logprob(model, chosen, chunk).item()
        lp_r = sequence_logprob(model, rejected, chunk).item()
    delta = (lp_c - ref_c) - (lp_r - ref_r)
    coef = -beta * torch.sigmoid(torch.tensor(-beta * delta)).item() * scale
    term = coef * sequence_logprob(model, chosen, chunk, grad=True)
    if sft_alpha:
        n_c = sum(1 for w in chosen["weights"] if w)
        term = term - sft_alpha * scale * sequence_logprob(model, chosen, chunk, grad=True) / n_c
    term.backward()
    (-coef * sequence_logprob(model, rejected, chunk, grad=True)).backward()
    return -torch.nn.functional.logsigmoid(torch.tensor(beta * delta)).item(), delta


def pair_margins(model, pairs, beta, chunk=2048):
    """Held-out check: beta * delta per pair and the accuracy (delta > 0)."""
    model.eval()
    out = []
    with torch.no_grad():
        for chosen, rejected, ref_c, ref_r in pairs:
            delta = (sequence_logprob(model, chosen, chunk).item() - ref_c) - (sequence_logprob(model, rejected, chunk).item() - ref_r)
            out.append(beta * delta)
    return {"mean_margin": round(sum(out) / max(len(out), 1), 4), "accuracy": round(sum(m > 0 for m in out) / max(len(out), 1), 3)}


# ── On-policy distillation (per-token reverse KL, Thinking Machines recipe) ─────────────────────────────

def _chunk_token_logprobs(hidden, targets, lm_head, softcap):
    logits = softcap(lm_head(hidden).float())
    return torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).squeeze(-1)


def token_logprobs(model, tokens, positions, chunk=2048):
    """Log-probs of tokens[p + 1] given tokens[:p + 1] for the given positions (no grad)."""
    lm = causal_lm(model)
    device = lm.lm_head.weight.device
    ids = torch.tensor(tokens, device=device)
    pos = torch.tensor(positions, device=device)
    with torch.no_grad():
        hidden = lm.model(input_ids=ids[None, :-1], use_cache=False).last_hidden_state[0]
        out = [_chunk_token_logprobs(hidden[p], ids[p + 1], lm.lm_head, lm.logit_softcap) for p in pos.split(chunk)]
    return torch.cat(out)


def _chunk_is_loss(hidden, targets, advantages, old_logprobs, lm_head, softcap):
    logp = _chunk_token_logprobs(hidden, targets, lm_head, softcap)
    return -(advantages * torch.exp(logp - old_logprobs)).sum()


def distill_loss(model, row, advantages, old_logprobs, chunk=2048):
    """Importance-sampled policy-gradient loss with per-token advantages on the supervised positions:
    L = -sum_t A_t * exp(lp_theta,t - lp_old,t); its gradient at lp_theta = lp_old is -sum_t A_t grad lp_t."""
    lm = causal_lm(model)
    device = lm.lm_head.weight.device
    ids = torch.tensor(row["tokens"], device=device)
    pos = torch.tensor(row["positions"], device=device)
    hidden = lm.model(input_ids=ids[None, :-1], use_cache=False).last_hidden_state[0]
    total = hidden.new_zeros((), dtype=torch.float32)
    for p, a, o in zip(pos.split(chunk), advantages.split(chunk), old_logprobs.split(chunk)):
        total = total + checkpoint(_chunk_is_loss, hidden[p], ids[p + 1], a, o, lm.lm_head, lm.logit_softcap,
                                   use_reentrant=False)
    return total


def teacher_logprobs_self(model, row, chunk=2048):
    """Self-teacher: the base weights (adapter disabled) on the teacher rendering (e.g. effort 0.9)."""
    model.eval()
    with model.disable_adapter():
        return token_logprobs(model, row["teacher_tokens"], row["positions"], chunk)


def distill_train(model, rows, lr, epochs, batch_size, seed=0, chunk=2048, clip=5.0, on_step=None):
    """On-policy distillation: A_t = lp_teacher,t - lp_student_old,t (negative per-token reverse KL estimate).

    rows: {"task", "tokens" (student rendering), "positions" (supervised positions: sampled assistant tokens),
           and either "teacher_logprobs" (precomputed, e.g. Inkling 2 Small via Tinker) or "teacher_tokens"
           (self-teacher rendering, scored here with the adapter disabled)}.
    The student-old log-probs are computed once per epoch with the current adapter (the behaviour policy of
    rollouts sampled from it). Advantages are clipped to [-clip, clip]. Loss is token-mean over the batch.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-12, weight_decay=0.0)
    teacher = []
    for row in rows:
        t = row.get("teacher_logprobs")
        teacher.append(torch.tensor(t, dtype=torch.float32, device=causal_lm(model).lm_head.weight.device)
                       if t is not None else teacher_logprobs_self(model, row, chunk))
    order, rng, history, step = list(range(len(rows))), random.Random(seed), [], 0
    for epoch in range(epochs):
        model.eval()
        old = [token_logprobs(model, r["tokens"], r["positions"], chunk) for r in rows]
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            model.train()
            opt.zero_grad(set_to_none=True)
            idx = order[start:start + batch_size]
            n_tokens = sum(len(rows[i]["positions"]) for i in idx)
            kl_sum = adv_sum = 0.0
            for i in idx:
                adv = (teacher[i] - old[i]).clamp(-clip, clip)
                loss = distill_loss(model, rows[i], adv, old[i], chunk) / n_tokens
                loss.backward()
                kl_sum += float((old[i] - teacher[i]).sum()); adv_sum += float(adv.sum())
            grad_norm = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])).item()
            opt.step()
            step += 1
            record = {"step": step, "epoch": epoch, "reverse_kl_per_token": round(kl_sum / n_tokens, 5),
                      "mean_advantage": round(adv_sum / n_tokens, 5), "grad_norm": round(grad_norm, 4)}
            if on_step:
                record.update(on_step(step, epoch) or {})
            history.append(record)
            print(json.dumps(record), flush=True)
    return history


def reverse_kl(model, rows, teacher, chunk=2048):
    """Mean per-token reverse-KL estimate (student - teacher log-prob on the sampled tokens)."""
    model.eval()
    total = n = 0.0
    for row, t in zip(rows, teacher):
        s = token_logprobs(model, row["tokens"], row["positions"], chunk)
        total += float((s - t).sum()); n += len(row["positions"])
    return total / max(n, 1)
