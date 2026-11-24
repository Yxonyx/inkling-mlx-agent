"""Memory-bounded attention for transformers' Inkling2-loop (5.24) on long single sequences.

Why: the reference implementation only has eager attention and materialises several (query x key) tensors
per layer (relative distance, additive mask, position bias, scores, fp32 softmax). At 16k tokens that is
tens of GB per layer visit; 40k-token rows do not fit on any single GPU.

What: when no cache is used, the full and sliding attention layers compute attention in query blocks.
Each block builds only its own (block x keys) bias, mask and scores, in the same dtype order as the eager
reference, and is wrapped in activation checkpointing during training, so peak memory is bounded by the
block size. With a cache (generation) the original forward runs unchanged.

IMPORTANT: call the model with use_cache=False; otherwise transformers creates a cache and the original
quadratic path runs silently. Check CALLS[0] > 0 after a forward.

Use:  from adaptation import inkling_attention; inkling_attention.install(block_size=1024)
Check: adaptation/test_inkling_attention.py compares logits and LoRA gradients with the reference.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_ORIGINAL = {}
BLOCK_SIZE = 1024
CALLS = [0]  # blockwise layer calls, so callers can prove the patch ran (pass use_cache=False!)


def _block(q, k, v, rel_states, proj, sinks, q_start, k_start, scaling, window, extent):
    """q [B,H,bq,D]; k,v [B,H,bk,D]; rel_states [B,bq,H,R]; sinks [B,H,bq,1] -> [B,H,bq,D]."""
    bq, bk = q.shape[-2], k.shape[-2]
    distance = (torch.arange(q_start, q_start + bq, device=q.device)[:, None]
                - torch.arange(k_start, k_start + bk, device=q.device)[None, :])
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    rel_logits = (rel_states @ proj).transpose(1, 2)  # [B,H,bq,extent]
    bias = rel_logits.gather(-1, distance.clamp(0, extent - 1).expand(*rel_logits.shape[:2], -1, -1))
    scores = scores + bias.masked_fill((distance < 0) | (distance >= extent), 0.0)
    blocked = distance < 0
    if window is not None:
        blocked = blocked | (distance >= window)
    scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
    probs = F.softmax(torch.cat([scores, sinks], dim=-1).float(), dim=-1)[..., :-1].to(q.dtype)
    return torch.matmul(probs, v)


def blockwise_attention(module, hidden_states, key_hidden_states, value_hidden_states, conv_mask, kwargs):
    from transformers.models.inkling2_loop.modeling_inkling2_loop import repeat_kv

    CALLS[0] += 1
    position_ids = kwargs["position_ids"]
    q = module.q_proj(hidden_states, position_ids)
    k = module.k_proj(key_hidden_states, cache=None, conv_mask=conv_mask, **kwargs)
    v = module.v_proj(value_hidden_states, cache=None, conv_mask=conv_mask, **kwargs)
    k, v = repeat_kv(k, module.num_key_value_groups), repeat_kv(v, module.num_key_value_groups)
    rel = module.relative_position_bias
    rel_states = rel.norm(rel.linear(hidden_states).view(*hidden_states.shape[:-1], rel.num_heads, -1), position_ids)
    sinks = module.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)
    window, extent, total = module.sliding_window, rel.rel_extent, q.shape[-2]
    use_checkpoint = module.training and torch.is_grad_enabled()
    outputs = []
    for start in range(0, total, BLOCK_SIZE):
        end = min(total, start + BLOCK_SIZE)
        k_start = 0 if window is None else max(0, start - window + 1)
        args = (q[:, :, start:end], k[:, :, k_start:end], v[:, :, k_start:end], rel_states[:, start:end], rel.proj,
                sinks[:, :, start:end], start, k_start, module.scaling, window, extent)
        outputs.append(checkpoint(_block, *args, use_reentrant=False) if use_checkpoint else _block(*args))
    attn_output = torch.cat(outputs, dim=2).transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
    return module.o_proj(attn_output, hidden_states), None


def _sliding_forward(self, hidden_states, attention_mask, relative_distance, conv_mask=None, cache=None,
                     key_hidden_states=None, value_hidden_states=None, cache_offset=0, **kwargs):
    if cache is not None:
        return _ORIGINAL["sliding"](self, hidden_states, attention_mask, relative_distance, conv_mask=conv_mask,
                                    cache=cache, key_hidden_states=key_hidden_states,
                                    value_hidden_states=value_hidden_states, cache_offset=cache_offset, **kwargs)
    _check_unpadded(attention_mask)
    key_hidden_states = hidden_states if key_hidden_states is None else key_hidden_states
    value_hidden_states = hidden_states if value_hidden_states is None else value_hidden_states
    return blockwise_attention(self, hidden_states, key_hidden_states, value_hidden_states, conv_mask, kwargs)


def _full_forward(self, hidden_states, key_hidden_states, value_hidden_states, attention_mask, relative_distance,
                  conv_mask=None, cache=None, cache_offset=0, **kwargs):
    if cache is not None:
        return _ORIGINAL["full"](self, hidden_states, key_hidden_states, value_hidden_states, attention_mask,
                                 relative_distance, conv_mask=conv_mask, cache=cache, cache_offset=cache_offset,
                                 **kwargs)
    _check_unpadded(attention_mask)
    return blockwise_attention(self, hidden_states, key_hidden_states, value_hidden_states, conv_mask, kwargs)


def _check_unpadded(attention_mask):
    if attention_mask is not None:
        raise ValueError("blockwise Inkling attention supports unpadded sequences only (pass attention_mask=None)")


def _no_quadratic_tensor(original):
    """Skip building (query x key) masks and distances when no cache is used; the blockwise path builds its own."""
    def wrapper(*args, **kwargs):
        if kwargs.get("cache") is None and (len(args) < 2 or args[1] is None):
            if kwargs.get("attention_mask") is not None:
                raise ValueError("blockwise Inkling attention supports unpadded sequences only")
            return None
        return original(*args, **kwargs)
    return wrapper


def install(block_size=1024):
    """Patch the module in place; idempotent."""
    global BLOCK_SIZE
    from transformers.models.inkling2_loop import modeling_inkling2_loop as m

    BLOCK_SIZE = int(block_size)
    if _ORIGINAL:
        return
    _ORIGINAL.update(sliding=m.Inkling2LoopSlidingAttention.forward, full=m.Inkling2LoopFullAttention.forward,
                     causal=m.create_causal_mask, sliding_mask=m.create_sliding_window_causal_mask,
                     distance=m.get_relative_distance)
    m.Inkling2LoopSlidingAttention.forward = _sliding_forward
    m.Inkling2LoopFullAttention.forward = _full_forward
    m.create_causal_mask = _no_quadratic_tensor(_ORIGINAL["causal"])
    m.create_sliding_window_causal_mask = _no_quadratic_tensor(_ORIGINAL["sliding_mask"])
    m.get_relative_distance = lambda inputs_embeds, cache, is_sliding: (
        None if cache is None else _ORIGINAL["distance"](inputs_embeds, cache, is_sliding))


def uninstall():
    from transformers.models.inkling2_loop import modeling_inkling2_loop as m

    if not _ORIGINAL:
        return
    m.Inkling2LoopSlidingAttention.forward = _ORIGINAL["sliding"]
    m.Inkling2LoopFullAttention.forward = _ORIGINAL["full"]
    m.create_causal_mask, m.create_sliding_window_causal_mask = _ORIGINAL["causal"], _ORIGINAL["sliding_mask"]
    m.get_relative_distance = _ORIGINAL["distance"]
    _ORIGINAL.clear()
