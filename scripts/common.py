"""Shared Phase 1 machinery: capture loading, pre-RoPE KV recompute,
cache surgery, two-pass behavioral evaluation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.cache_utils import DynamicCache


def load_model(name="Qwen/Qwen2.5-0.5B"):
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_capture(path):
    cap = torch.load(path)
    cap["hidden"] = cap["hidden"].float()    # (L+1, T, D)
    cap["keys"] = cap["keys"].float()        # (L, kvh, T, hd) post-RoPE
    cap["values"] = cap["values"].float()
    return cap


def pre_rope_kv(model, hidden):
    """Pre-RoPE K and V for all layers from the hidden trajectory.
    hidden: (L+1, T, D). Returns k_pre, v: (L, kvh, T, hd)."""
    cfg = model.config
    kvh = cfg.num_key_value_heads
    hd = cfg.hidden_size // cfg.num_attention_heads
    T = hidden.shape[1]
    ks, vs = [], []
    with torch.no_grad():
        for l, layer in enumerate(model.model.layers):
            x = layer.input_layernorm(hidden[l].unsqueeze(0))
            k = layer.self_attn.k_proj(x).view(1, T, kvh, hd).transpose(1, 2)
            v = layer.self_attn.v_proj(x).view(1, T, kvh, hd).transpose(1, 2)
            ks.append(k.squeeze(0))
            vs.append(v.squeeze(0))
    return torch.stack(ks), torch.stack(vs)


def rope_k(model, k_pre, position_ids):
    """Apply RoPE to pre-RoPE keys. k_pre: (L, kvh, T, hd)."""
    with torch.no_grad():
        dummy = torch.zeros(1, k_pre.shape[2], model.config.hidden_size)
        cos, sin = model.model.rotary_emb(dummy, position_ids)
        out = []
        for l in range(k_pre.shape[0]):
            _, k = apply_rotary_pos_emb(k_pre[l].unsqueeze(0), k_pre[l].unsqueeze(0), cos, sin)
            out.append(k.squeeze(0))
    return torch.stack(out)


def build_cache(keys, values, past_len):
    """Legacy-format cache from (L, kvh, T, hd) tensors, truncated to past_len."""
    legacy = tuple(
        (keys[l, :, :past_len].unsqueeze(0).contiguous(),
         values[l, :, :past_len].unsqueeze(0).contiguous())
        for l in range(keys.shape[0])
    )
    return DynamicCache.from_legacy_cache(legacy)


@torch.no_grad()
def eval_window(model, ids, keys, values, past_len):
    """Feed tokens [past_len:] with the given (possibly compressed) cache as
    past. Returns logits over the window. keys/values are post-RoPE (L, kvh, T, hd)."""
    T = ids.shape[1]
    cache = build_cache(keys, values, past_len)
    out = model(
        input_ids=ids[:, past_len:],
        past_key_values=cache,
        attention_mask=torch.ones(1, T, dtype=torch.long),
        position_ids=torch.arange(past_len, T).unsqueeze(0),
        use_cache=False,
    )
    return out.logits.squeeze(0).float()   # (T - past_len, vocab)


def behavior_metrics(base_logits, comp_logits, target_ids):
    """base/comp: (W, vocab). target_ids: (W-1,) true next tokens for the first
    W-1 window positions."""
    base_lp = torch.log_softmax(base_logits, dim=-1)
    comp_lp = torch.log_softmax(comp_logits, dim=-1)
    kl = (base_lp.exp() * (base_lp - comp_lp)).sum(-1)
    top1 = (base_logits.argmax(-1) == comp_logits.argmax(-1)).float()
    b5 = base_logits.topk(5, dim=-1).indices
    c5 = comp_logits.topk(5, dim=-1).indices
    top5_overlap = torch.tensor([
        len(set(b5[i].tolist()) & set(c5[i].tolist())) / 5.0 for i in range(b5.shape[0])
    ])
    n = target_ids.shape[0]
    base_nll = -base_lp[:n].gather(1, target_ids.unsqueeze(1)).squeeze(1)
    comp_nll = -comp_lp[:n].gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return {
        "kl_mean": kl.mean().item(),
        "kl_p95": kl.quantile(0.95).item(),
        "top1_agree": top1.mean().item(),
        "top5_overlap": top5_overlap.mean().item(),
        "base_nll": base_nll.mean().item(),
        "delta_nll": (comp_nll - base_nll).mean().item(),
    }
