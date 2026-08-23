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


@torch.no_grad()
def prefill_doc(model, ids):
    """Full prefill returning hidden trajectory, post-RoPE KV, and logits."""
    out = model(ids, output_hidden_states=True, use_cache=True)
    pkv = out.past_key_values
    legacy = pkv.to_legacy_cache() if hasattr(pkv, "to_legacy_cache") else pkv
    keys = torch.stack([kv[0].squeeze(0) for kv in legacy]).float()
    values = torch.stack([kv[1].squeeze(0) for kv in legacy]).float()
    hidden = torch.stack([h.squeeze(0) for h in out.hidden_states]).float()
    return hidden, keys, values, out.logits.squeeze(0).float()


# ---------------- analytic codecs (shared by A5/A6/A7 scripts) ----------------

def qsim(x, bits, dim):
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / qmax
    return (x / scale).round().clamp(-qmax, qmax) * scale


def pca_fit(X, r):
    mu = X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X - mu, full_matrices=False)
    return mu, Vh[:r].T


def stack_flat(k_pre, v_pre):
    """(L, kvh, T, hd) x2 -> (T, 2*L*kvh*hd)"""
    L, kvh, T, hd = k_pre.shape
    f = lambda x: x.permute(2, 0, 1, 3).reshape(T, L * kvh * hd)
    return torch.cat([f(k_pre), f(v_pre)], dim=1)


def stack_unflat(X, L, kvh, hd):
    T = X.shape[0]
    half = L * kvh * hd
    g = lambda x: x.reshape(T, L, kvh, hd).permute(1, 2, 0, 3).contiguous()
    return g(X[:, :half]), g(X[:, half:])


def fit_joint(k_pre, v_pre, fit_len, R, scale=None):
    """scale: optional (1, D) tensor replacing the variance normalizer -- PCA
    then minimizes error under the metric ||x/scale|| instead of whitened L2.
    Pass scale = 1/behavioral_sensitivity for behavior-weighted compression."""
    X = stack_flat(k_pre, v_pre)
    std = scale if scale is not None else \
        X[:fit_len].std(0, keepdim=True).clamp_min(1e-6)
    mu, W = pca_fit(X[:fit_len] / std, R)
    return {"std": std, "mu": mu, "W": W}


def apply_joint(codec, k_pre, v_pre, coeff_bits=16, keep_ranks=None):
    """keep_ranks: optional (T,) per-token component budget (A6 adaptive)."""
    L, kvh, _, hd = k_pre.shape
    Xs = stack_flat(k_pre, v_pre) / codec["std"]
    Z = (Xs - codec["mu"]) @ codec["W"]
    if coeff_bits < 16:
        Z = qsim(Z, coeff_bits, dim=0)
    if keep_ranks is not None:
        mask = torch.arange(Z.shape[1]).unsqueeze(0) < keep_ranks.unsqueeze(1)
        Z = Z * mask
    Xh = (Z @ codec["W"].T + codec["mu"]) * codec["std"]
    return stack_unflat(Xh, L, kvh, hd)


def fit_traj(hidden, fit_len, R):
    L1, T, D = hidden.shape
    X = hidden.permute(1, 0, 2).reshape(T, L1 * D)
    std = X[:fit_len].std(0, keepdim=True).clamp_min(1e-6)
    mu, W = pca_fit(X[:fit_len] / std, R)
    return {"std": std, "mu": mu, "W": W, "shape": (L1, D)}


def apply_traj(codec, hidden, coeff_bits=16):
    L1, T, D = hidden.shape
    Xs = hidden.permute(1, 0, 2).reshape(T, L1 * D) / codec["std"]
    Z = (Xs - codec["mu"]) @ codec["W"]
    if coeff_bits < 16:
        Z = qsim(Z, coeff_bits, dim=0)
    Xh = (Z @ codec["W"].T + codec["mu"]) * codec["std"]
    return Xh.reshape(T, L1, D).permute(1, 0, 2).contiguous()


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
