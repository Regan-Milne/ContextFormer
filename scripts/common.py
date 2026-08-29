"""Shared harness: capture, pre-RoPE KV recompute, cache surgery, two-pass
behavioral evaluation, analytic codecs.

Version- and architecture-robust by design:
  - transformers 4.45 (laptop, Qwen2.5) through current (4090, Qwen3+):
    cache construction/reading goes through compat shims, not legacy APIs.
  - Qwen3-family differences handled: config.head_dim decoupled from
    hidden_size/num_heads, and per-head q_norm/k_norm applied before RoPE.
  - Device/dtype aware: CPU fp32 by default; CUDA + bf16 for the scale gate.
    Codec math always runs in fp32; caches are cast to model dtype on build.

The safety net for any port is scripts/smoke.py: if the pre-RoPE recompute
matches the model's real cache, the harness understands the architecture.
"""

import time

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


def log(msg):
    """Timestamped, flushed progress line (long runs must stay monitorable)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

try:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
except ImportError:  # future refactor fallback: identical function in llama
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


# ---------------------------- model / geometry ----------------------------

def load_model(name="Qwen/Qwen2.5-0.5B", dtype=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def head_dim(cfg):
    """Qwen3+ sets head_dim explicitly and it need not equal hidden/heads."""
    hd = getattr(cfg, "head_dim", None)
    return hd if hd else cfg.hidden_size // cfg.num_attention_heads


def kv_geometry(cfg):
    """(L, kvh, hd, stack_dim, fp16_bytes_per_token)"""
    L, kvh, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, head_dim(cfg)
    stack_dim = 2 * L * kvh * hd
    return L, kvh, hd, stack_dim, stack_dim * 2


# ------------------------------ cache compat ------------------------------

def make_cache(pairs):
    """DynamicCache from an iterable of (k, v), each (1, kvh, T, hd)."""
    cache = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        cache.update(k, v, i)
    return cache


def cache_kv(pkv):
    """List of (k, v) per layer from whatever cache object the model returned."""
    if hasattr(pkv, "layers"):          # newer transformers
        return [(l.keys, l.values) for l in pkv.layers]
    if hasattr(pkv, "key_cache"):       # 4.4x-4.5x DynamicCache
        return list(zip(pkv.key_cache, pkv.value_cache))
    if hasattr(pkv, "to_legacy_cache"):
        return list(pkv.to_legacy_cache())
    return list(pkv)                    # legacy tuple


def build_cache(keys, values, past_len, model):
    """Cache from (L, kvh, T, hd) tensors truncated to past_len, cast to the
    model's device/dtype."""
    dt, dev = model.dtype, next(model.parameters()).device
    return make_cache(
        (keys[l, :, :past_len].unsqueeze(0).to(device=dev, dtype=dt).contiguous(),
         values[l, :, :past_len].unsqueeze(0).to(device=dev, dtype=dt).contiguous())
        for l in range(keys.shape[0])
    )


# --------------------------- capture / recompute ---------------------------

def load_capture(path):
    """Phase-0 on-disk capture (fp16) -> fp32 tensors."""
    cap = torch.load(path)
    cap["hidden"] = cap["hidden"].float()
    cap["keys"] = cap["keys"].float()
    cap["values"] = cap["values"].float()
    return cap


@torch.no_grad()
def prefill_doc(model, ids, chunk=0, want_logits=True):
    """Full prefill. Returns hidden (L+1,T,D), keys/values (L,kvh,T,hd) as the
    attention actually used them (post-RoPE, post-norm), logits (T,V). All
    fp32 on CPU regardless of model dtype/device.

    chunk > 0: prefill in segments of that many tokens, offloading hidden
    states and logits per segment so the peak device footprint is roughly
    weights + KV + one segment (the one-shot path peaks on the full fp32
    logits upcast: ~19 GB alone at 4B/32k). Identical output under causal
    attention; the KV accumulates on-device across segments."""
    dev = next(model.parameters()).device
    if not chunk:
        out = model(ids.to(dev), output_hidden_states=True, use_cache=True)
        kv = cache_kv(out.past_key_values)
        keys = torch.stack([k.squeeze(0).float().cpu() for k, _ in kv])
        values = torch.stack([v.squeeze(0).float().cpu() for _, v in kv])
        hidden = torch.stack([h.squeeze(0).float().cpu() for h in out.hidden_states])
        logits = out.logits.squeeze(0).float().cpu() if want_logits else None
        return hidden, keys, values, logits
    T = ids.shape[1]
    cache = None
    hid_parts, log_parts = [], []
    for s in range(0, T, chunk):
        seg = ids[:, s:s + chunk].to(dev)
        out = model(
            input_ids=seg,
            past_key_values=cache,
            attention_mask=torch.ones(1, s + seg.shape[1], dtype=torch.long,
                                      device=dev),
            position_ids=torch.arange(s, s + seg.shape[1],
                                      device=dev).unsqueeze(0),
            output_hidden_states=True, use_cache=True,
        )
        cache = out.past_key_values
        # offload first, upcast on CPU: bf16 -> fp32 is exact either way and
        # this avoids a per-segment fp32 logits copy on the device
        hid_parts.append(torch.stack([h.squeeze(0).cpu().float()
                                      for h in out.hidden_states]))
        if want_logits:
            log_parts.append(out.logits.squeeze(0).cpu().float())
        del out
    kv = cache_kv(cache)
    keys = torch.stack([k.squeeze(0).float().cpu() for k, _ in kv])
    values = torch.stack([v.squeeze(0).float().cpu() for _, v in kv])
    return (torch.cat(hid_parts, dim=1), keys, values,
            torch.cat(log_parts, dim=0) if want_logits else None)


def _proj_kv(layer, h, kvh, hd):
    """One layer's pre-RoPE K and V from its input hidden state (1, T, D),
    including Qwen3-style per-head k_norm when present."""
    attn = layer.self_attn
    T = h.shape[1]
    x = layer.input_layernorm(h)
    k = attn.k_proj(x).view(1, T, kvh, hd)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)
    v = attn.v_proj(x).view(1, T, kvh, hd)
    return k.transpose(1, 2), v.transpose(1, 2)


@torch.no_grad()
def pre_rope_kv(model, hidden):
    """Pre-RoPE K and V for all layers from the (L+1, T, D) fp32 trajectory.
    Returns fp32 CPU tensors (L, kvh, T, hd). Computation runs in the model's
    dtype/device so it matches the cache bit-for-bit."""
    cfg = model.config
    kvh, hd = cfg.num_key_value_heads, head_dim(cfg)
    dev, dt = next(model.parameters()).device, model.dtype
    ks, vs = [], []
    for l, layer in enumerate(model.model.layers):
        h = hidden[l].unsqueeze(0).to(device=dev, dtype=dt)
        k, v = _proj_kv(layer, h, kvh, hd)
        ks.append(k.squeeze(0).float().cpu())
        vs.append(v.squeeze(0).float().cpu())
    return torch.stack(ks), torch.stack(vs)


@torch.no_grad()
def rope_k(model, k_pre, position_ids):
    """Apply RoPE to pre-RoPE keys (L, kvh, T, hd), fp32 in/out."""
    cfg = model.config
    dev = next(model.parameters()).device
    T = k_pre.shape[2]
    dummy = torch.zeros(1, T, cfg.hidden_size, device=dev, dtype=torch.float32)
    cos, sin = model.model.rotary_emb(dummy, position_ids.to(dev))
    cos, sin = cos.float().cpu(), sin.float().cpu()
    out = []
    for l in range(k_pre.shape[0]):
        _, k = apply_rotary_pos_emb(k_pre[l].unsqueeze(0), k_pre[l].unsqueeze(0),
                                    cos, sin)
        out.append(k.squeeze(0))
    return torch.stack(out)


# ------------------------------- evaluation -------------------------------

@torch.no_grad()
def eval_window(model, ids, keys, values, past_len):
    """Feed tokens [past_len:] against the given (possibly reconstructed)
    post-RoPE cache. Returns fp32 CPU logits over the window."""
    dev = next(model.parameters()).device
    T = ids.shape[1]
    out = model(
        input_ids=ids[:, past_len:].to(dev),
        past_key_values=build_cache(keys, values, past_len, model),
        attention_mask=torch.ones(1, T, dtype=torch.long, device=dev),
        position_ids=torch.arange(past_len, T, device=dev).unsqueeze(0),
        use_cache=True,
    )
    return out.logits.squeeze(0).float().cpu()


def behavior_metrics(base_logits, comp_logits, target_ids):
    base_lp = torch.log_softmax(base_logits, dim=-1)
    comp_lp = torch.log_softmax(comp_logits, dim=-1)
    kl = (base_lp.exp() * (base_lp - comp_lp)).sum(-1)
    top1 = (base_logits.argmax(-1) == comp_logits.argmax(-1)).float()
    b5 = base_logits.topk(5, dim=-1).indices
    c5 = comp_logits.topk(5, dim=-1).indices
    top5 = torch.tensor([len(set(b5[i].tolist()) & set(c5[i].tolist())) / 5.0
                         for i in range(b5.shape[0])])
    n = target_ids.shape[0]
    tgt = target_ids.unsqueeze(1)
    base_nll = -base_lp[:n].gather(1, tgt).squeeze(1)
    comp_nll = -comp_lp[:n].gather(1, tgt).squeeze(1)
    return {
        "kl_mean": kl.mean().item(),
        "kl_p95": kl.quantile(0.95).item(),
        "top1_agree": top1.mean().item(),
        "top5_overlap": top5.mean().item(),
        "base_nll": base_nll.mean().item(),
        "delta_nll": (comp_nll - base_nll).mean().item(),
    }


# ------------------------ behavioral metric weights ------------------------

def _pair_symmetrize(w):
    half = w.shape[-1] // 2
    m = (w[..., :half] + w[..., half:]) / 2
    return torch.cat([m, m], dim=-1)


@torch.no_grad()
def behavioral_weights(model, hidden):
    """(wK, wV), each (L, kvh, hd): K channels by pooled post-RoPE query rms
    (pair-symmetrized so the diagonal metric commutes with RoPE); V channels
    by o_proj column norms. Handles Qwen3 q_norm."""
    cfg = model.config
    nh, kvh, hd = cfg.num_attention_heads, cfg.num_key_value_heads, head_dim(cfg)
    n_rep = nh // kvh
    dev, dt = next(model.parameters()).device, model.dtype
    L1, T, D = hidden.shape
    pos = torch.arange(T, device=dev).unsqueeze(0)
    dummy = torch.zeros(1, T, D, device=dev, dtype=torch.float32)
    cos, sin = model.model.rotary_emb(dummy, pos)
    wK, wV = [], []
    for l, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        h = hidden[l].unsqueeze(0).to(device=dev, dtype=dt)
        x = layer.input_layernorm(h)
        q = attn.q_proj(x).view(1, T, nh, hd)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        q = q.transpose(1, 2).float()
        q, _ = apply_rotary_pos_emb(q, q, cos.float(), sin.float())
        qrms = q.squeeze(0).pow(2).mean(1).sqrt().cpu()              # (nh, hd)
        wK.append(_pair_symmetrize(qrms.view(kvh, n_rep, hd).mean(1)))
        Wo = attn.o_proj.weight.float().cpu()                        # (D, nh*hd)
        onorm = Wo.view(Wo.shape[0], nh, hd).norm(dim=0)
        wV.append(onorm.view(kvh, n_rep, hd).mean(1))
    return torch.stack(wK), torch.stack(wV)


def metric_scale(wK, wV):
    w = torch.cat([wK.reshape(1, -1), wV.reshape(1, -1)], dim=1)
    w = w / w.mean()
    return 1.0 / w.clamp_min(1e-3)


# ------------------------------ analytic codec ------------------------------

def qsim(x, bits, dim):
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / qmax
    return (x / scale).round().clamp(-qmax, qmax) * scale


def _svd_components(Xc, r):
    """Top-r right singular vectors of an already-centered matrix."""
    # fp32 SVD on the GPU when available: same decomposition, minutes -> seconds
    # at gate scale (T x 147456). OOM falls back to the CPU path. The explicit
    # free-VRAM gate matters on Windows: WDDM pages an oversized allocation
    # into host memory instead of throwing, and a paged SVD thrashes for
    # hours without ever hitting the except branch.
    if torch.cuda.is_available() and Xc.numel() > 2 ** 24:
        free, _ = torch.cuda.mem_get_info()
        if free < 5 * Xc.numel() * 4 + 2 ** 30:
            log("pca_fit: not enough free VRAM for GPU SVD, using CPU")
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            return Vh[:r].T
        try:
            _, _, Vh = torch.linalg.svd(Xc.cuda(), full_matrices=False)
            return Vh[:r].T.cpu().contiguous()
        except RuntimeError:  # OOM (incl. capped) or cusolver workspace
            torch.cuda.empty_cache()
            log("pca_fit: GPU SVD unavailable, falling back to CPU")
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Vh[:r].T


def pca_fit(X, r):
    mu = X.mean(0, keepdim=True)
    return mu, _svd_components(X - mu, r)


def stack_flat(k_pre, v_pre):
    L, kvh, T, hd = k_pre.shape
    f = lambda x: x.permute(2, 0, 1, 3).reshape(T, L * kvh * hd)
    return torch.cat([f(k_pre), f(v_pre)], dim=1)


def stack_unflat(X, L, kvh, hd):
    T = X.shape[0]
    half = L * kvh * hd
    g = lambda x: x.reshape(T, L, kvh, hd).permute(1, 2, 0, 3).contiguous()
    return g(X[:, :half]), g(X[:, half:])


def fit_joint(k_pre, v_pre, fit_len, R, scale=None):
    # memory-lean: one big matrix alive at a time (the 32k fp32 stack is
    # ~9.4 GB per copy at 4B; the naive path holds three plus LAPACK's own)
    X = stack_flat(k_pre, v_pre)
    std = scale if scale is not None else \
        X[:fit_len].std(0, keepdim=True).clamp_min(1e-6)
    Xs = X[:fit_len] / std
    del X
    mu = Xs.mean(0, keepdim=True)
    Xs -= mu
    return {"std": std, "mu": mu, "W": _svd_components(Xs, R)}


def apply_joint(codec, k_pre, v_pre, coeff_bits=16, keep_ranks=None):
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


def behavioral_codec(model, hidden, k_pre, v_pre, fit_len, R):
    """One-call convenience: behavioral weights + joint fit."""
    wK, wV = behavioral_weights(model, hidden)
    return fit_joint(k_pre, v_pre, fit_len, R, scale=metric_scale(wK, wV))
