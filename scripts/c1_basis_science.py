"""C1: three cheap-but-sharp basis experiments at 2k tokens.

  A20  held-out-tokens: fit the basis on the FIRST 60% of the document only,
       compress everything (needles in the last 40% included). Deployment
       shape: at decode time new tokens must be encoded with a basis fit
       before they existed. All prior tests fit on the tokens they compress.
  ABL  metric ablation: K-weighting only vs V-weighting only vs both vs none.
  ND   non-diagonal metric: per-head query-covariance whitening for K
       (computed pre-RoPE; heuristic since rotation mixes channels) and exact
       o_proj-gram whitening for V, instead of the diagonal approximation.

Usage: python scripts/c1_basis_science.py
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

from common import (load_model, prefill_doc, pre_rope_kv, rope_k, head_dim,
                    behavioral_weights, metric_scale, fit_joint, apply_joint,
                    pca_fit, stack_flat, stack_unflat, qsim)
from a7_needle import NEEDLES, build_doc, probe_needle


@torch.no_grad()
def block_grams(model, hidden):
    """Per-(layer, kv-head) metrics: K from pre-RoPE query grams (pooled over
    the query heads sharing the kv head), V from o_proj grams. Returns
    (CK, CV): (L, kvh, hd, hd) Cholesky factors."""
    cfg = model.config
    nh, kvh, hd = cfg.num_attention_heads, cfg.num_key_value_heads, head_dim(cfg)
    n_rep = nh // kvh
    dev, dt = next(model.parameters()).device, model.dtype
    CK, CV = [], []
    for l, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        h = hidden[l].unsqueeze(0).to(device=dev, dtype=dt)
        x = layer.input_layernorm(h)
        q = attn.q_proj(x).view(1, -1, nh, hd)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        q = q.squeeze(0).transpose(0, 1).float().cpu()        # (nh, T, hd)
        T = q.shape[1]
        Wo = attn.o_proj.weight.float().cpu().view(-1, nh, hd)  # (D, nh, hd)
        ck, cv = [], []
        for g in range(kvh):
            qg = q[g * n_rep:(g + 1) * n_rep].reshape(-1, hd)
            M = qg.T @ qg / qg.shape[0]
            M = M + (M.diagonal().mean().clamp_min(1e-6) * 1e-2) * torch.eye(hd)
            ck.append(torch.linalg.cholesky(M))
            Wg = Wo[:, g * n_rep:(g + 1) * n_rep, :].reshape(-1, hd * n_rep)
            # V gram pooled over the group's heads
            Mv = torch.zeros(hd, hd)
            for r in range(n_rep):
                Wh = Wo[:, g * n_rep + r, :]                   # (D, hd)
                Mv = Mv + Wh.T @ Wh
            Mv = Mv / n_rep
            Mv = Mv + (Mv.diagonal().mean().clamp_min(1e-6) * 1e-2) * torch.eye(hd)
            cv.append(torch.linalg.cholesky(Mv))
        CK.append(torch.stack(ck))
        CV.append(torch.stack(cv))
    return torch.stack(CK), torch.stack(CV)


def block_apply(x, C):
    """x: (L, kvh, T, hd); C: (L, kvh, hd, hd). Returns x @ C per block."""
    return torch.einsum("lhtd,lhde->lhte", x, C)


def nondiag_codec(k_pre, v_pre, CK, CV, fit_len, R, coeff_bits):
    CKi = torch.linalg.inv(CK)
    CVi = torch.linalg.inv(CV)
    kt, vt = block_apply(k_pre, CK), block_apply(v_pre, CV)
    L, kvh, T, hd = k_pre.shape
    X = stack_flat(kt, vt)
    mu, W = pca_fit(X[:fit_len], R)
    Z = (X - mu) @ W
    if coeff_bits < 16:
        Z = qsim(Z, coeff_bits, dim=0)
    Xh = Z @ W.T + mu
    kh, vh = stack_unflat(Xh, L, kvh, hd)
    return block_apply(kh, CKi), block_apply(vh, CVi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--filler", default="../PREPRINT.md")
    ap.add_argument("--fit-frac", type=float, default=0.6)
    ap.add_argument("--report", default="reports/c1_basis_science.md")
    args = ap.parse_args()

    model = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids, depths = build_doc(tok, open(args.filler, encoding="utf-8").read())
    T = ids.shape[1]
    fit_len = int(T * args.fit_frac)
    n_out = sum(d >= fit_len for d in depths)
    print(f"doc {T} tokens; basis-fit boundary at {fit_len}; "
          f"{n_out} of {len(NEEDLES)} needles beyond it")

    hidden, keys_t, values_t, _ = prefill_doc(model, ids)
    k_pre, v_pre = pre_rope_kv(model, hidden)
    pos = torch.arange(T).unsqueeze(0)
    wK, wV = behavioral_weights(model, hidden)
    ones_K, ones_V = torch.ones_like(wK), torch.ones_like(wV)

    rows = []

    def recall_run(name, keys, values):
        hits, nlls, per = 0, [], []
        for (nm, num), d in zip(NEEDLES, depths):
            ok, nll = probe_needle(model, tok, keys, values, T, nm, num)
            hits += ok
            nlls.append(nll)
            per.append("Y" if ok else ".")
        rows.append((name, hits, sum(nlls) / len(nlls), "".join(per)))
        print(f"  {name:44s} {hits}/8  NLL {rows[-1][2]:.3f}  [{rows[-1][3]}]")

    def joint_run(name, scale, fl, R=768, cb=8):
        codec = fit_joint(k_pre, v_pre, fl, R, scale=scale)
        kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=cb)
        recall_run(name, rope_k(model, kh, pos), vh)

    recall_run("full KV", keys_t, values_t)
    sc = metric_scale(wK, wV)
    joint_run("behavioral 16x, fit on ALL tokens (ref)", sc, T)
    joint_run(f"A20: behavioral 16x, fit on first {args.fit_frac:.0%}", sc, fit_len)
    joint_run("ABL: K-weights only", metric_scale(wK, ones_V), T)
    joint_run("ABL: V-weights only", metric_scale(ones_K, wV), T)
    joint_run("ABL: no weighting (raw PCA)", metric_scale(ones_K, ones_V), T)

    print("computing non-diagonal grams...")
    CK, CV = block_grams(model, hidden)
    for R, cb, label in [(768, 8, "16x"), (384, 8, "32x")]:
        kh, vh = nondiag_codec(k_pre, v_pre, CK, CV, T, R, cb)
        recall_run(f"ND: non-diag metric joint{R} c{cb} ({label})",
                   rope_k(model, kh, pos), vh)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(f"# C1: held-out-tokens basis (A20), metric ablation, "
                f"non-diagonal metric\n\n{T}-token doc, needle depths {depths}, "
                f"fit boundary {fit_len} ({n_out} needles beyond it).\n\n"
                "| experiment | recall | gold-NLL | per-needle (shallow->deep) |\n"
                "|---|---|---|---|\n")
        for nm, h, nll, per in rows:
            f.write(f"| {nm} | {h}/8 | {nll:.3f} | `{per}` |\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
