"""Phase 1a: spectral structure of per-token state, and the context test.

Answers, from the captured data alone:
  1. How low-rank is the population of per-layer pre-RoPE K and V vectors?
  2. Does the *joint* cross-layer stack have more compressible structure than
     per-layer treatment (the "stack as one object" premise)?
  3. How smooth is the hidden trajectory across layers?
  4. Context test: how much of a token's mid-layer state is predictable from
     token identity alone vs identity + local context? (ridge, train/test split)

Usage: python scripts/analyze_spectra.py --capture data/capture_PREPRINT_2048.pt
"""

import argparse
import os

import torch

from common import load_model, load_capture, pre_rope_kv


def rank_for_energy(s, fracs=(0.90, 0.95, 0.99)):
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    return [int((e < f).sum().item()) + 1 for f in fracs]


def spectrum(x):
    """x: (T, D) rows = tokens. Centered singular values."""
    x = x - x.mean(0, keepdim=True)
    return torch.linalg.svdvals(x)


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=10.0):
    """Closed-form ridge; returns fraction of test variance explained."""
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xtr, Ytr = Xtr - mx, Ytr - my
    Xte, Yte = Xte - mx, Yte - my
    d = Xtr.shape[1]
    W = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(d), Xtr.T @ Ytr)
    resid = Yte - Xte @ W
    return 1.0 - (resid ** 2).sum().item() / (Yte ** 2).sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="data/capture_PREPRINT_2048.pt")
    ap.add_argument("--report", default="reports/phase1_spectra.md")
    args = ap.parse_args()

    model = load_model()
    cap = load_capture(args.capture)
    hidden = cap["hidden"]                       # (L+1, T, D)
    L1, T, D = hidden.shape
    L = L1 - 1
    k_pre, v = pre_rope_kv(model, hidden)        # (L, kvh, T, hd)
    kvh, hd = k_pre.shape[1], k_pre.shape[3]
    kv_dims = kvh * hd

    lines = [f"# Phase 1a: spectral structure ({cap['model']}, {T} tokens of {cap['doc']})\n"]

    # 1. per-layer spectra
    lines.append("## Per-layer rank (centered PCA over tokens, ranks for 90/95/99% energy, dim = "
                 f"{kv_dims})\n")
    lines.append("| layer | K90 | K95 | K99 | V90 | V95 | V99 |")
    lines.append("|---|---|---|---|---|---|---|")
    k_ranks, v_ranks = [], []
    for l in range(L):
        sk = spectrum(k_pre[l].permute(1, 0, 2).reshape(T, kv_dims))
        sv = spectrum(v[l].permute(1, 0, 2).reshape(T, kv_dims))
        rk, rv = rank_for_energy(sk), rank_for_energy(sv)
        k_ranks.append(rk)
        v_ranks.append(rv)
        lines.append(f"| {l} | {rk[0]} | {rk[1]} | {rk[2]} | {rv[0]} | {rv[1]} | {rv[2]} |")
    mk = torch.tensor(k_ranks, dtype=torch.float).mean(0)
    mv = torch.tensor(v_ranks, dtype=torch.float).mean(0)
    lines.append(f"| **mean** | {mk[0]:.0f} | {mk[1]:.0f} | {mk[2]:.0f} "
                 f"| {mv[0]:.0f} | {mv[1]:.0f} | {mv[2]:.0f} |")

    # 2. joint cross-layer stack vs per-layer, matched dims
    def flat(x):   # (L, kvh, T, hd) -> (T, L*kv_dims)
        return x.permute(2, 0, 1, 3).reshape(T, L * kv_dims)

    stack = torch.cat([flat(k_pre), flat(v)], dim=1)          # (T, 6144)
    std = stack.std(0, keepdim=True).clamp_min(1e-6)
    s_joint = spectrum(stack / std)
    j = rank_for_energy(s_joint)
    per_layer_sum = [int(mk[i].item() * L + mv[i].item() * L) for i in range(3)]
    lines.append(f"\n## Joint cross-layer stack (standardized, dim {stack.shape[1]})\n")
    lines.append(f"- joint rank for 90/95/99% energy: {j[0]} / {j[1]} / {j[2]}")
    lines.append(f"- sum of per-layer ranks (same energy): {per_layer_sum[0]} / "
                 f"{per_layer_sum[1]} / {per_layer_sum[2]}")
    lines.append(f"- cross-layer advantage at 95%: {per_layer_sum[1] / max(j[1], 1):.1f}x fewer "
                 "coefficients for the whole stack treated as one object")

    traj = hidden.permute(1, 0, 2).reshape(T, L1 * D)
    tstd = traj.std(0, keepdim=True).clamp_min(1e-6)
    tj = rank_for_energy(spectrum(traj / tstd))
    lines.append(f"- hidden-trajectory joint rank (dim {L1 * D}): {tj[0]} / {tj[1]} / {tj[2]}")

    # 3. trajectory smoothness
    cos = torch.nn.functional.cosine_similarity(hidden[:-1], hidden[1:], dim=-1)  # (L, T)
    lines.append("\n## Trajectory smoothness (mean cosine of adjacent-layer hidden states)\n")
    lines.append("| layers | " + " | ".join(str(l) for l in range(0, L, 4)) + " |")
    lines.append("|---|" + "---|" * len(range(0, L, 4)))
    lines.append("| cos | " + " | ".join(f"{cos[l].mean():.3f}" for l in range(0, L, 4)) + " |")
    lines.append(f"\nmean over all adjacent pairs: {cos.mean():.3f}")

    # 4. context test: predict mid-layer hidden state
    lines.append("\n## Context test: predicting h^(L/2) per token (ridge, "
                 "train first 75% of tokens, test last 25%)\n")
    mid = L // 2
    emb = model.model.embed_tokens(cap["token_ids"].squeeze(0)).detach()   # (T, D)
    Y = hidden[mid]
    n_tr = int(T * 0.75)

    def ctx_feats(width):
        cols = [emb]
        for w in range(1, width + 1):
            cols.append(torch.roll(emb, w, dims=0))   # previous tokens (causal)
        return torch.cat(cols, dim=1)

    tests = [
        ("token embedding only", emb),
        ("+ prev 2 tokens", ctx_feats(2)),
        ("+ prev 8 tokens", ctx_feats(8)),
        ("+ prev 8 + mean of prev 64", torch.cat([
            ctx_feats(8),
            torch.stack([emb[max(0, i - 64):max(1, i)].mean(0) for i in range(T)]),
        ], dim=1)),
    ]
    lines.append("| features | test R^2 on h^(mid) |")
    lines.append("|---|---|")
    for name, X in tests:
        r2 = ridge_r2(X[64:n_tr], Y[64:n_tr], X[n_tr:], Y[n_tr:])
        lines.append(f"| {name} | {r2:.3f} |")
    lines.append("\n(Caveat: 2048 tokens of one document; indicative, not conclusive. "
                 "The question is the *gap* between rows, i.e. whether context access "
                 "buys predictability beyond token identity.)")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
