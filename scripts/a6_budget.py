"""A6: do all tokens deserve the same budget?

Part 1 -- distribution: per-token reconstruction residual of the joint-stack
codec at rank 384; quantiles, tail shape, and correlation with (a) the
baseline model's surprisal at that position, (b) in-document token frequency,
(c) position.

Part 2 -- allocation: two-tier adaptive budgets (hard tokens get a bigger
rank, easy tokens smaller, mean bytes matched to the uniform rank-384 point)
vs uniform, scored behaviorally. Hard tokens are chosen by encode-time
reconstruction residual (legal: the encoder sees X_i).

Usage: python scripts/a6_budget.py
"""

import argparse
import os

import torch

from common import (load_model, load_capture, pre_rope_kv, rope_k, eval_window,
                    behavior_metrics, stack_flat, fit_joint, apply_joint)


def pearson(a, b):
    a = (a - a.mean()) / a.std().clamp_min(1e-9)
    b = (b - b.mean()) / b.std().clamp_min(1e-9)
    return (a * b).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="data/capture_PREPRINT_2048.pt")
    ap.add_argument("--past", type=int, default=1792)
    ap.add_argument("--report", default="reports/a6_budget.md")
    args = ap.parse_args()

    model = load_model()
    cap = load_capture(args.capture)
    ids = cap["token_ids"]
    hidden = cap["hidden"]
    keys_true, values_true = cap["keys"], cap["values"]
    T = hidden.shape[1]
    P = args.past
    k_pre, v_pre = pre_rope_kv(model, hidden)
    pos = torch.arange(T).unsqueeze(0)
    targets = ids.squeeze(0)[P + 1:]
    base_logits = eval_window(model, ids, keys_true, values_true, P)

    codec = fit_joint(k_pre, v_pre, P, 768)

    # ---- part 1: residual distribution at rank 384 ----
    Xs = stack_flat(k_pre, v_pre) / codec["std"]
    Z = (Xs - codec["mu"]) @ codec["W"]
    Xh384 = Z[:, :384] @ codec["W"][:, :384].T + codec["mu"]
    num = ((Xs - Xh384) ** 2).sum(1)
    den = ((Xs - codec["mu"]) ** 2).sum(1).clamp_min(1e-9)
    resid = (num / den)[:P]                                   # past tokens only

    with torch.no_grad():
        full_logits = model(ids).logits.squeeze(0).float()
    lp = torch.log_softmax(full_logits, dim=-1)
    surprisal = torch.zeros(T)
    surprisal[1:] = -lp[:-1].gather(1, ids.squeeze(0)[1:].unsqueeze(1)).squeeze(1)
    tid = ids.squeeze(0)
    counts = torch.bincount(tid, minlength=int(tid.max()) + 1).float()
    freq = counts[tid]

    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    quants = resid.quantile(q)
    corr_s = pearson(resid, surprisal[:P])
    corr_f = pearson(resid, freq[:P].log())
    corr_p = pearson(resid, torch.arange(P).float())

    print(f"residual fraction @rank384: median {quants[2]:.3f}, "
          f"p90 {quants[4]:.3f}, p99 {quants[5]:.3f}, max {resid.max():.3f}")
    print(f"corr(resid, surprisal) {corr_s:+.3f}  corr(resid, log-freq) {corr_f:+.3f}  "
          f"corr(resid, position) {corr_p:+.3f}")

    hard_ids = tid[:P][resid.topk(12).indices]
    hard_tokens = [repr(t) for t in
                   __import__("transformers").AutoTokenizer.from_pretrained(
                       model.config.name_or_path).batch_decode(hard_ids.unsqueeze(1))]

    # ---- part 2: adaptive two-tier vs uniform at matched mean bytes ----
    results = []

    def run(name, keep_ranks, mean_rank):
        kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=16, keep_ranks=keep_ranks)
        logits = eval_window(model, ids, rope_k(model, kh, pos), vh, P)
        m = behavior_metrics(base_logits, logits, targets)
        m.update(name=name, mean_rank=mean_rank)
        results.append(m)
        print(f"  {name:44s} KL {m['kl_mean']:.4f}  top1 {m['top1_agree']:.3f}  "
              f"dNLL {m['delta_nll']:+.4f}")

    resid_all = (num / den)   # includes window tokens (uncompressed anyway; harmless)
    run("uniform rank 384", torch.full((T,), 384), 384)
    for hi, lo in [(768, 192), (768, 96)]:
        frac = (384 - lo) / (hi - lo)
        n_hard = int(frac * T)
        keep = torch.full((T,), lo)
        keep[resid_all.topk(n_hard).indices] = hi
        run(f"adaptive {hi}/{lo} (top {frac:.0%} hard get {hi})", keep,
            int(keep.float().mean().item()))

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# A6: per-token budget distribution and adaptive allocation\n\n"
                "## Residual fraction at joint rank 384 (past tokens)\n\n"
                f"| p10 | p25 | median | p75 | p90 | p99 | max |\n|---|---|---|---|---|---|---|\n"
                f"| {quants[0]:.3f} | {quants[1]:.3f} | {quants[2]:.3f} | {quants[3]:.3f} "
                f"| {quants[4]:.3f} | {quants[5]:.3f} | {resid.max():.3f} |\n\n"
                f"Correlations: resid vs surprisal {corr_s:+.3f}, vs log in-doc freq "
                f"{corr_f:+.3f}, vs position {corr_p:+.3f}.\n\n"
                f"Hardest 12 tokens: {', '.join(hard_tokens)}\n\n"
                "## Adaptive vs uniform at matched mean rank (coeff fp16, 768 B/token)\n\n"
                "| allocation | mean rank | KL | top-1 | top-5 | dNLL |\n|---|---|---|---|---|---|\n")
        for m in results:
            f.write(f"| {m['name']} | {m['mean_rank']} | {m['kl_mean']:.4f} "
                    f"| {m['top1_agree']:.3f} | {m['top5_overlap']:.3f} "
                    f"| {m['delta_nll']:+.4f} |\n")
        f.write("\nA6 under test: 'every token deserves the same budget.' Adaptive "
                "winning at matched mean bytes falsifies it; the residual tail and its "
                "correlates say which tokens are the expensive ones.\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
