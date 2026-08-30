"""A25: corpus-basis rank sweep (pre-registered challenge to A5b).

Hypothesis under test: A5b's corpus-basis failure at 16x reflects
excessive compression, not fundamentally document-specific geometry.

Design: one shared basis, pre-fit on a training corpus the eval documents
are excluded from, evaluated at a rank ladder on held-out heterogeneous
documents; a per-document basis at matched rank is the comparator wherever
the rank is reachable per-doc (rank <= T). Ranks above T have no per-doc
comparator BY CONSTRUCTION -- the shared basis's fit samples are unlimited,
which is exactly the rank-cap dissolution the liberal-rank direction
predicts; those rows compare against full KV only.

The shared basis is fit on plain documents (no needles); eval documents
carry the standard typed needle battery. Behavioral scale is the mean of
per-train-doc behavioral weights.

Usage (0.5B primary):
  python scripts/a25_corpus_rank.py
"""

import argparse
import os
import random

import torch
from transformers import AutoTokenizer

from common import (load_model, prefill_doc, pre_rope_kv, rope_k,
                    kv_geometry, behavioral_weights, metric_scale,
                    fit_joint, apply_joint, stack_flat, _svd_components, log)
from battery import gen_needles, build_doc
from a7_needle import probe_needle

TRAIN = [("mobydick", "data/mobydick.txt", 30000),
         ("pride_prejudice", "data/pride_prejudice.txt", 2000),
         ("tale_two_cities", "data/tale_two_cities.txt", 2000),
         ("code_train", "data/code_train.txt", 0)]

EVAL = [("prose:frankenstein", "data/frankenstein.txt", 2000),
        ("dialogue:earnest", "data/earnest.txt", 2000),
        ("technical:origin_species", "data/origin_species.txt", 2000),
        ("code:code_eval", "data/code_eval.txt", 0),
        ("structured:logs", "data/structured.txt", 0)]


@torch.no_grad()
def doc_stacks(model, tok, text, n_tokens):
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens]
    hidden, _, _, _ = prefill_doc(model, ids, chunk=1024, want_logits=False)
    k_pre, v_pre = pre_rope_kv(model, hidden)
    return hidden, k_pre, v_pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target-tokens", type=int, default=1800)
    ap.add_argument("--train-tokens", type=int, default=1800)
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--ranks", default=None,
                    help="comma list; default = stack ranks for 2x/4x/8x/16x")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    model = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    L, kvh, hd, stack_dim, fp16_b = kv_geometry(model.config)
    ranks = ([int(r) for r in args.ranks.split(",")] if args.ranks else
             [min(fp16_b // r, stack_dim) for r in (2, 4, 8, 16)])
    rmax = max(ranks)
    print(f"{args.model}: stack {stack_dim}, ranks {ranks}", flush=True)

    # ---- phase A: shared corpus basis (plain docs, no needles) ----
    pools, wKs, wVs = [], [], []
    for name, path, skip in TRAIN:
        log(f"train doc: {name}")
        text = open(path, encoding="utf-8", errors="ignore").read()[skip:]
        hidden, k_pre, v_pre = doc_stacks(model, tok, text, args.train_tokens)
        wK, wV = behavioral_weights(model, hidden)
        wKs.append(wK); wVs.append(wV)
        pools.append(stack_flat(k_pre[:, :, ::args.train_stride],
                                v_pre[:, :, ::args.train_stride]))
        del hidden, k_pre, v_pre
    scale = metric_scale(torch.stack(wKs).mean(0), torch.stack(wVs).mean(0))
    X = torch.cat(pools, dim=0); del pools
    log(f"corpus fit: {X.shape[0]} pooled positions, rank {min(rmax, X.shape[0] - 8)}")
    Xs = X / scale; del X
    mu = Xs.mean(0, keepdim=True)
    Xs -= mu
    W = _svd_components(Xs, min(rmax, Xs.shape[0] - 8)); del Xs
    torch.save({"std": scale, "mu": mu, "W": W}, "reports/a25_corpus_basis.pt")
    log(f"shared basis: rank {W.shape[1]} saved")

    # ---- phase B: held-out evaluation ----
    rng = random.Random(args.seed)
    needles = gen_needles(rng, 3)
    types = sorted(set(nd["type"] for nd in needles))
    rows = []
    for name, path, skip in EVAL:
        log(f"eval doc: {name}")
        text = open(path, encoding="utf-8", errors="ignore").read()[skip:]
        ids, depths = build_doc(tok, text, needles, args.target_tokens)
        T = ids.shape[1]
        hidden, keys_t, values_t, _ = prefill_doc(model, ids, chunk=1024,
                                                  want_logits=False)
        k_pre, v_pre = pre_rope_kv(model, hidden); del hidden
        pos = torch.arange(T).unsqueeze(0)

        def probe_set(keys, values, label):
            ok_n, nlls = 0, []
            for nd in needles:
                ok, nll = probe_needle(model, tok, keys, values, T, "",
                                       nd["gold"], query_template=nd["query"])
                ok_n += ok; nlls.append(nll)
            row = (name, label, ok_n, len(needles), sum(nlls) / len(nlls))
            rows.append(row)
            print(f"  {name:26s} {label:22s} {ok_n:2d}/{len(needles)}"
                  f"  NLL {row[4]:.3f}", flush=True)

        probe_set(keys_t, values_t, "full KV")
        for r in ranks:
            if r <= W.shape[1]:
                codec = {"std": scale, "mu": mu, "W": W[:, :r]}
                kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=8)
                probe_set(rope_k(model, kh, pos), vh, f"shared r{r}")
            if r <= T - 8:
                codec = fit_joint(k_pre, v_pre, T, r, scale=scale)
                kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=8)
                probe_set(rope_k(model, kh, pos), vh, f"per-doc r{r}")
        del k_pre, v_pre, keys_t, values_t

    slug = args.model.split("/")[-1].replace(".", "")
    rpt = os.path.join(args.report_dir, f"a25_corpus_rank_{slug}.md")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"# A25 corpus-basis rank sweep -- {args.model}, "
                f"eval T~{args.target_tokens}, seed {args.seed}\n\n"
                f"| doc | basis | recall | gold-NLL |\n|---|---|---|---|\n")
        for name, label, ok, n, nll in rows:
            f.write(f"| {name} | {label} | {ok}/{n} | {nll:.3f} |\n")
    print(f"wrote {rpt}", flush=True)


if __name__ == "__main__":
    main()
