"""A28: two-level geometry tests (pre-registered).

P1 (containment): fraction of each per-doc rank-768 basis's energy inside
the shared basis's top-r span, for r in {768, 1536, 3072, 6144}.

P2 (selection): a codec restricted to shared axes but with per-document
axis SELECTION (top 768 shared axes ranked by the document's own mean
squared coefficient), evaluated on the standard typed battery. Compared
against the shared-prefix-768 and per-doc-PCA-768 rows from A25.

CPU-friendly at 0.5B; reuses reports/a25_corpus_basis.pt.
"""

import argparse
import random

import torch
from transformers import AutoTokenizer

from common import (load_model, prefill_doc, pre_rope_kv, rope_k,
                    kv_geometry, fit_joint, apply_joint, stack_flat, log)
from battery import gen_needles, build_doc
from a7_needle import probe_needle

EVAL = [("prose:frankenstein", "data/frankenstein.txt", 2000),
        ("dialogue:earnest", "data/earnest.txt", 2000),
        ("technical:origin_species", "data/origin_species.txt", 2000),
        ("code:code_eval", "data/code_eval.txt", 0),
        ("structured:logs", "data/structured.txt", 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target-tokens", type=int, default=1800)
    ap.add_argument("--k", type=int, default=768)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    model = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    shared = torch.load("reports/a25_corpus_basis.pt")
    scale, mu, W = shared["std"], shared["mu"], shared["W"]
    k = args.k
    print(f"shared basis rank {W.shape[1]}, selection k={k}", flush=True)

    rng = random.Random(args.seed)
    needles = gen_needles(rng, 3)
    results = []
    for name, path, skip in EVAL:
        log(f"doc: {name}")
        text = open(path, encoding="utf-8", errors="ignore").read()[skip:]
        ids, _ = build_doc(tok, text, needles, args.target_tokens)
        T = ids.shape[1]
        hidden, _, _, _ = prefill_doc(model, ids, chunk=1024,
                                      want_logits=False)
        k_pre, v_pre = pre_rope_kv(model, hidden); del hidden
        pos = torch.arange(T).unsqueeze(0)

        # P1: containment of the per-doc basis in shared top-r spans
        doc_codec = fit_joint(k_pre, v_pre, T, k, scale=scale)
        Wd = doc_codec["W"]                        # (D, k), orthonormal cols
        cont = {}
        for r in (768, 1536, 3072, W.shape[1]):
            C = W[:, :r].T @ Wd
            cont[r] = (C.pow(2).sum() / k).item()
        print(f"  containment of per-doc r{k}: " +
              "  ".join(f"top{r}: {v:.3f}" for r, v in cont.items()),
              flush=True)

        # P2: per-doc SELECTION of k shared axes by mean squared coefficient
        Xs = stack_flat(k_pre, v_pre) / scale
        Z = (Xs - mu) @ W; del Xs
        idx = Z.pow(2).mean(0).argsort(descending=True)[:k].sort().values
        sel_codec = {"std": scale, "mu": mu, "W": W[:, idx]}
        kh, vh = apply_joint(sel_codec, k_pre, v_pre, coeff_bits=8)
        keys, values = rope_k(model, kh, pos), vh
        ok_n, nlls = 0, []
        for nd in needles:
            ok, nll = probe_needle(model, tok, keys, values, T, "",
                                   nd["gold"], query_template=nd["query"])
            ok_n += ok; nlls.append(nll)
        print(f"  selection-{k} from shared:      {ok_n:2d}/{len(needles)}"
              f"  NLL {sum(nlls) / len(nlls):.3f}", flush=True)
        results.append((name, cont, ok_n, len(needles),
                        sum(nlls) / len(nlls)))

    with open("reports/a28_two_level.md", "w", encoding="utf-8") as f:
        f.write(f"# A28 two-level geometry tests -- {args.model}, "
                f"seed {args.seed}, k={k}\n\n"
                "| doc | cont@768 | cont@1536 | cont@3072 | "
                "selection recall | NLL |\n|---|---|---|---|---|---|\n")
        for name, cont, ok, n, nll in results:
            f.write(f"| {name} | {cont[768]:.3f} | {cont[1536]:.3f} | "
                    f"{cont[3072]:.3f} | {ok}/{n} | {nll:.3f} |\n")
    print("wrote reports/a28_two_level.md", flush=True)


if __name__ == "__main__":
    main()
