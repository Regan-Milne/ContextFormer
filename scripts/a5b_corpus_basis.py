"""A5b: does a corpus-fit basis transfer to held-out documents?

The decisive question for the 16x claim's accounting: if one shared basis
(fit across diverse documents) preserves needle recall on a document it never
saw, the basis ships with the model at zero marginal cost per document, and
the 16x figure is clean at every context length. If only per-doc bases work,
the claim needs long-context amortization or a learned encoder.

Fit corpus: Moby-Dick, Pride & Prejudice, Origin of Species, Federalist
Papers, kvspace demo source code (5 docs x 2048 tokens, diverse register).
Held out: the needle document (ML-preprint filler + planted facts) and a
Grimm's fairy-tale slice for clean-KL evaluation. Behavioral weights are
pooled over the corpus only -- the full pipeline is held-out.

Usage: python scripts/a5b_corpus_basis.py
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

from common import (load_model, pre_rope_kv, rope_k, prefill_doc, eval_window,
                    behavior_metrics, fit_joint, apply_joint, stack_flat,
                    pca_fit, qsim)
from a7_needle import NEEDLES, build_doc, probe_needle
from a8_behavior_basis import behavioral_weights, metric_scale

CORPUS = [
    ("mobydick", "data/mobydick.txt", 30000),
    ("pride", "data/pride.txt", 20000),
    ("origin", "data/origin.txt", 20000),
    ("federalist", "data/federalist.txt", 20000),
    ("kvspace-code", "../demo/kvspace.py", 0),
]


def doc_ids(tok, path, skip, n=2048):
    text = open(path, encoding="utf-8", errors="ignore").read()
    return tok(text[skip:], return_tensors="pt").input_ids[:, :n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filler", default="../PREPRINT.md")
    ap.add_argument("--past", type=int, default=1792)
    ap.add_argument("--report", default="reports/a5b_corpus_basis.md")
    args = ap.parse_args()

    model = load_model()
    tok = AutoTokenizer.from_pretrained(model.config.name_or_path)
    P = args.past

    # ---- build corpus stacks + pooled behavioral weights ----
    stacks, w2K_sum, w2V_sum = [], None, None
    for name, path, skip in CORPUS:
        ids = doc_ids(tok, path, skip)
        print(f"corpus doc {name}: {ids.shape[1]} tokens")
        hidden, _, _, _ = prefill_doc(model, ids)
        k_pre, v_pre = pre_rope_kv(model, hidden)
        stacks.append(stack_flat(k_pre, v_pre))
        wK, wV = behavioral_weights(model, hidden)
        w2K_sum = wK ** 2 if w2K_sum is None else w2K_sum + wK ** 2
        w2V_sum = wV ** 2 if w2V_sum is None else w2V_sum + wV ** 2
        del hidden, k_pre, v_pre
    scale_corpus = metric_scale((w2K_sum / len(CORPUS)).sqrt(),
                                (w2V_sum / len(CORPUS)).sqrt())
    X_corpus = torch.cat(stacks, dim=0)
    print(f"corpus stack: {tuple(X_corpus.shape)}")

    def corpus_codec(R):
        mu, W = pca_fit(X_corpus / scale_corpus, R)
        return {"std": scale_corpus, "mu": mu, "W": W}

    # ---- held-out needle doc ----
    n_ids, depths = build_doc(tok, open(args.filler, encoding="utf-8").read())
    Tn = n_ids.shape[1]
    hid_n, keys_n, values_n, _ = prefill_doc(model, n_ids)
    kn, vn = pre_rope_kv(model, hid_n)
    pos_n = torch.arange(Tn).unsqueeze(0)

    # per-doc reference codec (the a8 winner, fit on the eval doc itself)
    wKn, wVn = behavioral_weights(model, hid_n)
    scale_n = metric_scale(wKn, wVn)

    recall_rows = []

    def recall_run(name, keys, values):
        hits, nlls, per = 0, [], []
        for (nm, num), d in zip(NEEDLES, depths):
            ok, nll = probe_needle(model, tok, keys, values, Tn, nm, num)
            hits += ok
            nlls.append(nll)
            per.append("Y" if ok else ".")
        recall_rows.append((name, hits, sum(nlls) / len(nlls), "".join(per)))
        print(f"  {name:44s} recall {hits}/8  gold-NLL {recall_rows[-1][2]:.3f}  "
              f"[{recall_rows[-1][3]}]")

    print("needle recall on held-out doc:")
    recall_run("full KV", keys_n, values_n)
    for R, cb, label in [(768, 8, "16x"), (768, 4, "32x"), (384, 8, "32x")]:
        codec = corpus_codec(R)
        kh, vh = apply_joint(codec, kn, vn, coeff_bits=cb)
        recall_run(f"CORPUS basis joint{R} c{cb} ({label})",
                   rope_k(model, kh, pos_n), vh)
    codec_pd = fit_joint(kn, vn, Tn, 768, scale=scale_n)
    kh, vh = apply_joint(codec_pd, kn, vn, coeff_bits=8)
    recall_run("per-doc basis joint768 c8 (16x, reference)",
               rope_k(model, kh, pos_n), vh)

    # ---- held-out KL eval on Grimm ----
    g_ids = doc_ids(tok, "data/grimm.txt", 20000)
    Tg = g_ids.shape[1]
    hid_g, keys_g, values_g, _ = prefill_doc(model, g_ids)
    kg, vg = pre_rope_kv(model, hid_g)
    pos_g = torch.arange(Tg).unsqueeze(0)
    base_g = eval_window(model, g_ids, keys_g, values_g, P)
    targets_g = g_ids.squeeze(0)[P + 1:]

    kl_rows = []
    print("KL on held-out Grimm doc:")
    for R, cb, label in [(768, 8, "16x"), (384, 8, "32x")]:
        codec = corpus_codec(R)
        kh, vh = apply_joint(codec, kg, vg, coeff_bits=cb)
        m = behavior_metrics(base_g, eval_window(
            model, g_ids, rope_k(model, kh, pos_g), vh, P), targets_g)
        m["name"] = f"CORPUS basis joint{R} c{cb} ({label})"
        kl_rows.append(m)
        print(f"  {m['name']:44s} KL {m['kl_mean']:.4f}  top1 {m['top1_agree']:.3f}  "
              f"dNLL {m['delta_nll']:+.4f}")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# A5b: corpus-fit basis on held-out documents\n\n"
                f"Basis fit on 5 docs x 2048 tokens (fiction x2, science, political, "
                f"code), behavioral weights pooled over corpus. Eval docs never seen "
                f"by the basis.\n\n## Needle recall (held-out ML-preprint needle doc)\n\n"
                "| method | recall | gold-NLL | per-needle |\n|---|---|---|---|\n")
        for nm, h, nll, per in recall_rows:
            f.write(f"| {nm} | {h}/8 | {nll:.3f} | `{per}` |\n")
        f.write("\n## Teacher-forced eval (held-out Grimm doc)\n\n"
                "| method | KL | top-1 | top-5 | dNLL |\n|---|---|---|---|---|\n")
        for m in kl_rows:
            f.write(f"| {m['name']} | {m['kl_mean']:.4f} | {m['top1_agree']:.3f} "
                    f"| {m['top5_overlap']:.3f} | {m['delta_nll']:+.4f} |\n")
        f.write("\nIf the corpus basis holds recall at 16x, basis storage is a "
                "fixed model asset (zero marginal bytes/doc) and the 16x accounting "
                "is clean at every context length.\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
