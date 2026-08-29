"""A7: does mean KL hide exact-recall collapse?

Plants 8 exact facts (5-digit secret numbers) at varying depths in filler text,
compresses the whole document's KV with each Pareto-frontier method, then asks
the model to complete each fact's retrieval prompt. Scores greedy exact-match
of the digits and teacher-forced NLL of the gold digits, per method, against
the full-KV baseline. The needle tokens themselves ARE compressed.

Usage: python scripts/a7_needle.py
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

from common import (load_model, pre_rope_kv, rope_k, build_cache, prefill_doc,
                    qsim, fit_joint, apply_joint, fit_traj, apply_traj)

NEEDLES = [
    ("Aurora", "40917"), ("Halcyon", "82354"), ("Vesper", "17609"),
    ("Quillon", "93481"), ("Marlowe", "26075"), ("Tessera", "58212"),
    ("Orinth", "70346"), ("Calder", "61893"),
]
TEMPLATE = "\nFor the record, the secret number for Project {} is {}.\n"
QUERY = "\nFor the record, the secret number for Project {} is"


def build_doc(tok, filler_text, chunk_tokens=190):
    filler_ids = tok(filler_text, return_tensors="pt").input_ids.squeeze(0)
    pieces, depths = [], []
    for i, (name, num) in enumerate(NEEDLES):
        pieces.append(filler_ids[i * chunk_tokens:(i + 1) * chunk_tokens])
        needle_ids = tok(TEMPLATE.format(name, num), return_tensors="pt").input_ids.squeeze(0)
        depths.append(sum(p.shape[0] for p in pieces))
        pieces.append(needle_ids)
    pieces.append(filler_ids[len(NEEDLES) * chunk_tokens:(len(NEEDLES) + 1) * chunk_tokens])
    return torch.cat(pieces).unsqueeze(0), depths


@torch.no_grad()
def probe_needle(model, tok, keys, values, past_len, name, gold_num,
                 query_template=None):
    """Feed the retrieval prompt against the given cache; greedy-decode and
    teacher-force the gold continuation. Returns (exact_match, gold_nll).
    Device/dtype-safe: inputs are moved to the model's device; the cache is
    cast by build_cache."""
    dev = next(model.parameters()).device
    query = (query_template or QUERY).format(name)
    q_ids = tok(query, return_tensors="pt",
                add_special_tokens=False).input_ids.to(dev)
    gold_ids = tok(" " + gold_num, return_tensors="pt",
                   add_special_tokens=False).input_ids.squeeze(0).to(dev)

    # teacher-forced NLL of the gold continuation
    cache = build_cache(keys, values, past_len, model)
    tf_ids = torch.cat([q_ids, gold_ids.unsqueeze(0)], dim=1)
    out = model(input_ids=tf_ids, past_key_values=cache,
                attention_mask=torch.ones(1, past_len + tf_ids.shape[1],
                                          dtype=torch.long, device=dev),
                position_ids=torch.arange(past_len, past_len + tf_ids.shape[1],
                                          device=dev).unsqueeze(0),
                use_cache=True)
    lp = torch.log_softmax(out.logits.squeeze(0).float(), dim=-1).cpu()
    n = gold_ids.shape[0]
    nll = -lp[q_ids.shape[1] - 1:q_ids.shape[1] - 1 + n].gather(
        1, gold_ids.unsqueeze(1).cpu()).mean().item()

    # greedy decode (free-running for the answer span)
    cache = build_cache(keys, values, past_len, model)
    ids = q_ids
    generated = []
    pos = past_len
    for step in range(n + 2):
        out = model(input_ids=ids, past_key_values=cache,
                    attention_mask=torch.ones(1, pos + ids.shape[1],
                                              dtype=torch.long, device=dev),
                    position_ids=torch.arange(pos, pos + ids.shape[1],
                                              device=dev).unsqueeze(0),
                    use_cache=True)
        cache = out.past_key_values
        pos += ids.shape[1]
        nxt = out.logits[0, -1].argmax().item()
        generated.append(nxt)
        ids = torch.tensor([[nxt]], device=dev)
    text = tok.decode(generated)
    exact = text.strip().startswith(gold_num.strip())
    return exact, nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filler", default="../PREPRINT.md")
    ap.add_argument("--report", default="reports/a7_needle.md")
    args = ap.parse_args()

    model = load_model()
    tok = AutoTokenizer.from_pretrained(model.config.name_or_path)
    filler = open(args.filler, encoding="utf-8").read()
    ids, depths = build_doc(tok, filler)
    T = ids.shape[1]
    print(f"needle doc: {T} tokens, needles at depths {depths}")

    hidden, keys_true, values_true, _ = prefill_doc(model, ids)
    k_pre, v_pre = pre_rope_kv(model, hidden)
    pos = torch.arange(T).unsqueeze(0)

    jc768 = fit_joint(k_pre, v_pre, T, 768)
    tc384 = fit_traj(hidden, T, 384)

    def make(name):
        if name == "full KV":
            return keys_true, values_true
        if name == "quant K8 V4 (2.6x)":
            return rope_k(model, qsim(k_pre, 8, dim=2), pos), qsim(v_pre, 4, dim=3)
        if name == "joint768 c8 (16x)":
            kh, vh = apply_joint(jc768, k_pre, v_pre, coeff_bits=8)
        elif name == "joint768 c4 (32x)":
            kh, vh = apply_joint(jc768, k_pre, v_pre, coeff_bits=4)
        elif name == "traj384 c8 (32x)":
            h = apply_traj(tc384, hidden, coeff_bits=8)
            kh, vh = pre_rope_kv(model, h)
        elif name == "traj384 c4 (64x)":
            h = apply_traj(tc384, hidden, coeff_bits=4)
            kh, vh = pre_rope_kv(model, h)
        return rope_k(model, kh, pos), vh

    methods = ["full KV", "quant K8 V4 (2.6x)", "joint768 c8 (16x)",
               "joint768 c4 (32x)", "traj384 c8 (32x)", "traj384 c4 (64x)"]

    rows = []
    for m in methods:
        keys, values = make(m)
        hits, nlls, per = 0, [], []
        for (name, num), d in zip(NEEDLES, depths):
            ok, nll = probe_needle(model, tok, keys, values, T, name, num)
            hits += ok
            nlls.append(nll)
            per.append("Y" if ok else ".")
        rows.append((m, hits, sum(nlls) / len(nlls), "".join(per)))
        print(f"  {m:22s} recall {hits}/8  gold-NLL {rows[-1][2]:.3f}  [{rows[-1][3]}]")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(f"# A7: exact-recall (needle) test under compression\n\n"
                f"{T}-token document, 8 planted 5-digit facts (depths {depths}), "
                f"whole document compressed including the needles, greedy retrieval.\n\n"
                f"| method | recall | gold-digit NLL | per-needle (shallow->deep) |\n"
                f"|---|---|---|---|\n")
        for m, h, nll, per in rows:
            f.write(f"| {m} | {h}/8 | {nll:.3f} | `{per}` |\n")
        f.write("\nAssumption A7 under test: 'mean KL captures the damage that matters.' "
                "If recall collapses at ratios whose mean KL looked acceptable, A7 is "
                "falsified and every Pareto point must carry a recall score.\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
