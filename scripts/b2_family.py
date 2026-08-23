"""B2: does the mechanism transfer across model families?

Runs the core pipeline on another frozen model (any Llama/Qwen2-architecture
checkpoint transformers 4.45 supports): trajectory->KV roundtrip check,
whole-stack behavioral codec at matched 16x/32x ratios (rank scaled to the
model's own stack size), variance-metric control, needle recall, and
teacher-forced KL on a second document.

Usage:
  python scripts/b2_family.py --model TinyLlama/TinyLlama_v1.1
  python scripts/b2_family.py --model Qwen/Qwen2.5-1.5B
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

from common import (load_model, pre_rope_kv, rope_k, prefill_doc, eval_window,
                    behavior_metrics, fit_joint, apply_joint)
from a7_needle import build_doc, probe_needle
from a8_behavior_basis import behavioral_weights, metric_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--filler", default="../PREPRINT.md")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    model = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = model.config
    L = cfg.num_hidden_layers
    kvh = cfg.num_key_value_heads
    hd = cfg.hidden_size // cfg.num_attention_heads
    stack_dim = 2 * L * kvh * hd
    fp16_bytes = stack_dim * 2
    R16, R32 = fp16_bytes // 16, fp16_bytes // 32   # c8 coeffs -> 1 B each
    print(f"{args.model}: L={L} kvh={kvh} hd={hd} stack_dim={stack_dim} "
          f"fp16 {fp16_bytes} B/tok; 16x rank {R16}, 32x rank {R32}")

    ids, depths = build_doc(tok, open(args.filler, encoding="utf-8").read())
    T = ids.shape[1]
    hidden, keys_t, values_t, _ = prefill_doc(model, ids)
    k_pre, v_pre = pre_rope_kv(model, hidden)
    pos = torch.arange(T).unsqueeze(0)

    # roundtrip check (trajectory -> KV through frozen weights)
    kr = rope_k(model, k_pre, pos)
    k_err = (kr - keys_t).abs().max().item()
    v_err = (v_pre - values_t).abs().max().item()
    print(f"roundtrip max err: K {k_err:.2e} V {v_err:.2e} "
          f"(fp16 storage noise expected)")

    wK, wV = behavioral_weights(model, hidden)
    scale = metric_scale(wK, wV)
    rows = []

    def recall_run(name, keys, values):
        from a7_needle import NEEDLES
        hits, nlls, per = 0, [], []
        for (nm, num), d in zip(NEEDLES, depths):
            ok, nll = probe_needle(model, tok, keys, values, T, nm, num)
            hits += ok
            nlls.append(nll)
            per.append("Y" if ok else ".")
        rows.append((name, hits, sum(nlls) / len(nlls), "".join(per)))
        print(f"  {name:36s} recall {hits}/8  gold-NLL {rows[-1][2]:.3f}  "
              f"[{rows[-1][3]}]")

    recall_run("full KV", keys_t, values_t)
    for name, sc, R, cb in [
        (f"variance 16x (rank {R16})", None, R16, 8),
        (f"behavioral 16x (rank {R16})", scale, R16, 8),
        (f"behavioral 32x (rank {R32})", scale, R32, 8),
    ]:
        R = min(R, T - 8)
        codec = fit_joint(k_pre, v_pre, T, R, scale=sc)
        kh, vh = apply_joint(codec, k_pre, v_pre, coeff_bits=cb)
        recall_run(name, rope_k(model, kh, pos), vh)

    # KL eval on a second doc
    text = open("data/mobydick.txt", encoding="utf-8", errors="ignore").read()
    m_ids = tok(text[30000:], return_tensors="pt").input_ids[:, :2048]
    Tm, P = m_ids.shape[1], 1792
    hid_m, keys_m, values_m, _ = prefill_doc(model, m_ids)
    km, vm = pre_rope_kv(model, hid_m)
    pos_m = torch.arange(Tm).unsqueeze(0)
    base = eval_window(model, m_ids, keys_m, values_m, P)
    wKm, wVm = behavioral_weights(model, hid_m)
    codec = fit_joint(km, vm, P, min(R16, P - 8), scale=metric_scale(wKm, wVm))
    kh, vh = apply_joint(codec, km, vm, coeff_bits=8)
    kl = behavior_metrics(base, eval_window(model, m_ids, rope_k(model, kh, pos_m),
                                            vh, P), m_ids.squeeze(0)[P + 1:])
    print(f"  behavioral 16x KL {kl['kl_mean']:.4f}  top1 {kl['top1_agree']:.3f}")

    slug = args.model.split("/")[-1].replace(".", "")
    rpt = os.path.join(args.report_dir, f"b2_family_{slug}.md")
    os.makedirs(args.report_dir, exist_ok=True)
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"# B2: family transfer -- {args.model}\n\n"
                f"L={L}, kvh={kvh}, head_dim={hd}, fp16 KV {fp16_bytes} B/token. "
                f"Roundtrip max err K {k_err:.2e} / V {v_err:.2e}.\n\n"
                "| method | recall | gold-NLL | per-needle |\n|---|---|---|---|\n")
        for nm, h, nll, per in rows:
            f.write(f"| {nm} | {h}/8 | {nll:.3f} | `{per}` |\n")
        f.write(f"\nMoby-Dick teacher-forced window, behavioral 16x: KL "
                f"{kl['kl_mean']:.4f}, top-1 {kl['top1_agree']:.3f}, "
                f"dNLL {kl['delta_nll']:+.4f}.\n")
    print(f"wrote {rpt}")


if __name__ == "__main__":
    main()
