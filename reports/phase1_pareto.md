# Phase 1b: bytes/token vs behavior (Qwen/Qwen2.5-0.5B, doc ../PREPRINT.md)

Past = 1792 tokens compressed, window = 256 tokens evaluated teacher-forced against the full-KV baseline. fp16 full KV = 12288 B/token. Bases fit on past tokens (per-doc adaptive, amortized). Recompute-from-tokens baseline: 0 B/token marginal, full prefill compute -- every method below must justify its bytes by decode compute saved.

| method | B/token | ratio | KL mean | KL p95 | top-1 | top-5 | dNLL |
|---|---|---|---|---|---|---|---|
| full KV fp16 | 12288 | 1.0x | 0 | 0 | 1.000 | 1.000 | +0.0000 |
| quant K8 V8 | 6240 | 2.0x | 0.0003 | 0.0008 | 0.992 | 0.987 | +0.0012 |
| per-layer rank64 c16 | 6144 | 2.0x | 1.4920 | 4.6780 | 0.492 | 0.511 | +1.4942 |
| quant K4 V8 | 4704 | 2.6x | 0.1152 | 0.3812 | 0.789 | 0.819 | +0.1585 |
| quant K8 V4 | 4704 | 2.6x | 0.0029 | 0.0076 | 0.957 | 0.963 | +0.0051 |
| quant K4 V4 | 3168 | 3.9x | 0.1241 | 0.4438 | 0.797 | 0.799 | +0.1704 |
| per-layer rank32 c16 | 3072 | 4.0x | 2.4650 | 7.4584 | 0.406 | 0.417 | +2.5559 |
| quant K2 V4 | 2400 | 5.1x | 5.4701 | 10.7856 | 0.047 | 0.125 | +5.6082 |
| quant K4 V2 | 2400 | 5.1x | 0.6339 | 2.1426 | 0.609 | 0.645 | +0.7356 |
| quant K2 V2 | 1632 | 7.5x | 5.5161 | 10.6132 | 0.066 | 0.147 | +5.7412 |
| per-layer rank16 c16 | 1536 | 8.0x | 3.6619 | 10.0443 | 0.297 | 0.362 | +3.9258 |
| joint-stack rank768 c16 | 1536 | 8.0x | 0.0327 | 0.1065 | 0.875 | 0.902 | +0.0580 |
| per-layer rank8 c16 | 768 | 16.0x | 4.5868 | 11.5218 | 0.199 | 0.278 | +4.8362 |
| per-layer rank16 c8 | 768 | 16.0x | 3.6568 | 9.9817 | 0.297 | 0.365 | +3.9202 |
| joint-stack rank384 c16 | 768 | 16.0x | 0.5267 | 1.9140 | 0.637 | 0.680 | +0.6415 |
| joint-stack rank768 c8 | 768 | 16.0x | 0.0325 | 0.0950 | 0.887 | 0.899 | +0.0575 |
| trajectory rank384 c16 | 768 | 16.0x | 0.5104 | 2.5457 | 0.703 | 0.718 | +0.6300 |
| per-layer rank4 c16 | 384 | 32.0x | 4.2623 | 10.9521 | 0.164 | 0.264 | +4.5304 |
| per-layer rank8 c8 | 384 | 32.0x | 4.5856 | 11.4861 | 0.199 | 0.280 | +4.8325 |
| joint-stack rank192 c16 | 384 | 32.0x | 6.5384 | 14.5347 | 0.121 | 0.138 | +6.8257 |
| joint-stack rank768 c4 | 384 | 32.0x | 0.1705 | 0.5338 | 0.789 | 0.783 | +0.2444 |
| joint-stack rank384 c8 | 384 | 32.0x | 0.5204 | 1.8757 | 0.637 | 0.680 | +0.6374 |
| trajectory rank384 c8 | 384 | 32.0x | 0.6373 | 2.9971 | 0.641 | 0.682 | +0.7423 |
| trajectory rank192 c16 | 384 | 32.0x | 0.9218 | 4.0791 | 0.586 | 0.648 | +1.1126 |
| joint-stack rank96 c16 | 192 | 64.0x | 4.3446 | 10.2784 | 0.184 | 0.284 | +4.6624 |
| joint-stack rank192 c8 | 192 | 64.0x | 6.5641 | 14.6148 | 0.113 | 0.135 | +6.8466 |
| trajectory rank384 c4 | 192 | 64.0x | 0.6832 | 3.1522 | 0.625 | 0.648 | +0.7762 |
| trajectory rank192 c8 | 192 | 64.0x | 1.0648 | 4.9940 | 0.559 | 0.597 | +1.2161 |
| trajectory rank96 c16 | 192 | 64.0x | 1.3060 | 5.4244 | 0.492 | 0.565 | +1.4655 |
| joint-stack rank48 c16 | 96 | 128.0x | 3.7377 | 10.2066 | 0.227 | 0.320 | +3.9097 |
| trajectory rank96 c8 | 96 | 128.0x | 1.4295 | 5.6212 | 0.457 | 0.534 | +1.5536 |

Baseline NLL over window: 2.7526 nats/token.
