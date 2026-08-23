# A5: cross-document basis transfer (eval doc: Moby-Dick, 2048 tokens; foreign basis donor: kvspace preprint)

| method / basis | KL mean | KL p95 | top-1 | top-5 | dNLL |
|---|---|---|---|---|---|
| quant K8 V4 (basis-free, 2.6x) | 0.0022 | 0.0067 | 0.980 | 0.966 | +0.0030 |
| joint768 c8 16x, basis per-doc (fit on B) | 0.0111 | 0.0320 | 0.949 | 0.934 | -0.0044 |
| joint768 c8 16x, basis foreign (fit on A) | 0.2685 | 0.7781 | 0.730 | 0.737 | +0.3053 |
| traj384 c8 32x, basis per-doc (fit on B) | 0.2115 | 0.5898 | 0.727 | 0.752 | +0.2544 |
| traj384 c8 32x, basis foreign (fit on A) | 0.5058 | 1.5703 | 0.652 | 0.651 | +0.6276 |

Reading: per-doc >> foreign  => document context buys compression (supports A4 at doc granularity, per-doc adaptive bases worth their storage). per-doc ~= foreign => structure is global; a fixed basis ships with the model (deployability up, context evidence down).
