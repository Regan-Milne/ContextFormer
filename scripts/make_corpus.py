"""Assemble the code and structured-text corpus files for A25/DR2.

- code_train.txt: kvspace demo sources (train side of the corpus basis)
- code_eval.txt: EvolveR runtime scripts (held out from any fitting)
- structured.txt: deterministic synthetic config/log text, the
  identifier-heavy class (seeded; no wall-clock or randomness leaks)
"""

import os
import random

def cat(paths, out):
    with open(out, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"\n# ==== {os.path.basename(p)} ====\n")
            f.write(open(p, encoding="utf-8", errors="ignore").read())
    print(f"{out}: {os.path.getsize(out)} bytes")

KVS = r"C:\ComputerStuff\kvspace\demo"
EVR = r"C:\ComputerStuff\EvolveR"

cat([os.path.join(KVS, n) for n in
     ["runtime_r0.py", "runtime_r2.py", "kvspace.py", "tier_probe.py"]],
    "data/code_train.txt")

cat([os.path.join(EVR, n) for n in
     ["chat_server.py", "profile_decode.py", "wire_quant_test.py"]],
    "data/code_eval.txt")

rng = random.Random(11)
SERVICES = ["auth", "ledger", "ingest", "replica", "cache", "router",
            "billing", "archive", "metrics", "scheduler"]
lines = []
for i in range(3500):
    svc = rng.choice(SERVICES)
    h = "".join(rng.choice("0123456789abcdef") for _ in range(12))
    lines.append(
        f"[2026-0{rng.randint(1,9)}-{rng.randint(10,28)}T"
        f"{rng.randint(10,23)}:{rng.randint(10,59)}:{rng.randint(10,59)}Z] "
        f"{svc}_worker_{rng.randint(0,63)} req={h} "
        f"route=/api/v{rng.randint(1,4)}/{svc}/{rng.choice(['get','put','sync','flush'])} "
        f"status={rng.choice([200,200,200,404,500,503])} "
        f"lat_ms={rng.randint(1,900)} bytes={rng.randint(64,65536)}")
with open("data/structured.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"data/structured.txt: {os.path.getsize('data/structured.txt')} bytes")
