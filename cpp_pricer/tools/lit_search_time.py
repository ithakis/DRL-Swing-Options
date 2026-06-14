#!/usr/bin/env python3
"""Top-up probe batch for the COMPUTE-TIME front (need >=20). Appends to raw_hits.jsonl,
dedups against what's already there, and rewrites digest_time.md."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from lit_search import run_probe, norm_title, OUT

PROBES = [
    ("time", "SIMD vectorization small matrix multiply CPU neural network", "arxiv", 4),
    ("time", "network compression weight sharing quantization MLP", "arxiv", 4),
    ("time", "fast approximation activation function lookup exp sigmoid", "arxiv", 4),
    ("time", "training acceleration fewer iterations convergence deep network", "arxiv", 4),
    ("time", "depthwise separable bottleneck efficient architecture", "arxiv", 4),
    ("time", "early stopping budget reinforcement learning sample efficiency", "arxiv", 4),
    ("time", "tensor decomposition compress fully connected layer speedup", "arxiv", 3),
    ("time", "hardware aware efficient inference edge low latency network", "arxiv", 4),
]

raw = OUT / "raw_hits.jsonl"
seen = {}
order = []
if raw.exists():
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        k = norm_title(p.get("title"))
        seen[k] = p; order.append(k)

added = 0
for front, q, src, n in PROBES:
    sys.stderr.write(f"[probe+] {front} | {q}\n")
    for p in run_probe(front, q, src, n):
        k = norm_title(p.get("title"))
        if not k or k in seen:
            continue
        seen[k] = p; order.append(k); added += 1

with raw.open("w") as f:
    for k in order:
        f.write(json.dumps(seen[k]) + "\n")

# rewrite digest_time
items = [seen[k] for k in order if seen[k].get("_front") == "time"]
md = OUT / "digest_time.md"
with md.open("w") as f:
    f.write(f"# Lit digest — time ({len(items)} unique)\n\n")
    for p in items:
        yr = (p.get("published_date") or "")[:4]
        f.write(f"- **{p.get('title','?').strip()}** ({yr}) [{p.get('source')}:{p.get('paper_id')}] doi:{p.get('doi') or '—'}\n")
        ab = (p.get("abstract") or "").strip().replace("\n", " ")
        f.write(f"  - {ab[:480]}\n")

print(json.dumps({"added": added, "time_total": len(items), "grand_total": len(order)}))
