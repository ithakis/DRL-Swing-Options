#!/usr/bin/env python3
"""Fetch seminal/anchor papers by distinctive title and merge into raw_hits.jsonl.
Regenerates ALL three front digests. These anchors ground the experiment design."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from lit_search import run_probe, norm_title, OUT

# (front, distinctive-title-query, sources, n)
ANCHORS = [
    ("variance", "Deep Reinforcement Learning that Matters reproducibility", "arxiv", 2),
    ("variance", "Averaging Weights Leads to Wider Optima and Better Generalization", "arxiv", 2),
    ("variance", "Implementation Matters in Deep Policy Gradients case study PPO TRPO", "arxiv", 2),
    ("variance", "What Matters In On-Policy Reinforcement Learning large scale empirical", "arxiv", 2),
    ("variance", "Dropout regularization preventing co-adaptation neural networks", "arxiv", 2),
    ("variance", "Sharpness-Aware Minimization efficiently improving generalization", "arxiv", 2),
    ("accuracy", "Gaussian Error Linear Units GELUs Hendrycks", "arxiv", 2),
    ("accuracy", "Self-Normalizing Neural Networks SELU", "arxiv", 2),
    ("accuracy", "Exact solutions nonlinear dynamics learning deep linear neural networks", "arxiv", 2),
    ("accuracy", "Mish self regularized non-monotonic activation function", "arxiv", 2),
    ("accuracy", "deep equilibrium models implicit layers", "arxiv", 2),
    ("time", "Distilling the Knowledge in a Neural Network", "arxiv", 2),
    ("time", "The Lottery Ticket Hypothesis finding sparse trainable neural networks", "arxiv", 2),
    ("time", "Deep Compression pruning trained quantization Huffman coding", "arxiv", 2),
    ("time", "Predicting Parameters in Deep Learning low rank", "arxiv", 2),
    ("time", "mixed precision training half precision", "arxiv", 2),
    ("time", "rational activation functions efficient neural networks", "arxiv", 2),
]

raw = OUT / "raw_hits.jsonl"
seen, order = {}, []
if raw.exists():
    for line in raw.read_text().splitlines():
        if line.strip():
            p = json.loads(line); k = norm_title(p.get("title"))
            seen[k] = p; order.append(k)

added = 0
for front, q, src, n in ANCHORS:
    sys.stderr.write(f"[anchor] {front} | {q}\n")
    for p in run_probe(front, q, src, n):
        k = norm_title(p.get("title"))
        if not k or k in seen:
            continue
        seen[k] = p; order.append(k); added += 1

with raw.open("w") as f:
    for k in order:
        f.write(json.dumps(seen[k]) + "\n")

counts = {}
for fr in ("accuracy", "variance", "time"):
    items = [seen[k] for k in order if seen[k].get("_front") == fr]
    counts[fr] = len(items)
    md = OUT / f"digest_{fr}.md"
    with md.open("w") as f:
        f.write(f"# Lit digest — {fr} ({len(items)} unique)\n\n")
        for p in items:
            yr = (p.get("published_date") or "")[:4]
            f.write(f"- **{p.get('title','?').strip()}** ({yr}) [{p.get('source')}:{p.get('paper_id')}] doi:{p.get('doi') or '—'}\n")
            ab = (p.get("abstract") or "").strip().replace("\n", " ")
            f.write(f"  - {ab[:420]}\n")

print(json.dumps({"added": added, "counts": counts, "grand_total": len(order)}))
