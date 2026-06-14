#!/usr/bin/env python3
"""Batch literature search for the NN-architecture research front.

Runs a curated set of (front, query, sources) probes through the paper_search_mcp
CLI, dedups by normalized title, and writes a compact JSONL + a per-front markdown
digest.  Keeps full abstracts on disk (not in the agent's context) until curation.

Run:  ~/miniforge3/bin/python3 cpp_pricer/tools/lit_search.py
Out:  cpp_pricer/docs/research/lit/raw_hits.jsonl  (one paper per line, with `front`)
      cpp_pricer/docs/research/lit/digest_<front>.md
"""
import json, re, subprocess, sys, time
from pathlib import Path

PY = str(Path.home() / "miniforge3/bin/python3")
OUT = Path("cpp_pricer/docs/research/lit")
OUT.mkdir(parents=True, exist_ok=True)

# (front, query, sources, n_per_source)
PROBES = [
    # ---------------- Front A: ACCURACY (shallow-net function approximation) ---------------
    ("accuracy", "activation function comparison SiLU GELU Mish deep network accuracy", "arxiv", 5),
    ("accuracy", "value function approximation neural network reinforcement learning accuracy", "arxiv", 5),
    ("accuracy", "residual connection shallow MLP function approximation", "arxiv", 5),
    ("accuracy", "small neural network regression accuracy architecture", "arxiv,semantic", 4),
    ("accuracy", "layer normalization training stability accuracy MLP", "arxiv", 5),
    ("accuracy", "actor critic continuous control deep deterministic policy gradient accuracy", "arxiv", 5),
    ("accuracy", "input feature normalization neural network regression performance", "arxiv", 4),
    ("accuracy", "expressivity depth width tradeoff neural network approximation", "arxiv", 4),
    # ---------------- Front B: SEED-TO-SEED VARIANCE / reproducibility / stability ----------
    ("variance", "reproducibility variance random seed deep reinforcement learning", "arxiv", 5),
    ("variance", "stochastic weight averaging generalization flat minima", "arxiv", 4),
    ("variance", "ensemble neural network variance reduction prediction stability", "arxiv", 4),
    ("variance", "orthogonal initialization variance training stability deep network", "arxiv", 4),
    ("variance", "spectral normalization stability reinforcement learning", "arxiv", 4),
    ("variance", "exponential moving average target network stability off-policy RL", "arxiv", 4),
    ("variance", "weight initialization variance signal propagation deep network", "arxiv", 4),
    ("variance", "bootstrapped ensemble value function uncertainty variance DQN", "arxiv", 4),
    # ---------------- Front C: COMPUTATION TIME (shallow nets, CPU) -------------------------
    ("time", "neural network pruning width reduction inference speedup", "arxiv", 5),
    ("time", "knowledge distillation small student network speedup", "arxiv", 4),
    ("time", "low rank factorization neural network acceleration", "arxiv", 4),
    ("time", "lottery ticket hypothesis sparse training efficiency", "arxiv", 4),
    ("time", "extreme learning machine random feature closed form fast", "arxiv", 4),
    ("time", "efficient MLP CPU inference small model latency", "arxiv", 4),
    ("time", "structured matrix fast neural network multiplication", "arxiv", 3),
    # ---------------- Cross-cutting levers (init / batch / LR / approximation) -------------
    ("levers", "Fixup initialization residual networks without normalization", "arxiv", 3),
    ("levers", "LSUV layer-sequential unit variance initialization", "arxiv", 3),
    ("levers", "batch size learning rate linear scaling generalization", "arxiv", 4),
    ("levers", "Polyak averaging actor critic learning stability", "arxiv", 3),
    ("levers", "random Fourier features kernel approximation regression", "arxiv", 3),
    ("levers", "least squares Monte Carlo neural network optimal stopping", "arxiv", 4),
]


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def run_probe(front, query, sources, n):
    cmd = [PY, "-m", "paper_search_mcp.cli", "search", query, "-n", str(n), "-s", sources]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[timeout] {query!r}\n")
        return []
    # JSON is the last {...} block on stdout
    out = r.stdout
    i = out.find("{")
    if i < 0:
        sys.stderr.write(f"[no-json] {query!r}: {r.stderr[:200]}\n")
        return []
    try:
        data = json.loads(out[i:])
    except Exception as e:
        sys.stderr.write(f"[parse-fail] {query!r}: {e}\n")
        return []
    papers = data.get("papers", [])
    for p in papers:
        p["_front"] = front
        p["_query"] = query
    return papers


def main():
    seen = {}
    order = []
    for front, query, sources, n in PROBES:
        sys.stderr.write(f"[probe] {front:8s} | {query}\n")
        for p in run_probe(front, query, sources, n):
            key = norm_title(p.get("title"))
            if not key:
                continue
            if key in seen:
                # keep first front tag; record extra query hit
                seen[key].setdefault("_also", []).append(p["_query"])
                continue
            seen[key] = p
            order.append(key)
        time.sleep(0.4)

    raw = OUT / "raw_hits.jsonl"
    with raw.open("w") as f:
        for k in order:
            f.write(json.dumps(seen[k]) + "\n")

    # per-front compact digests
    by_front = {}
    for k in order:
        by_front.setdefault(seen[k]["_front"], []).append(seen[k])
    for front, items in by_front.items():
        md = OUT / f"digest_{front}.md"
        with md.open("w") as f:
            f.write(f"# Lit digest — {front} ({len(items)} unique)\n\n")
            for p in items:
                yr = (p.get("published_date") or "")[:4]
                f.write(f"- **{p.get('title','?').strip()}** ({yr}) "
                        f"[{p.get('source')}:{p.get('paper_id')}] doi:{p.get('doi') or '—'}\n")
                ab = (p.get("abstract") or "").strip().replace("\n", " ")
                f.write(f"  - {ab[:480]}\n")

    counts = {fr: len(v) for fr, v in by_front.items()}
    sys.stderr.write(f"\n[done] unique={len(order)} by_front={counts}\n")
    print(json.dumps({"unique": len(order), "by_front": counts}))


if __name__ == "__main__":
    main()
