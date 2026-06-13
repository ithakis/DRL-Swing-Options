#!/usr/bin/env python3
"""Collect benchmark + validation data for the C++ pricer notebook.

Subcommands:
  cpp-seeds    run the C++ binary over a seed range -> data/results_cpp_seeds.csv
  py-seeds     run the PyTorch baseline over a seed range -> data/results_py_seeds.csv
  cpp-scaling  C++ time vs #paths (train & eval grids) -> data/scaling_cpp.csv
"""
import os, sys, json, subprocess, csv, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repo root
CPP = os.path.join(HERE, "..", "build", "price_swing")
DATA = os.path.join(HERE, "..", "data")
PYBASE = os.path.join(HERE, "python_baseline.py")


def run_cpp(seed, n_train, n_eval, threads=8):
    out = subprocess.check_output([CPP, "--seed", str(seed), "--n_train", str(n_train),
                                   "--n_eval", str(n_eval), "--threads", str(threads),
                                   "--kernel", os.path.join(DATA, "kernel_v64.bin")],
                                  cwd=os.path.join(HERE, ".."))
    return json.loads(out)


def run_py(seed, n_train, n_eval, threads=8):
    out = subprocess.check_output([sys.executable, PYBASE, "--seed", str(seed),
                                   "--n_train", str(n_train), "--n_eval", str(n_eval),
                                   "--threads", str(threads)], cwd=ROOT,
                                  stderr=subprocess.DEVNULL)
    # baseline prints non-JSON banner lines then the JSON block; take from first '{'
    s = out.decode(); s = s[s.index("{"):]
    return json.loads(s)


def write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cpp-seeds", "py-seeds", "cpp-scaling"])
    ap.add_argument("--lo", type=int, default=11)
    ap.add_argument("--hi", type=int, default=25)
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--n_eval", type=int, default=65536)
    args = ap.parse_args()

    if args.cmd == "cpp-seeds":
        rows = []
        for seed in range(args.lo, args.hi + 1):
            r = run_cpp(seed, args.n_train, args.n_eval); rows.append(r)
            print(f"  cpp seed {seed}: price={r['price']:.4f} t={r['t_total']:.1f}s")
        write_csv(os.path.join(DATA, "results_cpp_seeds.csv"), rows)

    elif args.cmd == "py-seeds":
        rows = []
        for seed in range(args.lo, args.hi + 1):
            r = run_py(seed, args.n_train, args.n_eval); rows.append(r)
            print(f"  py  seed {seed}: price={r['price']:.4f} t={r['t_total']:.1f}s")
        write_csv(os.path.join(DATA, "results_py_seeds.csv"), rows)

    elif args.cmd == "cpp-scaling":
        rows = []
        for ne in [4096, 8192, 16384, 32768, 65536, 131072]:
            r = run_cpp(11, 64, ne); r["phase"] = "eval"; rows.append(r)
            print(f"  eval n={ne}: t_eval={r['t_eval']:.4f}s")
        for nt in [512, 1024, 2048, 4096, 8192]:
            r = run_cpp(11, nt, 4096); r["phase"] = "train"; rows.append(r)
            print(f"  train n={nt}: t_train={r['t_train']:.2f}s")
        write_csv(os.path.join(DATA, "scaling_cpp.csv"), rows)


if __name__ == "__main__":
    main()
