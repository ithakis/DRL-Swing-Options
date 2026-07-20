#!/usr/bin/env python
"""DP pricer — validation & performance suite (12 tests).

Drives the compiled grid-DP binary ``grid_dp_pricer/build/price_dp`` (FP64) and
``grid_dp_pricer/build_fp32/price_dp`` (FP32) via subprocess, runs the 12 V&V tests below, and
writes one tidy CSV per test into ``results/``.  The companion notebook ``DP_Validation.ipynb``
loads those CSVs and renders the figures/tables.  Nothing in ``src/`` / ``cpp_pricer/`` /
``grid_dp_pricer`` is modified — this only *calls* the binary (same pattern as
``grid_dp_pricer/tools/convergence_analysis.py`` and ``pareto.py``).

Tests
  T1  spatial order-of-accuracy        (nX, nY, nQ)          -> results/T1_spatial.csv (+T1_summary)
  T2  quadrature convergence           (Mx, Nmax, glU, glagJ)-> results/T2_quadrature.csv
  T3  domain-truncation error          (Xhi, Yhi)            -> results/T3_truncation.csv
  T4  interpolation-order study        (linear vs cubic)     -> results/T4_interp.csv
  T5  Richardson / GCI certified PV     (from T1)            -> results/T5_budget.csv
  T6  FP32 vs FP64 round-off floor                           -> results/T6_precision.csv
  T7  Greek bump-size study            (h_S)                 -> results/T7_bump.csv
  T8  Greek grid-convergence           (nX, nQ)              -> results/T8_greek_grid.csv
  T9  Greek spatial profile            (V, D, G vs S0)       -> results/T9_profile.csv
  T10 forward-MC self-consistency      (lower bound)         -> results/T10_selfconsistency.csv
  T10b large-N forward-MC bootstrap    (opt-in, ~5 min)       -> results/T10_mc_bootstrap(_summary).csv
  T11 limiting cases + cross-validation                      -> results/T11_monotonicity.csv (+ ordering)
  T12 speed-accuracy Pareto + complexity                     -> results/T12_pareto.csv

Greeks are noise-free: price_dp evaluates V0 = U0(log S0, 0, 0) and the backward array U is
independent of S0, so a central finite difference in --S0 (relative bump dS = h_S * S0, with a
(h, h/2) Richardson step) gives deterministic Delta = dV/dS0 and Gamma = d2V/dS02 — matching the
relative-bump / Richardson convention in src/greeks.py.

Usage
  python run_dp_validation.py                 # full run, all tests
  python run_dp_validation.py --quick         # coarser ladders (fast smoke run)
  python run_dp_validation.py --only T7,T9    # re-run selected tests only
  python run_dp_validation.py --mc-full 4000000
"""
import argparse
import csv
import json
import math
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))               # .../DRL-Swing-Options(/worktree)
RESULTS = os.path.join(HERE, "results")
GDP = os.path.join(REPO, "grid_dp_pricer")
BIN64 = os.path.join(GDP, "build", "price_dp")
BIN32 = os.path.join(GDP, "build_fp32", "price_dp")
SWEEP_CSV = os.path.join(GDP, "data", "dp_grid_sweep.csv")  # reused by T11 (DP vs LSM-D/RL grid)

# Reference grid held on the *non-swept* axes (each already converged ~1e-4, so it does not
# contaminate the order estimate of the swept axis) — mirrors tools/convergence_analysis.py.
REF = dict(nX=121, nY=121, nQ=151, Mx=24, Nmax=3, glU=16, glagJ=16,
           Xlo=-1.6, Xhi=1.6, Ylo=0.0, Yhi=4.0, interp=1)
H_S_DEFAULT = 1e-2          # default relative spot bump for Greeks (== src/greeks.py default)

_CACHE = {}                 # price() is deterministic in (bin, cfg) -> memoize across tests


# --------------------------------------------------------------------------- binary driver
def _key(binpath, cfg):
    return (binpath, tuple(sorted((k, str(v)) for k, v in cfg.items())))


def run(binpath=BIN64, threads=0, repeats=1, timed=False, **flags):
    """Call price_dp; return its parsed JSON dict.  Cached unless timed (T12 timing bypasses)."""
    cfg = {**REF, **flags}
    k = _key(binpath, cfg)
    if not timed and k in _CACHE:
        return _CACHE[k]
    args = [binpath, "--threads", str(threads)]
    for kk, vv in cfg.items():
        args += [f"--{kk}", str(vv)]
    best = None
    for _ in range(max(1, repeats)):
        out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
        j = json.loads(out)
        if best is None or j["t_total_ms"] < best["t_total_ms"]:
            best = j
    if not timed:
        _CACHE[k] = best
    return best


def price(binpath=BIN64, **flags):
    return run(binpath, **flags)["price"]


def value_at(S0, binpath=BIN64, **grid):
    return run(binpath, S0=S0, **grid)["price"]


def greeks(h_s=H_S_DEFAULT, S0=1.0, binpath=BIN64, **grid):
    """Central-difference Delta, Gamma at relative bump dS = h_s * S0 (noise-free; DP is
    deterministic in S0).  Returns (V0, Delta, Gamma)."""
    dS = h_s * S0
    vp = value_at(S0 + dS, binpath=binpath, **grid)
    v0 = value_at(S0, binpath=binpath, **grid)
    vm = value_at(S0 - dS, binpath=binpath, **grid)
    delta = (vp - vm) / (2.0 * dS)
    gamma = (vp - 2.0 * v0 + vm) / (dS * dS)
    return v0, delta, gamma


# --------------------------------------------------------------------------- math helpers
def fit_order(hs, errs):
    """Observed order p = LS slope of log|err| vs log h over points with err>0."""
    xs = [math.log(h) for h, e in zip(hs, errs) if e and e > 0]
    ys = [math.log(e) for e in errs if e and e > 0]
    n = len(xs)
    if n < 2:
        return float("nan")
    sx, sy = sum(xs), sum(ys)
    sxx, sxy = sum(x * x for x in xs), sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if den else float("nan")


def richardson3(hs, ps):
    """3-point Richardson on the finest triple (lists coarse->fine): (order, extrapolated)."""
    (h0, p0), (h1, p1), (h2, p2) = zip(hs[-3:], ps[-3:])
    d01, d12 = p1 - p0, p2 - p1
    if d01 == 0 or d12 == 0 or (d12 / d01) <= 0:
        return float("nan"), ps[-1]
    r = h0 / h1
    p_order = math.log(abs(d01 / d12)) / math.log(r)
    extr = p2 + d12 / (r ** p_order - 1.0)
    return p_order, extr


def gci_fine(hs, ps, p, Fs=1.25):
    """Roache Grid Convergence Index on the finest pair (relative, %).  Fs=1.25 for >=3 grids."""
    r = hs[-2] / hs[-1]
    f2, f1 = ps[-2], ps[-1]              # coarse, fine
    if f1 == 0 or not math.isfinite(p) or p <= 0:
        return float("nan")
    eps = abs((f2 - f1) / f1)
    return Fs * eps / (r ** p - 1.0) * 100.0


# --------------------------------------------------------------------------- io helpers
def write_csv(name, rows, fields):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"    wrote results/{name}  ({len(rows)} rows)")
    return path


def h_of(axis, n):
    span = {"nX": REF["Xhi"] - REF["Xlo"], "nY": REF["Yhi"] - REF["Ylo"],
            "nQ": 20.0}                              # Q_max - Q_min
    return span[axis] / (n - 1)


# --------------------------------------------------------------------------- T1: spatial order
def t1_spatial(quick):
    print("[T1] spatial order-of-accuracy (nX, nY, nQ)")
    # Established ladders (match grid_dp_pricer/tools/convergence_analysis.py and RESULTS.md so the
    # controlling-axis finding stays consistent); the reference resolution (121 / 151) is included on
    # every axis so resid@ref is well-defined.
    res = {"nX": [41, 61, 81, 121], "nY": [41, 61, 81, 121], "nQ": [51, 76, 101, 151]} if quick \
        else {"nX": [41, 61, 81, 121, 161, 241], "nY": [41, 61, 81, 121, 161, 241],
              "nQ": [51, 76, 101, 151, 201, 301]}
    rows, summ = [], []
    for axis, ns in res.items():
        hs, ps = [], []
        for n in ns:
            j = run(**{axis: n})
            hs.append(h_of(axis, n)); ps.append(j["price"])
            rows.append(dict(axis=axis, n=n, h=h_of(axis, n), price=j["price"],
                             t_backward_ms=j["t_backward_ms"]))
        p_rich, extr = richardson3(hs, ps)
        if not math.isfinite(extr):
            extr = ps[-1]                                   # fall back to the finest price
        # residual of the *reference-grid* resolution along this axis (the default the binary uses)
        ref_n = REF[axis]
        p_ref = ps[ns.index(ref_n)] if ref_n in ns else ps[-1]
        err_ref = abs(p_ref - extr)
        p_fit = fit_order(hs[:-1], [abs(p - extr) for p in ps[:-1]])
        p_for_gci = p_rich if (math.isfinite(p_rich) and p_rich > 0) else \
            (p_fit if (math.isfinite(p_fit) and p_fit > 0) else 1.0)
        g = gci_fine(hs, ps, p_for_gci)
        summ.append(dict(axis=axis, order_fit=p_fit, order_rich=p_rich, extrap=extr,
                         finest=ps[-1], err_at_ref=err_ref, gci_pct=g))
        print(f"     {axis}: p_rich~{p_rich:5.2f} p_fit~{p_fit:5.2f}  extrap={extr:.6f}  "
              f"resid@ref={err_ref:.2e}  GCI={g:.2e}%")
    write_csv("T1_spatial.csv", rows, ["axis", "n", "h", "price", "t_backward_ms"])
    write_csv("T1_summary.csv", summ,
              ["axis", "order_fit", "order_rich", "extrap", "finest", "err_at_ref", "gci_pct"])
    ctrl = max(summ, key=lambda s: s["err_at_ref"])["axis"]
    print(f"     controlling axis = {ctrl}")
    return summ


# --------------------------------------------------------------------------- T2: quadrature
def t2_quadrature(quick):
    print("[T2] quadrature convergence (Mx, Nmax, glU, glagJ)")
    axes = {"Mx": [4, 6, 8, 12, 16, 24, 32], "Nmax": [1, 2, 3, 4, 5, 6],
            "glU": [4, 8, 12, 16, 24], "glagJ": [4, 8, 12, 16, 24]}
    if quick:
        axes = {"Mx": [4, 6, 8, 12, 24], "Nmax": [1, 2, 3, 5], "glU": [4, 8, 16],
                "glagJ": [4, 8, 16]}
    rows = []
    for axis, ns in axes.items():
        ref_val = run(**{axis: ns[-1]})["price"]
        for n in ns:
            p = run(**{axis: n})["price"]
            rows.append(dict(axis=axis, n=n, price=p, err=abs(p - ref_val)))
        print(f"     {axis}: err(coarsest {ns[0]})={abs(rows[-len(ns)]['price']-ref_val):.2e} "
              f"-> ~0 at {ns[-1]}")
    write_csv("T2_quadrature.csv", rows, ["axis", "n", "price", "err"])


# --------------------------------------------------------------------------- T3: truncation
def t3_truncation(quick):
    print("[T3] domain-truncation (localization) error (Xhi, Yhi)")
    # Constant grid SPACING while extending the boundary, so truncation is isolated from
    # discretization: nodes = round(span / h_ref) + 1.
    hX = h_of("nX", REF["nX"]); hY = h_of("nY", REF["nY"])
    xhis = [0.8, 1.0, 1.2, 1.6, 2.0, 2.4] if not quick else [0.8, 1.2, 1.6, 2.4]
    yhis = [2.0, 3.0, 4.0, 5.0, 6.0] if not quick else [2.0, 4.0, 6.0]
    rows = []
    refX = run(nX=int(round(2 * max(xhis) / hX)) + 1, Xlo=-max(xhis), Xhi=max(xhis))["price"]
    for xh in xhis:
        nx = int(round(2 * xh / hX)) + 1
        p = run(nX=nx, Xlo=-xh, Xhi=xh)["price"]
        rows.append(dict(axis="Xhi", half_width=xh, n=nx, price=p, err=abs(p - refX)))
    refY = run(nY=int(round(max(yhis) / hY)) + 1, Yhi=max(yhis))["price"]
    for yh in yhis:
        ny = int(round(yh / hY)) + 1
        p = run(nY=ny, Yhi=yh)["price"]
        rows.append(dict(axis="Yhi", half_width=yh, n=ny, price=p, err=abs(p - refY)))
    write_csv("T3_truncation.csv", rows, ["axis", "half_width", "n", "price", "err"])
    print(f"     Xhi err @0.8={rows[0]['err']:.2e}  Yhi err @2.0="
          f"{[r for r in rows if r['axis']=='Yhi'][0]['err']:.2e}")


# --------------------------------------------------------------------------- T4: interp order
def t4_interp(quick):
    print("[T4] interpolation-order study (linear vs not-a-knot cubic)")
    # Refine BOTH spatial axes together (nX=nY=m) so the (X,Y) interpolation order is not masked
    # by a fixed-resolution residual on the un-swept axis; hold nQ/Mx fine so they don't floor it.
    ms = [31, 61, 121, 241] if not quick else [31, 61, 121]      # geometric (r=2 in m-1)
    fixed = dict(nQ=201, Mx=24)
    truth = run(nX=ms[-1], nY=ms[-1], interp=1, **fixed)["price"]  # shared h->0 limit
    rows = []
    for interp in (0, 1):
        for m in ms:
            p = run(nX=m, nY=m, interp=interp, **fixed)["price"]
            rows.append(dict(interp=interp, scheme="linear" if interp == 0 else "cubic",
                             m=m, h=1.0 / (m - 1), price=p, err=abs(p - truth)))
    for interp in (0, 1):
        sub = [r for r in rows if r["interp"] == interp][:-1]      # drop the truth point itself
        p = fit_order([r["h"] for r in sub], [r["err"] for r in sub])
        print(f"     {'cubic ' if interp else 'linear'}: observed order p~{p:.2f}")
    write_csv("T4_interp.csv", rows, ["interp", "scheme", "m", "h", "price", "err"])


# --------------------------------------------------------------------------- T5: Richardson / GCI
def _read_results(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _residual_at(rows, axis, ref_value):
    """|price at the reference setting - price at the finest setting| for one swept axis."""
    sub = [r for r in rows if r["axis"] == axis]
    if not sub:
        return None
    key = "n" if "n" in sub[0] else "half_width"
    by = {float(r[key]): float(r["price"]) for r in sub}
    finest = by[max(by)]
    ref = by.get(float(ref_value), finest)
    return abs(ref - finest)


def t5_budget(t1_summary):
    print("[T5] Richardson extrapolation -> certified PV +/- band (full error budget)")
    ref_price = run()["price"]                 # reference-grid focal price (binary default)
    rows, band = [], 0.0
    # (1) spatial discretization residuals (from T1, Richardson-extrapolated)
    for s in t1_summary:
        rows.append(dict(source="discretization", axis=s["axis"], order=round(s["order_rich"], 2)
                         if math.isfinite(s["order_rich"]) else "", residual=s["err_at_ref"]))
        band += s["err_at_ref"]
    # (2) quadrature residuals at the reference settings (from T2)
    t2 = _read_results("T2_quadrature.csv")
    for axis, ref in (("Mx", REF["Mx"]), ("Nmax", REF["Nmax"]),
                      ("glU", REF["glU"]), ("glagJ", REF["glagJ"])):
        d = _residual_at(t2, axis, ref)
        if d is not None:
            rows.append(dict(source="quadrature", axis=axis, order="spectral", residual=d))
            band += d
    # (3) domain-truncation residuals at the reference settings (from T3)
    t3 = _read_results("T3_truncation.csv")
    for axis, ref in (("Xhi", REF["Xhi"]), ("Yhi", REF["Yhi"])):
        d = _residual_at(t3, axis, ref)
        if d is not None:
            rows.append(dict(source="truncation", axis=axis, order="exp", residual=d))
            band += d
    rows.append(dict(source="TOTAL", axis="V0=%.6f" % ref_price, order="", residual=band))
    write_csv("T5_budget.csv", rows, ["source", "axis", "order", "residual"])
    print(f"     certified V0 = {ref_price:.6f} +/- {band:.1e}")


# --------------------------------------------------------------------------- T6: FP32 floor
def t6_precision(quick):
    print("[T6] FP32 vs FP64 round-off floor")
    if not os.path.exists(BIN32):
        print("     (skip: FP32 binary not built — cmake -DGRID_DP_FP64=OFF -B build_fp32)")
        return
    ladder = [(41, 51, 8), (61, 76, 12), (81, 101, 16), (121, 151, 24), (161, 201, 32)]
    if not quick:
        ladder += [(201, 251, 36)]
    # truth = Richardson extrapolation of the FP64 ladder (so neither endpoint is exactly 0)
    hs = [20.0 / (nq - 1) for (_, nq, _) in ladder]
    p64 = [run(BIN64, nX=nxy, nY=nxy, nQ=nq, Mx=mx)["price"] for (nxy, nq, mx) in ladder]
    _, truth = richardson3(hs, p64)
    rows = []
    for (nxy, nq, mx), h in zip(ladder, hs):
        for binpath, tag in ((BIN64, "fp64"), (BIN32, "fp32")):
            p = run(binpath, nX=nxy, nY=nxy, nQ=nq, Mx=mx)["price"]
            rows.append(dict(build=tag, nX=nxy, nQ=nq, Mx=mx, h=h, price=p, err=abs(p - truth)))
    write_csv("T6_precision.csv", rows, ["build", "nX", "nQ", "Mx", "h", "price", "err"])
    f32 = [r["err"] for r in rows if r["build"] == "fp32"]
    print(f"     FP32 floor ~ {min(f32):.2e}  (cannot beat single-precision round-off)")


# --------------------------------------------------------------------------- T7: Greek bump-size
def t7_bump(quick):
    print("[T7] Greek bump-size study (h_S)")
    grid = dict(nX=161, nY=161, nQ=201, Mx=24)        # fine grid so the Greek itself is converged
    hs = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4]
    if quick:
        hs = [1e-1, 2e-2, 1e-2, 2e-3, 5e-4]
    rows = []
    for h in hs:
        v0, d, g = greeks(h_s=h, **grid)
        rows.append(dict(h_s=h, V0=v0, delta=d, gamma=g))
    # Richardson (h, h/2) for the pairs that exist
    for r in rows:
        half = next((x for x in rows if abs(x["h_s"] - r["h_s"] / 2) < 1e-12), None)
        r["delta_rich"] = (4 * half["delta"] - r["delta"]) / 3 if half else ""
        r["gamma_rich"] = (4 * half["gamma"] - r["gamma"]) / 3 if half else ""
    write_csv("T7_bump.csv", rows, ["h_s", "V0", "delta", "gamma", "delta_rich", "gamma_rich"])
    mid = [r for r in rows if r["h_s"] <= 2e-2 and r["h_s"] >= 2e-3]
    print(f"     delta plateau ~ {sum(r['delta'] for r in mid)/len(mid):.5f}, "
          f"gamma plateau ~ {sum(r['gamma'] for r in mid)/len(mid):.5f}")


# --------------------------------------------------------------------------- T8: Greek grid-conv
def t8_greek_grid(quick):
    print("[T8] Greek grid-convergence (nX, nQ)")
    h_s = H_S_DEFAULT
    axes = {"nX": [31, 61, 121, 241], "nQ": [51, 101, 201, 401]}     # geometric (r=2 in n-1)
    if quick:
        axes = {"nX": [31, 61, 121], "nQ": [51, 101, 201]}
    rows = []
    for axis, ns in axes.items():
        ref = greeks(h_s=h_s, **{axis: ns[-1]})
        for n in ns:
            v0, d, g = greeks(h_s=h_s, **{axis: n})
            rows.append(dict(axis=axis, n=n, h=h_of(axis, n), V0=v0, delta=d, gamma=g,
                             err_delta=abs(d - ref[1]), err_gamma=abs(g - ref[2])))
        sub = [r for r in rows if r["axis"] == axis][:-1]
        pd = fit_order([r["h"] for r in sub], [r["err_delta"] for r in sub])
        pg = fit_order([r["h"] for r in sub], [r["err_gamma"] for r in sub])
        print(f"     {axis}: delta order~{pd:.2f}  gamma order~{pg:.2f}")
    write_csv("T8_greek_grid.csv", rows,
              ["axis", "n", "h", "V0", "delta", "gamma", "err_delta", "err_gamma"])


# --------------------------------------------------------------------------- T9: Greek profile
def t9_profile(quick):
    print("[T9] Greek spatial profile (V, Delta, Gamma vs S0)")
    s0s = [0.7 + 0.05 * i for i in range(17)] if not quick else [0.7 + 0.1 * i for i in range(9)]
    grids = {"coarse": dict(nX=41, nY=41, nQ=51, Mx=8),
             "fine": dict(nX=121, nY=121, nQ=151, Mx=24)}
    rows = []
    for tag, g in grids.items():
        for s0 in s0s:
            v0, d, gm = greeks(h_s=H_S_DEFAULT, S0=s0, **g)
            rows.append(dict(grid=tag, S0=round(s0, 3), V=v0, delta=d, gamma=gm))
    write_csv("T9_profile.csv", rows, ["grid", "S0", "V", "delta", "gamma"])


# --------------------------------------------------------------------------- T10: self-consistency
def t10_selfconsistency(quick, mc_full):
    print("[T10] forward-MC self-consistency (duality lower bound)")
    rows = []
    # (a) MC convergence at a fixed fine grid: CI shrinks ~1/sqrt(N), value sits below U0.
    fine = dict(nX=121, nY=121, nQ=151, Mx=24)
    mcs = [65536, 262144, 1048576, mc_full] if not quick else [65536, 262144]
    for mc in mcs:
        j = run(mc=mc, mc_seed=999, **fine)
        rows.append(dict(kind="paths", grid="121", mc=mc, U0=j["price"], mc_price=j["mc_price"],
                         ci_lo=j["mc_ci_lo"], ci_hi=j["mc_ci_hi"], gap=j["price"] - j["mc_price"]))
        print(f"     N={mc:>8}: U0={j['price']:.5f} MC={j['mc_price']:.5f} "
              f"gap={j['price']-j['mc_price']:+.4f} CI=[{j['mc_ci_lo']:.5f},{j['mc_ci_hi']:.5f}]")
    # (b) gap vs grid resolution at the tightest MC budget (from below as the grid refines).
    for nxy, nq, mx in ([(61, 76, 12), (81, 101, 16), (121, 151, 24)] if not quick else
                        [(61, 76, 12), (121, 151, 24)]):
        j = run(mc=mc_full, mc_seed=999, nX=nxy, nY=nxy, nQ=nq, Mx=mx)
        rows.append(dict(kind="grid", grid=f"{nxy}/{nq}", mc=mc_full, U0=j["price"],
                         mc_price=j["mc_price"], ci_lo=j["mc_ci_lo"], ci_hi=j["mc_ci_hi"],
                         gap=j["price"] - j["mc_price"]))
    write_csv("T10_selfconsistency.csv", rows,
              ["kind", "grid", "mc", "U0", "mc_price", "ci_lo", "ci_hi", "gap"])


# --------------------------------------------------------------------------- T10b: large-N MC bootstrap
def _mc_batch(task):
    """Worker: one single-threaded forward-MC batch (independent seed) on the given grid."""
    seed, n_paths, grid = task
    cfg = {**REF, **grid}
    cmd = [BIN64, "--threads", "1", "--mc", str(n_paths), "--mc_seed", str(seed)]
    for k, v in cfg.items():
        cmd += [f"--{k}", str(v)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return dict(seed=seed, n_paths=n_paths, U0=j["price"], mc_price=j["mc_price"],
                mc_stderr=j["mc_stderr"], t_total_ms=j["t_total_ms"])


def _bootstrap_mean_se(values, n_boot=2000, seed=0):
    """Resample-with-replacement bootstrap of the mean (same convention as the notebooks' boot():
    population std, ddof=0, of the resampled means)."""
    rng = random.Random(seed)
    n = len(values)
    boot_means = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)]
    return statistics.fmean(values), statistics.pstdev(boot_means)


def t10b_mc_bootstrap(n_batches=100, n_paths=1_000_000, seed0=30001):
    """Large-N independent cross-check of the DP reference (GOAL: is the near-exact grid solve
    actually correct, not just internally self-consistent?).  T5's Richardson/GCI budget only
    certifies DISCRETIZATION self-consistency of the backward solve; it cannot catch a modeling bug
    that would bias every grid resolution the same way.  forward_mc (dp.hpp) is a genuinely
    independent estimator: it rolls the DP-derived GREEDY POLICY forward via Monte-Carlo simulation
    on fresh OOS paths — an unbiased estimator of that policy's value, and hence a rigorous LOWER
    BOUND on the true optimum (a duality argument, not a symmetric two-sided estimate of U0).

    forward_mc has no internal parallelism (a plain serial loop in dp.hpp), so 100M total paths is
    split into 100 single-threaded 1M-path batches (independent seeds) run 8-way concurrent via
    ProcessPoolExecutor (mirrors the T12 8-way single-core pattern) — each batch redoes the cheap
    121/121/151/24 backward solve (~18-20s single-threaded incl. the 1M-path rollout; T1/T5 already
    certify this grid to ~1e-4 relative, so it stands in for the finest T12 reference at negligible
    extra bias). The 100 batch-level means are then bootstrapped (2000 resamples) for the pooled
    mean/SE/95% CI, which is what the DP_Validation notebook's T12a plot overlays as a reference-
    uncertainty floor.
    """
    print(f"[T10b] large-N forward-MC bootstrap: {n_batches} batches x {n_paths:,} paths "
          f"(threads=1 each, 8-way parallel)")
    grid = dict(nX=121, nY=121, nQ=151, Mx=24)
    tasks = [(seed0 + i, n_paths, grid) for i in range(n_batches)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(_mc_batch, tasks))
    write_csv("T10_mc_bootstrap.csv", rows,
              ["seed", "n_paths", "U0", "mc_price", "mc_stderr", "t_total_ms"])

    U0 = rows[0]["U0"]                                   # deterministic, identical across batches
    means = [r["mc_price"] for r in rows]
    boot_mean, boot_se = _bootstrap_mean_se(means)
    ci_lo, ci_hi = boot_mean - 1.959963985 * boot_se, boot_mean + 1.959963985 * boot_se
    gap = U0 - boot_mean
    summary = dict(n_batches=n_batches, n_paths_total=n_batches * n_paths, U0=U0,
                   boot_mean=boot_mean, boot_se=boot_se, ci_lo=ci_lo, ci_hi=ci_hi, gap=gap,
                   gap_in_se=gap / boot_se if boot_se else float("nan"))
    write_csv("T10_mc_bootstrap_summary.csv", [summary], list(summary.keys()))
    print(f"     U0={U0:.6f}  boot_mean={boot_mean:.6f}  boot_se={boot_se:.2e}  "
          f"95% CI=[{ci_lo:.6f},{ci_hi:.6f}]")
    print(f"     gap (U0 - boot_mean) = {gap:+.6f}  ({100*gap/U0:+.4f}% of U0)  "
          f"= {gap/boot_se:+.1f} bootstrap SE")
    print("     (expected sign/size: forward_mc rolls a LINEARLY-INTERPOLATED greedy policy on OOS "
          "paths, a genuine duality lower bound -- a small, non-vanishing positive gap is consistent "
          "with policy/interpolation sub-optimality, NOT necessarily a reference error; compare its "
          "magnitude to T5's ~1e-4 discretization budget.)")


# --------------------------------------------------------------------------- T11: limiting + cross-val
def t11_validation(quick):
    print("[T11] limiting cases + cross-validation")
    g = dict(nX=81, nY=81, nQ=101, Mx=16)             # moderate grid (trend, not 6th-digit, matters)
    rows = []
    sweeps = {
        "c":     ([0.01, 0.02, 0.04, 0.08, 0.15], lambda v: dict(c=v)),
        "gamma": ([1.0, 1.5, 2.0, 3.0], lambda v: dict(gamma=v)),
        "q_max": ([1.0, 1.5, 2.0, 2.5, 3.0], lambda v: dict(q_max=v)),
        "Q_max": ([10.0, 15.0, 20.0, 25.0, 30.0], lambda v: dict(Q_max=v)),
        "sigma": ([0.6, 0.9, 1.2, 1.5, 1.8], lambda v: dict(sigma=v)),
    }
    for param, (vals, mk) in sweeps.items():
        for v in vals:
            p = run(**g, **mk(v))["price"]
            rows.append(dict(param=param, value=v, price=p))
    write_csv("T11_monotonicity.csv", rows, ["param", "value", "price"])
    # zero-vol deterministic corner (sigma->0, lam=0): deep-ITM S0=1.5 decays toward 1.
    zv = run(**g, S0=1.5, sigma=1e-4, lam=0.0)["price"]
    print(f"     zero-vol deep-ITM(S0=1.5) DP price = {zv:.5f} (cf. test_dp_limits closed form)")
    # ordering grid: reuse the already-computed DP-vs-incumbents sweep if present.
    if os.path.exists(SWEEP_CSV):
        import shutil
        shutil.copyfile(SWEEP_CSV, os.path.join(RESULTS, "T11_ordering.csv"))
        print(f"     copied DP-vs-incumbents grid -> results/T11_ordering.csv")
    else:
        print(f"     (note: {SWEEP_CSV} missing — run tools/dp_grid_sweep.py for the ordering grid)")


# --------------------------------------------------------------------------- T12: Pareto + complexity
def _mx_coupled(nxy):
    """Quadrature mesh coupled to the spatial resolution (even, clipped) — matches the original
    balanced ladder's 8/10/12/16/20/24/28/32 progression.  The floor is 2 (not 6) so the coarse
    breakdown-regime rungs (nXY<31) get a correspondingly starved quadrature; no config with
    nXY>=31 is affected (the old floor of 6 only ever bound at nXY=31, where 2*round(3.1)=6)."""
    return int(min(32, max(2, 2 * round(nxy / 10.0))))


def _t12_configs(quick):
    """Build the (nX, nY, nQ, Mx, family) sweep.  Two families populate the time-accuracy plane:

      balanced   — a coupled diagonal ladder (nXY up, nQ~1.25*nXY, Mx coupled).  Well-allocated grids
                   that trace the Pareto frontier.
      unbalanced — off-diagonal misallocations at several base grids (starve/over-resolve one axis at
                   a time, including nX/nY spatial asymmetry).  These spend the same wall-clock on the
                   wrong axis, so they sit ABOVE the frontier — they are what turns the 1-D ladder into
                   a 2-D cloud and makes the frontier (and the cheapest-config-per-accuracy-bar) a
                   meaningful object.
    """
    if quick:
        bal_nxy = [41, 61, 81, 121]
        bases = [81]
    else:
        # The <31 rungs deliberately extend the ladder into the BREAKDOWN regime: e.g. 7/7/9/2 puts
        # only 9 nodes on the Q axis for a 22-right contract (nQ is the controlling axis, T1/T11),
        # so the exercise boundary is grossly under-resolved and the price error should blow up by
        # orders of magnitude — the point is to show WHERE the solver stops being trustworthy, not
        # to use these grids.
        bal_nxy = [7, 9, 11, 13, 15, 18, 21, 25,
                   31, 36, 41, 46, 51, 56, 61, 71, 81, 96, 111, 126, 141, 151, 161, 171]
        bases = [21, 31, 51, 61, 81, 101, 121, 141, 161]
    seen, out = {}, []                       # dedup on the grid quad; first family wins (balanced)

    def add(nx, ny, nq, mx, fam):
        key = (int(nx), int(ny), int(nq), int(mx))
        if key not in seen:
            seen[key] = fam
            out.append((*key, fam))

    for nxy in bal_nxy:                       # balanced (frontier-tracing) diagonal
        add(nxy, nxy, round(1.25 * nxy), _mx_coupled(nxy), "balanced")
    for base in bases:                        # unbalanced (off-diagonal) misallocations
        nq_b, mx_b = round(1.25 * base), _mx_coupled(base)
        # floor 7 (was 31): never binds for the original bases (>=51 give nQ>=32); for the new coarse
        # bases (21, 31) a floor of 31 would make the "starved" rung nQ-RICHER than its base grid.
        add(base, base, max(7, round(0.5 * nq_b)), mx_b, "unbalanced")      # nQ-starved (controlling axis)
        if not quick:
            add(base, base, min(280, round(1.6 * nq_b)), mx_b, "unbalanced")  # nQ-rich (over-resolved budget)
        add(base, base, nq_b, 4, "unbalanced")                             # Mx-starved (coarse quadrature)
        if not quick:
            add(base, base, nq_b, min(40, 2 * mx_b), "unbalanced")          # Mx-rich (fine quadrature)
            nxy_h = min(171, round(1.4 * base))
            add(nxy_h, nxy_h, round(0.6 * nq_b), mx_b, "unbalanced")        # nXY-heavy + nQ-coarse (wasted nodes)
            nx_a, ny_a = round(1.3 * base), max(11, round(0.7 * base))
            add(nx_a, ny_a, nq_b, mx_b, "unbalanced")                       # nX-heavy / nY-light (spatial asymmetry)
    return out


def _t12_one(task):
    """Worker: run ONE config single-threaded (--threads 1, verified a true single-thread fast-path
    in grid_dp_pricer/include/grid_dp/dp.hpp), `repeats` times, keep the min-wall-clock repeat (same
    min-over-repeats robustness practice as RESULTS.md/cpp_pricer DEVELOPMENT_NOTES.md), and report
    that repeat's CPU utilization.

    cpu_util_pct = 100 * cpu_seconds / wall_seconds, where cpu_seconds is the EXACT (not sampled)
    user+sys time of the one child process, taken from the POSIX wait()-rusage accounting via
    resource.getrusage(RUSAGE_CHILDREN) bracketed immediately around the subprocess.run() call. Each
    ProcessPoolExecutor worker only ever waits on the one binary subprocess it spawns per call, so this
    delta is not polluted by the other 7 concurrent workers — it robustly reflects how much of this
    config's own wall-clock was real CPU work vs. scheduler contention from the 8-way parallelism,
    independent of which P-/E-core the OS happened to place it on (Apple Silicon has no user-level core
    pinning, so placement itself is not controlled — see plan notes).
    """
    nx, ny, nq, mx, fam, repeats, ref_price = task
    cfg = {**REF, "nX": nx, "nY": ny, "nQ": nq, "Mx": mx}
    cmd = [BIN64, "--threads", "1"]
    for k, v in cfg.items():
        cmd += [f"--{k}", str(v)]
    best = None
    for _ in range(max(1, repeats)):
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.perf_counter()
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        t1 = time.perf_counter()
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        j = json.loads(out)
        wall_s = t1 - t0
        cpu_s = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        j["_cpu_util_pct"] = 100.0 * cpu_s / wall_s if wall_s > 0 else float("nan")
        if best is None or j["t_total_ms"] < best["t_total_ms"]:
            best = j
    err_pct = abs(best["price"] / ref_price - 1.0) * 100.0
    return dict(nX=nx, nY=ny, nQ=nq, Mx=mx, family=fam, nodes=nx * ny * nq, price=best["price"],
                err_pct=err_pct, t_total_ms=best["t_total_ms"], t_backward_ms=best["t_backward_ms"],
                cpu_util_pct=best["_cpu_util_pct"])


def t12_pareto(quick, threads, repeats):
    print(f"[T12] speed-accuracy Pareto + complexity "
          f"(1 thread/config, 8-way parallel OS processes, min of {repeats} repeats)")
    cfgs = _t12_configs(quick)
    ref_cfg = (161, 161, 201, 32) if quick else (201, 201, 251, 36)
    # Reference price: DP is bit-identical regardless of thread count, so this single call can use the
    # fast --threads flag (default 8) without affecting the err_pct computed below.
    rp = run(nX=ref_cfg[0], nY=ref_cfg[1], nQ=ref_cfg[2], Mx=ref_cfg[3], threads=threads)["price"]
    tasks = [(nx, ny, nq, mx, fam, repeats, rp) for nx, ny, nq, mx, fam in cfgs]
    if quick:
        rows = [_t12_one(t) for t in tasks]                  # smoke-test scale: skip pool overhead
    else:
        with ProcessPoolExecutor(max_workers=8) as ex:
            rows = list(ex.map(_t12_one, tasks))
    for r in rows:
        print(f"     [{r['family'][:3]}] {r['nX']}/{r['nY']}/{r['nQ']}/{r['Mx']}: "
              f"err={r['err_pct']:.4f}%  t={r['t_total_ms']:.1f}ms  cpu={r['cpu_util_pct']:.0f}%")
    write_csv("T12_pareto.csv", rows,
              ["nX", "nY", "nQ", "Mx", "family", "nodes", "price", "err_pct",
               "t_total_ms", "t_backward_ms", "cpu_util_pct"])
    # wall-clock scaling: t ~ nodes^p over ALL configs (balanced + off-diagonal).  If p~1 across the
    # cloud, wall-clock is set by the node count, not the allocation -> inner-control-bound (RESULTS §3).
    p = fit_order([1.0 / r["nodes"] for r in rows], [1.0 / r["t_total_ms"] for r in rows])
    cpu_vals = [r["cpu_util_pct"] for r in rows]
    print(f"     wall-clock scaling: t ~ nodes^{p:.2f}  (expect ~1.0, node-count linear)  "
          f"[{len(rows)} configs]")
    print(f"     cpu_util_pct (8-way contention, M1 4P+4E cores): "
          f"mean={sum(cpu_vals)/len(cpu_vals):.1f}%  min={min(cpu_vals):.1f}%  max={max(cpu_vals):.1f}%")


# --------------------------------------------------------------------------- main
ALL = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="coarser ladders (fast smoke run)")
    ap.add_argument("--only", default="", help="comma list e.g. T7,T9 (default: all)")
    ap.add_argument("--threads", type=int, default=8,
                     help="threads for T1-T11 and T12's reference-price call only; the T12 timed "
                          "sweep itself always runs each config single-threaded across an 8-way "
                          "process pool, regardless of this flag (price is thread-count invariant)")
    ap.add_argument("--repeats", type=int, default=5,
                     help="repeats per config (min wall-clock taken); for T12 this is per-config "
                          "repeats inside each of the 8 parallel single-threaded workers")
    ap.add_argument("--mc-full", type=int, default=4_000_000, help="largest MC path count (T10)")
    args = ap.parse_args()

    for b in (BIN64,):
        if not os.path.exists(b):
            sys.exit(f"missing binary {b} — build grid_dp_pricer first")
    want = [t.strip().upper() for t in args.only.split(",") if t.strip()] or ALL
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    print(f"DP validation suite  (quick={args.quick}, tests={want})\n  BIN64={BIN64}")

    t1s = None
    if "T1" in want or "T5" in want:
        t1s = t1_spatial(args.quick)
    if "T2" in want:
        t2_quadrature(args.quick)
    if "T3" in want:
        t3_truncation(args.quick)
    if "T4" in want:
        t4_interp(args.quick)
    if "T5" in want:
        t5_budget(t1s)
    if "T6" in want:
        t6_precision(args.quick)
    if "T7" in want:
        t7_bump(args.quick)
    if "T8" in want:
        t8_greek_grid(args.quick)
    if "T9" in want:
        t9_profile(args.quick)
    if "T10" in want:
        t10_selfconsistency(args.quick, args.mc_full)
    if "T10B" in want:                    # opt-in only (not in ALL) -- expensive, ~5 min
        t10b_mc_bootstrap()
    if "T11" in want:
        t11_validation(args.quick)
    if "T12" in want:
        t12_pareto(args.quick, args.threads, args.repeats)

    print(f"\ndone in {time.time()-t0:.1f}s  ->  {RESULTS}")


if __name__ == "__main__":
    main()
