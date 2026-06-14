# cpp_pricer — fast C++ swing-option pricer (v64 kernel-on D4PG)

A self-contained C++ reimplementation of the kernel-on D4PG swing-option pricer, tuned for
**Apple M1 (CPU-only)**. It reproduces the Python pipeline (`src/`) numerically and minimizes the sum
of two wall-clock costs:

1. **0 → 4k** — simulate training paths + closed-form warm-start + train 4096 episodes → a price.
2. **4k → 65k** — simulate 65 536 OOS paths + greedy rollout → final price + 95% CI.

Isolated from the rest of the repo; nothing under `src/` is modified.

## Build

```bash
cmake -S cpp_pricer -B cpp_pricer/build -DCMAKE_BUILD_TYPE=Release   # Accelerate BLAS ON by default
cmake --build cpp_pricer/build -j
```

Options: `-DCPP_PRICER_FP64=ON` (double precision, for parity/gradient checks),
`-DCPP_PRICER_ACCELERATE=OFF` (hand-rolled GEMMs instead of Accelerate).

## Run

```bash
cd cpp_pricer
./build/price_swing --seed 11 --n_train 4096 --n_eval 65536 --threads 8 --kernel data/kernel_v64.bin
```
Emits JSON: `price`, `ci95`, per-phase timings, `t_zero_to_4k`, `t_4k_to_65k`, `t_total`.

The kernel mesh `data/kernel_v64.bin` is produced once by the export step below (deterministic for the
config). If absent, the binary builds a close analytic approximation in C++ and warns.

## Validate against PyTorch

```bash
python cpp_pricer/tools/export_reference.py          # dump weights + reference tensors + kernel mesh
cmake -S cpp_pricer -B cpp_pricer/build_fp64 -DCMAKE_BUILD_TYPE=Release -DCPP_PRICER_FP64=ON
cmake --build cpp_pricer/build_fp64 -j
( cd cpp_pricer && ./build_fp64/test_parity && ./build_fp64/test_grad && ./build/test_sim )
```
- `test_parity` — forward / critic / kernel target vs PyTorch, max|Δ| < 1e-4 (achieved ~6e-11 in FP64).
- `test_grad` — hand-derived gradients vs finite differences, < 1e-5 (achieved ~5e-11).
- `test_sim` — HHK terminal moments vs closed form.

## Results (Apple M1)

| | 0→4k | 4k→65k | total |
|---|---:|---:|---:|
| PyTorch (eager CPU) | 379 s | 1.6 s | **381 s** |
| C++ (this work) | 43 s | 0.10 s | **43 s** |

~**9× end-to-end**. Price is statistically indistinguishable from PyTorch across seeds.
See `Jupyter Notebooks/9: C++ Pricer - Speed & Validation.ipynb` for the full story.

## Layout

```
include/  config, rng, linalg (BLAS/hand GEMM), mlp (actor/critic fwd+bwd),
          kernel (semi-analytical target), adam, replay, env, agent
src/      hhk_sim, mlp, kernel, agent
apps/     price_swing.cpp   (CLI)
bench/    bench_train.cpp   (isolated learn-step micro-benchmark)
tests/    test_parity, test_grad, test_sim
tools/    export_reference.py, python_baseline.py, collect_results.py, build_notebook.py
data/     kernel_v64.bin + exported reference fixtures
```
