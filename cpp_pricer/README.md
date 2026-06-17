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

## v67 — two-mode builds (kernel = v65, no-kernel = literal v61)

v67 ships the no-kernel lineage as a first-class second mode. The two modes differ only in two
**compile-time** knobs (hidden activation + actor-output β); every other v61 feature is a **runtime
flag that defaults to v65/off**, so the kernel build stays bit-identical to v65. See
[`docs/V61_CONFIG.md`](docs/V61_CONFIG.md) for the full literal-v61 recipe.

```bash
# Mode 1 — KERNEL (= v65 exactly): swish-β3 activation, beta_sigmoid 1.5
cmake -S cpp_pricer -B cpp_pricer/build_v67_kernel -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-DPRICER_GELU_FAST -DGELU_SLOPE=3"
cmake --build cpp_pricer/build_v67_kernel -j
# run (v65 "balanced" recipe):
./build_v67_kernel/price_swing --seed 11 --kernel data/kernel_v64.bin \
   --hidden 48 --actor_layers 2 --critic_layers 4 --hidden_actor 32 \
   --batch 64 --learn_number 3 --lr_c 5e-4 --n_train 4096 --n_eval 65536 --threads 8

# Mode 2 — NO KERNEL (literal v61): SiLU activation, beta_sigmoid 3.0
cmake -S cpp_pricer -B cpp_pricer/build_v67_nokernel -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-DACTOR_BETA_VAL=3"
cmake --build cpp_pricer/build_v67_nokernel -j
# run (literal v61 focal recipe):
./build_v67_nokernel/price_swing --seed 11 --kernel_off \
   --actor_layers 2 --critic_layers 2 --hidden 64 --init_method 1 \
   --lr_a 1.6e-4 --lr_c 9e-5 --wd_c 1.2e-4 --learn_every 2 --learn_number 1 \
   --batch 128 --tau 0.0032 --noise_sigma0 1.30 --noise_floor 0.26 \
   --noise_schedule hyperbolic --noise_plateau 3200 --adaptive_noise_scale 0.6 \
   --warmup_noise_fraction 0.4 --critic_warmup 1024 --weight_avg 2 --double_critic_step 1 \
   --target_policy_noise 0.15 --tpn_decay_start 20000 --tpn_floor 0.04 \
   --lr_schedule cosine --lr_warmup_episodes 1024 --lr_schedule_episodes 40000 \
   --final_lr_fraction 0.20 --min_lr 1e-6 --min_replay 18000 --max_replay 200000 \
   --n_train 4096 --n_eval 65536 --threads 8
```

New runtime flags (all default to v65/off): `--noise_schedule {linear,hyperbolic,const_floor}`,
`--noise_plateau`, `--weight_avg 2` (eval on raw local actor, no EMA), `--double_critic_step 1`,
`--target_policy_noise --tpn_decay_start --tpn_floor`, `--lr_schedule {const,cosine,linear}
--lr_warmup_episodes --lr_schedule_episodes --final_lr_fraction --min_lr`, `--min_replay --max_replay`.
PER (`--per`) is the one literal-v61 piece still pending; it is **inert (α=0 ⇒ pure uniform replay)** at
every budget ≤5000 episodes, so it does not affect the 4096-episode comparison/optimization budgets.

**Validated:** kernel build reproduces v65 bit-identically (price 1.975071 @ focal g2 seed 11/4k);
FP64 `ctest` (parity/grad/sim) all green; no-kernel build runs the literal-v61 recipe stably across
seeds (≈1.94 ± 1%, no collapse).

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
