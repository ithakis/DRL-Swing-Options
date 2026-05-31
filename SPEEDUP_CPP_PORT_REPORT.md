# Speeding up the D4PG Swing-Option Trainer (CPU-only) + C++ Port Plan

Scope: how to make the kernel-on D4PG trainer faster on a CPU-only box, what a C++
port buys you, and whether the network can be made "minimal." Grounded in a real
cProfile run, not guesswork.

## 1. Profile (the source of truth)

Config profiled: focal `c=0.04, gamma=2`, kernel ON (fast: `M_x=2, M_per_k=1, N_max=1`
→ M=4), `n_paths=3072`, `bs=128`, `learn_every=2`, single thread, `compile=0`.
Command: `python -m cProfile -o swing_prof.out run.py …` (3072 paths, eval off).

Total under cProfile: **140 s** (cProfile inflates Python-heavy code ~2-4×, so the
*shape* matters more than the absolute). `run_training` = 132 s, of which
`agent.step` = 115 s and `learn_` = **110 s**. Everything below is inside `learn_`.

### Self-time (where the CPU actually burns)

| Op | self s | calls | what it is |
|----|-------:|------:|------------|
| `torch._C._EngineBase.run_backward` | 18.0 | 64.7k | autograd backward (actor+critic) |
| `torch._C._nn.linear` | 9.7 | 690k | the matmuls (9→64, 64→64, 64→1) |
| `torch.layer_norm` | 8.7 | 460k | LayerNorm (2 per net per pass) |
| `torch._C._nn.silu_` | 7.7 | 460k | SiLU activation |
| `torch._foreach_mul_/add_/div_` | 10.0 | ~1.0M | **Adam** multi-tensor step |
| `Critic.forward` (networks.py:215) | 4.6 | 230k | critic body |
| `apply_profitability_gate` (469) | 3.1 | 165k | STE gate (recomputes payoff) |
| `torch.clamp` | 1.6 | 888k | noise/gate clamps |
| `torch.quantile` | 1.6 | 32.5k | **diagnostics only** (TD percentiles) |

### Cumulative (where the wall-clock lives, by subsystem)

| Subsystem | cum s | % of `learn_` |
|-----------|------:|--------------:|
| `expected_critic_target` (kernel TD target) | 29.7 | 27% |
| Actor forwards (`forward_preact`, gate) | ~26 | 24% |
| Critic forwards (`Critic.forward`) | ~23 | 21% |
| `run_backward` | 18.0 | 16% |
| Adam step (`_multi_tensor_adam`) | 20.9 | 19% |
| replay `sample` (PER Fenwick) | 2.5 | 2% |
| `soft_update` | 2.2 | 2% |

### The headline number

`torch.nn.Module._call_impl` shows **51.7 s cumulative across 2.9M calls**, and
`Module.__getattr__` fires **5.0M times**. The networks are 2×64 — the float work
per call is trivial; the cost is **Python + autograd dispatch overhead on tiny
tensors**. This single fact drives the whole optimization story: we are paying
framework tax, not FLOPs.

## 2. CPU wins available *without* leaving Python (do these first)

These are low-risk and stackable. Rough independent estimates:

1. **Kill diagnostics in the hot loop** (~2-4%). `torch.quantile` (1.6 s, 32.5k
   calls) and the TD-percentile / target-drift blocks run every
   `td_quantile_interval`. Gate them behind a `--diagnostics 0` flag or raise the
   interval 10×. Pure waste in production runs.
2. **Cheaper profitability gate** (~2-3%). `apply_profitability_gate` (3.1 s self,
   165k calls) recomputes the convex payoff `q*(S-K)+ - c*q^gamma` and a `clamp`
   each call. Precompute `(S-K)` once (already in the obs at `OBS_IDX_S_MINUS_K`),
   and fold the gate's clamp into the activation. It runs in the actor forward,
   the target path, *and* the kernel grid (B·M rows) — so it is hotter than it looks.
3. **Adam → lighter optimizer or fused step** (Adam = 19% of `learn_`). The
   `_foreach_*` ops (10 s) + `_init_group`/`_get_value` Python (≈5 s) are the
   multi-tensor Adam bookkeeping over ~30k tiny param tensors. Options: (a) hand-roll
   a fused SGD+momentum or a minimal Adam in one tensor-flattened step; (b) keep Adam
   but cache the param-group list (avoid `_init_group` rebuilding each call). On a
   2×64 net the optimizer should be <5% — it is ~19% purely from dispatch.
4. **Set thread env explicitly.** With `n_cores=1` we still hit BLAS/OMP. For these
   tiny GEMMs, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` avoids thread-pool overhead that
   can *exceed* the matmul itself; for the B·M kernel forward, 2-4 threads may help.
   Measure both — tiny-matrix regimes often prefer single-thread.
5. **`learn_every` / batch shape.** `learn_` runs 32k times. The kernel target builds
   a `(B, M, 9)` tensor via `unsqueeze().expand().contiguous()` every call (29.7 s
   subsystem). Preallocate the `(B·M, 9)` scratch buffer once and fill in-place; skip
   the `.contiguous()` copy by writing the 4 SXY columns into a persistent buffer.
6. **torch.compile** is off (`--compile 0`). Worth a measurement, but on CPU with
   tiny tensors the dispatch win is what `compile` targets — it may help 1.2-1.5×,
   but it is fragile with the STE gate. Lower priority than the C++ port.

Realistic stacked Python-side win: **~1.3-1.6×** wall-clock, no behavior change
(items 1-2, 5 are bit-identical; 3 changes the optimizer and must be re-validated).

## 3. The C++ port — why it pays, and the design

The profile is the argument: 2.9M `_call_impl` + 5M `__getattr__` on a 2×64 net.
A C++ port doesn't make the matmul faster — a 64×64 `gemv` is nanoseconds — it
**deletes the interpreter and autograd-graph tax entirely**. Expected end-to-end:
**5-15× over eager PyTorch CPU**, more on the forward-heavy kernel path.

### What the net actually is (so "minimal" is well-defined)

- **Actor**: `9 → 64 → 64 → 1`, SiLU, LayerNorm after each hidden layer, β-sigmoid(3.0)
  output, then the profitability-gate STE. ~4.7k params.
- **Critic**: `(state⊕action) → 64 → 64 → 1`, SiLU + LayerNorm. ~4.8k params.
- Both are pure feed-forward. No recurrence, no attention, no batchnorm running stats.

This is already minimal in *capacity*. The Stage-A→E approximator study confirmed you
**cannot shrink it further without losing accuracy** — poly/RFF/RBF/tiny_nn all lost
to the 2×64 net on Δ% (best alternative −1.3% vs the NN's +0.46% over LSM). So the
C++ port should reproduce the **2×64 SiLU+LayerNorm net exactly**, not replace it.

### Architecture of the port

1. **Inference-only forward in hand-written C++**, weights as flat `float`/`double`
   arrays. Each layer = one `gemv` (or small `gemm` for the B·M kernel batch) +
   bias + fused SiLU + fused LayerNorm. Use Eigen (header-only, expression-fused) or
   a tiny hand-rolled kernel; for 64-wide, hand-rolled with `-O3 -march=native`
   (FMA/AVX) often beats a BLAS call because there is no call overhead.
2. **Backward**: two choices.
   - (a) **Hand-derived gradients** for this fixed topology. The graph is 6 ops; the
     adjoint is mechanical (Linear, SiLU′ = σ(x)(1+x(1−σ(x))), LayerNorm Jacobian,
     β-sigmoid′, STE = identity on the gate). Most control, fastest, but you own the
     calculus. This is very tractable for a fixed 2×64 net.
   - (b) **LibTorch (C++ API)**. Keeps autograd, ~2-3× less Python tax but still
     graph overhead; lowest-effort but leaves performance on the table. Good as a
     migration checkpoint.
   Recommendation: ship (a) for the final, use (b) to cross-check gradients
   numerically during development.
3. **The DPG coupling matters.** Actor update needs `∂Q/∂a` through the critic. In
   C++ that is one extra backward through the critic w.r.t. its action input — cheap
   and already part of (a)'s derivation. The current code keeps `dQ/da` informative;
   the port must preserve it (don't detach the action in the critic forward used for
   the actor loss).
4. **The kernel (`expected_critic_target`) is the single hottest subsystem (27%)** and
   ports beautifully: it is a quadrature `sum_m w_m · Q(s'_m, π(s'_m))`. In C++ this is
   a `(B·M)` batched forward + a weighted reduce — no autograd needed on the target
   (it is under `no_grad`). The numpy grid build (`build_next_state_grid_batched`) is
   already njit; reimplement as a plain loop. With M=4 this is a `(512×9)` forward.
5. **Optimizer**: implement Adam in ~15 lines over the flat param vector (or SGD+mom
   if Stage F shows Adam is overkill once the target is deterministic).
6. **RNG / reproducibility**: match PyTorch's seeding only if you need bit-identical
   parity; otherwise seed a `std::mt19937_64` and re-validate Δ% statistically over
   seeds 11-22 (the harness already does this).
7. **PER replay**: the Fenwick-tree sampler is only 2% — port it straight (it is
   already Numba/array code), or even keep this part in Python if you wrap C++ only
   for the learn step.

### Port strategy (incremental, always-verifiable)

- **Step 1**: C++ inference forward for actor+critic; load PyTorch weights; assert
  max-abs diff vs PyTorch `< 1e-6` on a fixed batch.
- **Step 2**: C++ `expected_critic_target` (forward + reduce); same numeric assert.
- **Step 3**: C++ backward (hand-derived); `gradcheck` against `torch.autograd` in
  double precision (mirror `tools/test_approximators.py`'s float64 gradcheck).
- **Step 4**: C++ Adam + soft-update; run a full short training; compare Δ% to the
  Python baseline over seeds 11-22 (Welch p>0.05 = parity).
- **Step 5**: drop OpenMP/threads tuning and `-march=native`; re-profile.

Expose it to Python via pybind11 if you want to keep `run.py`/the sweep harness and
only swap the inner loop — that preserves all the existing tooling and CSV/stats
pipeline.

## 4. Recommended order of operations

1. Land the **config simplification first** (remove dead features, confirm Stage F
   removals). A smaller, fixed feature set is *far* cheaper to port — every feature
   you delete in Python is C++ you never write. Port the v63-kernel-minimal config,
   not v61.
2. Do the **free Python wins** (§2 items 1, 2, 5) — they are bit-identical and shrink
   the baseline you will compare the port against.
3. Port **inference + kernel target** first (biggest, cleanest share), measure.
4. Port **backward + optimizer**, validate Δ% statistically.

The net is already as small as the accuracy allows; the speed is in **deleting
framework overhead**, not shrinking the model. The C++ port is the right lever, and
the kernel-deterministic target makes it easier (the hottest path needs no autograd).
