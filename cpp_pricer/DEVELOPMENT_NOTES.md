# cpp_pricer — Development Notes & Lessons Learned

Knowledge transfer for whoever extends the C++ pricer next. Read this before touching
the hot loop or the validation harness. Companion to `README.md` (which is the *how to
build/run*); this file is the *why it's built this way + what to avoid*.

## What this is

A from-scratch C++ port of the **v64 kernel-on D4PG** swing-option pricer (`src/` in Python),
tuned for **Apple M1, CPU-only, float32**. Goal: minimize the sum of two wall-clocks —
(0→4k) train+price and (4k→65k) OOS eval. Result: **~8.8× end-to-end vs PyTorch eager**
(43 s vs 381 s), price statistically indistinguishable from Python (Welch p=0.37, seeds 11–25 vs 11–18).

The networks are tiny (3×64 actor/critic). The Python profile (`docs/SPEEDUP_CPP_PORT_REPORT.md`)
showed the cost is **framework/autograd dispatch tax on tiny tensors, not FLOPs** — that is exactly
what the C++ port deletes, and why the port pays off.

## Architecture decisions (and the reasoning)

- **Hand-written forward + hand-derived backward**, not LibTorch. The topology is fixed (6 ops);
  the adjoint is mechanical and removes all graph overhead. Layout mirrors the PyTorch `state_dict`
  exactly (`load_flat` order) so exported weights load 1:1.
- **float32 + `-ffast-math -mcpu=apple-m1`** for the headline price; a **separate FP64 build**
  (`-DCPP_PRICER_FP64=ON`) is used *only* for the parity/gradient checks (tight 1e-4 bar). This split
  is load-bearing — see "Gotchas".
- **Accelerate BLAS (`sgemm`/`dgemm`)** for every layer GEMM, forward and backward. Beats the
  hand-rolled kernel for B≥128. Toggle with `-DCPP_PRICER_ACCELERATE` (default ON).
- **Kernel mesh is a loaded artifact** (`data/kernel_v64.bin`), produced once by
  `tools/export_reference.py`. The single jump node depends on a scrambled-Sobol draw that is not
  worth reproducing in C++; loading the Python mesh guarantees bit-faithfulness. A C++
  `build_fast` analytic fallback exists (jump node = conditional mean) for when no file is present —
  it is *close but not bit-identical*, so don't use it for parity tests.
- **HHK simulation uses independent RNG** (xoshiro256**), antithetic pairs on the OU driver and the
  jump marks, plus terminal stratification. We do **not** reproduce NumPy's Sobol/PyTorch RNG — the
  validation bar is statistical (closed-form moments + price Welch test), not bit-identical, because
  RL training is stochastic across RNG streams anyway.
- **Eval is the only embarrassingly-parallel phase** → multithreaded across path-chunks with a
  per-thread Actor copy. Training is sequential (each learn step updates the weights the next uses).

## The optimization journey (measured, µs/learn-step on M1)

| stage | µs/step | note |
|------:|--------:|------|
| baseline (hand GEMV, `std::exp`) | 3444 | correct but slow |
| + fast float `exp` in SiLU/sigmoid | 1709 | **biggest single win (2×)** — `exp` was ~½ the compute |
| + `__restrict__` on kernels | 1702 | negligible (compiler already vectorized) |
| + Accelerate `sgemm` (forward) | 972 | |
| + Accelerate (backward GEMMs too) | **~490** | clean; **7× over baseline** |

End-to-end then dropped from 182 s → 43 s. Profiler split of the optimized step:
**kernel-target 49%, actor update 33%, critic update 18%, soft-update/EMA 0.1%**.

### What worked
- **Fast `exp`.** SiLU/sigmoid call `exp` ~3×10⁵ times/step. A degree-5 `2^x` poly + bit-trick
  (`fast_expf`, ~1e-5 rel.) roughly halved the step with no measurable price change. **Only in the
  FP32 build** — the FP64 parity build keeps `std::exp` so component parity stays at ~1e-11.
- **BLAS for both passes.** Forward `Y=X·Wᵀ` and backward (`gX=gY·W`, `gW+=gYᵀ·X`) map cleanly to
  3 GEMMs; Accelerate is well-tuned even at 64-wide once B≥128.
- **Moving all scratch buffers to members** (forward caches *and* backward temporaries). Per-call
  `std::vector` allocation in `backward()` was real overhead at ~90k steps/run.
- **One micro-benchmark (`bench/bench_train.cpp`)** that times K isolated learn-steps. Iterating on a
  ~15 s bench instead of a 3-min full run is what made the optimization tractable. Build it with
  `-DPRICER_PROFILE` to get the per-block percentage breakdown.

### What did NOT work / dead ends
- **`__restrict__`** — no measurable effect; `-ffast-math -mcpu=apple-m1` already vectorizes the dot products.
- **Threading the hot loop.** The kernel-target (512×64 GEMM) is 49% of the step, but it's too small
  for Accelerate to parallelize: forcing `VECLIB_MAXIMUM_THREADS` 1 vs 4 gives 486 vs 487 µs — no gain.
  A custom thread pool over the 512 rows would be eaten by per-step sync at 90k steps. **Training time
  has converged at this network size**; further gains need a smaller net or an algorithm change.
- **C++ `build_fast` kernel mesh for production** — fine for a quick run, but the approximate jump node
  shifts the target slightly; always prefer the exported `kernel_v64.bin`.

## Gotchas & bugs found during development (don't re-introduce these)

1. **SiLU backward must use the LayerNorm *output*, not the Linear output.** The forward order is
   `Linear → LayerNorm → SiLU`, so SiLU's input (needed for its derivative) is the LN output. An early
   version passed `*_lin_out_` to `silu_backward`; gradcheck caught it (critic grad off by 0.12). The
   actor happened to be correct already, but its DPG gradcheck *also* failed because it routes through
   `critic.backward` — fix the critic and both pass.
2. **Dangling-pointer-on-resize.** `Actor::forward` passed `u_.data()` into `forward_preact`, which then
   called `ensure(B)` and *resized* `u_` — invalidating the pointer (segfault, caught by ASan). Lesson:
   call `ensure(B)` **before** taking the address of any member buffer you pass down.
3. **STE gate breaks finite-difference gradcheck on purpose.** The profitability gate uses a
   straight-through estimator (`q_raw + (q_proj − q_raw).detach()`), so the analytic grad ≠ the true
   (clamped) derivative wherever the gate is active — exactly as PyTorch. Gradcheck the actor with the
   gate **disabled** (`c=0` → identity); the STE wrapper is identity-by-construction and is covered by
   the gate-ON forward parity test.
4. **Finite-difference gradcheck is only meaningful in FP64.** In FP32 the FD error (ε=1e-5, float
   precision ~1e-7) is ~1e-3 — the gradients are correct, the *check* is noise-limited. `test_grad`
   uses a precision-aware tolerance (1e-5 FP64 / 5e-3 FP32) for this reason.
5. **AdamW weight-decay grouping.** Decoupled WD applies to **2-D weight matrices only**; biases and
   LayerNorm params get WD=0. This mirrors `_build_optimizer` in `src/agent.py`. `collect_params`
   encodes it via the per-block `weight_decay` field.
6. **Robust normalizer is effectively fixed.** v64 sets `use_robust_normalization=1` but with default
   `median=0, iqr=1`, so `HHKInputLayer` is just log-moneyness at idx 5 + a ±10 clamp at idx 6,7. We
   hardcode this; no data-dependent calibration to match.
7. **CTest working directory.** Tests resolve fixtures relative to cwd; CMake passes the absolute
   `data/` path as `argv[1]` so `ctest` works from the build dir.

## Validation methodology (keep this bar)

- **Component parity (exact, deterministic):** `tools/export_reference.py` dumps PyTorch weights +
  fixed-batch forward/critic/kernel outputs; FP64 build asserts max|Δ| < 1e-4 (achieved ~6e-11).
- **Gradients:** FP64 finite-difference gradcheck < 1e-5 (achieved ~5e-11), incl. the DPG path.
- **End-to-end:** float32 65k price over seeds vs PyTorch baseline (`tools/python_baseline.py`), Welch
  two-sample p > 0.05. Achieved p = 0.37 (C++ 1.982±0.009 n=15, Python 1.985±0.006 n=8).
- Re-run all of this after any change to `mlp.cpp`, `kernel.cpp`, or the activation/exp code.

## Reproducing the data / notebook

`tools/collect_results.py {cpp-seeds,py-seeds,cpp-scaling}` → CSVs in `data/`; `tools/build_notebook.py`
regenerates `Jupyter Notebooks/9: C++ Pricer - Speed & Validation.ipynb`. **Measure timings without
other load** — the seed sweeps were run concurrently and their timings are contention-polluted (prices
are not), which is why the notebook's speed bars use clean single-seed numbers.

## Ideas for future work (untried or deferred)

- Shrink the network (the approximator study said the 3×64 NN is needed for accuracy — revisit only if
  a faster target is acceptable). A smaller net is the most direct training speedup left.
- A persistent thread pool *might* help the kernel-target if the batch grows (M larger, or larger B),
  but not at the current 512×64.
- pybind11 wrapper to swap the C++ learn-step into `run.py` and reuse the existing sweep/CSV tooling
  (deliberately skipped here: pybind11 not installed, and the standalone CLI keeps full isolation).
- FP16 storage for the replay buffer (memory only; risky for parity — measure first).
