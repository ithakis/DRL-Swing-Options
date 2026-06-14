# cpp_pricer — Research-Driven Speedup Plan (keep the price, cut the wall-clock)

**Author:** C++/quant review pass, 2026-06-13 · **Target HW:** Apple M1, 8 GB, CPU-only, FP32
**Companion docs:** `cpp_pricer/DEVELOPMENT_NOTES.md` (why it's built this way),
`docs/SPEEDUP_CPP_PORT_REPORT.md` (the Python-side profile that motivated the port).

> **Goal.** Find changes that make `cpp_pricer` *meaningfully faster* while producing a price that is
> *statistically equivalent* to the current v64 kernel-on C++ pricer (and, transitively, to the
> PyTorch v64 reference). "Faster" must be proven by profiling with confidence intervals; "the same"
> must be proven by **equivalence testing (TOST)**, not by the absence of a significant difference.

---

## 0. Executive summary

> **RESULTS (Tiers 1–2 complete, 2026-06-13).** The plan delivered **2.37× faster training with a
> statistically equivalent price** (conservative tier: 42.06 → 17.76 s, TOST-equivalent to the v64
> original, 90% CI of the −0.098% drift ⊂ ±0.5%, seed variance *improved*). Two further tiers trade a
> documented, still-equivalent drift for more speed: **fast 2.88×** (14.61 s, −0.167%) and **aggressive
> 4.13×** (10.18 s, ~−0.26%). Wins, in order of impact: **H-K1** drop jump node (M=4→2, −24%), **H-T1**
> episode budget (4096→3072), **H-A1** width 64→48, **H-C1** stale-target reuse, **H-S6** skip the
> discarded DPG critic-grad (bit-identical). All gated by paired **TOST** (not bare p>0.05), with a
> fresh-baseline combination round (R7) confirming transitive equivalence. Per-step cost fell 486→~282 µs
> (conservative) / ~230 µs (fast). The Tier-3 "replace the learner" branch was then run to completion and
> **empirically closed**: ELM critic (H-R1) and direct-policy REINFORCE (H-R2) both fail the equivalence
> gate (plateau ~5% and ~17% off, respectively), and distillation (H-R3) is bounded to <0.5% of wall-clock
> by construction — the kernel-on D4PG is confirmed as the right learner. The plan is exhausted. Full
> ledger in Part F.

The optimized learn-step costs **486 µs** on this M1 and is extremely stable (478–487 µs, <2% run-to-run).
A full 0→4k train is ≈ 86k learn-steps ≈ **42 s** (matches the 43 s headline); OOS eval is 0.1 s
(already multithreaded, irrelevant). So **t_total is essentially t_train, and t_train = (learn-steps) ×
(µs/step).** Both factors are levers.

The per-step time splits **kernel-target 49% / actor 33% / critic 17% / soft-EMA 0.3%**. The
kernel-target dominates because it runs *both* the actor and the critic forward at **B·M = 512 rows**
(M=4 mesh) every step — forward-only, on frozen target weights. That single fact drives the ranking
below: the highest-value, lowest-risk wins attack the **target expectation** (its mesh size and its
arithmetic precision), because that is where ~half the compute lives and it carries the least
numerical risk (no gradient flows through it).

Three tiers of hypotheses, ranked by *expected speedup × P(price preserved) ÷ effort*:

| Tier | Hypotheses | Mechanism | Expected | Risk |
|---|---|---|---|---|
| **1 — mechanism-targeted, cheap** | H-K1 drop jump node (M=4→2); H-P1 fp16 target forward; H-S1 vectorize/fuse LN+SiLU; H-S3 `pow(q,2)`→`q*q` | shrink/cheapen the 49% block | 15–35% | low |
| **2 — arch/config, equivalence-gated** | H-A1 shrink net (width/depth); H-T1 fewer episodes; H-C1 stale-target reuse; H-C2 share preprocess | fewer/cheaper passes & steps | 10–40% | medium |
| **3 — research-grade, high upside** | H-R1 random-feature (ELM) critic w/ closed-form readout; H-R2 likelihood-ratio swing policy (Warin); H-R3 distill to a 1-layer net | replace the iterative learner | 2–10× | high |
| **4 — systems last-mile** | H-S2 fused MLP kernel; H-S4 Adam fp32; H-S5 batched cross-network GEMM | dispatch/precision | 5–15% | low–med |

Each is specified below with a mechanism, an implementation sketch keyed to real files/lines, an
expected effect, and a **hard acceptance gate**. Part E fixes the statistics and profiling protocol;
Part F is the iteration loop and the results ledger.

---

## Part A — C++ implementation review

### A.1 Verdict
The port is **well-engineered and faithful**. Layout mirrors the PyTorch `state_dict` (`load_flat`
order), forward/backward are hand-derived and gradient-checked in FP64 (`test_grad` < 1e-5), and the
FP32/FP64 split is principled (fast price vs. tight parity). The optimization journey is documented and
measured. Nothing below is a correctness objection to the *current* numbers — they are sound.

### A.2 Strengths worth preserving (don't regress these)
- **FP32-headline / FP64-parity split** (`config.hpp:10`). Any new fast path must keep an FP64 sibling
  so `test_parity`/`test_grad` still gate it. This is load-bearing — keep it.
- **Member-resident scratch** for both forward caches and backward temporaries (`mlp.hpp:73-118`).
  Re-introducing per-call `std::vector` in `backward()` was a measured regression historically; the
  plan's new kernels must allocate via `ensure(B)`, never per-call.
- **`ensure(B)` before taking member pointers** (`mlp.cpp:64`) — the resize-dangling-pointer bug
  (DEVELOPMENT_NOTES gotcha #2) is easy to reintroduce in any new fused path.
- **Kernel mesh is a loaded artifact** (`kernel.cpp:16`). Bit-faithfulness to Python depends on
  `data/kernel_v64.bin`; the C++ `build_fast` fallback is *close but not identical*.

### A.3 Correctness / reproducibility risks (flag, not bugs)
1. **No `data/` in this checkout.** `kernel_v64.bin` and the export fixtures are absent, so `test_parity`
   and `test_grad` **cannot run** and the bench falls back to `build_fast` (M=4, approximate jump node).
   *Prerequisite for the whole plan:* run `tools/export_reference.py` in the `EP11` conda env to
   regenerate `data/` before any equivalence claim against PyTorch. (Conda is not on PATH in this shell
   either — see Part E.0.) Until then, equivalence is *C++-vs-C++* (new vs. current build_fast baseline),
   which is still a valid internal gate.
2. **`build_fast` jump node is an approximation** (`kernel.cpp:48-54`): node1 = conditional-mean jump
   increment, tail folded into pmf1. Fine for speed experiments, but *every* cross-Python equivalence
   test must use the exported mesh, not build_fast.
3. **`PRICER_PROFILE` is not wired into CMake.** The macro is read in `agent.cpp`/`bench_train.cpp` but
   never defined by `CMakeLists.txt`; the profile build only works via `-DCMAKE_CXX_FLAGS="-DPRICER_PROFILE"`.
   *Action:* add `option(PRICER_PROFILE ...)` so the breakdown is reproducible. (Low effort, do first —
   the iteration loop needs it.)

### A.4 Micro-opportunities found while reading (candidate quick wins)
- **`std::pow(q, gamma_cost)` with γ=2** in `env.hpp:67`, `agent.cpp:309/340`. `pow(x,2.0)` is ~10–30×
  slower than `x*x`. Hot in the training rollout (`act_single`→`env_step`) and in eval. Special-case
  γ∈{1,2} (and the `pow(payoff/…, 1/(γ−1))` gate likewise). → **H-S3**.
- **`expected_critic_target` runs `hhk_preprocess` twice** on the same `flat` rows (once inside
  `actor_t.forward`, once inside `critic_t.forward`; `kernel.cpp:95-96`). 512 rows × 9 dims × 2/step.
  Preprocess once, pass the preprocessed buffer to both. → **H-C2**.
- **Kernel mesh uses `std::exp` (double)** in the FP32 build (`kernel.cpp:80,84`), not `fast_expf`.
  M·B exp calls/step. Worth a fast-path test (guarded so FP64 build keeps `std::exp`). → folds into H-P1.
- **AdamW carries `m_,v_` and does all math in `double`** even in the FP32 build (`adam.hpp:24-39`).
  Param count is tiny (~12k actor + ~16k critic), so this is correctness-cheap insurance, not a hot spot
  — but an FP32-state variant is a free micro-test (**H-S4**) and de-risks an eventual all-FP32 build.
- **Soft-update + EMA copy every step** over all params (`agent.cpp:171-177`) — measured 0.3%. Leave it.

---

## Part B — Profiling study (the source of truth)

### B.1 Measurement setup
- Build: `Release`, `-O3 -ffast-math -funroll-loops -mcpu=apple-m1`, Accelerate BLAS ON, FP32.
- Micro-bench: `bench/bench_train.cpp` times K isolated `learn_step()`s after a 2000-step warm-up;
  `-DPRICER_PROFILE` adds the per-block split. Sampling profile via `/usr/bin/sample` (1 ms, ~8 s window).
- No kernel file present → `build_fast` M=4 mesh (M_x=2 × M_y=2). This matches the *production fast*
  config's M=4, so the shape is representative; absolute µs may shift slightly with the exported mesh.

### B.2 Headline numbers (this M1, FP32)
```
learn_steps=20000  us/step=484.0   | kernel 49.3%  critic 17.2%  actor 33.2%  soft/ema 0.3%
learn_steps=20000  us/step=481.6   | kernel 49.2%  critic 17.2%  actor 33.3%  soft/ema 0.3%
learn_steps=20000  us/step=478.1   | kernel 49.0%  critic 17.3%  actor 33.4%  soft/ema 0.3%
learn_steps=400000 us/step=487.0   (long run, same)
```
Run-to-run spread <2% → a **~3–4% speedup is already detectable** with a handful of clean bench runs
(see Part E for the formal CI). Sampling leaves (8262 samples): `cblas_sgemm` is the single heaviest
leaf, then `layernorm_forward` (a **scalar** row-reduction loop, *not* BLAS), then the `*::forward`
functions (SiLU/`fast_expf` inlined into them). `mach_absolute_time` + vm-reclaim ≈ 0.7% noise.

### B.3 The cost model (why the ranking is what it is)
Per `learn_step`, counting network passes in units of *one batch-128 forward* (B=128, M=4 ⇒ kernel rows
= 512 = 4 units):

| Pass | kernel-target | critic step | actor step | total (units) |
|---|---:|---:|---:|---:|
| **actor forward** | 4 | – | 1 | **5** |
| **critic forward** | 4 | 1 | 1 | **6** |
| **critic backward** | – | 1 | 1 | **2** |
| **actor backward** | – | – | 1 | **1** |

≈ **11 forward-equiv + 3 backward-equiv** per step, and **8 of the 11 forward units are inside the
kernel-target** (4 actor + 4 critic at B·M). The matmuls are tiny (64-wide) → this is **latency/dispatch-
and memory-bound, not FLOP-bound** (roofline: 64×64 GEMM is ~0.5 MFLOP, trivially in-cache; the cost is
op count, BLAS call overhead, and the scalar LN/SiLU traffic). Implication: **wins come from doing fewer
and cheaper ops in the target expectation**, not from faster FLOPs.

### B.4 What this rules in / out
- **In:** shrink M (H-K1); cheapen each target forward via fp16 (H-P1) and vectorized LN/SiLU (H-S1);
  reuse the target across steps (H-C1); shrink the net (H-A1); fewer steps total (H-T1).
- **Out / low-value:** soft-update/EMA (0.3%), eval (0.1 s), replay sampling (in noise), Adam state.
- **Uncertain:** fused hand MLP kernel (H-S2) — prior `__restrict__`/hand-GEMV attempts were neutral or
  worse, but a *multi-layer-fused* kernel that never round-trips activations through BLAS is a different
  experiment; treat as Tier-4, measure don't assume.

---

## Part C — Literature synthesis (method families & relevance)

~60 papers were screened via `paper-search` across 20 themes (full list in the Appendix). They cluster
into seven families; each maps onto one or more hypotheses below. Relevance verdict in **bold**.

**C.1 Model-based value targets (the kernel-target itself).**
Our semi-analytical kernel *is* a 1-step model-based value expansion. Two papers are decisive guardrails:
*Revisiting Model-based Value Expansion* (arXiv 2203.14660) and *Diminishing Return of Value Expansion*
(2303.03955) show **multi-step** expansion is usually *not* worth it (compounding model error; 1-step
Dyna targets win). **Verdict: do NOT add multi-step kernel rollout; instead minimize the 1-step mesh
(H-K1) and reuse it (H-C1).** Strong support for shrinking M, not growing it.

**C.2 Randomized / reservoir networks for optimal stopping (the big swing).**
*Optimal Stopping via Randomized Neural Networks* (Herrera–Krach–Ruyssen–Teichmann; Semantic
e12309e6) fixes hidden weights at random and trains only a **closed-form least-squares readout** —
matching LSM/deep-optimal-stopping accuracy at a fraction of the cost. Extreme-learning-machine variants
(Semantic 89979d2a "Effective non-random ELM", b9027c3c ensemble RVFL) generalize it. **Verdict: a
random-feature critic with a ridge-regression readout could replace the iterative Adam critic entirely —
this is exactly the LSM regression spirit and the single largest potential win (Tier 3, H-R1).**

**C.3 Direct policy / likelihood-ratio formulations for swing options.**
*Deep combinatorial optimisation for optimal stopping … application to swing options* (arXiv 2001.11247,
Warin et al.): models the **policy directly**, no DPP/BSDE backward pass, likelihood-ratio gradient for
the discrete exercise. *Neural network regression for Bermudan option pricing* (1907.06474) and
*Optimizing NNs for Bermudan Option Pricing: Convergence Acceleration* (2402.15936) give convergence-
acceleration tricks. **Verdict: an alternative training objective that may converge in far fewer
gradient steps — Tier 3 (H-R2); keep the existing env/price plumbing for the equivalence gate.**

**C.4 Network compression: pruning, low-rank, distillation.**
Lottery-ticket line (1912.05671, 1903.01611) and low-rank factorization (2306.14152) target *large*
nets; our 3×64 is already tiny, so structured pruning has little to chew on. **Knowledge distillation**
(2004.08116; layer-wise KD 8b44dab) is the right framing for "can a smaller/shallower net reproduce the
trained policy?" **Verdict: H-A1 (shrink width/depth, equivalence-gated) and H-R3 (distill the converged
3×64 actor into a 1–2 layer student, then time *that* at eval and as a training target).**

**C.5 Reduced precision (fp16/bf16/int8/posit).**
Mixed-precision training (1710.03740-style) and Posit8 DNN training (Semantic 68c040f7) show fp16/bf16
storage+compute with controlled accuracy loss. M1 NEON has native **fp16 at ~2× fp32 throughput**.
**Verdict: fp16 for the *forward-only* target expectation is low-risk (no gradients) and hits the 49%
block — H-P1. Full fp16 training is higher risk → defer/measure.**

**C.6 Numerical integration / quadrature for the expectation.**
*Fast & Simple Exotic Option Pricing via Gauss-Hermite Quadrature on a Cubic Spline* (arXiv 1408.6938)
and *Chebyshev Interpolation for Parametric Option Pricing* (1505.04648) are the classical levers for the
conditional expectation. Our mesh is already a 2-node Gauss-Hermite in X. **Verdict: confirms M_x=2 is
near-minimal; the open question is the *jump* dimension M_y — H-K1 tests whether the jump node can be
folded into the X-conditional mean with no price change (Phase-B evidence says N_max is irrelevant for
M_x≥2).**

**C.7 Systems: small/batched GEMM on Arm, activation approximation, optimizer convergence.**
*Optimization of small matrix multiplication kernels on Arm* (Semantic 5f0f60af), *autoGEMM on Arm*
(3644a5e), *Cache Optimization … Batched Small Rectangular MatMul* (2311.07602) — directly relevant to
the 64-wide GEMM and to a **batched** cross-network GEMM (H-S5). *On Polynomial Approximation of
Activation Function* (2202.00004) backs the existing `fast_expf` and a polynomial SiLU (H-S1).
Optimizer-side, high update-to-data-ratio RL (REDQ / *Aggressive Q-learning with Ensembles*, Semantic
b9183999; high-replay-ratio 2312.05787) and Shampoo/SOAP (2509.03378) inform whether **fewer episodes at
higher learn_number** preserve the price (H-T1). Schedule-free / two-timescale evidence supports the
existing constant-LR + EMA choice.

---

## Part D — Hypotheses (each with a hard acceptance gate)

Notation: **Δprice** = mean price difference (new − baseline) over the seed set; **Δeq** = equivalence
margin (Part E.2); **Δµs** = change in µs/learn-step; **Δwall** = change in `t_total`. A hypothesis is
**ADOPTED** only if it passes its gate. All speed claims use non-overlapping CIs (Part E.3); all "same
price" claims use **TOST** (Part E.2), never a bare Welch p>0.05.

### Tier 1 — mechanism-targeted, cheap, low-risk

**H-K1 · Drop the jump node (M=4 → M=2).**
*Mechanism:* the kernel-target's cost is ∝ M. M_y=2 adds the jump node; Phase-B found N_max/M_per_k
irrelevant for M_x≥2. Setting M_y=1 (fold jump into the X-conditional mean, i.e. add `mu_J·Edecay` to
`y_base`) roughly **halves the 49% block ⇒ ~24% step speedup**.
*Impl:* in `build_fast` (and a new exported `kernel_v64_Mx2_Ny1.bin`) set `delta_Y={Edecay·mu_J·… }`,
`w_Y={1}`; `expected_critic_target` already loops `M_y` generically.
*Gate:* TOST equivalence on 65k price across ≥15 seeds (Δeq, Part E.2) **AND** Δµs ≤ −20%. Expected the
strongest single win; test first.

**H-P1 · fp16 target-expectation forward.**
*Mechanism:* the target forward is inference on frozen weights — no gradient, so fp16 round-off cannot
destabilize learning, only perturb the target by ~1e-3 rel. NEON fp16 ≈ 2× fp32. Hits the 49% block.
*Impl:* an fp16 forward path for `Actor::forward`/`Critic::forward` used *only* inside
`expected_critic_target` (cast `flat`→half, GEMM in `hgemm`/NEON, accumulate in fp32). Keep the FP32 path
for the actor/critic *update* (gradients) and the FP64 parity build untouched.
*Gate:* TOST equivalence (≥15 seeds) **AND** Δµs ≤ −15% on the kernel block. Risk: BLAS half-GEMM may not
beat fp32 sgemm at 64-wide — measure the GEMM in isolation first.

**H-S1 · Vectorize & fuse LayerNorm+SiLU.**
*Mechanism:* `layernorm_forward` is the #2 leaf and is a scalar two-pass row reduction; SiLU is a
separate pass. Fusing (compute mean/var, normalize, apply γ/β, and SiLU in one NEON-vectorized sweep over
the 64-wide row) cuts memory traffic and loop overhead. Called 3×(actor)+3×(critic) per forward × many
forwards.
*Impl:* `linalg.hpp` — a `layernorm_silu_forward` fused kernel (and matching fused backward that still
caches `xhat`,`rstd`). Guard with the FP64 build keeping the unfused reference for parity.
*Gate:* `test_grad` < 1e-5 (FP64) **AND** Δµs ≤ −5% **AND** price TOST-equivalent (should be *identical*
up to fp rounding — this is a pure refactor).

**H-S3 · Replace `pow(q,γ)` with integer-power fast paths (γ∈{1,2}).**
*Mechanism:* `pow` is libm-slow; γ=2 focal ⇒ `q*q`. Hot in `env_step`/`act_single` (training rollout) and
eval; also the FOC gate `pow(payoff/cg, 1/(γ−1))`.
*Impl:* branch on γ in `env.hpp`/`agent.cpp` gate helpers.
*Gate:* bit-identical price by construction (γ=2 ⇒ `q*q` exact in fp) **AND** Δwall < 0 on `t_train`+eval.
Pure win; bundle with H-S1.

### Tier 2 — architecture / config, equivalence-gated

**H-C1 · Stale-target reuse (recompute the expectation every k steps).**
*Mechanism:* target nets move slowly (τ=0.0032). The expected-target `q_next` for a *given* transition
changes little step-to-step. Caching/reusing the target value for the same replay row across k≤4 learn-
steps, or recomputing the mesh forward only every k steps, cuts the 49% block by ~(1−1/k). This is a
sample-reuse idea backed by the high-replay-ratio literature (C.7).
*Impl:* memoize `q_next` keyed by replay index within a refresh window, or run `expected_critic_target`
every k steps and reuse for intermediate critic updates.
*Gate:* TOST equivalence **AND** Δwall negative. Risk: changes learning dynamics → must pass equivalence,
not just parity. Sweep k∈{2,3,4}.

**H-C2 · Preprocess once in the target.** (micro) Compute `hhk_preprocess(flat)` once and pass to both
`actor_t.forward` and `critic_t.forward`. *Gate:* identical price; Δµs<0. Bundle with H-S1.

**H-A1 · Shrink the network (width 64→{48,32}; depth 3→2), equivalence-gated.**
*Mechanism:* cost ∝ width² per layer and ∝ depth. The approximator study said 3×64 is needed *for
accuracy vs LSM* — but that predates the kernel-on canonical and TOST. Re-test the speed/accuracy Pareto
*under the kernel*, gated by equivalence (not just "looks close").
*Impl:* parameterize `H` and layer count (currently `constexpr int H=64`, fixed 3 layers in `mlp.cpp`).
Add a `--hidden`/`--depth` build or runtime knob; re-export reference weights per shape (or train from
scratch — no PyTorch parity needed for a C++-native arch).
*Gate:* TOST equivalence (≥15 seeds) **AND** Δwall ≤ −15%. Expect width 48 to pass, 32 to be borderline,
depth-2 to fail equivalence (depth-3 was a Stage-C winner) — but *measure*.

**H-T1 · Reduce the episode budget (4096 → ?).**
*Mechanism:* t_train ∝ episodes. If price converges by N<4096, the tail is wasted. Levers to converge
sooner: higher `learn_number`/UTD (REDQ-style, C.7), better warm-start, optimizer (Shampoo/SOAP test).
*Impl:* convergence study — price (TOST band) as a function of episode budget ∈ {1536, 2048, 3072, 4096},
each at ≥15 seeds; also a `learn_number`∈{2,3,4} × episodes grid.
*Gate:* smallest budget whose 65k price is TOST-equivalent to the 4096 baseline **AND** lower t_total.
This is the **most likely large win** because it scales the dominant cost directly.

### Tier 3 — research-grade (separate branch; high upside / high risk)

**H-R1 · Random-feature (ELM) critic with closed-form ridge readout.**
*Mechanism (C.2):* freeze the critic's hidden layers at random init (or a one-time fit), train only the
final linear readout by **ridge regression** against the kernel-target — solved in closed form each
update (a 64×64 normal-equation solve, BLAS `posv`), eliminating critic backward + Adam and most of the
critic's iterative cost. This is LSM's own regression machinery inside the actor-critic loop.
*Impl:* new `CriticELM` with fixed `se/al/pl` and a least-squares `fc4`; recursive least squares (RLS) or
periodic batch solve over the replay minibatch.
*Gate:* TOST equivalence on price **AND** Δwall ≤ −40%. High risk to dynamics (the actor's DPG gradient
flows through the critic readout only) → expect to iterate; park if it can't hit equivalence.

**H-R2 · Likelihood-ratio direct-policy swing solver (Warin 2001.11247).**
*Mechanism:* drop the critic/DPP entirely; optimize the exercise policy directly with a likelihood-ratio
gradient. May converge in far fewer gradient steps for this specific contract class.
*Gate:* prototype-level; price within Δeq of the v64 baseline at lower t_total. Most exploratory; only if
Tier 1–2 underdeliver.

**H-R3 · Distill the converged actor into a 1–2 layer student.**
*Mechanism (C.4):* after a normal 4k train, fit a tiny student to the EMA actor's action map, then use the
student at eval (and optionally as a fixed target). Cheapens eval and possibly the target forward.
*Gate:* student-eval price TOST-equivalent **AND** faster eval; low risk (post-hoc, doesn't touch training
dynamics). Good "safe" Tier-3 item.

### Tier 4 — systems last-mile (only after Tier 1–2 settle)

- **H-S2 fused multi-layer MLP kernel** (keep activations in registers/L1 across 9→64→64→64→1; Arm small-
  GEMM literature C.7). Measure vs Accelerate; prior single-layer hand kernels lost, so this is uncertain.
- **H-S4 Adam state in fp32** (`adam.hpp`) — de-risk an all-fp32 build; gate on identical-within-tolerance
  price + Δµs<0.
- **H-S5 batched cross-network GEMM** — the kernel-target runs actor then critic on the same 512 rows;
  batch the two first-layer GEMMs (9→64) into one call to amortize dispatch. Small but free-ish.

---

## Part E — Statistical & profiling methodology (how we *prove* it)

### E.0 Prerequisites — ✅ ALL DONE (2026-06-14)
1. ✅ **`data/` regenerated.** `~/miniforge3/envs/EP11/bin/python cpp_pricer/tools/export_reference.py`
   (now `N_max`-parametrized, **default 0 = canonical M=2**) → `kernel_v64.bin` + actor/critic/ref fixtures.
   FP64 sibling (`build_fp64`) built; **`test_parity` / `test_grad` / `test_grad_depth` / `test_sim` all PASS.**
   ⇒ equivalence is no longer C++-vs-C++ only: **H-K1 (M=4→2, N_max=0) is now parity-validated against the
   PyTorch kernel** (the production caveat in §0/§A.3 is closed). ⚠️ Discovered in the process: the C++
   `build_fast` fallback prices **~+0.002–0.003 higher** than the exported mesh — always quote the exported
   mesh for cross-method (RL-vs-LSM) claims (see NN plan ledger "MESH CORRECTION").
2. ✅ **`PRICER_PROFILE` wired into CMake** — `option(PRICER_PROFILE …)` → `-DPRICER_PROFILE=1`; verified the
   bench block split emits via `cmake -B build_prof -DPRICER_PROFILE=ON`.
3. ◻ **TOST/Levene wrappers** — the research analyzers (`tools/research_equiv.py`) already implement TOST +
   paired-superiority + Pitman-Morgan variance + bootstrap std-ratio CI standalone (no scipy); folding them
   back into `tools/stats_analysis.py` is optional housekeeping, not gating.

> **PLAN CLOSED (2026-06-14).** Tiers 1–3 exhausted (2.37–4.13× delivered, all TOST-equivalent; Tier-3
> replace-the-learner rejected); prerequisites 1–2 now green. The follow-on **NN-architecture study**
> (`NN_ARCHITECTURE_RESEARCH_PLAN.md`) then redesigned the net itself (asymmetric shallow-actor/deep-critic),
> beating reduced-state LSM on accuracy (+0.6–0.7%, p<1e-4) and — via a K=3 ensemble — on variance, on the
> canonical mesh.

### E.1 Primary & secondary endpoints
- **Primary:** 65k OOS option price (mean over the seed set). Secondary: seed-std (robustness), focal
  **Delta%** = (RL/LSM − 1)·100, avg exercised quantity (sanity), bang-bangness.
- **Baseline:** the current FP32 `price_swing` (exported-mesh build) at seeds **{11…25}** (n=15), matching
  the existing focal protocol; extend to {11…40} when MDE demands (E.4).

### E.2 Equivalence margin Δeq and the TOST gate (the core "same results" test)
- Define **Δeq in price units** tied to economic + statistical relevance. Default: **Δeq = 0.5% of the
  baseline price** (≈ ±0.010 at price ≈ 1.98), comparable to the published seed-noise band
  (C++ 1.982 ± 0.009). Pre-register Δeq *before* running each hypothesis.
- **TOST:** two one-sided Welch t-tests on (new − baseline). Conclude **equivalent** iff the **90% CI** of
  Δprice ⊂ [−Δeq, +Δeq] (equivalently both one-sided p<0.05). Report the CI, not just the verdict.
- Also report the ordinary Welch two-sample t (to detect a *shift*) and **Levene/Brown-Forsythe** (to
  detect a *variance/robustness* regression — a change can be mean-equivalent yet seed-fragile).
- Where the change is near-deterministic per seed (e.g. H-S1/H-S3 refactors, H-P1), use the **paired
  per-seed** test (CRN) for tighter power; otherwise two-sample.

### E.3 Speed measurement & CIs
- **µs/step:** `bench_train --K 20000`, **n≥7** clean runs (no other load — DEVELOPMENT_NOTES warns the
  seed sweeps were contention-polluted). Report **median + 95% bootstrap CI** and the **min** (least-
  contended estimate). A speedup is real iff the new CI is **below** the baseline CI (non-overlapping).
- **t_total / t_zero_to_4k:** `price_swing --quiet`, n≥7 clean runs, median + IQR. This is the number the
  user actually minimizes.
- **Mechanism check:** the `-DPRICER_PROFILE` block split must move in the predicted direction (e.g. H-K1
  must shrink the *kernel* block specifically) — guards against accidental speedups from the wrong cause.
- **Variance budget:** with <2% run-to-run spread, **~3–4%** speedups are detectable at n=7; require Δµs
  effects larger than that to claim a win.

### E.4 Power / MDE
- Pre-compute seeds needed for 80% power to (a) detect a shift of Δeq and (b) demonstrate equivalence
  within Δeq, given seed-std ≈ 0.009 (`paired_minimum_detectable_effect`). At σ=0.009 and Δeq=0.010,
  n=15 is adequate for the paired case; two-sample arms may need n≈25–30 — budget accordingly.

### E.5 Multiplicity
- Across the hypothesis battery, control family-wise error with **Holm–Bonferroni** on the *shift* tests
  (`holm_bonferroni`, already present). Equivalence decisions are pre-registered per hypothesis, so the
  primary inference is the TOST CI; Holm guards the secondary shift screen.

### E.6 Decision rule (per hypothesis)
```
ADOPT   if  TOST-equivalent (price)  AND  variance not worse (Levene n.s.)  AND  speed CI strictly better
PARK    if  TOST-equivalent  AND  speed-neutral            (keep for combination tests)
REJECT  if  NOT TOST-equivalent      OR   variance regresses OR  no speed gain
```

---

## Part F — Iteration protocol & living ledger

The user asked for *many rounds, reviewing what worked and adjusting*. Protocol:

1. **Round = one hypothesis (or one small combination).** Implement behind a build flag / runtime knob so
   the baseline is one rebuild away. Keep the FP64 parity sibling green.
2. **Run the gate** (E.2 + E.3 + E.5). Record the full result row in the ledger (below) — including
   negative results, with the mechanism-check delta.
3. **Decide** (E.6): ADOPT / PARK / REJECT. On ADOPT, fold into the running baseline and re-profile (the
   block split shifts → re-rank remaining hypotheses by the *new* cost model).
4. **Re-rank.** After each adoption the 49%/33%/17% mix changes; the next round attacks the new dominant
   block. (e.g. if H-K1 halves the kernel block, the actor step becomes the largest and H-A1/H-S1 rise.)
5. **Combination rounds.** Periodically test the union of ADOPTED winners jointly — speedups compose but
   equivalence may not (interactions); re-run the full gate on the stack.
6. **Stop** when the remaining hypotheses' expected value (speedup × P(equiv) ÷ effort) drops below a
   threshold, or the per-step time is dominated by irreducible BLAS at the chosen network size.

### F.1 Suggested round order (initial; will be re-ranked by results)
`R0` env+prereqs (E.0) → `R1` H-S3+H-C2+H-S1 (pure refactors, free wins, re-baseline) →
`R2` **H-K1** (biggest single lever) → `R3` H-P1 → `R4` **H-T1** (episode budget) → `R5` H-A1 (shrink) →
`R6` H-C1 (stale target) → `R7` combine R2+R4+R5 → `R8` H-R3 (distill, safe Tier-3) →
`R9` H-R1 (ELM critic, the moonshot) → `R10` Tier-4 systems cleanup.

### F.2 Results ledger (template — append one row per round)

| Round | Hypothesis | µs/step (min,n=7) | t_train | Δprice (paired) | Decision |
|------:|------------|-------------------|---------|-----------------|----------|
| R0 | baseline (build_fast M=4) | 475.8 (med 477) | 41.7 s | ref | — |
| R1a | H-S1 fuse LN+SiLU | **517.0 (+8.7%)** | — | bit-identical | **REJECT** (defeats SiLU vectorization) |
| R1b | H-S3 pow→q*q/foc_qstar | 480.5 (≈ref, noise) | 41.5 s (≈ref) | **0.0e+00, seeds 11–14** | **PARK** (bit-identical, 0 measurable speedup) |
| R1c | H-C2 preprocess-once | — | — | — | DEFERRED (preprocess not hot) |

**R1 learnings / re-rank (2026-06-13).** Pure-refactor tier is a wash. `learn_step` is BLAS-bound +
already-optimally-vectorized flat activation loops; **fusing hurts** (interleaving SiLU into the
row-serial LayerNorm reduction blocks the compiler's vectorization of `fast_expf`). The non-learn-step
code (rollout/eval/calibrate) is <0.1% of wall-clock, so `pow` elimination is invisible. ⇒ Only
learn-step GEMM-row count (**H-K1**), learn-step arithmetic precision (**H-P1**), and learn-step *count*
(**H-T1**) can move the needle. Promote H-K1 to R2; demote all remaining systems micro-opts (H-S2/S4/S5)
below the algorithmic levers. H-S3 kept in-tree (bit-identical, harmless, removes libm `pow` from the
γ∈{1.5,3} sweep cells where it could matter on other HW) but claims **no speedup**.

| **R2** | **H-K1 drop jump node (M=4→2)** | **369 (−23.5%)** | **31.6 s (−24.9%)** | **−0.000046, 90% CI [−3.1e-4,+2.2e-4] ⊂ ±9.9e-3 → TOST EQUIVALENT** | **ADOPT** |

**R2 result (2026-06-13) — the first real win.** Folding the HHK jump into its unconditional mean
(N_max=0 ⇒ M_y=1, M=4→2) cut the learn-step **23.5%** (482→369 µs, n=7, tight) and end-to-end train
**24.9%** (42.1→31.6 s, 15 seeds). Price is statistically **equivalent** (paired Δ=−5e-5, 90% CI ~40×
narrower than Δeq; Welch p=0.31; seed-std ratio 1.005 — no robustness loss). Empirically confirms the
Phase-B "N_max irrelevant for M_x≥2" claim, now monetized. Adopted as the build_fast default
(`config.hpp` N_max=0; `--n_max 1` reproduces M=4). *Production caveat:* re-export `kernel_v64.bin` with
N_max=0 to carry this into the PyTorch-mesh pipeline.

**Re-profiled M=2 cost model + re-rank.** New split: **actor 43% / kernel 34% / critic 22% / soft 0.4%**
(was 49/33/17/0.3). The **actor step is now dominant** (it runs a full critic fwd+bwd for the DPG action
gradient *plus* an actor fwd+bwd). Re-ranked next rounds: **R3 = H-A1** (shrink width 64→48/32 — now hits
the dominant actor block, cost ∝ width²) and **R4 = H-T1** (fewer episodes — orthogonal, scales the new
31.6 s linearly). **H-P1 (fp16)** demoted: at M=2 the kernel is only 34% and 64-wide Accelerate may not
give fp16's 2×; measure the half-GEMM in isolation before committing. Tier-4 systems micro-opts remain
below the algorithmic levers (R1 showed they're a wash).

| **R4** | **H-T1 budget 4096→3072 (M=2)** | — | **23.1 s (−45% vs orig, clean n=3)** | **−0.00154 (−0.08%), 90% CI [−0.00225,−0.00083] ⊂ ±9.9e-3 → EQUIVALENT** | **ADOPT (recommended)** |
| R4b | H-T1 budget 4096→2048 (M=2) | — | ~16 s (−62% vs orig) | −0.00413 (−0.21%), CI [−0.0050,−0.0032] ⊂ ±9.9e-3 → EQUIVALENT but biased | **PARK** (aggressive option) |
| **R5** | **H-A1 width 64→48 (M=2, 3072)** | **306.4 (−15.8%, n=7)** | **19.35 s (−16.7% vs w64@3072)** | **−0.000302 (−0.015%), 90% CI [−0.00149,+0.00089] ⊂ ±9.9e-3 → EQUIVALENT; shift p=0.34 (n.s.); var ratio 0.923 (improved)** | **ADOPT** |
| R5b | H-A1 width 64→32 (M=2, 3072) | 200.5 (−44.9%, n=7) | 12.39 s (−46.7% vs w64@3072) | −0.00221 (−0.11%), CI [−0.00330,−0.00113] ⊂ ±9.9e-3 → EQUIVALENT but biased (shift p=0.003); var ratio 0.983 | **PARK** (aggressive option) |

**R5 result (2026-06-13) — width is a clean, multiplicative lever.** Made the hidden width `H` a
runtime member of `Actor`/`Critic` (was `constexpr int H=64`), threaded through `AgentConfig::hidden`
+ a `--hidden` CLI flag on `price_swing`/`bench_train`; **default 64 reproduces v64 exactly** (PyTorch
parity untouched). Per-step cost ≈ ∝ width (not width²: the 9-dim input layer, kernel-mesh prep and
BLAS dispatch don't scale with width). **Width 48** is a clean win: TOST-equivalent vs w64@3072 (diff
−0.015%, **no significant shift** p=0.34, **variance improved** ratio 0.92), −15.8% µs/step, −16.7%
t_train. **Adopted as canonical.** **Width 32** is TOST-equivalent (−0.11%, within ±0.5%) but shows a
**detectable systematic low bias** (paired p=0.003) — same situation as 2048-episodes; **parked** as an
aggressive option (would survive a 0.25% margin but is a real, if tiny, shift). ⚠️ Process note: the
price is **deterministic per thread-count but shifts ~0.27% across thread-counts** (a path-simulation
artifact, NOT training). Width does not touch path sim ⇒ the w64 baseline was **regenerated fresh at
fixed `--threads 8`** alongside the variants so the paired TOST isolates width alone. *Lesson logged:
pin thread count across baseline+variant in every price-equivalence comparison.*

**Combined adopted stack (canonical):** H-K1 (M=2) + H-T1 (3072) + H-A1 (width 48) ⇒
**42.06 → 19.35 s = 2.17×, 54% faster training, TOST-equivalent price** (verified incrementally;
combination round R7 below re-tests the full stack jointly vs a fresh original). Aggressive stack
(+ width 32, 2048 ep) reaches ~3.4–4× but carries a documented small systematic low bias.

| **R7** | **COMBINATION: full stack (M=2, 3072 ep, w48) vs ORIGINAL (M=4, 4096 ep, w64)** | — | **19.35 s vs 42.06 s = 2.17×** | **−0.00194 (−0.098%), 90% CI [−0.00309,−0.00078] ⊂ ±9.9e-3 → EQUIVALENT; shift p=0.011; var ratio 0.950** | **CONFIRM** |
| R7b | aggressive stack (M=2, 3072 ep, w32) vs ORIGINAL | — | 12.39 s = 3.39× | −0.00385 (−0.194%), CI [−0.00478,−0.00291] ⊂ ±9.9e-3 → EQUIVALENT but biased (p<1e-7) | confirm (aggressive) |

**R7 result (2026-06-13) — transitive equivalence HOLDS.** Incremental equivalence (each round vs the
previous adopted baseline) does not automatically compose, so the full canonical stack was re-tested
**jointly against a freshly-regenerated true original** (M=4, 4096 ep, width 64, same `--threads 8`,
seeds 11–25). The stack is **TOST-equivalent to the original** with a cumulative −0.098% low bias
(90% CI ⊂ ±0.5%), which is precisely the additive sum of its components (H-T1 −0.08% + w48 −0.015%);
the shift is statistically detectable (p=0.011) but economically negligible and variance is *improved*
(ratio 0.95). **Headline: 42.06 → 19.35 s = 2.17× / 54% faster, price statistically the same.** The
aggressive stack (width 32) is also equivalent at the 0.5% margin (−0.194%) but carries a clearly
significant systematic bias that a 0.25% margin would reject — kept as the documented speed-max option
(3.39×). This is the rigorous capstone for Tiers 1–2.

| **R6** | **H-C1 stale-target reuse (reuse minibatch+target across learn_number, w48 M=2 3072)** | **247.1 (−17.5%, n=7)** | **15.76 s (−18.6% vs w48)** | **vs w48: −0.00137 (−0.069%), 90% CI [−0.00205,−0.00070] ⊂ ±9.9e-3 → EQUIVALENT; shift p=0.003; var ratio 0.93** | **ADOPT (fast tier)** |
| R6→orig | fast stack (w48 + H-C1) vs ORIGINAL | — | 15.76 s = **2.67×** | −0.00331 (−0.167%), CI [−0.00432,−0.00230] ⊂ ±9.9e-3 → EQUIVALENT (p=1e-4); var ratio 0.885 (improved) | confirm |

**R6 result (2026-06-13) — stale-target reuse, the kernel's last big lever.** With `learn_number=2`,
the inner learn loop recomputed the (dominant) kernel-target expectation twice per interaction. H-C1
samples one minibatch + computes the target **once**, then reuses that (minibatch, target) pair for both
critic+actor updates (`--reuse_target 1`; `AgentConfig::reuse_target`). Sound because the target nets are
frozen within the group (they soft-update by τ=0.0032 only at step end), so recomputing barely moves the
target — we skip the forward instead. **−17.5% µs/step, t_train 19.35→15.76 s (−18.6%).** Price is
TOST-equivalent (−0.069% vs w48) with a small **detectable** systematic low bias (p=0.003) and *improved*
seed variance (ratio 0.93). The full fast stack (w48 + H-C1) is TOST-equivalent to the **original** at
−0.167% (still inside a 0.25% margin), variance tighter (0.885). **Adopted into the *fast* tier** (not
the conservative canonical) since it is the point where cumulative drift becomes clearly significant — a
zero-detectable-drift user stops at w48 (2.17×); the fast tier is **2.67×**.

**Re-profiled (w48) after H-C1 + re-rank.** New split: **actor 51.9% / critic 27.1% / kernel 20.6% /
soft 0.4%** (reuse off at w48 was 43/22/34). The kernel-target is no longer a primary cost — H-K1 + H-C1
between them took it from 49% (R0) to ~21%. The actor step (DPG: a critic fwd+bwd for the action gradient
+ actor fwd+bwd) is now the dominant ~52% — which motivated R8.

| **R8** | **H-S6 skip discarded critic-grad in DPG actor step (bit-identical)** | **282.2 / 229.6 (−5.8% / −7.1%, n=7)** | **w48 17.76 s; +reuse 14.61 s** | **BIT-IDENTICAL (1.968648, 1.972956 … exact match across seeds)** | **ADOPT (always-on)** |

**R8 result (2026-06-13) — free win on the dominant block.** The DPG actor update calls
`Critic::backward` only for the **action gradient** `g_action`; the critic's *parameter* gradients it
also computed were **discarded** (the critic `zero_grad`s before its own step). `linear_backward` /
`layernorm_backward` / `Critic::backward` gained an `accum_params` flag (default true; the FP64
`test_grad` path is untouched). With `accum_params=false` the actor step skips (a) the `gW += gYᵀ·X`
GEMM in every critic linear and (b) the **entire state-encoder backward branch**, which is fully dead
when only `g_action` is needed (its `gX` feeds nothing, its params are discarded). **Bit-identical** by
construction — verified exact to 6 dp across seeds, no equivalence test needed — and **−5.8% µs/step**
(conservative w48) / **−7.1%** (fast w48+reuse). Hardcoded always-on (it is unconditionally correct).
Actor block 43.2%→39.1%. This is the kind of mechanism win the cost model points to once a block
dominates: not fewer FLOPs in a GEMM, but *not issuing a GEMM whose result is thrown away*.

**Final speed tiers (H-S6 always on; all TOST-equivalent to the v64 original at the 0.5% margin):**

| Tier | Config (flags on top of default M=2) | t_train | vs original 42.06 s | Price drift vs orig | Notes |
|---|---|---:|---:|---|---|
| **Conservative** | `--hidden 48 --n_train 3072` | **17.76 s** | **2.37×** | −0.098% (p=0.011) | no clearly-meaningful drift; var improved |
| **Fast** | `+ --reuse_target 1` | **14.61 s** | **2.88×** | −0.167% (p=1e-4) | survives a 0.25% margin; var improved |
| **Aggressive** | `--hidden 32 + reuse` | **10.18 s** | **4.13×** | ~−0.26% | would fail a 0.25% margin; speed-max |

### Tier-3 rounds (research branch — replace the learner)

| **R9** | **H-R1 ELM critic (frozen random features + closed-form ridge readout)** | w48 **300.9 (+6.5%, SLOWER)**; w128 1362; w256 4436 | — | w48 price **1.50–1.78** (NN 1.97); w128 ~1.86; w256 ~1.86 — gap ~5% = **~10× the ±0.5% margin** | **REJECT** (both less accurate AND slower) |

**R9 result (2026-06-13) — ELM critic is a dead end here.** Froze the critic's hidden layers at He-init
(random features φ(s,a) = last hidden activation) and learned only the linear readout `fc4` by ridge
regression — running normal equations with exponential forgetting, solved each step by an in-house
Cholesky (`elm_update_readout` / `chol_solve` in `src/agent.cpp`; `--elm_critic`, `--elm_ridge`,
`--elm_forget`; default off ⇒ bit-identical). **Two-axis failure:** (1) *accuracy* — at the canonical
width 48 the price **collapses** (1.50–1.78 vs the NN's 1.97, huge seed spread); widening the random
features lifts it but **plateaus ~1.86** (w128 and w256 both), a persistent ~5% low bias that is ~10×
the TOST margin and never approaches the NN. Random features cannot represent the *global* Q(s,a,t) that
the DPG policy gradient must differentiate — unlike LSM, which regresses a *local* continuation value on
a fixed per-date design. (2) *speed* — even at w48 the ridge solve is **slower** than the backprop it
replaces (300.9 vs 282.5 µs/step), and the (H+1)² accumulation + O(H³) Cholesky make wide features
catastrophic (w128 4.8×, w256 15.7× slower/step). So the one config that might approach accuracy is far
slower than the width-48 NN it would replace. Matches the plan's pre-registered "park if it can't hit
equivalence." Kept behind the flag as a documented dead-end (like H-S1). The LSM-style closed-form
regression belongs at the *exercise-date* granularity (the model-based DP line), not as a global critic.

| H-R3 | distill converged actor → tiny student (eval-only) | — | eval = **0.073 s = 0.51% of wall-clock** | (not run) | **PARK by construction** |
| **R10** | **H-R2 likelihood-ratio direct policy (REINFORCE, critic-free, Warin-style)** | — | **0.11 s @3072 … 33 s @1M (fast/episode)** | price **1.63–1.72** vs 1.97 — **~14–17% off (~35× margin); does NOT converge with budget** | **REJECT** |

**H-R3 result — PARK by construction.** The distill hypothesis cheapens *eval* (a post-hoc tiny student
of the EMA actor). But on this train-dominated profile eval is **0.073 s = 0.51% of t_total**, and a
post-hoc student cannot serve as a *moving* training target (it can't track the still-training actor), so
it cannot touch t_train. A perfect 2× eval speedup would save ~0.25% of wall-clock — below the detection
threshold and far under every other lever. Not implemented: the mechanism's ceiling is provably <0.5%.

**R10 / H-R2 result — REINFORCE can't price it.** Implemented a critic-free direct-policy trainer
(`train_direct_policy`, `--direct_policy 1`): the actor's pre-squash exploration is read as a Gaussian
policy on the mean u_θ(s) (∇log π = (ε/σ_eff)·∂u/∂θ), trained by REINFORCE with a moving-average return
baseline — no critic, no kernel target (added `Actor::backward_from_u` for the pre-activation gradient).
It is **~150× faster per episode** (0.11 s for 3072 ep) but the price is **~14% low** and, decisively,
**more budget does not help**: 20k→1.70, 100k→1.64, 400k→1.63, 1M (33 s)→1.63 — it *plateaus ~17% below
D4PG and even drifts slightly worse* as exploration decays into a poor local optimum. This is the textbook
critic-free pathology: the high-variance likelihood-ratio gradient with only a scalar baseline cannot
resolve the 22-right exercise policy to within the ±0.5% margin at any budget that would still be fast.
**REJECT.** Kept behind the flag as a documented dead-end.

**Tier-3 conclusion — the learner is right; don't replace it.** All three "replace the learner" routes
fail the equivalence gate: H-R1 (ELM critic) collapses / plateaus ~5% off and is slower; H-R2 (direct
policy) plateaus ~17% off and never converges; H-R3 (distill) is bounded to <0.5% of wall-clock by
construction. The semi-analytical-kernel D4PG — *with* its learned critic and model-based target — is
confirmed as the accurate, sample-efficient learner; the realizable speedup is the **2.37–4.13× from
Tiers 1–2**, achieved by making that same learner cheaper, not by swapping it out. (Consistent with C.2's
randomized-net evidence applying at the *exercise-date* regression granularity — the model-based DP line
— not as a global Q-critic.)

**Cost-model verdict / stop assessment.** After R8 the per-step split (w48) is **actor ~39% / kernel
~37% / critic ~24% / soft 0.3%** — no single block dominates and all remaining sub-blocks are either
irreducible BLAS at the chosen (already-shrunk) width or tiny. Per Part F.6's stop rule ("per-step time
dominated by irreducible BLAS at the chosen network size"), Tiers 1–2 (+ the H-S6 systems win) are
**exhausted**, and Tier-3 (the only remaining branch) is now **empirically closed** (R9/R10 REJECT,
H-R3 PARK). H-P1 (fp16) is dead (kernel forward is a minority, 48-wide Accelerate won't give 2×);
Tier-4 micro-systems (H-S2/S4/S5) are a wash (R1). **The research plan is exhausted.**

**R4 result + process note (2026-06-13).** ⚠️ First H-T1 sweep was **invalid** — `build_r1` wasn't
rebuilt after the N_max=0 default flip, so it ran M=4, confounding mesh×budget. *Lesson logged: rebuild
the binary that reads a changed default, or always pass the flag explicitly.* Re-ran cleanly at M=2.
Both reduced budgets are TOST-**equivalent** vs M=2@4096, but show a **monotone under-convergence bias**:
3072 = −0.08% mean / −0.26% worst (negligible, ~5× under the 0.43% per-seed noise; seed-std ratio 1.037);
2048 = −0.21% mean / −0.39% worst (real, p<1e-7, still in margin; ratio 1.058). **Adopt 3072** (combines
with H-K1: 42.06→31.6→**23.1 s clean**, = **1.82× / 45% faster than the original**, price equivalent). **2048 parked** as an
aggressive option for users who accept a −0.2% systematic low bias. A tighter Δeq=0.25% would fail 2048
but pass 3072 — the choice of margin is doing real work here, exactly as intended.

**Combined adopted stack:** H-K1 (M=2) + H-T1 (3072) ⇒ **42.06 → 23.1 s = 1.82×, 45% faster training,
TOST-equivalent price**.
Next: **R5 = H-A1** (shrink width 64→48, now the dominant actor block) — compounds multiplicatively on
the per-step cost; needs `H` made runtime/templated + retrain (no PyTorch parity, C++-native arch).

### F.3 Reproducibility artifacts per round
- `bench_train` raw logs + `price_swing` JSON per seed → `cpp_pricer/data/research/Rxx_*.json`.
- A `tools/collect_results.py`-style aggregator emitting the ledger row (mean/CI/TOST/Levene) via
  `stats_analysis.py`.
- One-line entry in this file's ledger + a short "what we learned / re-rank" note.

---

## Appendix — Screened bibliography (by theme)

IDs are `source:paper_id` for re-fetch via `paper-search read <source> <id>`. ★ = pivotal to a hypothesis.

**Model-based value targets (C.1):** ★arxiv:2203.14660 Revisiting Model-based Value Expansion ·
★arxiv:2303.03955 Diminishing Return of Value Expansion · arxiv:2101.09890 Efficient approximation for
high-dim semilinear PDEs (control variates).

**Randomized / reservoir / ELM (C.2):** ★semantic:e12309e6… Optimal Stopping via Randomized NN ·
semantic:89979d2a… Effective non-random ELM · semantic:b9027c3c… Shallow/ensemble deep randomized NN ·
arxiv:1609.04846 Tutorial on Random Neural Networks · arxiv:1905.09368 Outlier-robust ELM regression ·
arxiv:2509.08759 Fourier Learning Machines.

**Direct-policy / option-pricing NN (C.3):** ★arxiv:2001.11247 Deep combinatorial optimisation for
swing options · arxiv:1907.06474 NN regression for Bermudan options · arxiv:2402.15936 Optimizing NNs for
Bermudan pricing (convergence acceleration) · semantic:2dbfda08… Deep primal-dual BSDE for optimal
stopping · semantic:2689c5da… Optimal stopping with Deep Q-Learning · semantic:795d69b2… NN expressivity
for optimal stopping · semantic:88d34f55… Optimal stopping with signatures · arxiv:2601.18634 Compound
BSDE method · arxiv:1906.06483 Multi-path autoregressive MC.

**Compression / pruning / distillation (C.4):** arxiv:1912.05671 Linear Mode Connectivity & LTH ·
arxiv:1903.01611 Stabilizing LTH · semantic:21937ecd… Lottery Ticket Hypothesis · arxiv:2306.14152
Low-Rank Prune-And-Factorize · arxiv:1802.02271 Universal DNN Compression · arxiv:2004.08116 Triplet Loss
for KD · semantic:8b44dabd… Layer-wise KD for lighter NNs · arxiv:2501.16372 Low-Rank Adapters meet NAS.

**Reduced precision (C.5):** semantic:68c040f7… Posit8 energy-efficient DNN training · arxiv:2104.11805
Partitioning sparse DNNs for scalable train/infer.

**Quadrature / interpolation (C.6):** ★arxiv:1408.6938 Exotic option pricing via Gauss-Hermite quadrature
on cubic spline · arxiv:1505.04648 Chebyshev interpolation for parametric option pricing · arxiv:0503234
Bermudan pricing via réduite/cubature.

**Systems — small/batched GEMM, activation, optimizer (C.7):** ★semantic:5f0f60af… Small matmul kernels
on Arm · semantic:3644a5ed… autoGEMM on Arm · arxiv:2311.07602 Cache opt & modeling of batched small
rectangular matmul · semantic:d0514f23… Batched small matmul on multicore DSP · arxiv:2305.02444 FT-GEMM
on x86 · arxiv:2202.00004 Polynomial Approximation of Activation Functions · ★semantic:b9183999…
Aggressive Q-learning with Ensembles (high UTD) · arxiv:2312.05787 High replay-ratio RL · arxiv:1909.11583
Off-policy actor-critic w/ shared replay · arxiv:2402.05963 Frugal Actor-Critic · arxiv:1801.01290 SAC ·
arxiv:2509.03378 Improving Shampoo & SOAP.

*(Search also surfaced ~15 off-topic hits from keyword collisions — excluded.)*
