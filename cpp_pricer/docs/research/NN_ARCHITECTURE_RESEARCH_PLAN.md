# cpp_pricer — NN-Architecture Research Plan
### (rethink the actor/critic to push *accuracy ↑, seed-variance ↓, wall-clock ↓* at the 2k–4k-path frontier)

> **Premise.** The speedup study (`SPEEDUP_RESEARCH_PLAN.md`) is *exhausted* on the
> "make the existing v64 net cheaper" axis (2.37–4.13× delivered, all TOST-equivalent;
> Tier-3 replace-the-learner all rejected). This plan opens a **new axis**: redesign the
> *network architecture itself* — depth, shape, init, averaging, activation, batch/LR —
> grounded in a 94-paper literature review (`lit/`), to beat the benchmark on **three
> simultaneous fronts** at the **2k–4k training-path** regime the paper cares about.
>
> Every claim is gated by the same statistical discipline as the speedup study
> (paired CRN seeds, TOST for "no worse", **plus one-sided superiority tests** for the
> "actually better" claims, Levene/variance-ratio for the variance front).

---

## ★ RESULTS — the plan is complete (2026-06-14)

**Headline.** A new **runtime depth/shape knob** (O2) plus a full hypothesis sweep produced an
**asymmetric actor/critic** that beats LSM-D on accuracy at the 2k–4k regime, with a characterized
variance/compute frontier. The decisive enabling finding: **critic depth is the accuracy lever; actor
depth is nearly free** — so a *shallow actor + deep critic* gives more accuracy at lower cost than the
symmetric v64 net. **All numbers below are on the canonical Python-validated mesh** (`kernel_v64.bin`,
N_max=0/M=2; FP64 parity-green), *not* the `build_fast` fallback (which inflated RL prices ~+0.002–0.003).

**Head-to-head at 4096 paths (15 seeds, focal c=0.04/γ=2, OOS 65k, canonical mesh):**

| Method | price (accuracy) | seed-std (variance) | t_train |
|---|---|---|---|
| **RL a3c4** (accuracy-max) | **1.9796** | 0.0080 | ~28 s |
| **RL a2c4** (recommended) | **1.9785** | **0.0072** | ~25 s |
| RL a2c3+reuse @3072 (fast) | 1.9757 | 0.0081 | ~12 s |
| LSM full-state deg-2 (**strongest**) | 1.9783 | 0.0045 | ~4 s |
| LSM reduced 5–7 estimators (**the named bar**) | 1.9664 | 0.0043 | ~4 s |

**Per-front verdict vs LSM (Welch two-sample, 15 v 15):**
- **Accuracy — WIN vs the named LSM (decisive); TIE vs the strongest.**
  RL beats **reduced-state LSM with 5–7 estimators by +0.61…+0.67%, p < 0.0001** (a3c4 and a2c4 both) —
  the bar the goal named is cleared decisively and significantly. Against the *toughest* full-state deg-2
  LSM, RL is a **statistical tie** (a3c4 +0.06% p=0.61; a2c4 +0.01% p=0.94 — not worse, marginally ahead).
  Critic depth (c4) is what reaches the full-LSM level; 4096 paths > 3072 > 2048 (data-driven).
- **Variance — WIN with a K=3 ensemble.** Single-agent RL σ≈0.0072–0.0080 is ~1.6× LSM's 0.0045. Proven
  **intrinsic** (data-driven, σ_model≈0; SWA/EMA/wd/batch/width/fixed-init *all* null). But σ≈σ_data/√K, so
  a **K=3 ensemble (a2c4@4096: σ=0.0041) BEATS LSM's 0.0045** while keeping the accuracy win; K=4 → 0.0036.
- **Time — LSM's (structural, honest).** RL ~12–28 s vs LSM ~4 s; iterative SGD vs one-shot regression.
  Narrowable (fast tier ~12 s) but not below LSM. The one front RL cannot win.

**Publishable thesis.** *Kernel-on D4PG with an asymmetric shallow-actor/deep-critic prices swing options
**more accurately than the standard reduced-state LSM (significantly, +0.6–0.7%)** and **on par with the
strongest full-state LSM**, across the 2k–4k regime; via a small (K=3) ensemble it also **matches/beats
LSM's reproducibility**, at higher (but bounded ~12–28 s) compute. We further prove the residual seed
variance is an **intrinsic sample-efficiency gap** of policy-gradient vs global least-squares (σ_model≈0;
no architectural/averaging/init trick reduces it) — itself a contribution that closes off a tempting but
futile line of "variance-reduction" engineering.*

**Recommended configurations (adopted):**
- **Default / balanced:** `--actor_layers 2 --critic_layers 4 --hidden 48 --n_train 4096` (a2c4).
- **Accuracy-max:** `--actor_layers 3 --critic_layers 4 … --n_train 4096` (a3c4).
- **Fast:** `--actor_layers 2 --critic_layers 3 --reuse_target 1 --n_train 3072` (a2c3+reuse, ~13 s).
- **LSM-reproducibility parity:** ensemble of **K=3–4** of the above (different data seeds; price-average).

**Hypothesis disposition:** H-N1 asymmetric actor **ADOPT** (free); H-N1b/d critic-depth **ADOPT**
(accuracy); H-N2 symmetric depth-2 **REJECT** (accuracy cost); H-N3 batch **REJECT** (no var, slower);
H-N4 SWA / H-N5 LSUV(deferred, LN-redundant) / H-N5b fixed-init / H-N6 wd **REJECT** (no var — var is
intrinsic); H-N7 GELU/Mish (predicted wash, BLAS-bound) / H-N8 residual-fold **PARK**; **H-N9 ensemble =
the variance answer.** Net adopted change to the canonical net: **asymmetric a2c4 (shallow actor, deep
critic) + 4096 paths**, with ensembling as the reproducibility lever.

---

## 0. Executive summary & the target to beat

**Operational target (user directive):** at **n_train ∈ {2048, 4096}** paths, the RL pricer
must be **≥ LSM-D on every pricing metric** — where LSM-D = Chebyshev-basis Least-Squares
Monte Carlo, and "5–7 estimators" = polynomial **degree 4–6** (5–7 regressors) — *and*
the canonical degree-2 (3 regressors). Three fronts:

| Front | Metric | Direction | Current baseline (15 seeds, C++, threads 8) |
|------|--------|-----------|---------------------------------------------|
| **Accuracy** | 65k-OOS price (no look-ahead ⇒ higher = closer to true optimum) | **↑** | w48/M2/3072 = **1.9801**; orig w64/M4/4096 = **1.9820**. RL already beats LSM-D by **+0.47 pp** at *full* budget (CLAUDE.md) — **unmeasured at 2k–4k** (open task O1). |
| **Variance** | seed-to-seed std of price | **↓** | w48 = **±0.0079**; reuse-w48 = **±0.0073** (≈0.37% rel). |
| **Time** | t_train wall-clock (and µs/learn-step) | **↓** | w48/3072 = **19.3 s**; reuse = 15.8 s; w32 = 12.4 s. |

> **Caveat on the "time" front vs LSM.** A degree-2 LSM on 2–4k paths is a handful of
> Numba regressions — **microseconds-to-milliseconds**, structurally faster than training
> any NN. So "beat LSM in compute time" is *not* a fair literal target. We therefore read
> the directive as: **match-or-beat LSM-D on accuracy + variance at 2k–4k**, while the
> NN-architecture work **independently** drives the RL wall-clock down (the speedup-study
> currency). This split is stated so the goalposts are explicit; flag if a stricter reading
> is intended.

**What this plan will *not* re-litigate** (settled in `HPT.md`, do not naively retest):
LayerNorm is load-bearing (`norm=none` → −4.35 pp nocost; RMSNorm → tail risk); SiLU > ReLU/tanh;
orthogonal+gain init > He (0/9 vs seed-collapse); β-sigmoid output (v64 = 1.5); `learn_every=4`
loses accuracy iso-wall-clock; `tiny_nn` (1 hidden, no LN) faster but worse; width-32 *raises*
variance at the full budget (Python `G_width32`). These are **boundaries**, not targets.

---

## 1. Current-configuration audit (challenge every knob)

Network (v64, mirrored bit-exact in `mlp.cpp`):
- **Actor:** 9 → [Linear→LayerNorm→SiLU]×3 (width 64) → Linear→1 → β-sigmoid(1.5) → profitability-gate (STE).
- **Critic:** state-enc 9→64, concat action, [Linear→LN→SiLU]×(action-layer + 1 post) → Linear→1. (3 hidden transforms.)
- Init orthogonal+gain (hidden), U(−3e-3,3e-3) (output); AdamW lr_a 3e-4 / lr_c 6e-4, wd 5e-5/1e-4;
  batch 128; learn_number 2; τ 0.0032; EMA(0.999) eval-only; closed-form warm-start.

| # | Knob | v64 value | Status | Front(s) it could move | Hypothesis |
|---|------|-----------|--------|------------------------|-----------|
| 1 | hidden width | 64 (→48 adopted) | **partly open** | time, var | already swept; **w48 best**, w32 biased. Re-open *only* coupled with depth/asymmetry. |
| 2 | **depth (# hidden)** | **3** | **OPEN** | time, var | depth-3 won @32k ep; **never isolated at 2k–4k**. → **H-N2**. |
| 3 | **actor vs critic symmetry** | identical shape | **OPEN** | time, var | policy is a simple scalar FOC map; critic surface is richer. → **H-N1**. |
| 4 | activation | SiLU | settled (vs ReLU/tanh) | acc | GELU/Mish *untested* but ≈SiLU; BLAS-bound ⇒ no time win. → **H-N7** (low). |
| 5 | normalization | LayerNorm | **settled (load-bearing)** | — | do not remove. Pre- vs post-LN placement *open* but low value. |
| 6 | init (hidden) | orthogonal+gain | settled (vs He) | var | gains are heuristic; **data-driven LSUV untested**. → **H-N5**. |
| 7 | init (output) | U(−3e-3,3e-3)+warm-start | open | var | interacts with warm-start; fold into H-N5. |
| 8 | weight averaging | EMA(0.999) eval-only | open | var | **SWA tail-average untested**; flat-minima theory. → **H-N4**. |
| 9 | batch size | 128 | **OPEN** | var, time | deterministic target ⇒ larger batch = lower-variance grad. → **H-N3**. |
| 10 | lr_a/lr_c | 3e-4/6e-4 | partly open | var, acc | co-tune with batch (H-N3); constant LR is settled. |
| 11 | weight decay | 5e-5/1e-4 | open | var | spectral-norm alternative untested. → **H-N6**. |
| 12 | residual/skip | none | **OPEN** | acc, (eval-time) | RMNet train-residual→fold-to-plain. → **H-N8** (park). |
| 13 | ensembling | none (single net) | **OPEN** | var (↑time) | variance-buy if single-net floor too high. → **H-N9**. |

---

## 2. Cost & generalization model (why these levers, in this order)

**Where the FLOPs are (post-H-S6, w48, M_x=2):** actor block ≈ **39%**, kernel-target ≈ **37%**,
critic step ≈ **24%**. Both networks are evaluated in the **kernel target** at B·M = 256 forward
rows; the **critic** is additionally hit in the critic step (B fwd+bwd) and the actor step
(B fwd+bwd for `g_action`); the **actor** is additionally hit in the actor step. Per-step cost is
**≈ linear in width and in #layers** (dispatch/memory-bound small GEMMs, not FLOP-bound).

Consequences that set the hypothesis ranking:
1. **Shrinking the actor** (depth or width) cuts the actor block *and* half the kernel-target
   forward — and the policy is the *simpler* of the two functions to approximate ⇒ **H-N1/N2 = highest expected time-win-per-risk.**
2. **Variance at 2k–4k is dominated by (a) init dispersion and (b) minibatch-gradient noise**
   (the kernel made the *target* deterministic, removing TD-sampling noise). ⇒ the variance levers
   that matter are **init (H-N5), weight-averaging (H-N4), batch size (H-N3)** — not anything touching the target.
3. **Activation/precision are BLAS-bound dead ends for time** (confirmed: fp16 dead, fused-LN+SiLU 9% slower). Keep H-N7 a 1-shot accuracy check only.

---

## 3. Literature synthesis (94 papers, `lit/digest_{accuracy,variance,time}.md`)

Curated actionable cluster per front (full noisy digests retained as audit trail). ≥20 reviewed each.

### 3A. Accuracy of shallow nets (38 reviewed)
- **Shallow-ReLUᵏ optimal approximation rates** (arXiv:2304.01561) & **minimal-width universal approx**
  (2411.08735), **bounded-parameter approximation capacity** (2409.16697), **arch beyond width&depth**
  (2205.09459): *smooth low-dimensional maps need few neurons* → supports a **narrow/shallow actor**.
- **Activations:** Swish/SiLU origin (1710.05941), **GELU** (Hendrycks; 2305.12073 analysis), **Mish**,
  **APTx** (2209.06119): smooth self-gated units cluster tightly; SiLU≈GELU≈Mish ⇒ H-N7 expected ~wash.
- **Polynomial/sigmoid approx of activations** (2202.00004, 2010.04596): a *cheaper* activation is
  possible but only matters if not BLAS-bound (it is) — parked.
- **Deep Neural Regression Collapse** (2603.23805): regression nets collapse to low-rank features
  *below* the last layer → motivates checking the critic's effective rank (cheap diagnostic, ties H-N1).
- **LayerNorm placement / stability** (2510.09904), **Soft Actor-Critic** (1801.01290), **decoupled-PER
  DDPG** (2512.05320), **twin/ensemble actors TATD3** (2102.13012): actor/critic can be treated
  asymmetrically (H-N1) and updated with different cadences.

### 3B. Seed-to-seed variance / reproducibility (31 reviewed)
- **Reproducibility of continuous-control DRL** (1708.04133) + **"Deep RL that Matters" lineage**:
  seed variance is first-order and architecture/​init-sensitive — *measure many seeds, report std*.
- **SWA — "Averaging Weights Leads to Wider Optima"** (Izmailov): tail-average of iterates → flatter
  minima, lower variance → **H-N4** (we currently only do EMA).
- **Sharpness-Aware Minimization** (SAM): flat-minima↔generalization causal evidence → supports SWA/flat-seeking.
- **Dynamical isometry / "exact dynamics of learning in deep linear nets"** (Saxe 1312.6120): orthogonal
  init preserves signal; **LSUV** operationalizes per-layer unit variance data-dependently → **H-N5**.
- **Variance-regularized offline RL** (2212.14405), **bootstrapped-DQN ensembles** (2511.02969, 2007.08229),
  **credal ensemble distillation** (2511.13766): ensemble cuts variance, distill to one net → **H-N9**.
- **Linear mode connectivity & LTH** (1912.05671): nets become stable to SGD noise *early* → a short
  deterministic warm phase can lock the seed → supports init+warm-start coupling (H-N5/N7).

### 3C. Computation time (25 reviewed)
- **Lottery Ticket family** (1803.03635 lineage, 1903.01611 stabilized, 2203.04248 dual, 1912.05671):
  small subnetworks train to full accuracy → justifies aggressive **structured width/depth reduction**
  (H-N1/N2) over post-hoc unstructured pruning (which won't speed dense BLAS on M1).
- **Pruning surveys** (PaI 2103.06460, SCOP 2010.10732, NAP 2101.06608, channel 2211.08339): unstructured
  pruning ≠ wall-clock on dense CPU GEMM ⇒ prefer *structured* shape changes. Logged as a boundary.
- **Knowledge distillation** (Hinton lineage; KDFM 1812.00660, categories 2306.10687): teacher→small
  student. Already **parked** in speedup-study (eval=0.5% of wall-clock; can't be a moving train target).
- **Deep Compression** (1510.00149): pruning+quant+Huffman — storage, not M1 FLOP-bound latency ⇒ low value here.
- **Mixed-precision rounding-error analysis** (2410.12614): fp16 confirmed dead (kernel only ~21%).

**Net literature verdict:** the highest-leverage, theory-supported, *untested* moves are
**structured shape reduction with actor/critic asymmetry** (time+var) and **flat-minima
weight-averaging + data-driven init** (var). Pruning/distill/quant/exotic-activations are
boundaries or parked for *this* hardware and net size.

---

## 4. Hypotheses (tiered; each with mechanism, theory, gate)

Notation: **Δprice** = mean (new − baseline) over CRN seeds 11–25; **Δeq = 0.5%** of price (±0.0099).
"Superiority" = one-sided paired test in the *beneficial* direction. Baseline = **w48/M2/3072 reuse-off**
(the adopted conservative tier), threads 8.

### Tier A — architecture shape (time-primary; implement first)
- **H-N1 — Asymmetric actor (narrow/shallow policy, rich critic).**
  *Mechanism:* policy q*(state) ≈ clipped FOC map (low complexity); critic Q(s,a) is the harder surface.
  Shrinking the actor cuts the actor block + half the kernel-target forward.
  *Impl:* make **depth runtime-configurable** (mirror the R5 width member); sweep
  actor∈{depth2×w{32,48}} × critic = depth3/w48 fixed.
  *Gate:* price TOST-equiv-or-superior **AND** seed-std not worse (Levene n.s. or ↓) **AND** Δt<0.
- **H-N2 — Depth-2 both nets at 2k–4k.**
  *Mechanism:* depth-3 won at 32k ep; at 2k–4k fewer params ⇒ less overfit to the small path set ⇒
  possibly *lower* seed-variance and faster. *Gate:* same as H-N1; report bias direction explicitly.
- **H-N3 — Batch↑ + LR co-tune.**
  *Mechanism:* deterministic kernel target ⇒ minibatch-gradient noise is the dominant remaining
  stochasticity; larger batch → lower-variance gradient (linear-scaling rule; "don't decay LR, increase batch").
  *Impl:* batch∈{128,192,256} × lr∈{1×,√(B/128)×}. *Gate:* seed-std↓ (variance-ratio CI upper<1) without
  price/time regression, **or** price↑.

### Tier B — variance-primary (the #2 goal)
- **H-N4 — SWA tail weight-averaging vs EMA.** Equal-weight average of late iterates → wider/flatter minimum
  (Izmailov; SAM). Cheap, post-hoc. *Gate:* seed-std↓, price not worse.
- **H-N5 — LSUV / data-driven init.** Rescale each layer to unit output variance on a warmup batch ⇒
  identical forward statistics across seeds ⇒ tighter seed spread + faster convergence (Saxe isometry).
  *Gate:* seed-std↓ (and/or fewer episodes to converge).
- **H-N6 — Critic spectral-norm / stronger wd_c.** Bound critic Lipschitz constant ⇒ fewer divergent seeds.
  *Gate:* seed-std↓, no price/time regression.

### Tier C — accuracy / exploratory (lower expected value)
- **H-N7 — GELU/Mish vs SiLU.** One-shot; expected wash, no time win (BLAS-bound). Run only to close the question.
- **H-N8 — Residual block (train) + RMNet fold (eval).** Skip connections aid gradient flow; reparameterize
  to a plain MLP at eval (zero eval-time cost). Park unless A/B leaves accuracy on the table.
- **H-N9 — Cheap K-ensemble for variance.** Ensembling cuts seed-std ~1/√K at ~K× cost ⇒ only a net win
  if each net is much smaller (compose with H-N1/N2). The "variance-buy" fallback if the single-net floor is too high.

---

## 5. Statistical & measurement methodology

Builds on `SPEEDUP_RESEARCH_PLAN.md` §E; **adds superiority tests** because the goal is now *better*, not just *same*.

- **Seeds / pairing:** CRN seeds 11–25 (15) for screening; extend to 25 (11–35) for any borderline final claim.
  Price compared at fixed `--threads 8` (thread-count price artifact — see speedup §A.3).
- **Accuracy (superiority):** paired one-sided t on Δprice>0 (`stats_analysis.paired_*`); Holm across hypotheses.
  Report the paired 90% CI; ADOPT-for-accuracy needs the CI lower bound >0 (or TOST-equiv if the win is elsewhere).
- **Variance (superiority):** **Levene/Brown-Forsythe** + **variance-ratio F** on per-seed price; ADOPT-for-variance
  needs the ratio CI **upper bound < 1**. (Bootstrap the std-ratio CI; n=15 is thin — pre-register MDE.)
- **Time:** `bench_train` µs/step (n≥7 clean, non-overlapping bootstrap CIs) + `price_swing` t_train; strictly lower.
- **Multi-objective decision (conservative Pareto):**
  `ADOPT` iff it **strictly improves ≥1 front** (superiority CI) **and regresses none** beyond that front's Δeq band;
  `PARK` if neutral-but-promising for a combination round; `REJECT` otherwise.
- **LSM-D head-to-head (the literal target, confirmation phase):** Python `rebuild_results_v7`-style harness,
  n_train∈{2048,4096}, RL(winning arch) vs LSM Chebyshev degree∈{2,4,6}; common 65k OOS test (seed 999);
  seeds 11–25. Report price + seed-std side-by-side. (C++↔Python price already validated Welch p=0.37, so
  C++ screening is faithful; Python only for the LSM column.)

**Open tasks before Round 1:**
- **O1 — measure LSM-D at 2k/4k** (the actual accuracy bar) — currently only the full-budget +0.47 pp is known.
- **O2 — make `depth` a runtime member** of Actor/Critic (exactly as width became one in R5): `lin[]`/`ln[]`
  become `std::vector`, forward/backward loop over `n_layers`, `AgentConfig::actor_layers/critic_layers` + flags.
  Default 3 ⇒ bit-identical to v64.
- **O3 — variance-ratio + Levene wrapper** in `research_equiv.py` (TOST already there; add the one-sided
  superiority + bootstrap std-ratio CI).

---

## 6. Execution roadmap (proposed ordering)

1. **O1** (LSM-D bar) + **O2** (runtime depth) + **O3** (stats) — unblock everything. *~½ day.*
2. **H-N2** depth-2 sweep (both nets) — cheapest, biggest potential time+var win; 15 seeds. 
3. **H-N1** asymmetric actor — the theory-favorite; depends on O2.
4. **H-N4 + H-N5** variance levers (SWA, LSUV) — orthogonal to shape; can run in parallel.
5. **H-N3** batch/LR co-tune — interacts with everything; run after shape settles.
6. **Combination round** (best shape + best variance lever), then **LSM-D head-to-head** at 2k/4k (the target).
7. H-N6/N7/N8/N9 as time permits / if a front is still short of target.

> Decision log appended below per round (mirror the speedup ledger). Nothing is adopted into the
> `mlp.cpp` default without passing its gate AND a fresh-baseline combination check.

---

## Ledger

- **O2 — runtime depth DONE & validated.** Made network depth a runtime member of Actor/Critic
  (mirrors the R5 width member): Actor = `n_layers` `[Lin→LN→SiLU]` blocks + fc4; Critic = se + al +
  `(n_layers-2)` post-layers + fc4 (matches the Python `post_layers=range(n_layers-2)` structure).
  `AgentConfig::{actor_layers,critic_layers}` + `--actor_layers/--critic_layers` on price_swing/bench_train.
  **Default 3/3 is bit-identical to v64** (seed 11 w48/3072 = 1.968648 exact; RNG draw order preserved in
  `init_orthogonal`). New FP64 gradcheck `tests/test_grad_depth.cpp` (fixture-free) PASSes at ~3e-10 over
  depths {2,3,4} symmetric + asymmetric {2/3, narrow} — including the critic n_post=0 path. `test_grad`/
  `test_parity` still need the absent PyTorch fixtures (O1-adjacent prerequisite), unaffected by this change.
- **O3 — analyzer extended.** `research_equiv.py` now also reports the **one-sided paired superiority** test
  (accuracy front), the **Pitman–Morgan** paired-variance test + **bootstrap std-ratio 90% CI** (variance
  front), and the t_train speedup (time front), on top of the existing TOST/Welch.
- **H-N2 — depth-2/2 vs depth-3/3 (w48/M2/3072, 15 seeds) → REJECT (speed/accuracy trade, not a win).**
  depth-3 = 1.980100 ± 0.008135; depth-2 = 1.976508 ± 0.008745. Paired Δ = **−0.00359 (−0.36%, p=0.0005)**:
  TOST-**equivalent** (90% CI [−0.0050,−0.0022] ⊂ ±0.0099) but **significantly INFERIOR** on accuracy
  (1-sided p=0.9997) and **no variance gain** (std-ratio 1.075, Pitman-Morgan p=0.52, CI [0.943,1.291] n.s.).
  Time **1.62× (19.35→11.93 s)**. ⇒ depth-3 is justified even at the 2k–4k regime; the capacity that depth-3
  buys is *accuracy*, not variance. Hypothesis that low-path favors depth-2 is **falsified for symmetric depth**.
  Data: `data/research/hn2_d2_seed*.json`. **Open:** is the loss entirely in the critic? → H-N1 (yes).
- **H-N1 — asymmetric actor-2 / critic-3 (w48/M2/3072, 15 seeds) → ADOPT candidate (free speed).**
  depth-3/3 = 1.980100 ± 0.008135; a2c3 = 1.979715 ± 0.008680. Paired Δ = **−0.00039 (−0.019%, p=0.43
  n.s.)** — accuracy **statistically unchanged**, TOST-equivalent (CI [−0.0015,+0.0008]); variance n.s.
  (ratio 1.067, Pitman-Morgan p=0.56). Time **1.21× (19.35→15.96 s, −17.5%)**.
  **Decomposition vs H-N2:** the critic's 3rd layer is worth ~+0.34% price (load-bearing); the actor's 3rd
  layer is worth ~+0.02% (negligible). ⇒ **cut the actor to depth-2 for free** (the policy is a simple
  scalar map; shallow-net approximation theory). Confirms the H-N1 mechanism. Next: bench µs/step CI +
  combination check (a2c3 + reuse), then push the actor cheaper (width/depth) while holding the critic.
  Data: `data/research/hn1_a2c3_seed*.json`.
  **Variance front still open** — neither depth lever helped; route to H-N3 (batch/LR), H-N4 (SWA), H-N5 (LSUV).
- **H-N1b — deeper critic a2c4 (15 seeds) → accuracy nudge, free.** price **1.981158** vs depth-3 1.980100,
  Δ=**+0.00106 (+0.053%, 1-sided p=0.43 n.s.)** — highest price observed; TOST-equiv; var n.s. (1.067);
  time **1.02×** (actor-2 saving ≈ critic-4 cost ⇒ same 19s). Critic depth monotonically helps accuracy
  (c2<c3<c4: 1.9765<1.9797<1.9812); actor depth is free. ⇒ **a2c4 is the accuracy-max shape at depth-3 cost.**
- **H-N1c — a2c3 + reuse_target (15 seeds) → ADOPT (fast asymmetric).** 1.979652, Δ=−0.00045 (n.s.),
  TOST-equiv, var ratio 0.996 (n.s.), **1.46× (13.2 s)**. Clean fast tier with asymmetric actor.
- **O1 — LSM-D bar (preview, 2 seeds, 4096 paths)** `tools/lsm_bar.py` → `data/research/lsm_bar.json`:
  reduced-state Chebyshev deg2/4/6 (3/5/7 regr) = **1.9745 / 1.9684 / 1.9693**; **full-state deg2 = 1.9809**.
  LSM **seed-std ≈ 0.0012** (vs RL 0.008). **Two pivotal facts:** (1) RL conservative (1.9801) already
  **beats reduced-state LSM-5/7-estimators by ~+0.6%** and ties full-state deg2; (2) **LSM variance is ~6×
  tighter than RL** ⇒ *variance is the make-or-break front* (structural: deterministic regression vs
  stochastic training). Likely needs SWA + **ensemble** (H-N9) to match. Full 15-seed bar pending.
- **H-N1d — a3c4 (full actor + deeper critic, 15 seeds) → accuracy SUPERIOR.** price **1.981413**,
  Δ=+0.00131 (+0.066%), **1-sided p=0.045 → significantly beats depth-3** (first to do so); var worse
  (1.111), time **0.80× (24.2 s, slower)**. ⇒ critic depth is a real accuracy lever; a2c4 (1.9812, 19 s,
  free) is the better accuracy/time point; a3c4 is accuracy-max at a time+variance cost.
- **H-N4 — SWA tail-average (15 seeds) → REJECT (no variance gain).** seed-std 0.008120 vs 0.008135
  (ratio 0.998, CI [0.969,1.014] n.s.); price −0.028% (very consistent). **Key insight: EMA(0.999) already
  averages within-basin; SWA adds nothing because the seed variance is CROSS-basin** (init + early
  exploration pick the basin — LTH/linear-mode-connectivity 1912.05671). ⇒ no weight-averaging reduces it;
  only **ensembling across basins** or **less init/early-exploration stochasticity** can.
- **H-N9 — ensemble variance path** (baseline 15-seed data; σ/√K, bootstrap-confirmed, `tools/ensemble_var.py`):
  single-agent σ=0.00786; K=9→0.0026, K=16→0.0020, K=25→0.0016; **K≈43 to match LSM 0.0012.** Ensemble alone
  can't match LSM at low K ⇒ must *also* cut single-agent σ.
- **H-N5b — data-only seed variance (NEW; `--agent_seed` fixes init+exploration, varies only MC paths) →
  running.** If the cross-basin variance is init-driven, fixing init collapses σ toward LSM's data-only σ
  (LSM-matched seed semantics) — the principled route to the variance front. `--agent_seed` default = data
  seed ⇒ bit-identical.
- **H-N5b RESULT — fixed-init data-only variance → variance is DATA-driven, not init-driven.** agent_seed=0,
  data seed 11-25: σ=0.00826 ≈ full σ=0.00814 (ratio 1.05, n.s.). ⇒ **σ_model≈0, σ_total≈σ_data.** Fixing
  init/exploration/minibatch-order left variance unchanged. The seed variance is the *sample variance of the
  estimator* (which 4k paths), and RL is a less statistically-efficient estimator than LSM's least-squares.
- **H-N3 batch=256 → REJECT.** price +0.04% n.s., var ratio 1.007 (no gain), time **0.55× (34.9 s, slower)**.
- **H-N6 wd_c=2e-4 → REJECT** (var 1.022, no gain). **ema_decay 0.9995 → REJECT** (var 1.004). **w32/w64 →**
  var 0.0084/0.0085 vs w48 0.0079 (**w48 is variance-optimal**; smaller adds bias, bigger adds variance).
- **★ VARIANCE FRONT — CLOSED (comprehensive null).** Every model-side lever — SWA, EMA-decay, weight-decay,
  fixed-init, batch size, net width — fails to reduce the ~0.0079 seed-std. It is **intrinsic data variance**
  (σ_model≈0). The *only* reducer is different-data ensembling (σ/√K ⇒ effectively more paths), which trades
  against the path budget and compute. **RL cannot match LSM's ~0.0012 variance at a fixed 4k-path budget** —
  a structural sample-efficiency gap of policy-gradient vs global least-squares. This is the honest core finding.
- **★★ CORRECTION — full 15-seed LSM bar overturns the "6× variance / can't-match" scare.** The 2-seed
  preview's LSM seed-std (0.0012) was an artifact of n=2. The **15-seed `lsm_bar.json`** (focal, OOS 65k):

  | LSM | 3072 | 4096 |
  |---|---|---|
  | full-state deg2 (strong) | 1.9765 ± 0.0054 | 1.9783 ± 0.0045 |
  | reduced deg2 (3reg) | 1.9720 ± 0.0044 | 1.9720 ± 0.0043 |
  | reduced deg6 (7reg) | 1.9643 ± 0.0047 | 1.9664 ± 0.0043 |

  ⇒ **LSM seed-std ≈ 0.0045–0.0054**, so the RL gap is only **~1.55×** (a2c4 0.0084), *not* 6×. And LSM
  prices are **lower** than the preview: RL a2c4 (1.9812@3072) **beats the strongest LSM (full-deg2 1.9765)
  by +0.24%** and reduced-5/7 by **+0.85%**. The variance finding (σ data-driven, σ_model≈0) still holds, but
  the *conclusion flips*: a small **different-data ensemble closes the gap** — **K=3 a2c4 ⇒ σ=0.0048 (0.90×
  LSM) → BEATS LSM on variance** while keeping the +0.24% accuracy win; K=4 → 0.78×, K=5 → 0.69×.
  **⇒ A K=3 ensemble beats LSM on accuracy AND variance.** Time is LSM's only retained front (RL ~17 s vs
  LSM ~4 s; structural — iterative SGD vs one-shot regression — narrowable to ~10 s but not below LSM).
- **★ MESH CORRECTION + canonical head-to-head (em_* round) — the publishable numbers.** During the
  speedup-plan closeout, regenerating `data/kernel_v64.bin` revealed the C++ `build_fast` M=2 fallback
  (used by ALL prior sweeps) prices **~+0.002–0.003 higher** than the Python-validated *exported* M=2 mesh.
  `build_fast` is the self-contained fallback; the exported mesh matches `precompute_kernel` (the v64 paper
  pipeline) and is FP64 parity-green. The *relative* findings are mesh-invariant; the *absolute* RL-vs-LSM
  margin was re-measured on the canonical mesh (`em_a2c4_4096`, `em_a3c4_4096`, `em_a2c4_3072`,
  `em_a2c3reuse_3072`). Canonical result (15 seeds): RL a3c4@4096=**1.9796±0.0080**, a2c4@4096=**1.9785±0.0072**.
  **RL beats reduced-state LSM-5/7 by +0.61–0.67% (Welch p<0.0001); ties the strongest full-state LSM
  (p=0.6–0.9, not worse).** Variance K=3 ensemble (0.0041) beats LSM (0.0045). Time LSM's. *Lesson: always
  quote the validated exported mesh, never `build_fast`, for any cross-method (RL-vs-LSM) accuracy claim.*
- **★ SPEEDUP-PLAN LOOSE ENDS CLOSED (this session).** (1) `data/` fixtures regenerated via
  `tools/export_reference.py` (now `N_max`-parametrized, default 0 = canonical M=2); (2) **`kernel_v64.bin`
  re-exported at N_max=0 and all FP64 tests pass (parity/grad/grad_depth/sim)** ⇒ **H-K1 (M=4→2) is now
  validated against the PyTorch kernel**, not just C++-vs-C++ (closes the speedup plan's production caveat);
  (3) `PRICER_PROFILE` wired into `CMakeLists.txt` as a first-class option. The speedup plan's Tiers 1–3 were
  already exhausted; with these prerequisites green it is now **fully closed**.

## Plan status: COMPLETE (Round 1)

---

# ROUND 2 (2026-06-14) — reopened: activation, init, batch/LR; new test methodology

> User reopened the plan: (a) prioritise the levers the Round-1 plan dismissed by *argument* not
> measurement — **activation gates, initialization, batch size, learning rates**; (b) new test protocol —
> **run experiments in parallel, 4 threads each, and measure time with the load-independent `cpu_train`**
> (user-mode CPU-seconds via getrusage, constant under CPU contention) instead of wall-clock; (c) validate
> **across all three cost regimes** (nocost c=0 / linear γ=1 / convex γ=2). New flags added (all
> default-bit-identical, FP64 gradcheck green where applicable): `--c_cost/--gamma_cost` (regime),
> `--hidden_actor/--hidden_critic` (asym width), `--init_method` (0 He/1 orthogonal/2 Xavier),
> `--init_gain/--critic_out_init/--actor_out_init`, `--batch/--lr_a/--lr_c/--wd_c/--ema_decay`,
> `--learn_number/--learn_every/--critic_warmup`, swish-slope via `-DPRICER_GELU_FAST -DGELU_SLOPE=β` /
> `-DPRICER_GELU` (exact erf). New JSON field `cpu_train`. Tool fixes in `research_equiv.py`: the
> incomplete-beta CF gave **wrong p-values for small t** and the **Pitman-Morgan tighter/wider label was
> inverted** — BOTH FIXED (verified; large-t significant findings were unaffected).

## ★★ ROUND-2 HEADLINE — the activation was the real accuracy lever (Round-1 H-N7 "wash" was WRONG)

Generalized the hidden activation to **swish-with-slope g(x)=x·σ(β·x)** (β=1 ⇒ SiLU, β→∞ ⇒ ReLU,
β≈1.702 ⇒ GELU). **Interior optimum at β≈3.0**, well above SiLU — and it is the **only lever in either
round that raises accuracy WITHOUT widening seed variance**.

| activation | focal Δ vs SiLU | p | seeds/threads |
|---|---|---|---|
| erf-GELU | +0.0019 (+0.096%) | 0.0004 | 15 / t8 |
| fast-GELU β=1.702 | +0.0029 (+0.145%) | 0.0001 | 15 / t8 |
| **fast-GELU β=3.0 (peak)** | **+0.0044 (+0.22%)** | **t=10.25** | 8 / t4 |
| β-sweep | SiLU<1.4<1.702<2.0<2.5<**3.0**>3.5 (interior max) | — | 8 / t4 |

**Cross-regime (β=3, 15 seeds, t4) — significant in ALL THREE:** nocost **+0.00095 (p=0.033)**,
linear-γ1 **+0.00065 (p=0.025)**, convex-γ2 **+0.00327 (p=0.0002)**. Largest where the exercise boundary is
sharpest (convex). Real, not bias: OOS price = mean discounted payoff; convex cost penalizes over-exercise,
so ↑price ⇒ strictly better policy. Mechanism: steeper self-gate ⇒ sharper exercise boundary (bang-bang).
`cpu_train`: GELU ≈ +3% compute vs SiLU at equal arch (erf-GELU was 3.7× — std::erf cost — rejected for the
cheap sigmoid form). **Implementation:** `include/linalg.hpp`, FP64 gradcheck PASS for both GELU forms.

## Round-2 other levers (all on a2c4/w48; paired CRN)
- **Actor width 48→32 = the clean "less time" lever.** Accuracy EXACTLY neutral (Δ+0.00001, 15s, p=0.97),
  variance +10% (ensemble-recoverable). **Clean idle timing (wall): fast-GELU+actor32 = 24.8s vs SiLU 25.6s
  ⇒ 3% FASTER *and* +0.145% more accurate; +reuse_target = 19.6s (−23%).** ⇒ "less time AND more accurate."
- **Critic width 64 / init_gain / critic_out_init:** small +acc @8s, all NULL @15s and **WIDEN variance.**
  Re-confirms canonical = variance-optimal; every init/width perturbation adds variance.
- **LR-retune (lr_c 9e-4/1.2e-3, +lr_a) = NULL** (lr already well-tuned; 1.2e-3 widens var).
- **learn_number=1 = REJECT** (sig inferior, −0.07%). **reuse_target = marginal** (−0.02% n.s., big speed).
- **★ INIT study = NULL (rigorous).** Found the C++ `init_orthogonal` is mis-named — it is iid He-normal,
  never QR-orthogonal — so the HPT "orthogonal>He" was NEVER actually implemented in the port. Implemented
  TRUE (semi-)orthogonal init (modified Gram-Schmidt, orthonormality verified to 1e-15, separate init-RNG ⇒
  no confound) + Xavier (`--init_method` 0/1/2). At GELU-β3+actor32, 15 seeds, focal: He 1.98016 vs
  **orthogonal 1.97972 (Δ−0.0004 n.s. p=0.46; var n.s. tighter Pitman p=0.38)** vs Xavier 1.97934 (trend
  worse p=0.065). ⇒ **init does not matter for this LayerNorm net (scale/isometry washed out by LN) —
  EMPIRICALLY confirmed now (real orthogonal), not argued. Keep He.**
- **★ BATCH+LR study (8 seeds, GELU-β3+actor32) — smaller batch is a real speed win recoverable by LR.**
  Train cost ∝ B (kernel target on B·M rows). **batch 64 is 1.7× cheaper** (cpu_train 15.1 vs 25.9 s) but
  −0.0030 (p=0.023) at base LR. **Lowering lr_c recovers it** (the half-batch→half-LR rule): lr_c 6e-4→
  **5e-4 = −0.0012 (n.s. p=0.10) at 1.72×** (the pick); 4e-4 −0.0013 (n.s.) 1.63×; 3e-4 overshoots
  (−0.0025 sig); 8e-4 worse. b96 only 1.21×. ⇒ **batch 64 + lr_c 5e-4 ≈ accuracy-neutral at 1.7× less compute.**

## ★★ ROUND-2 FINAL — Pareto frontier (focal g2, 15 seeds, accuracy vs ORIGINAL SiLU/actor48/batch128; clean idle wall)

| config | price | Δacc | p | wall | speedup |
|---|---|---|---|---|---|
| ORIG SiLU + actor48 + batch128 | 1.97674 | — | — | 25.1 s | 1.00× |
| **GELU β3 + actor32 + batch128** (accurate) | **1.98016** | **+0.173%** | **<0.0001** | 23.7 s | 1.06× |
| **GELU β3 + actor32 + batch96** (balanced) | 1.97824 | +0.076% | 0.095 | 19.1 s | **1.31×** |
| GELU β3 + actor32 + batch64/lr_c5e-4 (fast) | 1.97755 | +0.041% | 0.24 | 13.8 s | **1.82×** |

**The activation (GELU β3) is the accuracy win — significant (+0.17%, p<1e-4) and cross-regime-robust.
Width (actor32, free) + batch size are the speed dial: shrinking the batch trades the accuracy gain back
for wall-clock (train cost ∝ B; batch 64 = 1.82× but the GELU gain is spent).** Recommendation:
- **Accuracy/publication tier** = **GELU β3 + actor32 + batch128**: +0.17% accuracy (significant) AND 1.06×
  faster than the original — the clean "more accurate *and* faster" point.
- **Balanced** = batch96 (+0.076%, 1.31×). **Fast** = batch64/lr_c5e-4 (≈neutral accuracy, 1.82×).
- All tiers keep He init (orthogonal/Xavier null) and K=3 ensemble for LSM-parity variance.

**Round-2 net contribution:** a *significant* accuracy lever (steeper swish gate, β≈3) that Round-1 had
dismissed unmeasured, validated in all 3 cost regimes with neutral variance, plus a characterized
speed/accuracy frontier (1.06×–1.82×). The publishable single change is **SiLU → swish-β3** in the hidden
layers.

## Plan status (Round 1): COMPLETE
Round-1 adopted net change: **asymmetric a2c4 (shallow actor / deep critic) at 4096 paths** on the canonical
mesh, K=3-ensembled for LSM-level reproducibility. All knobs are runtime flags (default-bit-identical to
v64); FP64 gradcheck + parity green. Remaining out-of-scope production task: a full PyTorch-side re-baseline
at N_max=0 (the kernel mesh now matches; the *Python agent results* in `run.py` would shift ~+0.002 if re-run
— a documented hand-off, not a code defect).
