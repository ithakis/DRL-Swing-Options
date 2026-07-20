# Alternative Numerical Methods for Pricing HHK Swing Options under Convex Costs

**Purpose.** Decide whether to implement a *third* pricer to benchmark against the paper's RL (D4PG)
and LSM‑D. This survey ranks every viable numerical family by accuracy and by the compute needed to
*match* LSM‑D / RL accuracy on **the exact same contract and dynamics**: the HHK two‑factor
OU‑with‑jumps spot, discrete swing rights with local/global volume constraints and refraction, and a
**convex exercise cost** `c·qᵞ (γ>1)`.

Evidence base: 33 targeted `paper-search` queries → **548 unique papers** deduplicated
(`aggregated_papers.json`), covering LSM, PDE/PIDE, trees/lattices, quantization, Fourier/COS/SWIFT,
duality/upper bounds, deep learning/BSDE/RL, stochastic mesh, sparse grids, tensor/Chebyshev, MLMC,
and stochastic programming.

---

## 0. The one fact that organizes the whole ranking: convex cost kills bang‑bang

The classical swing literature (Carmona–Touzi *Optimal multiple stopping and valuation of swing
options*; Bardou–Bouthemy–Pagès; Bender; Meinshausen–Hambly duals) rests on a structural result: when
the per‑right payoff is **linear in the exercised quantity**, the optimal exercise is **bang‑bang**
— `q* ∈ {q_min, q_max}` — so swing collapses to a *multiple optimal stopping* problem (a stack of
coupled American options). Almost every "swing‑native" classical method exploits this.

**Your convex cost `c·qᵞ`, γ>1, destroys bang‑bang.** The per‑step objective
`q(S−K) − c·qᵞ` is strictly concave in `q`, so the optimum is **interior**:
`q*(S) = clip( ((S−K)₊/(cγ))^{1/(γ−1)}, q_min, q_max )` (subject to the remaining global budget). The
problem is therefore a genuine **continuous‑control finite‑horizon stochastic dynamic program**, not a
multiple‑stopping problem. This single fact splits the candidate methods into two classes:

* **Class A — native continuous control** (handle convex cost *for free*, by a pointwise 1‑D convex
  solve inside the Bellman step): grid/kernel backward DP, PIDE‑HJB, quantization, Chebyshev/tensor
  DP, control‑randomization MC, info‑relaxation dual, and (already done) RL & LSM‑D.
* **Class B — built on bang‑bang / discrete stopping** (need real surgery to admit interior `q`, and
  some lose their headline guarantees): multiple‑stopping duals, marginal‑value/forest‑of‑trees,
  deep optimal stopping, multistage LP. Penalize these in the ranking.

The second organizing fact: **your state space is small.** The Markov state is
`(X_t, Y_t, Q_t)` — two stochastic factors (OU diffusion `X`, mean‑reverting jump `Y`) plus the
endogenous cumulative‑volume `Q`, with discrete right/refraction counters. This is **2+1 dimensions**,
not the 50‑D basket regime. Deterministic grid methods do **not** hit the curse of dimensionality here
— which is decisive for the ranking.

Third fact — **you already own the hard part.** `src/transition_kernel.py` is an analytical HHK
transition kernel on a quadrature mesh, and `src/greeks.py::_roll_from` does closed‑form OU
propagation. A grid/kernel backward DP is therefore *mostly already implemented inside your repo*.

---

## 1. Adaptability check (quant reasoning) — does each method survive HHK + convex cost + swing?

For each method I confirm the three load‑bearing requirements before it earns a rank.

| Requirement | What it demands of a method |
|---|---|
| **HHK dynamics** | Must handle a 2‑factor *affine* process: Gaussian OU `X` **plus** a mean‑reverting compound‑Poisson (exp‑jump) `Y`. The joint conditional characteristic function is known in closed form ⇒ transform & kernel methods apply; the jump term ⇒ a *non‑local* (integro‑) operator for PDE/tree methods. |
| **Convex cost** | Must support an **interior, continuous** `q*` per step (concave 1‑D inner problem), not just on/off exercise. |
| **Swing** | Must carry the **volume state `Q`** (global `Q_min/Q_max`), per‑step `q_min/q_max`, refraction, discrete dates, and the budget coupling across dates. |

---

## 2. Ranked methods, with a suitability paragraph each

Ranking criterion = **suitability as an independent, trustworthy benchmark for *this* contract**:
accuracy ceiling × ease/safety of adapting to HHK+convex+swing × marginal scientific value beyond
what LSM‑D and RL already give you. (Incumbents LSM‑D and RL are listed for calibration, not ranked
as "new".)

### #1 — Backward Dynamic Programming on a grid with the analytical HHK kernel  *(Class A)*
**Verdict: implement this one.** Discretize `(X,Y,Q)` on a tensor grid; backward‑induct
`V_t(x,y,Q) = max_{q} [ q(S−K) − c·qᵞ + DF·E_{kernel}[ V_{t+1}(X',Y',Q−q) | x,y ] ]`, where the
conditional expectation is the **analytical HHK kernel you already have** and the inner `max` over `q`
is the closed‑form FOC above (one 1‑D convex solve per node, clipped to the residual budget). It is the
*natural ground truth*: deterministic (zero Monte‑Carlo noise), low‑dimensional (2+1), and convex cost
is handled exactly. Boogert–de Jong and Warin use exactly this template for gas storage; your repo's
`transition_kernel.py` + `_roll_from` mean ~70% of the code exists. Accuracy is limited only by grid
resolution and value interpolation (Richardson‑extrapolable), so it pins down the **true price** that
both LSM‑D (low‑biased) and RL can be measured against — which is precisely the missing third leg
today. Highest accuracy‑per‑effort by a wide margin.

### #2 — Information‑Relaxation / Martingale **Dual** upper bound  *(Class A, as a stochastic‑control dual)*
**Verdict: the highest *scientific* value‑add — implement alongside #1 if you want a referee‑proof
result.** This is not a standalone pricer; it produces a rigorous **upper bound** `V̄` to complement
the **lower bounds** that LSM‑D and RL already give. Rogers / Andersen–Broadie / Haugh–Kogan, and for
the *multiple‑exercise* case Meinshausen–Hambly, Bender, Schoenmakers ("a pure martingale dual for
multiple stopping", "robust multiple stopping — a duality approach"). **Crucial adaptation:** the *pure
multiple‑stopping* dual assumes bang‑bang, so for convex cost you must use the **information‑relaxation
dual for stochastic control** (Brown–Smith–Sun / Rogers' controlled form): penalize a candidate policy
(your trained RL actor or the #1 DP value) with a martingale built from the kernel, then maximize the
inner deterministic control problem pathwise. The payoff: you can report `V_lower ≤ V_true ≤ V̄` and
**certify the optimality gap of RL** (e.g. "RL within 0.4% of the dual bound"). For a paper under
review this is the single most persuasive thing you can add. Moderate–high effort; reuses the kernel.

### #3 — Optimal Quantization of the HHK process  *(Class A)*
**Verdict: strong, swing‑native, but more build than #1.** Bardou–Bouthemy–Pagès, *Optimal
quantization for the pricing of swing options*, is **literally this problem** (swing, volume state).
Replace the continuous `(X_t,Y_t)` at each date by an optimal `N`‑point Voronoi quantizer with its
companion transition‑probability matrix; backward‑induct on the finite quantization tree with the same
pointwise convex `q`‑solve. Comes with *a‑priori* error bounds (`O(N^{-1/d})`). HHK wrinkle: building
good quantizers for the **jump** factor `Y` (heavy/asymmetric) is more work than for the Gaussian `X`
(CLVQ/Lloyd on simulated `Y`). Accuracy is excellent and it is purpose‑built for swing — but given #1
reuses your kernel and reaches comparable accuracy, quantization is the strong *runner‑up* reference
rather than the first build.

### #4 — PIDE / Finite‑Difference HJB (variational inequality with control)  *(Class A)*
**Verdict: the most "classical‑finance‑referee‑friendly" alternative; heavy to build.**
Calvo‑Garrido, Vázquez et al. ("Numerical methods for solving PIDEs arising in swing option pricing
under a two‑factor mean‑reverting model"; "Pricing swing options ... two stochastic factors using a
PDE") solve essentially your model. The value solves a 2‑D **PIDE** (OU 2nd‑order operator in `X`
+ **non‑local jump integral** in `Y`) on a stack of `Q`‑layers coupled through the control. Convex cost
enters as a smooth `sup_q[ q(S−K) − c·qᵞ + 𝓛V ]` Hamiltonian (no obstacle/penalty gymnastics needed —
the convexity is a gift here). Spectral‑grade accuracy. Costs: jump integral (dense/quadrature or
FFT), ADI/operator‑splitting in 2‑D, boundary conditions, `Q`‑coupling, and stability tuning — weeks
of careful numerical PDE work, easy to get subtly wrong. Best when you specifically want a PDE‑side
cross‑check; otherwise #1 gives you the same accuracy for far less risk.

### #5 — Dynamic Chebyshev / Low‑rank Tensor backward DP  *(Class A)*
**Verdict: an accelerated, higher‑accuracy variant of #1.** Glau et al. (*Chebyshev interpolation for
parametric option pricing*; dynamic Chebyshev) interpolate the continuation value on a Chebyshev tensor
grid in `(X,Y,Q)` and propagate expectations with the kernel; spectral convergence in smooth regions.
Hierarchical‑Tucker / TT formats (*Pricing high‑dimensional Bermudan options with hierarchical tensor
formats*) extend it if you later add factors. Convex cost: identical pointwise `q`‑solve. It is
genuinely excellent but it is *the same idea as #1 with a fancier interpolant* — implement only if #1's
grid accuracy proves insufficient (unlikely at 2+1 dims).

### #6 — Control‑Randomization / Regress‑Later Monte Carlo  *(Class A)*
**Verdict: the cleanest MC upgrade of your incumbent, but overlaps LSM‑D.** Kharroubi–Langrené–Pham's
control randomization and Glasserman–Yu regress‑later are MC‑regression schemes designed for
**continuous‑control** DP — so unlike vanilla LSM‑D's discretized exercise, they regress the value as a
function of a *randomized* continuous `q` and recover `q*` by optimizing the fitted continuation. HHK by
simulation, convex cost natively, volume state as a regressor. Accuracy is LSM‑class (low‑biased,
regression‑basis‑limited). Because it lives in the same MC‑regression family as your current LSM‑D, its
marginal value as an "independent" benchmark is limited — more a refinement than a new axis.

### #7 — Deep BSDE / Deep Galerkin (DGM) PDE solvers  *(Class A, NN)*
**Verdict: a mesh‑free PDE solver, but it's an NN method like your RL — weak independence.** DGM
(Sirignano–Spiliopoulos) and deep‑BSDE (Han–E) solve the same HJB‑PIDE as #4 with a neural ansatz,
sidestepping the 2‑D mesh. They handle the jump term and continuous control. But as a *benchmark for an
NN method (RL)* they share NN failure modes (optimization noise, seed variance, no exactness
guarantee) — so they don't give the "deterministic ground truth" you actually want. Lower priority for
*this* purpose.

### #8 — Deep Optimal Stopping / Neural multiple‑stopping  *(Class B → adapted)*
**Verdict: skip for this contract.** Becker–Cheridito–Jentzen *Deep optimal stopping* and
randomized‑NN optimal stopping, and "deep combinatorial optimisation ... application to swing options",
are built around **stopping decisions** (bang‑bang). To admit interior `q` they must be rebuilt into a
continuous‑control policy network — at which point you have re‑derived your own D4PG. Redundant with the
incumbent RL.

### #9 — Fourier‑COS backward DP (Fang–Oosterlee)  *(Class A‑ish; awkward under convex cost)*
**Verdict: elegant, but the convex cost fights the method.** COS does Bermudan backward induction via
the conditional **characteristic function** (HHK has one in closed form) and reaches spectral accuracy
cheaply — *in low state dimension*. Two frictions: (i) you need **2‑D COS** for `(X,Y)` **plus** the
`Q`‑layers ⇒ 3 effective dimensions, near the practical ceiling of cosine methods; (ii) COS's speed
comes from *analytic payoff cosine coefficients*, which assume the clean early‑exercise `max`; the
**continuous `q`‑optimization recomputes the "payoff" per node**, eroding the analytic advantage.
Workable but fiddly; #1 dominates it here.

### #10 — SWIFT / Shannon‑wavelet DP  *(same niche as #9)*
**Verdict: a more robust cousin of COS, same limitations.** SWIFT (Ortiz‑Gracia–Oosterlee) fixes some
COS truncation/oscillation issues and inherits the same characteristic‑function backward DP — and the
same 2‑factor‑plus‑volume dimensional strain and convex‑cost friction. No reason to prefer it over #1
for this problem.

### #11 — Trinomial / jump‑augmented Lattice; Forest of Stochastic Trees  *(Class B)*
**Verdict: the jumps break the method's main virtue.** A recombining trinomial tree is clean for the
OU factor `X`, but HHK's **jump factor `Y` breaks recombination** (you need a jump‑augmented or 2‑D
lattice that grows fast), and the volume `Q` adds another layer. "Forest of Stochastic Trees" values
multiple‑exercise but is exponential. Convex cost itself is fine (node‑wise `q`‑solve). The
analytical kernel makes the lattice's whole reason‑for‑being (approximating the transition) obsolete —
use #1 instead.

### #12 — Multistage Stochastic Programming / LP (+ swarm/heuristics)  *(Class B for cost, but scenario‑tree limited)*
**Verdict: convex cost is *ideal* for it, but HHK scenario trees explode.** "Valuation of electricity
swing options by multistage stochastic programming" and "From linear programming to particle swarm":
discretize uncertainty into a scenario tree and solve one big convex program — and your `c·qᵞ` makes
each stage a **convex** program (a genuine plus). The killer is the **scenario tree**: 22 stages of a
continuous 2‑factor‑plus‑jump process need either a coarse tree (biased) or an explosive one. Gives a
serviceable approximation, not a high‑accuracy reference. Niche.

### #13 — Stochastic Mesh (Broadie–Glasserman)  *(Class A but mis‑sized)*
**Verdict: built for high dimension you don't have.** The mesh's weight‑based estimators target
regression‑free high‑D American pricing; at 2+1 dimensions it is strictly dominated on accuracy/cost by
grid‑DP (#1) and carries notorious bias and weight‑variance issues. Skip.

### #14 — Multilevel Monte Carlo  *(accelerator, not a pricer)*
**Verdict: a speed multiplier to bolt onto #2 or LSM‑D, not a standalone method.** "A multilevel Monte
Carlo method for the valuation of swing options" shows MLMC cuts the cost of *nested/biased* estimators
(e.g., the inner simulations of the dual upper bound, or fine time‑grids). Worth knowing **if** you
build the dual (#2); irrelevant on its own.

---

## 3. The master comparison table (mental map)

Columns chosen to be decision‑complete. **Compute‑to‑parity** = order‑of‑magnitude wall‑clock to reach
LSM‑D/RL price accuracy on this 2+1‑D contract (★ = cheap … ★★★★★ = very expensive). **Output** tells
you whether it bounds the true price. **Reuse** = how much of your existing repo it leverages.

| # | Method (family) | Class | Accuracy ceiling | Compute→parity | Bias / output type | Curse of dim? | HHK fit | Convex‑cost fit | Reuses repo | Impl. effort/risk | Marginal value vs LSM‑D+RL | Key refs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **Grid backward DP + analytical kernel** | A | **Very high** (grid/interp‑limited; Richardson‑able) | ★★ | **~Exact**, deterministic point value | No (2+1 D fine) | ✓ kernel = exact `E` | ✓ closed‑form FOC node‑wise | **High** (`transition_kernel.py`, `_roll_from`) | **Low–Med** | **★★★★★ — the missing ground truth** | Boogert–de Jong; Warin; Bardou–Pagès |
| **2** | **Info‑relaxation / martingale dual** | A | High (upper bound; tightness∝penalty) | ★★★ | **Upper bound** (pairs w/ LSM‑D, RL lowers) | No | ✓ kernel martingale | ✓ (use control‑dual, *not* multi‑stop dual) | Med (kernel) | Med–High | **★★★★★ — certifies RL optimality gap** | Rogers; Andersen–Broadie; Meinshausen–Hambly; Brown–Smith–Sun; Bender |
| **3** | **Optimal quantization** | A | Very high (`O(N^{-1/d})` bounds) | ★★★ | Slightly biased, deterministic | No | ✓ (quantize jump `Y` carefully) | ✓ node‑wise FOC | Low–Med | Med | ★★★★ — swing‑native 2nd reference | Bardou–Bouthemy–Pagès; Bally–Pagès |
| **4** | **PIDE / FD‑HJB (control VI)** | A | **Highest** (spectral on smooth) | ★★★★ | ~Exact, deterministic | No | ✓ jump = non‑local term | ✓ smooth `sup_q` Hamiltonian | Low | **High** | ★★★★ — PDE‑side cross‑check | Calvo‑Garrido; Vázquez et al.; Forsyth |
| **5** | **Dynamic Chebyshev / tensor DP** | A | Very high (spectral) | ★★ | ~Exact, deterministic | No (TT scales up) | ✓ kernel | ✓ node‑wise FOC | Med (kernel) | Med | ★★★ — accelerated #1 | Glau et al.; Hierarchical‑Tucker Bermudan |
| **6** | **Control‑randomization / regress‑later MC** | A | LSM‑class (basis‑limited) | ★★ | Low‑biased (lower bound) | Mild | ✓ simulate | ✓ continuous control native | High (sim+LSM code) | Low–Med | ★★ — refines, doesn't diversify | Kharroubi–Langrené–Pham; Glasserman–Yu |
| **7** | **Deep BSDE / DGM** | A (NN) | High (no exactness cert) | ★★★★ | NN point estimate | No (mesh‑free) | ✓ PIDE residual | ✓ continuous control | Med (torch) | High | ★★ — shares NN weaknesses w/ RL | Han–E; Sirignano–Spiliopoulos |
| **8** | **Deep optimal stopping / neural multi‑stop** | B→A | High | ★★★★ | NN lower bound | No | ✓ | ✗→ must rebuild for interior `q` | Med | High | ★ — re‑derives your D4PG | Becker–Cheridito–Jentzen; deep‑combinatorial swing |
| **9** | **Fourier‑COS backward DP** | A* | Spectral *in low‑D* | ★★ | ~Exact in covered dims | **Yes** at 2‑factor+`Q` | ✓ char. fn closed form | ✗ awkward (per‑node payoff) | Low | Med–High | ★★ — elegant but strained | Fang–Oosterlee |
| **10** | **SWIFT / wavelet DP** | A* | Spectral *in low‑D* | ★★ | ~Exact in covered dims | Yes (as COS) | ✓ char. fn | ✗ as COS | Low | Med–High | ★ — COS cousin | Ortiz‑Gracia–Oosterlee |
| **11** | **Trinomial / jump tree; forest of trees** | B | Moderate | ★★★ | Biased, deterministic | Yes (jumps break recomb.) | ✗ jumps break recombination | ✓ node‑wise | Low | Med | ★ — kernel makes it obsolete | Hahn–Dyer; Forest of Stochastic Trees |
| **12** | **Multistage SP / LP** | B | Low–Moderate | ★★★★ | Biased (scenario‑tree) | **Yes** (tree blowup) | △ via scenario tree | ✓ stage = convex program | Low | Med–High | ★ — coarse | Multistage‑SP swing; LP‑to‑PSO |
| **13** | **Stochastic mesh** | A | Moderate (bias/variance) | ★★★★ | Two‑sided, noisy | Built for high‑D | ✓ simulate | ✓ node‑wise | Med | Med | ☆ — mis‑sized for 2+1 D | Broadie–Glasserman |
| **14** | **Multilevel MC** | accel. | n/a (variance tool) | — | Speeds nested estimators | — | ✓ | ✓ | Med | Med | ★ only *with* #2 | Giles; MLMC‑swing |
| — | *LSM‑D (incumbent)* | A | Low‑biased baseline | ★ | **Lower bound** | Mild | ✓ | ✓ inner `q`‑opt | — | — | baseline | Longstaff–Schwartz; Carmona–Touzi; Boogert–de Jong |
| — | *RL / D4PG (incumbent)* | A | Policy‑limited | ★★ | Lower bound (policy value) | No | ✓ | ✓ | — | — | your method | this paper |

\* COS/SWIFT are continuous‑state spectral methods but become dimension‑strained at 2 factors + volume.

---

## 4. Bottom line / recommendation

1. **Build the grid + analytical‑kernel backward DP (#1) first.** It is the natural *deterministic
   ground truth* for this 2+1‑D problem, convex cost is handled by your existing closed‑form FOC, and
   ~70% of the machinery (`transition_kernel.py`, `_roll_from`) already exists. It converts your story
   from "RL beats a low‑biased LSM" into "RL is within X% of the *true* price." Best value‑to‑effort.

2. **If you want a referee‑proof result, add the information‑relaxation dual (#2)** to get a rigorous
   **upper bound**. With #1 (or LSM‑D / RL) as lower bounds you then *bracket* the true price and
   **certify the RL optimality gap** — the most persuasive single addition for a paper under review.
   Use the *stochastic‑control* dual, not the bang‑bang multiple‑stopping dual.

3. **If you specifically want a non‑DP, classical‑finance cross‑check**, pick **optimal quantization
   (#3)** (swing‑native, error‑bounded) or **PIDE‑HJB (#4)** (PDE referees love it, but weeks of work).

4. **Do not** invest in trees (#11), COS/SWIFT (#9/#10), stochastic mesh (#13), multistage SP (#12),
   or deep stopping (#8) for *this* contract: jumps, volume‑dimension strain, convex‑cost friction, or
   redundancy with your existing RL make them dominated here.

**Provenance note (convex cost):** every Class‑A recommendation handles `c·qᵞ` via the *same*
mechanism — a pointwise strictly‑concave 1‑D maximization `max_q [q(S−K) − c·qᵞ + continuation]` with
the closed‑form interior root clipped to `[q_min,q_max]` and the residual global budget. That is the
identical FOC the paper already uses in LSM‑D and in the actor warm‑start, so all four top methods are
*provably* consistent with your contract definition — they differ only in how they compute the
continuation `E[V_{t+1}]` (exact kernel vs. quantizer vs. PDE operator vs. martingale penalty).
