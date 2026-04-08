# LSM vs RL Runtime Tradeoff

## Scope

- Goal: approximate how expensive a stronger discretized-action LSMC benchmark would be relative to the current RL pipeline.
- Target question: if I improve LSM enough to approximate the continuous partial-exercise policy, is that cheaper or more expensive than making RL more efficient?
- Output requested: six runtime values.
  - Discretized LSM: lower, best, upper.
  - Faster RL: lower, best, upper.
- Unit of comparison used here: one convex-cost configuration.
  - LSM: one fit + one out-of-sample evaluation.
  - RL: one seed.
- Important fairness note:
  - The paper reports RL over three seeds, so publication-grade RL cost is roughly `3x` the single-seed numbers below.

## Executive Answer

### Six values

| Method | Lower | Best | Upper |
|---|---:|---:|---:|
| Discretized LSMC, pruned to the partial-exercise region | `45 s` | `100 s` | `240 s` |
| RL, using your requested "end-only OOS eval + 10x faster training" assumption | `100 s` | `125 s` | `170 s` |

### Short interpretation

- Under an optimistic but still defensible LSM implementation, the two methods are in the same time class.
- On the best estimate, they are almost tied: about `1.7 min` for discretized LSMC versus about `2.1 min` for faster RL.
- If the LSM implementation has to discretize inventory explicitly rather than handle inventory through regression features, its cost blows up toward tens of minutes, and RL becomes clearly more attractive.
- If your aim is a stronger benchmark with limited engineering risk, the cheapest path is still manuscript-level clarification plus the current full-state bang-bang LSM comparison.
- If your aim is a genuinely stronger classical benchmark, a carefully pruned discretized-action LSMC is feasible, but it is not obviously cheaper than just making RL evaluation lighter.

## Repository Measurements Used

### 1. Current full-state LSM runtime

Measured from [logs/lsm_state_mode_comparison.csv](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/logs/lsm_state_mode_comparison.csv):

- Mean full-state runtime: `13.12 s`
- Min full-state runtime: `11.70 s`
- Max full-state runtime: `15.80 s`

This is the best anchor for current LSM cost because it already runs the fitted LSM and the out-of-sample price on the repo's actual convex-cost configurations.

### 2. RL evaluation runtime

Benchmarked directly on the repo's saved `c=0.04, gamma=2` policy, using the same `65,536`-path test size and the same evaluation batch size (`512`) that `run.py` uses:

- Final-policy evaluation time: `3.37 s`

### 3. RL training throughput

Benchmarked by running the steady-state part of the training loop after replay warm-up, with logging disabled and no periodic evaluation:

- `500` steady-state episodes: `18.16 s`
- Episodes per second: `27.53`
- Extrapolated time for `32,768` episodes:

$$
T_{\text{RL,train}} \approx \frac{32768}{27.53} \approx 1191\text{ s} \approx 19.9\text{ min}
$$

This is the cleanest measured approximation of the core RL training cost.

### 4. Partial-exercise region frequency

For the focal convex-cost case `c=0.04`, `gamma_c = 2`, `q_max = 2`, the immediate-payoff objective is

$$
g(q) = q\Delta - c q^{\gamma_c}, \qquad \Delta = (S-K)^+.
$$

The interior myopic optimum is

$$
q^*(\Delta) = \left(\frac{\Delta}{c\gamma_c}\right)^{1/(\gamma_c-1)}.
$$

Partial exercise is relevant when

$$
0 < q^*(\Delta) < q_{\max}
\quad \Longleftrightarrow \quad
0 < \Delta < c\gamma_c q_{\max}^{\gamma_c-1}.
$$

For `c=0.04`, `gamma_c=2`, `q_max=2`, this threshold is

$$
\Delta < 0.16.
$$

Using the common `65,536`-path test set:

- Partial-exercise region as a fraction of all states: `28.5%`
- Partial-exercise region as a fraction of in-the-money states: `57.6%`

These two numbers are the key lever for pruning the LSM action grid.

## How the Current LSM Complexity Works

The current LSM recursion is effectively:

$$
T_{\text{LSM,0}} = \Theta(n_t \cdot n_Q \cdot C_{\text{reg}}),
$$

where:

- $n_t$ is the number of decision dates.
- $n_Q$ is the number of bookkeeping states.
- $C_{\text{reg}}$ is the cost of the regression solve for one continuation fit.

In the current code:

- `n_t = 22`
- bookkeeping is bang-bang, so the inventory/rights state is tiny.
- the measured full-state runtime is about `13 s`.

### Basis-size effect

The current "full-state" LSM in code is still not Sven's requested full state, because the implementation regresses on `(S, X, Y)` rather than `(S, X, Y, Q)`.

For a total-degree-2 polynomial/Chebyshev product basis, the number of terms is

$$
P(f,d) = \binom{f+d}{d}.
$$

So:

- Current full-state code: `f = 3`, `d = 2`, so $P = \binom{5}{2} = 10$.
- Sven-style full state with inventory feature: `f = 4`, `d = 2`, so $P = \binom{6}{2} = 15$.

Regression cost usually scales between linear and quadratic in the basis width once `N >> P`, so a reasonable basis-width multiplier is:

$$
\left(\frac{15}{10}\right)^\alpha, \qquad \alpha \in [1,2].
$$

That gives a factor between:

- `1.5x` optimistic
- `2.25x` conservative

## Discretized LSMC: Cost Model

### Practical design I would recommend

If the aim is to make LSMC stronger without letting cost explode, the implementation should do all of the following:

- Regress on `(S, X, Y, Q)`.
- Keep `Q` as a regression feature rather than exploding the dynamic program into a massive explicit inventory grid.
- Add interior action candidates only in the partial-exercise region.
- Outside that region, keep the cheap logic:
  - zero exercise when not profitable,
  - full exercise when the local objective clearly favors `q_max`.

That gives the following runtime multiplier:

$$
\kappa_{\text{LSM}}(M,\rho,\alpha)
=
\left(\frac{15}{10}\right)^\alpha
\left(1 + \frac{\rho M}{2}\right),
$$

where:

- $M$ is the number of interior action bins in the partial region.
- $\rho$ is the fraction of states where partial exercise is relevant.
- $\alpha \in [1,2]$ captures the basis-width cost.

The new runtime estimate is then

$$
T_{\text{LSM}}(M,\rho,\alpha)
\approx T_{\text{LSM,0}} \cdot \kappa_{\text{LSM}}(M,\rho,\alpha).
$$

### Lower estimate

Use the most favorable reasonable assumptions:

- baseline runtime `11.70 s`
- basis factor `1.5x`
- `M = 10`
- `\rho = 0.285`

Then:

$$
T_{\text{LSM,low}}
\approx 11.70 \times 1.5 \times \left(1 + \frac{0.285 \times 10}{2}\right)
\approx 42.5\text{ s}.
$$

Rounded lower estimate: `45 s`.

### Best estimate

Use midpoint assumptions:

- baseline runtime `13.12 s`
- basis factor `1.8x`
- `M = 15`
- `\rho = 0.43`

Then:

$$
T_{\text{LSM,best}}
\approx 13.12 \times 1.8 \times \left(1 + \frac{0.43 \times 15}{2}\right)
\approx 99.8\text{ s}.
$$

Rounded best estimate: `100 s`.

### Upper estimate

Use the expensive but still plausible end:

- baseline runtime `15.80 s`
- basis factor `2.25x`
- `M = 20`
- `\rho = 0.576`

Then:

$$
T_{\text{LSM,high}}
\approx 15.80 \times 2.25 \times \left(1 + \frac{0.576 \times 20}{2}\right)
\approx 240.3\text{ s}.
$$

Rounded upper estimate: `240 s`.

## RL Cost Model

### Measured core RL training cost

Measured steady-state training throughput gives:

$$
T_{\text{RL,train,measured}} \approx 1191\text{ s}.
$$

Measured final evaluation cost gives:

$$
T_{\text{RL,eval}} \approx 3.37\text{ s}.
$$

### Requested assumption

You asked to assume that removing periodic out-of-sample evaluations and doing them only once at the end makes RL training effectively one order of magnitude faster.

I therefore use:

$$
T_{\text{RL,fast-train}} \approx 0.1 \times 1191 \approx 119\text{ s}.
$$

Adding the final evaluation only:

$$
T_{\text{RL,fast}} \approx 119 + 3.37 \approx 122.4\text{ s}.
$$

I then widen that slightly to absorb:

- bias-calibration overhead,
- dataset pre-generation,
- small benchmark/setup effects,
- benchmark noise from measuring only a steady-state slice.

### Lower estimate

- Training a bit faster than the measured slice.
- Minimal setup overhead.

Rounded lower estimate: `100 s`.

### Best estimate

- `119 s` fast training by your assumption.
- `3.4 s` final evaluation.
- a few seconds of fixed overhead.

Rounded best estimate: `125 s`.

### Upper estimate

- Slightly slower extrapolation from the measured slice.
- some extra warm-up/setup overhead.

Rounded upper estimate: `170 s`.

## Convergence of Discretized LSMC Toward the Continuous Policy

### Local action-discretization error

If the Bellman objective in `q` is smooth and strictly concave near the interior optimum, then choosing the nearest grid point to the true optimizer gives:

$$
V(q^*) - V(\hat q)
\approx
\frac{|V''(q^*)|}{8} (\Delta q)^2,
$$

where `\Delta q` is the grid spacing.

That is the optimistic local behavior: second-order value loss.

### Global practical error

In practice, the full LSMC error is not purely local because:

- the action switches between `0`, interior `q`, and `q_max`,
- the profitability boundary is non-smooth,
- continuation values are themselves estimated by regression.

So the practical convergence rate is more realistically bracketed by:

$$
O(M^{-1}) \quad \text{to} \quad O(M^{-2}),
$$

where `M` is the number of interior bins.

### What that means numerically

- Going from `10` to `20` interior bins should improve approximation quality.
- But it is unlikely to halve the pricing error in a perfectly clean way because regression noise and switching boundaries dominate part of the error.
- So `10` bins probably captures most of the qualitative gain.
- `20` bins is more likely a polishing step than a game-changing accuracy jump.

## How Computational Cost Scales With More Discretizations

### Recommended implementation

With continuous `Q` in the regression state and action pruning, cost scales approximately like:

$$
T_{\text{LSM}}(M) \propto \left(1 + c_1 M\right),
$$

with a fairly large constant because each extra action level requires another continuation-value fit.

This is why going from `10` to `20` interior bins roughly doubles the action-related part of the runtime.

### Straightforward explicit-inventory-grid implementation

If you discretize inventory explicitly as well, then the number of inventory states also scales with `M`, so the runtime moves toward:

$$
T_{\text{LSM,grid}}(M) \propto O(M^2).
$$

That is the dangerous implementation path.

Using the current measured `13.1 s` full-state runtime as the anchor, this explicit-grid version would likely move from the few-minute range to the tens-of-minutes range.

That is why, if you do implement a stronger LSMC, I would strongly avoid an explicit fine inventory grid unless you really need it.

## What This Implies Strategically

### If your goal is the fastest path to a better paper

- Improving RL efficiency is the lower-risk engineering path.
- You already have the RL pipeline, saved models, and evaluation workflow.
- A faster RL regime with end-only evaluation is easy to justify operationally.

### If your goal is the strongest response to Sven's benchmark criticism

- A discretized-action LSMC is the more direct answer.
- But it is not a trivial add-on.
- Even with pruning and continuous-`Q` regression, it is probably in the same runtime class as your faster RL scenario.
- If the implementation slips into explicit inventory discretization, it becomes much more expensive.

### My bottom-line recommendation

- If you want the cheapest engineering win: make RL evaluation lighter and communicate the current LSM limitations more explicitly.
- If you want the cleanest reviewer-facing benchmark answer: implement a pruned discretized-action LSMC, but do it with continuous `Q` regression and mathematical pruning of the partial-exercise region.
- I would not recommend a naive explicit inventory-grid LSMC. That version is exactly where the classical method starts losing its computational appeal.

## Final Six Values Again

| Method | Lower | Best | Upper |
|---|---:|---:|---:|
| Discretized LSMC | `45 s` | `100 s` | `240 s` |
| Faster RL | `100 s` | `125 s` | `170 s` |

## Caveats

- These are approximation ranges, not exact profiled end-to-end production timings.
- The RL numbers use your explicit `10x` faster-training assumption.
- The LSM numbers assume a careful implementation with:
  - continuous `Q` in the regression state,
  - pruning to the partial-exercise region,
  - `10` to `20` interior bins.
- A naive discretized-inventory implementation would be materially slower than the LSM range reported above.