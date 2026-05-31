# Hedging Integration Recommendations

## Integration Path

The hedging contribution is **not yet ready to be integrated into the main paper as a claimed implemented result**.

The repository currently implements **pricing only**:

- HHK path generation
- swing-option exercise environment
- pricing actor-critic training
- batched out-of-sample valuation
- LSM-D benchmark

The relevant implemented components are:

- [run.py](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/run.py#L1353)
- [src/swing_env.py](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/src/swing_env.py#L408)
- [src/agent.py](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/src/agent.py#L618)
- [src/agent_evaluation.py](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/src/agent_evaluation.py#L93)
- [src/lsm_swing_pricer.py](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/src/lsm_swing_pricer.py#L740)

The main manuscript already reflects this correctly by treating hedging as future work:

- [Paper/DRL_Swing_Options.tex](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/Paper/DRL_Swing_Options.tex#L695)

By contrast, the separate hedging draft contains substantial theory but no implemented experiment pipeline:

- [Paper/Hedging.tex](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/Paper/Hedging.tex#L701)
- [Paper/Hedging.tex](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/Paper/Hedging.tex#L1469)

### Most Credible Integration Strategy

The clean path is a **two-stage research program**:

1. Keep the current pricing problem unchanged under $\mathbb{Q}$.
2. Train the holder policy exactly as now.
3. Freeze that holder policy.
4. Add a separate writer-side hedging layer that treats the learned exercise strategy as the liability-generation mechanism.

This is the strongest scientifically because it:

- matches the actual codebase
- preserves the current paper's pricing problem
- uses the strongest part of the hedge draft, namely conditional continuation-value and forward-delta construction
- avoids prematurely mixing pricing and risk management into a single underspecified objective

### What Is Already Implemented

- Risk-neutral HHK simulation from time 0 on a fixed decision grid
- 9-dimensional pricing state
- scalar continuous exercise action
- discounted exercise reward
- policy training with D4PG-style actor-critic
- batched evaluation
- LSM-D comparator

### What Exists Only in Draft Form

- forward-curve representation of hedge exposures
- quoted-product delta formulas
- self-financing hedge recursion
- variance-optimal projection into restricted forward instruments
- rollout-based continuation-value and delta estimation from the trained pricing policy

### What Still Requires New Theory, Code, or Experiments

**New code**

- conditional simulation from an arbitrary intermediate post-exercise state
- forward and quoted-forward pricing functions under HHK
- hedge backtesting engine
- hedge diagnostics and evaluation pipeline
- hedging benchmarks

**New experiments**

- hedge P&L backtests
- delta-validation experiments
- exact-spanning versus restricted-instrument hedge experiments
- RL-liability versus LSM-liability hedgeability comparison

**New theory or at least sharper exposition**

- measure consistency between pricing and hedging
- precise specification of the traded hedge universe
- conditional differentiability assumptions behind the delta formulas
- incompleteness interpretation and what “exact” means in the hedging results

---

## Clean Problem Formulations

### 1. Pricing Only

This is the current project.

Formulation:

$$
\sup_q \; \mathbb{E}^{\mathbb{Q}}\left[\sum_i \delta^i \Pi_i(q_i)\right].
$$

This is already implemented and coherent.

### 2. Hedging Only

This is the cleanest extension.

Fix a holder policy $q^*$, learned by RL or produced by LSM-D. Then, from the writer's perspective, choose hedge positions $\eta_i$ in a traded forward universe to minimize a risk functional of terminal hedged P&L.

This is scientifically clean because:

- the liability is well-defined
- pricing and hedging are separated
- comparisons between liabilities become meaningful

### 3. Combined Pricing-and-Hedging

This is only clean if you explicitly move to a **writer utility** or **indifference-pricing** problem.

Otherwise it mixes:

- a valuation problem under $\mathbb{Q}$
- a risk-management problem typically interpreted under $\mathbb{P}$

without a coherent economic objective.

**Conclusion:** combined pricing-and-hedging is not the right next step for this repository.

---

## Exact Algorithmic Extensions Needed

## A. Hedging with Frozen Holder Policy

### State

The writer-side hedge state should include:

- current HHK factors or equivalent observable forward-state variables
- current contract state after observing the holder's realized exercise history
- current hedge inventory
- current bank account or cumulative cash position if transaction costs are included
- current quoted forward vector

Minimum defensible version:

- post-exercise contract state
- current quoted forward vector
- current time index

### Action Space

Two clean options:

1. **Analytic/model-based hedge**
   - no RL action
   - hedge position is computed from delta formulas or variance-optimal projection
2. **RL hedger**
   - action is hedge trade or hedge position in one or more forward instruments

For the next paper, the first option is materially stronger.

### Reward / Objective

For model-based hedging:

- no training reward is needed
- evaluate terminal hedged P&L and hedge error diagnostics

For RL hedging:

- reward must be tied to marked-to-market hedged P&L
- if transaction costs are present, subtract them explicitly
- if optimizing tail risk, a distributional critic or terminal-risk objective is needed

### Training Setup

For the recommended first extension:

- train holder policy exactly as now
- freeze holder policy
- simulate hedging paths conditionally on that fixed holder policy
- compute deltas and hedge positions by conditional Monte Carlo or projection
- no second training loop is required

### Evaluation Pipeline

Need a new pipeline that reports:

- unhedged liability P&L
- hedged P&L
- hedge-error time series
- residual-risk metrics
- turnover if rebalancing costs are included

## B. If You Insist on a Secondary RL Hedger

### State

Must include at least:

- market state or quote vector
- contract state induced by holder actions
- current hedge inventory
- optionally bank account / cumulative costs

### Action

- hedge position or hedge increment in one or more quoted forward products

### Reward

- one-step hedged P&L minus transaction cost
- or terminal P&L optimized through a risk-sensitive objective

### Training

The only partially reusable ingredient is the optional distributional critic path in:

- [src/agent.py](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/src/agent.py#L768)

But this is **not** close to plug-and-play. The current environment and data model are pricing-specific.

### Evaluation

Would require:

- new environment
- new simulator under a chosen hedging measure
- new benchmarks
- extensive risk-metric reporting

---

## Extension Options

## 1. Frozen-Policy Model-Based Hedge in Forwards

**Scientific value:** high

**Why it matters:**

- directly uses the strongest part of the hedge draft
- preserves the current pricing contribution
- avoids over-claiming deep hedging before the market model is properly specified

**Implementation difficulty:** medium

**Required new experiments:**

- delta validation by bump-and-reprice
- exact quoted-product hedge backtest
- restricted-instrument hedge backtest

**Likely publishability:** good

## 2. Hedgeability Comparison: RL Liability vs LSM-D Liability

**Scientific value:** very high

**Why it matters:**

The present pricing results do not support a strong value-dominance claim for RL. A better question is:

> Does the continuous RL exercise policy produce a liability that is easier to hedge than the LSM-D liability?

That is a stronger and more original scientific angle.

**Implementation difficulty:** medium-high

**Required new experiments:**

- same hedge engine for both liabilities
- same market model
- same instrument set
- same evaluation paths

**Likely publishability:** very good

## 3. Restricted-Instrument Variance-Optimal Hedge

**Scientific value:** high

**Why it matters:**

This is the most realistic energy-market extension in the draft and is already formalized in:

- [Paper/Hedging.tex](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/Paper/Hedging.tex#L1277)

**Implementation difficulty:** medium

**Required new experiments:**

- one-instrument hedge
- two-instrument hedge
- bucketed hedge universe
- exact-spanning vs non-spanning comparison

**Likely publishability:** good

## 4. Secondary RL Hedger for the Writer

**Scientific value:** potentially high

**But only if you specify clearly:**

- the scenario measure
- the traded instruments
- transaction costs
- hedge inventory dynamics
- terminal risk objective

**Implementation difficulty:** very high

**Required new experiments:** large

**Likely publishability:** good as a separate paper, weak as a quick add-on here

## 5. Joint Pricing-and-Hedging RL

**Scientific value:** low relative to cost

**Problem:** it changes the economics rather than extending the existing setup.

**Implementation difficulty:** extreme

**Likely publishability:** weak unless reformulated as utility indifference or risk-adjusted writer pricing

---

## Recommended Direction

## Recommendation 1

Build the **frozen-policy hedge engine** first.

This is the best next step because it is:

- scientifically clean
- closest to the existing codebase
- aligned with the hedge draft theory
- implementable without inventing a second poorly specified RL problem

## Recommendation 2

Make the main new empirical question:

> Which liability is easier to hedge: the RL-derived liability or the LSM-D-derived liability?

This is stronger than asking whether RL prices slightly better, because the current results do not support broad pricing dominance.

## Recommendation 3

After exact quoted-product deltas, add the **restricted-instrument variance-optimal projection**.

That gives a realistic market story and a publishable bridge from theory to practice.

### What Not to Do Yet

Do **not** fold the full contents of [Paper/Hedging.tex](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/Paper/Hedging.tex#L1469) into the main paper as if it were already part of the implemented contribution.

The current shorter future-work paragraph in:

- [Paper/DRL_Swing_Options.tex](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/Paper/DRL_Swing_Options.tex#L703)

is scientifically safer than a long unvalidated hedging section.

---

## Concrete Results to Target

## Tables

### Table 1. Hedging Performance by Method

Columns:

- liability type
- hedge method
- mean terminal P&L
- P&L standard deviation
- VaR$_{95}$
- CVaR$_{95}$
- hedge-error RMSE
- turnover

### Table 2. RL Liability vs LSM-D Liability Under Common Hedge Engine

Columns:

- $(c, \gamma_c)$
- option value
- unhedged risk metrics
- hedged risk metrics
- percentage risk reduction

### Table 3. Instrument-Set Ablation

Columns:

- hedge universe
- spanning status
- variance reduction
- CVaR reduction
- turnover

## Figures

### Figure 1. Delta Validation

- Monte Carlo delta estimator vs finite-difference bump-and-reprice

### Figure 2. Terminal Hedged P&L Distributions

- RL liability vs LSM-D liability
- same hedge engine

### Figure 3. Time Series of Hedge Exposures

- forward or bucket deltas through the contract life
- representative cost regimes

### Figure 4. Residual Risk vs Number of Hedge Instruments

- exact-spanning and restricted-instrument cases

### Figure 5. If Transaction Costs Are Added

- risk-cost frontier

## Metrics That Should Be Reported

- hedge-error RMSE
- terminal P&L standard deviation
- VaR$_{95}$
- CVaR$_{95}$
- variance-reduction ratio relative to unhedged
- turnover
- delta-estimation error vs finite differences

---

## Gaps and Objections

## 1. Measure Consistency Is Underspecified

Pricing is currently under $\mathbb{Q}$.

Hedging performance is usually interpreted under $\mathbb{P}$.

If you hedge under $\mathbb{Q}$, call it **model-based sensitivity hedging**, not realistic economic hedging.

## 2. The Traded Hedge Universe Is Not Fixed

The draft moves between:

- instantaneous forwards
- quoted delivery-period forwards

You need one precise market convention before running any hedging experiment.

## 3. The Current Code Cannot Yet Implement the Draft Algorithm

The algorithm in:

- [Paper/Hedging.tex](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/Paper/Hedging.tex#L1469)

needs conditional simulation from arbitrary post-exercise states. The current simulator starts from time 0 only:

- [src/simulate_hhk_spot.py](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/src/simulate_hhk_spot.py#L54)

## 4. The Delta Theory Is Conditional, Not Operationally Verified

The forward-delta formulas are mathematically plausible, but in practice you still need finite-difference validation.

Without that, the draft remains theoretical rather than demonstrated.

## 5. Nested Monte Carlo Cost May Become the Main Engineering Bottleneck

If continuation values and deltas are recomputed by nested rollouts at every hedge date, runtime can explode.

This is a real implementation risk, not a minor detail.

## 6. Incompleteness Must Be Stated Clearly

The draft is correct to note market incompleteness:

- [Paper/Hedging.tex](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/Paper/Hedging.tex#L845)

The hedge is first-order exact in sensitivity space, not a replication strategy.

That distinction must remain explicit.

## 7. The Current Deep-Hedging Paragraph in the Main Paper Is Too Thin to Count as a Real Method

The future-work paragraph in:

- [Paper/DRL_Swing_Options.tex](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/Paper/DRL_Swing_Options.tex#L703)

still leaves unspecified:

- hedge inventory dynamics
- bank-account recursion beyond a single formula
- measure choice
- traded instrument universe
- transaction costs
- benchmark design

That is acceptable as future work, but not as a real algorithm section.

## 8. Joint Pricing-and-Hedging Would Need a Different Economic Thesis

If you move to a combined problem, you need a new paper objective such as:

- utility-indifference pricing
- risk-adjusted writer valuation
- capital-constrained issuance pricing

Without that, the combined formulation is conceptually weak.

---

## Bottom Line

The strongest scientifically defensible extension is **not** “add deep hedging” in the abstract.

It is:

1. use the learned swing policy to define a liability,
2. derive and implement forward-curve hedges for that liability,
3. compare hedgeability of the RL liability and the LSM-D liability,
4. then add restricted-instrument variance-optimal hedging as the realistic market extension.

That is the path most likely to yield a coherent and publishable next contribution.