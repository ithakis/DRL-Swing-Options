# Restricted-Instrument Variance-Optimal Hedge

## Idea

A swing option is naturally exposed to an entire **forward curve**, not just to one price.

In the idealized theory, you would hedge it with a full strip of forwards, one for each relevant delivery date. In practice, that is usually impossible because the market only offers a **restricted set of quoted hedge instruments**, such as monthly or quarterly delivery-period forwards.

The **restricted-instrument variance-optimal hedge** answers this question:

> If I cannot hedge the full forward-curve exposure exactly, what positions in the available instruments minimize the residual hedge error?

---

## Why It Is Needed

The exact chain-rule hedge works only if the available quoted products span the relevant forward-curve deformations.

If they do not, then some risk remains unhedgeable. In that case, the right objective is no longer:

- match the true delta exactly

but instead:

- choose the hedge that makes the remaining one-step error as small as possible in mean-square sense

This is a **local variance-minimization problem**.

---

## Setup

At hedge date $t_i$, suppose the available hedge universe is a vector of quoted products

$$
\Delta \mathbf{G}_i = (\Delta G_i^1, \dots, \Delta G_i^p)^\top,
$$

where each $\Delta G_i^a = G_{t_{i+1}}^a - G_{t_i}^a$ is the one-step price change of an available traded forward.

Let the one-step centered change in the swing continuation value be

$$
\Delta \widetilde V_i := V_{i+1} - \mathbb{E}[V_{i+1} \mid \mathcal{F}_{t_i}].
$$

We want hedge positions $\eta \in \mathbb{R}^p$ such that

$$
\eta^\top \Delta \mathbf{G}_i
$$

tracks $\Delta \widetilde V_i$ as closely as possible.

---

## Optimization Problem

The restricted-instrument variance-optimal hedge solves

$$
\eta_i^{\mathrm{vo}}
=
\operatorname*{arg\,min}_{\eta\in\mathbb{R}^p}
\mathbb{E}\left[
\left(
\Delta \widetilde V_i - \eta^\top \Delta \mathbf{G}_i
\right)^2
\middle|
\mathcal{F}_{t_i}
\right].
$$

Interpretation:

- $\Delta \widetilde V_i$ is the liability shock you want to hedge
- $\eta^\top \Delta \mathbf{G}_i$ is the hedge portfolio shock you can produce
- the objective minimizes the conditional one-step residual variance

---

## Closed-Form Solution

If the conditional covariance matrix of hedge instruments is invertible, then the solution is

$$
\eta_i^{\mathrm{vo}} = C_i^{-1} b_i,
$$

where

$$
C_i := \mathbb{E}\left[\Delta \mathbf{G}_i \Delta \mathbf{G}_i^\top \mid \mathcal{F}_{t_i}\right],
$$

and

$$
b_i := \mathbb{E}\left[\Delta \mathbf{G}_i \, \Delta \widetilde V_i \mid \mathcal{F}_{t_i}\right].
$$

This is just conditional least squares:

- $C_i$ measures how the available hedge instruments move with each other
- $b_i$ measures how the liability moves with the available hedge instruments

So $\eta_i^{\mathrm{vo}}$ is the best linear projection of the liability shock onto the span of tradable hedge shocks.

---

## Relation to the Exact Hedge

If the chosen hedge instruments span the relevant forward-curve moves exactly, then the variance-optimal hedge collapses to the exact chain-rule hedge.

So:

- **exact hedge**: possible when the market basis is rich enough
- **variance-optimal hedge**: fallback when the hedge basis is restricted

This is why the variance-optimal hedge is the practical version of the ideal forward-curve hedge.

---

## Intuition

Think of the exact forward-curve delta as the “true” exposure vector in an infinite-dimensional space.

But in the real market, you are only allowed to trade a few instruments. That means you can only hedge inside a smaller subspace.

The restricted-instrument variance-optimal hedge is the **orthogonal projection** of the true exposure onto that smaller tradable subspace, with orthogonality defined by conditional covariance.

---

## Why It Matters for Swing Options

This is especially relevant for swing options because:

- the liability depends on many future exercise dates
- the hedge exposure is distributed over the forward curve
- electricity markets quote bucket products, not a full continuum of forwards
- jumps and non-storability make the market incomplete anyway

So exact replication is generally impossible. Variance-optimal hedging is the right practical objective.

---

## In This Project

In [Paper/Hedging.tex](/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options/Paper/Hedging.tex#L1277), this hedge is introduced after deriving the ideal forward-curve delta and the quoted-product chain-rule hedge.

The intended workflow is:

1. Compute the swing option's forward-curve exposure.
2. Map that exposure into quoted products when possible.
3. If the quoted products are too coarse, use the variance-optimal projection instead.

So the restricted-instrument variance-optimal hedge is not a different theory of hedging. It is the practical projection of the ideal hedge into the smaller set of instruments the market actually gives you.

---

## Bottom Line

The restricted-instrument variance-optimal hedge is the hedge that uses the **available quoted forwards** to minimize the **conditional variance of the residual one-step hedge error**.

It is the right object when:

- the true forward-curve exposure cannot be matched exactly
- the market offers only a restricted hedge universe
- you want the best feasible hedge rather than an unattainable exact one
