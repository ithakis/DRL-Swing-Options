# 1) Time grid, state, and controls

Dates: $0=t_0<t_1<\dots<t_{N}=T$.
Spot (or day-ahead) price $S_k$ and *known* strike/index $K_k$ (e.g., quarter-ahead index fixed within the quarter). The key signal is the **instantaneous spread** $\psi_k:=S_k-K_k$ (or capped as we do now $\psi_k:=\left(S_k-K_k\right)^+$). 

Decision (lift) at $t_k$: $q_k$ with **local bounds** $q_k\in[q^{\min},q^{\max}]$.
Cumulative offtake: $Q_{k+1}=Q_k+q_k$.

---

# 2) From firm corridor to **soft penalties**

Classically, the global corridor is **hard**: $Q_N\in[Q_{\min},Q_{\max}]$. Here we relax it with a convex terminal penalty

$$
\Phi(Q_N)
:= \kappa_{\downarrow}\, (Q_{\min}-Q_N)_+^{p}
 + \kappa_{\uparrow}\, (Q_N-Q_{\max})_+^{p},
\qquad p\in\{1,2\}.
\tag{C}
$$

* $p=2$ is smooth and works well with gradient methods;
* letting $\kappa_{\downarrow},\kappa_{\uparrow}\!\to\infty$ recovers firm constraints.
* Penalty means that in case of underconsumption or
overconsumption, the holder of the contract is penalized with a penalty depending on the gas price at
the expiry.
* Terms underconsumption and overconsumption mean respectively
that, at the expiry, the total purchased volume (in case of Swing) or the inventory (in case of storage) is
respectively under $Q_{min},Q_{max}$. 

---

# 3) **Frictions** and operating costs (convex)

Per-date *net* cash flow is

$$
\Pi_k(q_k;S_k)=
q_k\,\psi_k
\;-\; c_{\text{op}}(q_k)
\;-\; c_{\text{lin}}(q_k)
\;-\; c_{\text{ramp}}(q_k,q_{k-1}),
\tag{P\_k}
$$

with

$$
\begin{aligned}
c_{\text{op}}(q) &= \tfrac{b}{2}\,q^2
\quad\text{(smooth operating/“wear” cost)},\\
c_{\text{lin}}(q)&= \tau\,|q|
\quad\text{(linear fee / bid–ask / scheduling cost)},\\
c_{\text{ramp}}(q,q_{-})&=\tfrac{\nu}{2}\,(q-q_{-})^2
\quad\text{(inertia/ramping penalty)}.
\end{aligned}
\tag{F}
$$

All three are **convex** ⇒ the one-step objective is concave in $q_k$, which guarantees a unique optimizer at each step and turns off pure bang-bang behavior.

*(If you prefer hard ramping:* add $|q_k-q_{k-1}|\le R$ alongside or instead of $c_{\text{ramp}}$.)

---

# 4) The **penalized-frictional swing** pricing problem

Under a risk-neutral measure (discounting omitted for clarity), define

$$
\boxed{
V_0
=\sup_{q\ \text{adapted}}
\ \mathbb{E}\!\left[\sum_{k=0}^{N-1}\Pi_k(q_k;S_k)\;-\;\Phi(Q_N)\right]
}
% \tag{P\*}
$$

subject to
$q_k\in[q^{\min},q^{\max}]$ (and optional hard ramp $|q_k-q_{k-1}|\le R$), with
$Q_{k+1}=Q_k+q_k,\ Q_0$ given, and $q_{-1}$ given (often $0$).

This nests the standard firm-constraint problem (via $\kappa\to\infty$) used as the baseline in the PV-strat paper, and it reflects the **indexed-strike** reality.

---

# 5) Dynamic programming (state & recursion)

The Markov state needs to carry last lift for ramping:

$$
Y_k=(S_k,Q_k,q_{k-1}),\qquad V_N(Y_N)=-\Phi(Q_N).
$$

Then

$$
V_k(Y_k)=
\max_{q\in\mathcal A_k(Y_k)}
\Big\{\,q\,\psi_k - \tfrac{b}{2}q^2 - \tau|q| - \tfrac{\nu}{2}(q-q_{k-1})^2
+\mathbb E_k\big[V_{k+1}(S_{k+1},\,Q_k+q,\,q)\big]\Big\},
\tag{DP}
$$

where $\mathcal A_k$ enforces local boxes (and optional hard ramp).
*Interpretation.* The term inside the braces is **today’s margin** minus **frictions**, plus the **continuation value** of the remaining corridor.

---

# 6) First-order condition & closed-form **single-step optimizer**

Let the (shadow) marginal value of one extra unit carried forward be

$$
\lambda_{k+1} \;:=\;
\mathbb E_k\!\left[\partial_Q V_{k+1}(S_{k+1},Q_k+q,\,q)\right].
$$

Ignoring the box/ramp *constraints* for a moment (interior solution), the subgradient optimality condition for (DP) yields

$$
0
=\psi_k
-\underbrace{(b+\nu)\,q_k - \nu\,q_{k-1}}_{\text{smooth+inertia}}
-\underbrace{\tau\,\operatorname{sgn}(q_k)}_{\text{linear friction}}
+\lambda_{k+1}.
\tag{FOC}
$$

Equivalently (soft-threshold form),

$$
\boxed{
q_k^{\text{int}}
=
\frac{1}{b+\nu}\;
\mathcal S\!\big(\ \psi_k+\lambda_{k+1}+\nu q_{k-1}\,,\ \tau\ \big),
\qquad
\mathcal S(m,c)=\operatorname{sgn}(m)\,[|m|-c]_+,
}
\tag{S}
$$

and the **feasible** optimizer is the projection

$$
q_k^\star
=\operatorname{Proj}_{[q^{\min},\,q^{\max}] \cap [q_{k-1}-R,\,q_{k-1}+R]}\big(q_k^{\text{int}}\big).
\tag{Proj}
$$

**Economic reading.**

* $\psi_k$ (spread): pushes you **up** when $S_k$ is rich vs. the index.
* $\lambda_{k+1}$ (shadow price of volume): **negative** if you risk *failing $Q_{\min}$* ⇒ lifts more today; **positive** if you risk *hitting $Q_{\max}$* ⇒ lifts less.
* $\tau$ creates an **inaction band** $|\psi_k+\lambda_{k+1}+\nu q_{k-1}|\le \tau$: don’t change lift for small advantages—fees would eat it.
* $b$ and $\nu$ **smooth** decisions: big, abrupt lifts are discouraged even when spreads are attractive.

With firm global constraints instead of $\Phi$, $\lambda_{k+1}$ is the **KKT multiplier** of the corridor; with penalties it is the **marginal penalty** induced by $\Phi$.

---

# 7) Exercise boundaries (“boundary prices”)

The **exercise frontier** at date $k$ is the set of $(\psi_k,Q_k,q_{k-1})$ where the interior optimizer (S) *just* hits a bound in (Proj). For example, the switch to the **upper** bound occurs when $q_k^{\text{int}}=q^{\max}$:

$$
\psi_k^{\uparrow}(Q_k,q_{k-1})
=(b+\nu)\,q^{\max}-\nu q_{k-1}-\lambda_{k+1}+\tau\,\operatorname{sgn}(q^{\max}).
$$

Analogously for the lower bound. 

---

# 8) Why these extensions matter (economics)

1. **No-free-lunch for “churning”.** Linear frictions $\tau|q|$ create a *do-nothing region*—you avoid tiny, back-and-forth lifts that would degrade realized value after fees.
2. **Asset care & operational realism.** Quadratic $bq^2$ and ramp $\nu(q-q_{-})^2$ capture wear, compressor/fuel effort, or operational inertia; they **replace** knife-edge bang-bang by **smooth, interior** lifts when spreads are only moderately attractive.
3. **Contract compliance vs. economics.** With $\Phi$, you price the *trade-off* between missing the corridor and exploiting spreads: large $\kappa_{\downarrow}$ mimics a strict take-or-pay obligation; smaller $\kappa$ lets the optimizer skip bad markets if they are expensive to serve.
4. **Indexation reality.** When $K_k$ is fixed within a quarter, $\psi_k$ is truly tradeable today (buy at $K_k$, sell spot $S_k$).

---

# 9) Further **swing-only** extensions you can add next

* **Asymmetric penalties** $(\kappa_{\downarrow}\ne\kappa_{\uparrow})$ to reflect contractual asymmetry of shortfall vs. overflow.
* **Time-varying frictions** $\tau_k,b_k,\nu_k$ (weekends, maintenance, capacity outages).
* **Seasonal/local bounds** $q_k^{\min/\max}$ and **per-period quotas** (e.g., weekly/monthly minimums) via additional penalties $\Phi_{\text{per}}(\sum_{k\in\text{week}}q_k)$.
* **Risk-adjusted objective**: add $-\tfrac{\lambda}{2}\operatorname{Var}[\sum_k \Pi_k]$ or an entropic/CVaR term; the DP and PV-strat training both carry through.
* **Add parameter/model uncertainty**, where RL could be more robust compared to LSM. This would mean randomizing the parameter of the Exp-OU process with jumps throughout the contracts lifespan. 

---

# References

1. **Lemaire, Pagès, Yeo (2024).**
   *Swing contract pricing: with and without Neural Networks.* arXiv:2306.03822v4, 28 Mar 2024. Sorbonne Université.&#x20;

2. **KYOS Energy Analytics (2016).**
   *How to make more money with a gas swing contract (Backtest).* Practitioner report describing quarter-ahead indexation, daily exercise, and delta-hedging workflow. ([Kyos][1])

3. **Yeo, C. (2025).**
   *Numerical Methods for the Pricing of Swing and Storage Contracts.* PhD thesis, Sorbonne Université. (Bibliographic listing) ([Google Scholar][2])

4. **Barrera-Esteve, Bergeret, Dossal, Gobet, Meziou, Munos, Reboul-Salze (2006).**
   *Numerical methods for the pricing of swing options: a stochastic control approach.* *Methodology and Computing in Applied Probability,* 8(4), 517–540. (Journal version of the INRIA/HAL report linked.) ([IDEAS/RePEc][3])

[1]: https://www.kyos.com/wp-content/uploads/2016/08/KYOS-Gas-Swing-Contract-Backtest-2016.pdf "How to make more money with a gas swing contract"
[2]: https://theses.hal.science/tel-05110013v1/file/144576_YEO_2025_archivage.pdf "Christian Yeo"
[3]: https://ideas.repec.org/a/spr/metcap/v8y2006i4d10.1007_s11009-006-0427-8.html "Numerical Methods for the Pricing of Swing Options"
