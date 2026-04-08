## Response to Sven

### 1. Benchmark strength
I strengthened the benchmark discussion in two ways. First, the manuscript now states clearly that the default comparator is still a bang-bang LSM benchmark, so it should not be read as a full continuous-control classical solution. Second, I added a full-state bang-bang rerun against the older reduced-state spot-only version. Across the 25 convex-cost configurations, the full-state regression is higher in 20 of 25 cases, with mean uplift `0.0065` and median uplift `0.0046`, so the paper now documents that conditioning on `(S, X, Y, Q)` improves the classical fit.

### 2. Main quantitative effect
This only partially resolves the benchmark critique. The current LSM comparator still restricts actions to `{0, q_max}`, so it cannot represent the interior exercise quantities that matter most under convex costs. The revised manuscript now says this explicitly and avoids claiming that the current comparison isolates the value of continuous actions from the value of a richer state representation. A stronger follow-up benchmark would still be a discretized-action LSMC or randomized LSMC variant.

### 3. State representation
I rewrote the state-description paragraph to present the 9-feature state as a non-minimal engineering choice rather than a theoretically minimal basis. The manuscript now says that the redundant time and inventory features are retained for optimization stability, cites recent RL hedging literature only for that narrower engineering point, and leaves a minimal-state ablation as future work.

### 4. Profitability gate scope
I kept the profitability-gate justification restricted to the regime actually studied in the experiments: optional exercise, no ramping, and no active refraction lag. The revised manuscript avoids suggesting that the same hard gate is already justified for richer path-coupled constraints.

### 5. Evaluation protocol — now resolved
The reporting issue raised by Sven is now fully addressed. The manuscript now includes an explicit paragraph (orange-coloured in the draft) describing the evaluation protocol:

- Each configuration is trained with seeds {11, 12, 13} for 32,768 episodes; **final-episode** actor weights are saved (no cherry-picking).
- A single common test set of **N_test = 65,536 paths** is generated from a dedicated seed (s_test = 999) that was never used during training or hyperparameter selection.
- Every saved actor **and** the LSM baseline are evaluated on this identical test set, so all reported prices are true out-of-sample estimates.
- Table 5 reports the **cross-seed mean** and **standard deviation** of the three test-set prices; the ±CI column propagates the cross-seed uncertainty.

This protocol is also reflected in Table 5's caption, which states it clearly. The "Convex Costs Results 7.csv" artifact was rebuilt from logs using `tools/rebuild_results_v7.py` using this same protocol, and the focal robustness study (Section 6.5) confirms low seed sensitivity (σ = 0.009, CV = 0.45%) across 15 independent seeds at c = 0.04, γ_c = 2.