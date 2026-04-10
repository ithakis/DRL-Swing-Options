# Sven Feedback Status Review

## Short Executive Summary
- Sven's four comments are no longer in the same state as when this note was first drafted.
- Comment 3 on profitability-gate scope is now resolved in the manuscript text.
- Comment 4 on reporting bias is now resolved in the manuscript text and evaluation protocol.
- Comment 1 on redundant state variables is mostly resolved at the manuscript-framing level, but still has no dedicated ablation.
- Comment 2 on benchmark weakness is mostly resolved because the paper now uses a much stronger discretized-action full-state LSM baseline, though it remains a finite-grid classical approximation rather than an ultimate benchmark.

## Per-Comment Status Review

### Comment 1: Redundant state variables
- Supervisor point: The paper should reduce redundant time and inventory features or, at minimum, stop presenting them as theoretically elegant.
- Current status: Mostly resolved.
- What changed:
- The state subsection now explicitly calls the 9-feature state intentionally non-minimal.
- The duplicated volume and time features are justified as an engineering choice for optimization stability.
- The text now leaves a minimal-state ablation to future work instead of implying theoretical necessity.
- Why this is not fully resolved:
- The state itself is unchanged.
- There is still no minimal-state ablation demonstrating that a reduced sufficient basis preserves performance.
- Practical conclusion: Sven's critique is answered honestly in the manuscript, but not closed experimentally.

### Comment 2: Weak benchmark
- Supervisor point: The old comparison used a weak LSM baseline and therefore did not cleanly isolate the benefit of continuous control.
- Current status: Mostly resolved.
- What changed:
- The paper now documents a discretized-action LSM baseline with $M=5$ exercise levels instead of the old bang-bang benchmark.
- The baseline now uses the fuller HHK state in its continuation regressions and explicit remaining-capacity bookkeeping in the dynamic program.
- The manuscript also reports a reduced-state versus fuller-state comparison and no longer frames the old spot-only bang-bang LSM as the main benchmark.
- Why this is still not perfectly final:
- The classical comparator is still limited by a finite action grid and a polynomial continuation approximation.
- Sven's broader suggestion of even stronger randomized or finer-grid LSM variants remains open.
- Practical conclusion: The benchmark criticism is substantially addressed and the comparison is now much more convincing, but there is still room for follow-up benchmark strengthening if needed.

### Comment 3: Profitability gate under ramping / refraction constraints
- Supervisor point: The hard profitability gate is not generically justified once actions are path-coupled by ramping or refraction constraints.
- Current status: Resolved.
- What changed:
- The profitability-gate discussion now limits the dominance argument to the actual experimental regime: optional exercise, no ramping, and no active refraction lag.
- The text now explicitly says that under mandatory take obligations, active ramping constraints, or hard refraction limits, the hard gate may need to be relaxed.
- Practical conclusion: This comment is cleanly resolved at the manuscript level.

### Comment 4: Best-seed / best-checkpoint reporting bias
- Supervisor point: Publication tables should not be based on best-seed or best-checkpoint selection on held-out paths.
- Current status: Resolved.
- What changed:
- The experimental-setup paragraph now states that each configuration is trained with seeds {11, 12, 13} for 32,768 episodes and that final-episode weights are saved.
- The manuscript now states that all actors and the LSM baseline are evaluated on one common untouched test set of 65,536 paths with seed 999.
- Table 5 is now described as reporting the cross-seed mean and standard deviation on that common test set.
- Residual note:
- Some CSV artifacts still retain legacy audit columns such as best-seed fields, but the manuscript no longer relies on them for reported results.
- Practical conclusion: Sven's statistical-rigor objection is resolved for the paper as written.

## Recommended Current Position
- Comments 3 and 4 can be described as resolved.
- Comments 1 and 2 should be described as substantially or mostly resolved, not fully closed in the strongest possible experimental sense.
- The manuscript is now in a defensible state for response to Sven without overclaiming.

## Optional Follow-Up Work
- Run a minimal-state ablation if you want to fully close comment 1 experimentally.
- Test a finer-grid or randomized LSMC variant if you want to go beyond the current response to comment 2.
- Remove or archive legacy best-seed fields from downstream summary artifacts if you want the supporting files to mirror the manuscript narrative even more closely.