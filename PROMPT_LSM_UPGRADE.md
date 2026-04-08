# Prompt: Upgrade LSM Baseline to Discretized-Action Full-State LSMC

You are working on the repository at `/Users/alexanderithakis/Documents/GitHub/DRL-Swing-Options`. This is a research-grade implementation of D4PG for pricing swing options in energy markets. The paper benchmarks RL against a Least-Squares Monte Carlo (LSM) baseline.

The co-author Sven raised two comments which interact with each other. Read `SVEN_FEEDBACK_PLAN.md` and `SVEN_RESPONSE.md` for the full context. The two comments that are relevant to this work are:

**Comment 1 (State representation)**: The paper should present the 9-feature RL state as a non-minimal engineering choice. This was already addressed in the current revision with purple-colored text in the paper. The LSM was also upgraded from reduced-state (spot-only, degree-7 Chebyshev) to full-state (S, X, Y with degree-2 Chebyshev product basis). The paper already documents this full-state upgrade. **Do not break or revert any of this.**

**Comment 2 (Weak benchmark)**: The LSM baseline is bang-bang (exercises either 0 or q_max). Sven requests discretized actions like {0, q_max/4, q_max/2, 3q_max/4, q_max} so the comparison isolates the value of continuous control from the baseline's structural inability to represent interior exercises. This is the primary change you must implement.

**Critical constraint**: Both comments must remain addressed simultaneously in the final paper and code. The LSM must keep its full-state regression AND gain discretized actions.

---

## Color Legend for Paper Edits

The manuscript uses colored text to track provenance of changes. Existing colors in `Paper/DRL_Swing_Options.tex` (lines 14–18):

```latex
\definecolor{revisiongreen}{RGB}{0,100,0}     % Green: new additions independent of Sven
\colorlet{green}{revisiongreen}
\definecolor{revisionpurple}{HTML}{6A0DAD}     % Purple: addressing Sven's review comments
\colorlet{purple}{revisionpurple}
\definecolor{revisionorange}{HTML}{B35A00}     % Orange: new statistical methodology/results
```

Add a new color for this round of changes (discretized-action LSM upgrade). Use a dark brown:

```latex
\definecolor{revisionbrown}{HTML}{6B3A2A}
```

Wrap ALL new text you write in `\textcolor{revisionbrown}{...}`. Do not change the color of existing green, purple, or orange text. Do not touch the `\textcolor{red}` Sven comments — they stay as-is for traceability.

---

## Task 1: Identify Minimal Paper Edits

Before writing any code, identify which specific sections of `Paper/DRL_Swing_Options.tex` need modification. The LSM upgrade is significant algorithmically but the paper's focus is the RL method, so edits should be minimal and surgical. 

Sections that will need updating (read them first):
- **Line ~675–695**: `\subsection{Bang-Bang LSM Baseline}` — rename/rewrite to describe discretized-action LSM. The equation and description of the exercise rule must change to reflect M action candidates instead of binary.
- **Line ~688**: The purple paragraph comparing reduced-state vs full-state — keep this, update numbers if they change.
- **Line ~760–790**: Table `\ref{tab:results}` — all LSM column values will change. The RL values do NOT change.
- **Line ~830–870**: Exercise heatmap figure caption and discussion — LSM behavior description changes because it's no longer strictly bang-bang.
- **Line ~890–920**: Bang-Bangness discussion — LSM now has B < 100%, update accordingly.
- **Line ~930–960**: Discussion subsection — the three qualitative patterns and the caveat paragraph need light rewording since the baseline is now stronger.
- **Line ~960–975**: Seed robustness subsection — LSM baseline value changes, verify the "every seed exceeds LSM" claim still holds.
- **Abstract** (line ~69): "bang-bang least-squares Monte Carlo baseline" → update phrasing.
- **Line ~112**: "bang-bang LSM baseline" in contributions list.
- **Conclusion** (line ~910+): references to "bang-bang LSM baseline" → update.

Make a plan of what each section should say differently. Do not make edits yet.

---

## Task 2: Modify the LSM Code

### 2a. Implement Discretized-Action LSM in `src/lsm_swing_pricer.py`

The current LSM has three main entry points:
1. `price_swing_option_lsm()` — single-function combined fit+evaluate (used rarely, has a Numba fast path for reduced-state OLS)
2. `fit_lsm_estimators()` — fits regression coefficients on training data → returns `LSMEstimators` dataclass
3. `price_swing_option_lsm_oos()` — evaluates frozen estimators on out-of-sample data

The **canonical pipeline** used by `tools/rebuild_results_v7.py` and `run.py` is: `fit_lsm_estimators()` → `price_swing_option_lsm_oos()`. Focus your changes on these two functions.

**What to change:**

Currently the code loops over `(cooldown, rights)` states and makes a binary keep-vs-exercise decision at each backward step. Replace this with a loop over M candidate exercise quantities. For each candidate action `q_m ∈ {0, Δq, 2Δq, ..., q_max}` where `Δq = q_max / (M-1)`:

1. Each non-zero candidate has its own continuation-value regression `betas_ex[m]` and its own payoff `q_m * max(S - K, 0) - c * q_m^γ`.
2. At each backward step, compute the net value `payoff_net(q_m) + cont_ex(q_m)` for each candidate and compare against `cont_keep` (the zero-exercise continuation). Pick the `q_m` that maximizes total.
3. The inventory bookkeeping changes: instead of tracking "rights remaining" in units of q_max, track remaining volume capacity. After exercising `q_m`, remaining capacity decreases by `q_m`. A feasibility check must ensure `q_m ≤ Q_max - Q_used`.
4. Apply the net-profitability gate per candidate action: only consider `q_m` if `q_m * max(S - K, 0) - c * q_m^γ > 0`.

**Design constraints:**
- Keep lines of code minimal. Remove boilerplate.
- Use vectorized NumPy operations. Avoid Python-level path loops in the backward induction (the forward "cash flow accumulation" loop can remain scalar for correctness but vectorize if straightforward).
- Use Numba `@jit(nopython=True)` for any critical inner loops that can't be vectorized.
- The default number of action candidates `M = 5` (matching Sven's suggestion: {0, q_max/4, q_max/2, 3q_max/4, q_max}). Make M configurable via a parameter `n_actions` with default 5.
- The `_extract_state_paths()` function currently returns `(S, X, Y)` for `state_mode="full"`. This is fine for the regression features. Inventory/volume state is handled through the DP grid, not through regression features (the DP state tracks remaining capacity).
- The `LSMEstimators` dataclass and `_build_lsm_feature_matrix()` stay the same. Update `LSMEstimators` to store `n_actions` and the action grid.
- Keep support for the old bang-bang mode accessible (e.g., `n_actions=2` gives the original {0, q_max} behavior). This lets us validate backward compatibility.
- The old `_lsm_core_ols` Numba function (line ~66) is the fast path for reduced-state OLS bang-bang. You may leave it as-is or remove it. It is not used in the canonical pipeline.

**Performance notes:**
- The current `fit_lsm_estimators` takes ~13 seconds for one configuration (65536 paths, 22 steps). With M=5 actions, the regression count per (c,r,step) increases from 2 to M+1. Target: under 60 seconds per configuration. If slower, optimize with Numba or batched linear algebra.
- The OOS evaluation `price_swing_option_lsm_oos` should also support M actions. The forward pass must pick the best action at each step using the frozen estimators.

### 2b. Validate the Implementation

After modifying the code, run these validation checks:

1. **Backward compatibility**: Run with `n_actions=2` and compare against the current bang-bang results. Values should be very close (within bootstrap CI) to the current `Convex Costs Results 7.csv` LSM_full column.
2. **Monotonicity in M**: For a single config (c=0.04, γ=2), run with M=2, M=3, M=5. Values should be non-decreasing (more action choices ≥ fewer).
3. **Linear cost sanity**: With c=0 (or c with γ=1), the M=5 result should be very close to M=2 because bang-bang is optimal under linear costs.
4. **Convex cost improvement**: With c=0.04, γ=3, M=5 should meaningfully improve over M=2 because interior actions become profitable.

Run these checks before proceeding. If any fail, debug and fix.

### 2c. Tune LSM Parameters (2 iterations)

Use 4 test configurations:
- Config A: c=0.00, γ=1.0 (linear, no cost)
- Config B: c=0.04, γ=1.0 (linear cost)
- Config C: c=0.04, γ=1.5 (mild convexity)
- Config D: c=0.04, γ=2.0 (moderate convexity)

**Parameters to tune** (the action grid is fixed at M=5):
- `poly_degree`: integer in [2, 5]
- `basis_type`: one of {chebyshev, laguerre, power, hermite}
- `reg_type`: one of {none, ridge}
- `reg_alpha`: float (only matters if ridge)
- `n_training_paths`: the number of paths used to fit the LSM (currently 32768)

Constraint: the total number of basis functions (product of multi-indices with total degree ≤ poly_degree across 3 features S,X,Y) must not exceed 50.

**Iteration 1**: Pick 10 diverse setups covering the parameter space. For each, fit LSM on a training set (seed=998) and evaluate OOS on the test set (seed=999, 65536 paths) for all 4 configs. Record all prices.

**Iteration 2**: Analyze iteration 1 results. Identify which settings worked best for which configs. Design 10 more setups concentrated near the best performers. Run them. Select the single setup that maximizes the average LSM price across the 4 configs (higher is better since LSM is a lower bound).

The winning setup becomes the default parameter set used for all subsequent experiments.

---

## Task 3: Re-run All LSM Experiments

Using the winning LSM parameters from Task 2c, regenerate all LSM benchmark values:

1. Modify `tools/rebuild_results_v7.py`:
   - Update the `evaluate_lsm_on_test_set()` function to use the new discretized-action LSM with the winning parameters.
   - The function currently hard-codes `state_mode="full"`, `lsm_degree=2`, `lsm_basis="chebyshev"`. Update to also pass `n_actions=5` (and any other changed parameters).
   - The `lsm_bangbangness` is currently hard-coded to `1.0`. Compute it from the actual exercise decisions.
   
2. Also update `tools/compare_lsm_state_modes.py` if it's used anywhere, but it's a diagnostic tool and lower priority.

3. Also update `run.py` where it calls `fit_lsm_estimators` (around line 1320-1340) so that live LSM benchmarks during RL training use the upgraded LSM.

4. Run: `python tools/rebuild_results_v7.py` — this regenerates `Jupyter Notebooks/Convex Costs Results 7.csv` with updated LSM values.

5. Also run the focal study: `python tools/rebuild_results_v7.py --focal_config SwingOption_20_c0.04_gamma2 --focal_seeds 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 --output "Jupyter Notebooks/Convex Costs Results 7 focal.csv"`

6. Validate: The new LSM values should be ≥ the old ones for all 25 configs (since M=5 ⊃ M=2). The improvement should be largest for high γ configs. If any LSM value decreased, something is wrong — debug.

---

## Task 4: Update Notebooks and Generate Results 8

The paper figures come from `Jupyter Notebooks/6: Convex costs 0.04 Analysis.ipynb`. Read the paper-figure-regen skill at `.github/skills/paper-figure-regen/SKILL.md` before touching the notebook.

1. **Generate new LSM parquet files**: The notebook reads LSM exercise data from parquet files in `logs/lsm_full_state/`. Run the LSM for the c=0.04 case studies (γ ∈ {1.0, 1.5, 2.0, 3.0}) and save the parquets to `logs/lsm_full_state/` (matching existing filename conventions). The parquets contain per-path per-timestep exercise records (path, time_step, spot, q_exercised_so_far, q_t, payoff, payoff_gross, exercise_cost).

   **Critical**: With discretized actions, the `q_t` column will no longer be just 0 or q_max — it can be any of the M action values. The notebook plotting code must handle this correctly.

2. **Update notebook cells**: Any cell that assumes LSM is bang-bang (e.g., hard-coded LSM BangBangness = 1.0, or filters on q_t == q_max) needs updating. The notebook should compute LSM BangBangness from data, not assume it.

3. **Regenerate figures**: The paper has 4 figures:
   - Figures 1–3 come from notebook 6 (exercise heatmap, spot-income plot, bang-bangness chart)
   - Figure 4 comes from `tools/generate_seed_robustness_figure.py`
   
   Re-run the relevant notebook cells to regenerate figures 1–3. Also re-run the seed robustness figure script (it reads from the focal CSV which was updated in Task 3).

4. **Generate `Convex Costs Results 8.csv`**: Copy the structure of `Convex Costs Results 7.csv` but with the new LSM values. Keep `Convex Costs Results 7.csv` intact for reference. The new file should be `Jupyter Notebooks/Convex Costs Results 8.csv`.

5. **Validate**: Open the regenerated figures and verify:
   - The exercise heatmap (Figure 1) now shows non-trivial LSM exercise patterns at high γ (not pure bang-bang).
   - The spot-income plot (Figure 2) shows LSM dots at intermediate q values.
   - The bang-bangness chart (Figure 3) shows LSM B < 1.0 at high γ.
   - The seed robustness figure (Figure 4) still shows all seeds above LSM line (which may have shifted up).

---

## Task 5: Review Results and Plan Paper Strategy

Before editing the paper, analyze the new results:

1. **Compare old vs new LSM**: How much did the LSM improve? Which configs improved most?
2. **RL gap**: How does the RL-vs-LSM gap change? The gap should shrink (stronger baseline), especially at high γ. Is the RL advantage still clear for γ ≥ 2? If the gap vanished for some configs, that's actually a valid finding — it means the baseline was artificially inflating the RL advantage.
3. **Narrative impact**: 
   - If RL still dominates at high γ → the paper's main claim strengthens (we ruled out baseline weakness as the explanation)
   - If RL advantage shrinks substantially → the paper should honestly report this and position the main contribution as the framework/methodology rather than magnitude of outperformance
   - If LSM now exceeds RL for some configs at low γ → this is expected and even desirable (shows the baseline is now competitive)

4. Based on the analysis, update your Task 1 plan with specific quantitative claims. Write down the exact phrases/numbers you will change.

---

## Task 6: Edit the Paper

Apply your planned edits to `Paper/DRL_Swing_Options.tex`. Wrap ALL new text in `\textcolor{revisionbrown}{...}`.

Key changes:
1. Add the `\definecolor{revisionbrown}` line near the other color definitions (around line 18).
2. Rewrite the LSM subsection (~line 675) to describe the discretized-action baseline. Keep the purple paragraph about reduced-vs-full state comparison but update numbers.
3. Update Table `\ref{tab:results}` with new LSM values from Results 8.
4. Update figure captions and discussion to reflect that LSM is no longer strictly bang-bang.
5. Update the caveat paragraph. With a 5-point action grid, the caveat is weaker but still valid (a finer grid or randomized LSMC could be even better).
6. Update abstract, contributions list, conclusion to replace "bang-bang" with appropriate phrasing.
7. **Do not** change RL numbers, training methodology, or section structure.

After editing, build the paper:
```bash
./tools/build_latex.sh "$PWD/Paper" DRL_Swing_Options.tex
```

Check the build log for errors. If the PDF builds, view key pages to verify:
- Table formatting and alignment
- Figure placement
- No overflow/underflow warnings on the results table
- New brown text is visible and correctly placed

Iterate if needed.

---

## Task 7: Deliver and Clean Up

1. Verify the final PDF looks correct.
2. Ensure `Convex Costs Results 7.csv` and `Convex Costs Results 7 focal.csv` are updated with the new LSM values (Task 3 already did this).
3. Ensure `Convex Costs Results 8.csv` exists as the new canonical file.
4. Update `tools/rebuild_results_v7.py` to reference the new LSM parameters as defaults.
5. Verify `run.py`'s LSM benchmark call uses the upgraded LSM.
6. Do NOT delete any old result files. Keep Results 1–7 for audit trail.
7. Run `make paper` one final time to confirm the build is clean.

---

## Reference: Current File Structure

| File | Role |
|------|------|
| `src/lsm_swing_pricer.py` | LSM implementation — **primary edit target** |
| `tools/rebuild_results_v7.py` | Regenerates results CSV — **must update LSM call** |
| `tools/compare_lsm_state_modes.py` | Diagnostic tool — update if time permits |
| `tools/generate_seed_robustness_figure.py` | Generates Figure 4 from focal CSV |
| `run.py` | Main RL training script — update LSM benchmark call (~line 1320) |
| `Paper/DRL_Swing_Options.tex` | Paper source — **edit target for text** |
| `Jupyter Notebooks/6: Convex costs 0.04 Analysis.ipynb` | Generates Figures 1–3 |
| `Jupyter Notebooks/Convex Costs Results 7.csv` | Current results — will be overwritten |
| `Jupyter Notebooks/Convex Costs Results 7 focal.csv` | Focal study results — will be overwritten |
| `.github/skills/paper-figure-regen/SKILL.md` | Skill file for figure regeneration |
| `.github/copilot-instructions.md` | Repo-level instructions |
| `SVEN_FEEDBACK_PLAN.md` | Full analysis of Sven's 4 comments |
| `SVEN_RESPONSE.md` | Current response to Sven |

## Reference: Sven's Exact Comment (preserved at line 690 of paper)

> The comparative analysis relies on a weak baseline as we compare a 9-dimensional continuous DRL method against a 1-dimensional discrete LSMC. This makes it harder to disentangle the benefit of continuous actions under convex costs from the poor performance of the baseline (ignoring X, Y, Q). A full-state LSMC baseline with discretized actions (q ∈ {0, q_max/4, q_max/2, ..., q_max}) regressing on (X, Y, Q) would provide a stronger and more convincing comparison.

## Reference: Contract Parameters

- Strike K = 1.0, Maturity T ≈ 0.0833 years, n_rights = 22
- q_min = 0, q_max = 2.0, Q_max = 20.0
- Risk-free rate r = 5%
- Cost grid: c ∈ {0.00, 0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15}, γ ∈ {1.0, 1.5, 2.0, 3.0}
- With M=5 actions: exercise grid = {0, 0.5, 1.0, 1.5, 2.0}
- DP inventory states: with Δq=0.5 and Q_max=20, there are Q_max/Δq = 40 discrete inventory levels (vs 10 with bang-bang)

## Reference: Python Environment

Use the conda environment `EP11`. Run Python commands as:
```bash
conda run -p /path/to/miniforge3/envs/EP11 --no-capture-output python ...
```
or activate with `conda activate EP11`.

## Execution Order

Tasks must be executed **strictly sequentially**: 1 → 2a → 2b → 2c → 3 → 4 → 5 → 6 → 7. Each task depends on the previous. Do not skip validation steps.
