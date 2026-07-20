# DP → Paper integration plan (gap analysis, 2026-07-15)

Goal: reach the state "the DP development process and comparison results are ready; the only thing
left is surgical edits to the PDF." Target deliverable in the manuscript: **one DP subsection with
1–2 tables/figures + a DP reference column in Table 5 (`tab:results`)**.

---

## 0. BLOCKING — recover the Results-9 paper state from the stash

The manuscript on disk (`Paper/DRL_Swing_Options.tex` at `9c1931c`) is the **old 3-seed table**.
The entire Results-9 rebuild (8-seed BCa table, three-way LSM-D/AC-sample/AC-kernel format,
narrative edits) lives only in **`git stash@{0}`** ("epitaxy: pre-switch from main", base `c13107c`),
together with its untracked companion commit **`7dd4be0`** holding:

- `Jupyter Notebooks/Convex Costs Results 9.csv` (canonical 8-seed results — exists nowhere else)
- `episode_efficiency.csv` (the "lost" Validation-3 CSV — found!)
- `gen_results9_v67.py`, `Convex_Costs_Results_9_METHODOLOGY.md`, `tools/colorize_results_table.py`,
  sample-efficiency figs, bangbang_v67, table6_lsm_grid_v67, render scripts.

**Action**: create a branch, commit the current WIP (DP-validation notebook/driver edits), then
`git stash apply` (expect conflicts in notebooks 3/5 — both modified in tree and stash), commit the
recovered CSVs immediately (they are gitignore-whitelisted but were never committed — the exact bug
`c13107c` fixed for the other CSVs). Everything below edits the *stashed* tex, not the on-disk one.

---

## 1. Sharpen/validate the DP numbers (science gaps)

### 1.1 One canonical publication sweep with per-cell error bands  ← biggest gap
Two DP grid sweeps exist and they **disagree at γ=1**:

| cell | RESULTS.md (121/121/151/24) | dp_desk_sweep.csv (81/81/101/16) | |Δ| |
|---|---:|---:|---:|
| c=0.04, γ=1 | 2.23859 | 2.23472 | **3.9e-3 (0.17%)** |
| c=0.15, γ=1 | 1.33060 | 1.33235 | 1.7e-3 |
| c=0.02, γ=1 | 2.44558 | 2.44690 | 1.3e-3 |
| γ ≥ 1.5 cells | — | — | ≤ 3e-4 (fine) |

The advertised accuracy (±1e-4 at the 81-grid, ±5.4e-5 at reference) was measured **at the focal
γ=2 cell only** and does not transfer to γ=1. Consequences if unfixed:
- At γ=1 the DP–LSM-D gap is 0.01–0.10% (Results 9), i.e. **the same order as the DP's own γ=1
  discretisation error** → the "DP ≥ every incumbent in all 28 cells" claim is not resolved at γ=1.
- At c=0.04, γ=1 the DP point estimate crosses LSM-D depending on grid choice.
- Zero-cost row: desk-sweep DP 2.66351 sits *below* the paper's LSM-D 2.6649 (±0.006 — covered by
  MC noise, but the point ordering flips; not publishable as is).
- `grid_dp_pricer/data/dp_grid_sweep.csv` (the 121-grid sweep RESULTS.md cites) is **not in the
  repo** — regenerate and commit; the desk sweep CSV lacks provenance columns (no X/Y ranges).

**Action**: one script → one committed CSV `dp_publication_sweep.csv` covering all 28 cells + c=0,
with per-cell **three-grid ladder** (e.g. 61/81/121 family, ~6 s/cell → minutes total), per-cell
Richardson-extrapolated value, observed order, and **GCI numerical-uncertainty band** (see §2), plus
full config provenance columns (nX,nY,nQ,Mx,X/Y ranges, FP64, threads, binary/commit hash).

### 1.2 γ=1 and c=0 cells: exact-lattice ("snap") DP
At γ=1 (and c=0) the per-step profit is linear in q ⇒ the optimum is bang-bang (q∈{0,q_max}),
so reachable Q states are exactly {0, 2, 4, …, 20} — an 11-point lattice. Running these cells with
Q snapped to the reachable lattice and actions restricted to {0, q_max} makes the Q-interpolation
error (the controlling first-order axis) **vanish identically**; residual error is spatial +
quadrature (~1e-6 per T1). This turns the weakest cells into the most exact ones, at trivial cost.
Validate: snap-DP vs fine-nQ Richardson limit agree within the spatial band.

### 1.3 Forward-MC self-consistency beyond the focal cell
T10 (focal): backward U₀ = 1.99032944 vs 100M-path forward MC, gap 1.5e-4 = 0.67 SE. Cheap
insurance for the reviewer: repeat at 2–3 more cells — one γ=1 cell (post-snap), one high-cost
corner (c=0.15, γ=3, where LSM-D is 5% off), the zero-cost row. Greedy-policy forward MC is a true
lower bound ⇒ "forward CI must sit at/below backward value" is the falsifiable check.

### 1.4 Fix the monotonicity claim
RESULTS.md §1.4 claims "price ↓ in γ" — **violated on the grid itself**: c=0.15 gives V(γ=2)=1.0659
< V(γ=3)=1.0997 (when optimal q<1, cq^γ *decreases* in γ). Scope the claim (holds where exercised
quantities ≥1 / at the focal c) or drop it; make sure neither T11 nor the paper text asserts global
γ-monotonicity. This is a nice economic observation, not a bug — say so.

### 1.5 Finish the in-flight T12 "breakdown regime" work
Working tree has uncommitted edits to `run_dp_validation.py` (+ notebook): coarse rungs (nXY 7–25),
`_mx_coupled` floor 6→2, unbalanced-floor 31→7. `results/T12_pareto.csv` is stale relative to the
driver. Finish, re-run, commit — the "where does it stop being trustworthy" boundary is a good
robustness sentence for the paper subsection. Keep the ladder **non-geometric** (per the earlier
finding that a geometric ladder flips the nQ-controlling conclusion) — which forces the
least-squares GCI variant (§2).

---

## 2. CI methodology — how to report DP "uncertainty" next to MC CIs

Literature answer (what referees will expect):

1. **PDE/lattice pricing papers do not report CIs.** Standard practice (Forsyth–Vetzal, d'Halluin
   et al.; Jaillet–Ronn–Tompaidis for swing) is a **convergence table**: value at successive
   refinements, successive differences, ratio of differences (⇒ observed order p), and the
   Richardson-extrapolated value quoted to converged digits.
2. **The formal numerical-uncertainty standard is the Grid Convergence Index** (Roache 1994,
   "Perspective: A Method for Uniform Reporting of Grid Refinement Studies", J. Fluids Eng.;
   codified in ASME V&V 20-2009): U_num = Fs·|δ_RE|/(r^p̂ − 1) with safety factor Fs = 1.25 for a
   three-grid study with observed order. T1 already computes this (`gci_pct`). Because the ladder
   is non-geometric, use the **least-squares GCI** procedure (Eça & Hoekstra, J. Comput. Phys.
   2014) for the per-cell bands.
3. **MC-side bracket (optional, reviewer-proof)**: for multiple-exercise options the primal–dual
   literature (Meinshausen–Hambly 2004 — already in the bibliography; Bender 2011; Schoenmakers
   2012; Andersen–Broadie 2004 for the CI convention) reports a *statistically valid* two-sided
   bracket [lower-bound CI, dual upper-bound CI]. You already have the primal leg (T10 forward MC).
   A dual upper bound would bracket V₀ independently of any grid argument — **optional**, flag as
   nice-to-have only if a referee pushes.

**Table convention to adopt** (resolves the transition cleanly):
- MC-based entries (LSM-D, AC-sample, AC-kernel): 95% BCa CIs — statistical uncertainty (unchanged).
- DP entry: point value with **U_num (GCI) band**, footnoted as *deterministic discretisation
  uncertainty, not a statistical CI*.
- One sentence closing the gap: U_num ≤ 1e-4 in value (~0.005%) is 1–2 orders below the MC CI
  half-widths (~6e-3, ~0.3%), so **gaps-to-DP inherit each method's own MC CI unchanged**, and the
  backward value is confirmed by an independent 100M-path forward-MC lower bound to within 0.7 SE.
- Precondition: §1.1/§1.2 must first make "U_num ≤ 1e-4" true **per cell** (currently false at γ=1).

---

## 3. Table 5 (`tab:results`) edit

Recommended minimal-risk design: **add one leading DP column** to the stashed three-way table:

```
c  γc | DP V* (±U_num) | LSM-D ±CI | AC-sample ±CI  Δ%[BCa] | AC-kernel ±CI  Δ%[BCa]
```

- Keep Δ% defined vs LSM-D (the paired per-seed BCa machinery stays intact — don't re-anchor the
  bootstrap).
- Caption gains two sentences: DP definition + U_num footnote (§2), and the observation the DP
  certifies: every method sits below the DP; AC-kernel's Δ%>0 cells are certified moves *toward*
  the optimum, LSM-D's shortfall grows with convexity (0.01% at γ=1 → 5.05% at c=0.15, γ=3).
- Distance-to-optimum per method belongs in the DP subsection (compact per-γ summary or the small
  table below), not as extra Table-5 columns (keeps width manageable — it's already `table*`).
- Update `tools/colorize_results_table.py` → emit the DP column from `dp_publication_sweep.csv`
  so the table stays script-generated end-to-end.
- Numbers change slightly everywhere (121-grid/extrapolated values replace desk-sweep values).

## 4. The DP subsection (new, in §Numerical Experiments)

Suggested title: *"A near-exact dynamic-programming reference"*. Content budget = 1–2 tables/figs:

- **Table D1 — head-to-head at the focal cell**: method | value (±uncertainty, typed: GCI vs BCa) |
  wall-clock | notes. Rows: DP (three accuracy tiers or just production+reference), LSM-D, AC-kernel,
  AC-sample. Timing footnote: M1, 8 threads, min-over-repeats, deterministic price (bit-identical
  across runs); state DP is model-specific (needs the HHK kernel) vs simulation-based incumbents —
  the fair-comparison caveat.
- **Figure D1 — two panels**: (a) convergence: per-axis error vs resolution with fitted orders
  (nQ ≈ 1 controlling, nY ≈ 4.75, Mx spectral) + Richardson band; (b) accuracy–time Pareto with
  LSM-D / AC-kernel / AC-sample overlaid. Source figs exist (`docs/figs/convergence.png`,
  `pareto.png`) but must be regenerated **publication-grade**: PDF, column width 3.30 in (or 6.85 in
  if full-width), font sizes matched to the manuscript, consistent method colors with the rest of
  the paper.
- Text: (i) method recap (2-D exact-transition grid + budget axis, quadrature kernel — point back
  to the §Classical Methods recursion, eq. grid_bellman); (ii) V&V summary: kernel parity ≤1e-10,
  limiting cases, forward-MC self-consistency, GCI bands; (iii) the certified findings: LSM-D bias
  grows with convexity; AC-kernel uniformly 0.13–0.30% from optimal; kernel-vs-sample gap certified
  against an exact reference; (iv) scope sentence: feasible because HHK is 2-factor — the DP is the
  testbed's ground truth, not a competitor at scale.

## 5. Ripple edits elsewhere in the tex (the part that's easy to miss)

1. **§Classical Methods**: "We do not implement the full HHK grid method in the numerical section
   below" — now **false**. Rewrite: we *do* implement it as the reference; forward-pointer to the
   DP subsection.
2. **"Why Continuous-Control RL is Attractive"** bullets: currently argue grid methods are
   impractical (dimensionality, jumps, continuous actions, convex costs) — the DP subsection
   *refutes the practical force of each* on this testbed (quadrature handles jumps; inner golden-
   section handles continuous actions & convex costs; 2-D grid is cheap). Reframe honestly around
   scalability + model-agnosticism, or Sven will.
3. **Abstract + intro contributions + conclusion**: add the reference-pricer contribution ("we
   certify the benchmark gaps against a near-exact DP reference"); the headline "AC-kernel beats
   LSM-D by up to +5.1%" gets its missing justification ("…and LSM-D is verifiably 5.05% below the
   true value there").
4. **Table 8 (`tab:lsm_grid`, LSM action-grid study)**: caption says LSM-D "converges upward to
   essentially the AC-kernel value" — can now say *converges toward the DP reference*; add DP line
   to the figure/caption. (This table was already flagged as stale in the Results-9 checklist —
   resolve both at once.)
5. **Reproducibility appendix**: DP manifest — binary commit, grid config incl. X/Y ranges, FP64,
   thread count, ctest suite (kernel parity, moments, limits), sweep CSV name, regeneration one-liner.
6. **Bibliography**: add Roache (1994); ASME V&V 20-2009; Eça & Hoekstra (2014). Optional: Bender
   (2011)/Schoenmakers (2012) if the dual-bracket remark is kept. Jaillet et al. and
   Meinshausen–Hambly are already in.
7. **Hedging §**: unaffected (no DP there), but re-verify its "consistent with Table 5" PV sentence
   still holds after the table gains the DP column.
8. **RESULTS.md / HPT.md**: sync the headline numbers to the publication sweep (they'll shift at
   γ=1), fix the γ-monotonicity sentence.

## 6. Definition of done

- [ ] Stash applied on a branch; Results 9 CSV + episode_efficiency.csv committed.
- [ ] `dp_publication_sweep.csv` committed: 29 cells × 3-grid ladder, Richardson value, observed
      order, least-squares GCI band, full provenance columns.
- [ ] γ=1 + c=0 cells re-priced on the exact Q-lattice (snap) and consistent with the ladder limit.
- [ ] Per-cell U_num ≤ ~1e-4 verified (γ=1 included) — the "DP treated as exact vs MC noise"
      footnote is then honest; DP ≥ incumbent orderings hold cell-by-cell **outside** the band.
- [ ] Forward-MC self-consistency at focal + γ=1 + high-c corner (+ c=0), all gaps ≲ 1 SE.
- [ ] T12 breakdown-regime edits finished, results regenerated, committed.
- [ ] Monotonicity claims scoped (γ non-monotone at high c — documented as economics, not bug).
- [ ] Table 5 regenerated by script with DP column + dual-uncertainty footnote.
- [ ] Figure D1 (convergence + Pareto) regenerated as publication PDFs at manuscript widths/fonts.
- [ ] Table D1 head-to-head with typed uncertainties + timing-methodology footnote.
- [ ] Ripple edits 1–8 of §5 done; `make paper` builds clean (check rc — it keeps a stale PDF on
      failure); colored-provenance convention applied to the new text.
