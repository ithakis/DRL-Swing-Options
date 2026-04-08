---
name: paper-figure-regen
description: "Use when regenerating or editing paper figures from Jupyter Notebooks, especially the three manuscript figures produced by 6: Convex costs 0.04 Analysis.ipynb. Covers the minimum safe rerun path, which cells to rerun after edits, and when Bang-Bangness or path_stats dependencies must be refreshed."
---

# Paper Figure Regeneration

Use this skill when working on the paper figures generated from `Jupyter Notebooks/6: Convex costs 0.04 Analysis.ipynb`.

## Preconditions

- Configure the notebook kernel before running cells.
- Keep changes limited to figure-generation work unless the user explicitly asks for data or benchmark regeneration.
- If the request is styling-only, rerun only the smallest safe subset of cells.

## Figure Outputs

- Figure 1: `hist_exercise.pdf`
- Figure 2: `spot_income_pv_hist.png`
- Figure 3: `bang_bangness_rl.pdf`

## Minimum Reliable Rerun Path From A Fresh Kernel

1. Run cells 2 through 14 to load paths, helper functions, evaluation data, derived path statistics, and figure labels.
2. For Figure 1, run cell 22.
3. For Figure 2, run cell 25 after the setup cells.
- For Figure 3, run cell 27 to load Bang-Bangness values from `Jupyter Notebooks/Convex Costs Results 7.csv`, then run cell 28.

## Smallest Safe Rerun Paths With State Already Loaded

- Figure 1 only: rerun cell 22.
- Figure 2 only: rerun cell 25 unless `path_stats` was invalidated; if it was, rerun cells 11 through 14 first.
- Figure 3 only: rerun cell 28 for styling-only edits; rerun cells 27 and 28 if the Bang-Bangness data changed.

## After Editing Specific Figure Cells

- If you edit the Figure 1 cell only, rerun cell 22 only.
- If you edit the Figure 2 cell only, rerun cell 25 only as long as cells 2 through 14 are still valid in the current kernel.
- If you edit the Bang-Bangness data update cell only, rerun cells 27 and 28.
- If you edit the Figure 3 plotting cell only, rerun cell 28 only as long as the Bang-Bangness CSV is already current.
- If you edit any setup cell feeding `path_stats`, rerun cells 11 through 14 and then cell 25.

## Practical Guardrails

- Prefer the smallest rerun set that matches the scope of the edit.
- Do not regenerate unrelated figures or rerun the full experiment sweep when the task is only about paper figures.
- If the figure depends on refreshed benchmark-derived data, state that clearly before rerunning the dependent plotting cells.
