# Copilot Instructions for DRL-Swing-Options

## Purpose

Research-grade implementation of **D4PG** (Deep Deterministic Policy Gradient) for pricing **swing options** in energy markets. The algorithm prices options with **convex exercise costs** using the **Hambly-Howison-Kluge (HHK)** stochastic process and benchmarks against **Least-Squares Monte Carlo (LSM)**.

## Agent Execution Guardrails

- Prefer named repo commands (`make ...`) or VS Code tasks over reconstructing long shell commands from memory.
- Do not run the full convex-cost sweep unless the user explicitly asks for it.
- Use the currently selected Python environment; do not assume a machine-specific absolute path.
- Keep searches focused on source, tools, docs, and paper files unless the task explicitly targets generated outputs in `logs/`, `runs/`, or notebook artifacts.

---

## Color Legend (manuscript diffs)

When reviewing the manuscript we annotate inserted text with colors to indicate provenance. Use the following legend when giving feedback:

- **Purple**: Text inserted specifically to address Sven's review comments (benchmark strength, state-space, profitability-gate scope, or reporting bias). If you want to give targeted feedback on reviewer-driven edits, quote or reference the purple text.
- **Orange**: New statistical methodology and results added in the current revision (evaluation protocol, Results table orange columns, Seed Robustness section). Use `\textcolor{revisionorange}{...}` (defined as `\definecolor{revisionorange}{RGB}{179,90,0}`).
- **Green**: New material or analysis added independently of Sven's comments (e.g., new figures, metrics, or explanatory paragraphs). These are broader content additions rather than direct responses.
- **Red**: Sven's original inline reviewer comments preserved in the draft for traceability.

When you want to provide feedback on a purple passage, please either (a) paste the purple sentence(s) into your message, or (b) specify the file and line range (for example: [Paper/DRL_Swing_Options.tex](Paper/DRL_Swing_Options.tex#L320-L330)). This makes it easy to locate the targeted fragment and apply edits.

---

## Running Experiments

### Main Experiment Sweep
```bash
make sweep
```
This runs individual scripts from `Convex Cost Experiments/` (e.g., `SwingOption_20_c0.05_gamma2.sh`).

### Single Experiment
```bash
make single-exp EXP_SCRIPT="Convex Cost Experiments/SwingOption_20_c0.05_gamma2.sh"
```

### Direct Python Training
```bash
make train ARGS='-name "Test" -seed 42 -n_paths 8192 --c_cost 0.05 --gamma_cost 2 [other args]'
```

### Evaluation
```bash
make eval RUN_NAME=<name> RUNS=100
```

### TensorBoard
```bash
make tensorboard
```

---

## Codebase Map

| File | Purpose |
|------|---------|
| `run.py` | Training orchestration: CLI parsing, env/agent setup, training loop, evaluation |
| `src/agent.py` | D4PG agent: networks, PER, noise schedules, optimizer configuration |
| `src/swing_env.py` | Gymnasium environment: state representation, reward calculation, contract constraints |
| `src/networks.py` | Neural architectures: Actor (profitability-gated), Critic, IQN |
| `src/lsm_swing_pricer.py` | LSM benchmarks: discretized-action ($M$-level) Longstaff-Schwartz with volume-based DP |
| `src/simulate_hhk_spot.py` | HHK simulation: 2-factor OU with jumps, Sobol sequence |
| `src/swing_contract.py` | Contract specification: exercise bounds, refraction, discounting |
| `src/replay_buffer.py` | Replay buffers: Circular and PER with Fenwick tree |
| `tools/compare_lsm_state_modes.py` | Compare reduced-state vs full-state LSM outputs and summary metrics |
| `tools/update_convex_costs_results.py` | Refresh convex-cost result tables/CSVs after benchmark reruns |
| `tools/rebuild_results_v7.py` | Regenerate `Convex Costs Results 7.csv` and `Convex Costs Results 7 focal.csv` from logs |
| `tools/generate_seed_robustness_figure.py` | Generate Figure 4 (seed robustness strip+box plot) from the focal results CSV |

### Paper Layout
- Manuscript sources live in `Paper/`
- Main paper entrypoint: `Paper/DRL_Swing_Options.tex`
- `Paper/sven.tex` preserves Sven's original red-comment review version; when working on manuscript claims, tables, or figures, treat Sven's feedback there as active review context we are trying to address
- Paper-local assets: `Paper/Bibliography.bib`, `Paper/cas-dc.cls`, `Paper/cas-common.sty`, `Paper/model1-num-names.bst`
- Generated paper artifacts live in `Paper/build/`
- Figures remain in `figs/` and are referenced from the manuscript via `../figs/...`

---

## Default Configuration (Convex Cost Experiments)

### Swing Contract
- Strike: `K = 1.0`
- Maturity: `~1 month (0.0833 years)`
- Decision dates: `n_rights = 22`
- Exercise bounds: `q ∈ [0, 2.0]` per period, `Q ∈ [0, 20.0]` total
- Risk-free rate: `r = 5%`

### HHK Process
- `S0 = 1.0`, `alpha = 12.0`, `sigma = 1.2`
- `beta = 150.0`, `lambda = 6.0`, `mu_J = 0.3`

### Convex Costs (experiment sweep)
- `c_cost ∈ {0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.15}`
- `gamma_cost ∈ {1, 1.5, 2, 3}`

### Training
- Episodes: `32768`, Batch: `128`, Seeds: `{11, 12, 13}` (standard sweep)
- **Focal robustness study**: c=0.04, γ=2 uses seeds 11–25 (15 seeds); see `Convex Cost Experiments/SwingOption_20_c0.04_gamma2_focal.sh`
- PER: soft alpha ramp (0.1 → 0.2), beta ≈ 1.0
- Noise: pre-squash Gaussian, plateau + hyperbolic decay
- Networks: 2×64 SiLU MLPs, LayerNorm, orthogonal init

---

## Output Locations

| Path | Contents |
|------|----------|
| `logs/<run>/` | Training/evaluation CSVs, parquet files |
| `runs/<name>.pth` | Saved actor weights |
| `runs/<name>.json` | Run hyperparameters |
| `Jupyter Notebooks/Convex Costs Results 7.csv` | Aggregated results (standard 3-seed sweep) |
| `Jupyter Notebooks/Convex Costs Results 7 focal.csv` | Focal 15-seed results for c=0.04, γ_c=2 |
| `Jupyter Notebooks/Convex Costs Results *.csv` | Earlier result snapshots (1–6) |

---

## Key CLI Flags

**Contract**: `--strike`, `--maturity`, `--n_rights`, `--q_min`, `--q_max`, `--Q_max`, `--c_cost`, `--gamma_cost`

**Training**: `-n_paths`, `-eval_every`, `-n_paths_eval`, `-seed`, `-name`

**Agent**: `-per`, `--per_alpha`, `-noise_sigma0`, `-noise_floor`, `-lr_a`, `-lr_c`, `-layer_size`, `--activation`

**LSM**: `--lsm_basis`, `--lsm_degree`, `--lsm_reg`

**System**: `-n_cores`, `--compile`, `--disable_csv_logging`

---

## Development Resources

| Resource | Purpose |
|----------|---------|
| `HPT.md` | Hyperparameter tuning history (v1 → v62) |
| `Jupyter Notebooks/` | Validation and results analysis |
| `results.md` | Quick results summary |
| `Convex Cost Experiments/` | Pre-configured experiment scripts |

### Paper Build
```bash
make paper
```
This compiles in a temporary directory and copies the final artifacts to `Paper/build/`.

### Paper Clean
```bash
make clean-paper
```

### Paper Figure Regeneration
The paper has **four figures**:
- **Figures 1–3** (HHK paths, main results grid, Bang-Bangness) are generated from `Jupyter Notebooks/6: Convex costs 0.04 Analysis.ipynb`.
- **Figure 4** (Seed Robustness strip+box plot) is generated from the standalone script:
  ```bash
  /path/to/conda run -p /path/to/EP11 python tools/generate_seed_robustness_figure.py
  ```
  or via `make` if a target is added.

Use the `paper-figure-regen` skill for the detailed notebook choreography, minimum rerun paths, and figure-specific rebuild rules. Keep the global instructions focused on the codebase and invoke the skill when the task is specifically about figure regeneration.

---

## Critical Notes

1. **Profitability gate**: Actor enforces `Pi(q) > 0` before exercise (STE for gradients)
2. **Reward includes discounting**: Use `gamma = 1` in D4PG
3. **LSM uses out-of-sample evaluation**: Separate train/test datasets
4. **Pre-squash noise**: Exploration noise added before tanh01/beta-sigmoid squashing
5. **LSM convex cost caveat**: Both terminal and non-terminal steps must gate on **net profitability** (`payoff_net > 0`), not just ITM (`payoff_gross > 0`). With `c > 0`, exercising can be unprofitable even when in-the-money, so the net-payoff gate must be applied consistently throughout the backward induction.
6. **Bang-Bangness metric**: `B = fraction of exercises at q_max`. Decreases with γ. Column `RL_BangBangness_mean` in `Convex Costs Results 7.csv`. In-text estimates: ~66% at γ=1, ~21% at γ=2.
7. **Results 7 is canonical**: The current paper uses `Convex Costs Results 7.csv` (not the older Results 1–6). Regenerate with `tools/rebuild_results_v7.py`.
