# Deep RL Mid/Late-Stage Tuning Prompt

## Role

You are a senior deep RL researcher with strong experience in distributional actor–critic methods (D4PG/TD3/DDPG), prioritized replay, and continuous‑control RL in noisy, heavy‑tailed environments (e.g., finance and energy).

You are working inside the `DRL-Swing-Options` repository, which implements a D4PG‑style agent for pricing swing options under the Hambly–Howison–Kluge (HHK) process.

---

## High‑Level Goal

Diagnose and improve **mid‑ to late‑stage training** (roughly episodes 20k–32k) of the current best agent so that:

- The **in‑sample training performance** (`Average100`: mean episode return over last 100 episodes) keeps improving or at least does not degrade late in training.
- The **out‑of‑sample performance** (`delta_percent`: RL vs LSM relative price delta) is:
  - as close to 0 as possible (slight negative is acceptable),
  - **stable across seeds** (low seed‑to‑seed variance late in training),
  - and does **not deteriorate after ~20k–25k episodes** as it currently does.

Observed behavior from current experiments:

- **In‑sample (Average100)**  
  - Ramps up rapidly and then **starts to flatten and mildly degrade after ~25k episodes**.
- **Out‑of‑sample (delta_percent)**  
  - Improves during early and mid training but **tends to worsen and become more seed‑sensitive starting around ~20k–25k episodes**, suggesting late‑stage instability or overfitting.

You should assume:

- **Gray curves** in the provided TensorBoard screenshots are **version 1 (v26)** with multiple seeds.
- **Pink curves** are **version 2 (v26_SiLU / 131k)** with improvements aimed at fixing action variance collapse. These changes address action‑variance collapse but both versions still show mid/late‑stage issues (flattening or degradation in performance and increased seed variance in `delta_percent`).

Your job is to:

1. Understand the current algorithm and schedules.
2. Carefully analyze the existing training curves.
3. Form deep, evidence‑backed hypotheses for what goes wrong in the **20k–32k episode regime**, with particular focus on:
   - `Average100` degradation after ~25k.
   - `delta_percent` degradation and variance increases after ~20k–25k.
4. Design concrete diagnostics and short experiments to confirm/deny these hypotheses.
5. Propose targeted hyperparameter/algorithm tweaks for **mid/late‑stage fine‑tuning** only (keep early‑stage behavior as close as possible).
6. End with a clear, prioritized TL;DR action plan.

---

## Files / Repo Context

Start by reading:

- `README.md` – high‑level overview of the problem, algorithm, and CLI flags.
- `HPT.md` – chronological hyperparameter‑tuning log (v1–v31), with conclusions about PER, noise schedules, LR schedules, and where v26 currently sits.

Core implementation:

- `src/agent.py` – D4PG/DDPG‑style agent, PER scheduling, LR schedulers, noise/epsilon schedule, diagnostics logging.
- `src/networks.py` – actor/critic/IQN architectures and activations.
- `src/replay_buffer.py` – `PrioritizedReplay` implementation and its Fenwick‑tree / priority statistics.
- `src/swing_env.py`, `src/swing_contract.py`, `src/simulate_hhk_spot.py` – environment dynamics, reward structure, and HHK process.

Experiment scripts:

- `runv26.sh` – baseline v26 configuration (ReLU, 2×64, batch 128, PER schedule, noise/LR schedule, etc.).
- `runv26SiLUinit.sh` – variant with `--activation=silu` (pink runs), same overall hyperparameters but different activation and evaluation paths.

Evaluation / results:

- `evaluate_agent.py` – how pricing and `delta_percent` are computed.
- `results.md` – example RL vs LSM comparisons and `delta_percent` behavior.

Logs and runs for concrete evidence:

- `logs/SwingOption_20_v26_*` and `logs/SwingOption_20_v26_131k_*`
- `runs/SwingOption_20_v26_*` and `runs/SwingOption_20_v26_131k_*`

Assume TensorBoard scalar screenshots are available for at least:

- `Average100`
- `Exploration/Epsilon`, `Exploration/Noise_Scale`, `Exploration/Plateau_Active`
- `Policy/Action_variance_mean`, `Policy/Actions_at_lower_pct`, `Policy/Actions_at_upper_pct`
- `Critic_loss`, `Actor_loss`
- `PER/priority_entropy`, `PER/priority_max`, `PER/priority_mean`, `PER/priority_min`, `PER/priority_std`
- `TD_Error/p50`, `TD_Error/p90`, `TD_Error/p99`
- `Stability/Target_drift`
- `nstep/bootstrap_mask_mean`, `nstep/done_mean`

---

## Phenomenon to Explain

From the graphs and prior notes (README/HPT + screenshots):

- **Training return (`Average100`)**
  - In both versions, `Average100` ramps up quickly and then **flattens**, with signs of **mild degradation after ~25k episodes**.
- **Out‑of‑sample performance (`delta_percent`)**
  - After ~20k–25k episodes, RL–LSM price delta (`delta_percent`) is **not steadily improving**; for some seeds it worsens, and the **seed‑to‑seed spread widens**, indicating instability or overfitting.
- **Exploration / action statistics**
  - Exploration (`Epsilon`, `Noise_Scale`, `Plateau_Active`) decays according to the v26 schedule; noise floor and epsilon are small by the time we enter the 20k–32k episode regime.
  - `Policy/Action_variance_mean` and `Policy/Actions_at_upper_pct` differ between gray (v26) and pink (v26_SiLU). The pink version fixes action‑variance collapse but may show **reduced variance and more conservative behavior** late in training.
- **Losses / stability**
  - `Critic_loss` peaks early then decays; `Actor_loss` becomes more negative and then flattens.
  - `TD_Error/*` percentiles and `PER/priority_*` show how PER focuses on high‑TD transitions; their late‑stage behavior suggests **strong or changing emphasis** on a subset of experiences.
  - `Stability/Target_drift` generally decreases, but its late‑stage values differ between gray and pink versions.

Treat these as **hard evidence** and cross‑reference them with the implementation and hyperparameter schedules.

---

## What to Investigate

### 1. Deep Understanding of Current Training Dynamics

Use the repo and logs to build a precise picture of how training actually runs:

- **Algorithm & schedules**
  - How `Agent` configures:
    - PER: `per_alpha`, `per_beta_start`, `per_beta_frames`, `per_alpha_final`, `per_alpha_ramp_start`, `per_alpha_ramp_end`, `per_beta_final`, `per_alpha_sigmoid`.
    - Noise and epsilon schedules: `noise_sigma`, `noise_anneal_power`, `noise_plateau`, `min_action_noise`, `epsilon`, `epsilon_decay`, `get_noise_scale()` in `Agent`.
    - LR schedules: `final_lr_fraction`, `warmup_frac`, `min_lr`, and the `LambdaLR` schedulers in `Agent.__init__`.
    - Regularization knobs still present: gradient clipping, weight decay, target policy smoothing; action L2 has been removed (handled instead by activations and noise floors).
  - For v26 vs v26_SiLU:
    - Precisely what changed (activation function, eval paths, any subtle side effects).
    - How those changes manifest in the logged metrics (especially action variance, priorities, and losses).

- **Mapping episodes ↔ steps**
  - Understand the mapping from episodes (`-n_paths`) to timesteps (x‑axis in most TensorBoard plots) given 22 decision points per episode (`--n_rights=22`) and `-learn_every=2`.
  - Identify **what “early”, “mid”, and “late” mean** in terms of both episodes and environment steps (e.g., early: <5k episodes; mid: 5k–20k; late: >20k).

### 2. CLI‑Based Inspection of Schedules

Explicitly compute the **actual numeric values** of noise and learning rate across training for the v26/v26_SiLU runs.

Use the CLI and a short Python script to inspect values at representative episodes:

- Episodes to probe: e.g. `0, 1000, 3000, 5000, 8000, 12000, 16000, 20000, 24000, 28000, 32000`.
- For each probed episode:
  - `epsilon_t`, `noise_scale_t = agent.get_noise_scale()`.
  - Actor and critic learning rates from the LR schedulers.
  - PER `alpha`/`beta` as actually applied inside `PrioritizedReplay` (based on the ramp logic in `Agent._maybe_update_per_schedule()`).

The **key requirement** is:

> Use the CLI to **print concrete numeric values of noise, epsilon, LR, and PER hyperparameters at early/mid/late episodes** for v26 and v26_SiLU to understand whether **insufficient exploration or too‑low learning rates** are contributing to the observed degradation after ~20k–25k episodes.

You may implement this either by instantiating `Agent` with the v26 arguments or by directly evaluating the schedule formulas in `agent.py`.

### 3. Detailed Analysis of Metrics and Behavior

Given the graphs plus logs in `logs/` and `runs/`, analyze:

- **Action statistics**
  - How do `Policy/Action_variance_mean` and `Policy/Actions_at_upper_pct` evolve over time in gray vs pink runs?
  - Are pink runs becoming too conservative late (variance dipping, fewer boundary actions than optimal), or still flirting with boundary lock‑in?
  - Compare these to changes in `Average100` and `delta_percent` to understand whether high or low variance correlates with good pricing.

- **PER dynamics**
  - Use `PER/priority_*` and `TD_Error/*` to understand how PER behaves mid/late:
    - Are `priority_max` and `priority_std` spiking (too strong focus on a few rare transitions)?
    - Is `priority_mean` drifting up/down in a way consistent with over‑fitting or under‑training?
  - Relate these patterns to the PER schedule (alpha/beta ramp) and LR/epsilon schedules.

- **Losses and stability**
  - `Critic_loss` and `Actor_loss` trends: are we seeing signs of:
    - Over‑confident critic (very low loss, low TD error percentiles) combined with under‑exploration?
    - Under‑training (loss not decaying, high variance)?
  - `Stability/Target_drift`: how does drift change mid/late? Does target drift correlate with deteriorating `delta_percent`?

- **In‑sample vs out‑of‑sample**
  - Using `evaluate_agent.py` and `results.md`, inspect how `Average100` vs `delta_percent` move around key checkpoints (e.g., 10k, 20k, 26k, 30k+ episodes).
  - Look for evidence of **overfitting to the training distribution**, e.g., `Average100` still creeping up slightly while `delta_percent` worsens or becomes more seed‑sensitive.

### 4. Literature / Theory Research

Conduct targeted research on:

- **RL in stochastic and heavy‑tailed reward settings**, particularly:
  - Over‑estimation/under‑estimation issues and the “deadly triad”.
  - How high variance and rare large rewards interact with PER and n‑step learning.
- **Mid‑ / late‑stage fine‑tuning for actor–critic methods**:
  - Strategies for adjusting PER, LR, and exploration late in training.
  - Known pathologies of Adam/AdamW in tiny‑LR regimes and with strong weight decay.
- **Action‑space regularization and boundary behavior** in continuous control, especially in finance/energy settings.

For each relevant idea from the literature, explicitly map it back to:

- which part of our code or schedule it touches (PER, LR, noise, action regularization, target updates), and
- what specific failure signature it would leave in the logged metrics.

---

## Deliverables

Produce a **structured report** with the following sections:

1. **Context recap (brief)**  
   - Summarize the problem, algorithm, and key hyperparameters (v26 + v26_SiLU), relying on `README.md`, `HPT.md`, and the run scripts.  
   - Clarify what “gray” vs “pink” runs are and which logs they correspond to.

2. **Observed behavior and metric‑level diagnosis**  
   - Describe, with references to specific curves, what happens in:
     - Early training (<5k episodes),
     - Mid training (5k–20k episodes),
     - Late training (20k–32k episodes), with emphasis on:
       - `Average100` flattening/degradation after ~25k.
       - `delta_percent` degradation and variance growth starting around ~20k–25k.
   - Highlight how seed‑to‑seed variance behaves in this regime.
   - Explain what the PER, TD error, action variance, and target‑drift metrics suggest about the critic, policy, and replay buffer at each stage.

3. **Most probable failure modes (ranked)**  
   For each hypothesis, provide:
   - A short name (e.g., “LR too low late”, “PER over‑focusing on stale tails”, “exploration floor too low”, “critic over‑confidence + under‑exploration”, “AdamW + weight decay over‑regularizing late”, “overfitting to specific exercise patterns”).
   - Mechanism: why this failure mode is plausible in this environment.
   - Predicted signatures in our metrics.
   - Evidence from the actual graphs/logs supporting or contradicting it.
   - Confidence level (high/medium/low).

4. **Diagnostics and tests to run**  
   For each main hypothesis, propose **concrete, cheap tests** that can be run from the CLI, preferably with shortened runs (e.g., 5k–10k episodes) or by re‑analyzing existing logs:

   - Schedule probes: numeric inspection of noise/LR/PER at selected episodes (as above).
   - PER ablations: v26‑like runs with:
     - PER frozen at low alpha/beta after a certain episode,
     - PER turned off after 20k–25k episodes while keeping the same replay buffer contents.
   - LR tweaks: runs where LR decays more slowly in the last third of training, or where LR is partially “re‑warmed” after 20k–25k episodes.
   - Exploration tweaks: modest bumps to `min_action_noise` after a certain episode or a small epsilon reset/plateau mid‑run.
   - Regularization tweaks: adjusting weight decay or gradient clipping for actor/critic to change late‑stage bias/variance (action L2 is retired).

   For each proposed test:
   - Specify which file(s) to edit (`runv26.sh`, `runv26SiLUinit.sh`, or `Agent`), and exactly which flags/lines to change.
   - Define what metrics and behavior would confirm or reject the hypothesis.

5. **Candidate Mid/Late‑Stage Tuning Strategies**  
   Based on your analysis, recommend a small set of **concrete tuning strategies** for mid/late training (episodes > ~15k), such as:

   - PER: e.g., “keep alpha≈0 until 7k, ramp to ~0.35 by 15k, then **freeze or gently decay** alpha after 20k–25k; adjust beta accordingly.”
   - LR: e.g., “increase `final_lr_fraction` and/or add a second‑phase slow decay or LR floor from 20k onward; avoid decaying below some minimum for the actor.”
   - Noise: e.g., “raise `min_action_noise` slightly after 20k–25k or introduce a second small plateau; tie this to observed action variance.”
   - Regularization: e.g., “tune weight decay or mild gradient clipping; rely on tanh01/SILU for variance control instead of action L2.”
   - Any algorithmic tweaks justified by evidence (e.g., mild changes to target smoothing, per‑phase PER behavior), but **avoid wholesale algorithm swaps** (no entirely new algorithms; focus on refining the existing D4PG/DDPG setup).

   For each strategy, specify:
   - Which metrics should improve (and in what direction).
   - Possible risks or trade‑offs.

6. **TL;DR Action Plan (Prioritized)**  
   End with a short, prioritized checklist, for example:

   - Step 1: Quick schedule diagnostics (noise/LR/PER) at selected episodes; summarize numeric results.
   - Step 2: Run 2–3 carefully chosen ablation experiments (describe them precisely with command lines and changed flags) focusing on the 20k–32k regime.
   - Step 3: Select the best mid/late‑stage schedule tweak based on `Average100` and `delta_percent` across seeds.
   - Step 4: Confirm stability with a full 32k‑episode run for the best candidate settings.

   The TL;DR should be **immediately actionable**: concrete hyperparameter changes and short run commands that can be executed, along with what to look for in the resulting TensorBoard plots to judge success, especially around **20k–32k episodes** where the current degradation appears.

---

## Style and Expectations

- Be **technical and concrete**, grounded in both the code and the graphs.
- When you speculate, **label it as a hypothesis** and tie it to specific evidence.
- Favor **small, staged changes** and diagnostics over sweeping rewrites.
- Assume you have time for deep thinking and careful analysis; cite insights from relevant RL literature when useful, but keep the final recommendations focused and practical.
