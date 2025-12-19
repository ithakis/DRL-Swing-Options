# LAP v49 (Loss-Adjusted Priorities) — Implementation Prompt for Codex

Copy/paste this entire file as a single prompt for Codex.

---

You are GPT-5.2 running in Codex CLI inside `DRL-Swing-Options`. Implement **Loss-Adjusted Priorities (LAP)** for Prioritized Experience Replay with **minimal, easily removable code changes**, keeping the code concise and quant-friendly.

<design_and_scope_constraints>
- Implement EXACTLY what I request; no unrelated refactors, no new features.
- Keep changes localized: prefer `src/agent.py` (+ small `run.py` arg plumbing) over new modules.
- Backward-compatible: default behavior must remain identical unless LAP is enabled via a new flag.
- Reduce bloat: avoid new classes/files; add at most one small helper function if needed.
</design_and_scope_constraints>

<context_you_must_use>
- PER base priorities are computed in `src/agent.py`:
  - standard critic: `Agent.learn_()` currently sets `priorities = td.abs().clamp_min(1e-6)` and calls `self.memory.update_priorities(idx, ...)`.
  - IQN critic: `Agent.learn_distribution()` currently sets priority from mean abs TD.
- Sampling exponent `alpha` is applied inside `src/replay_buffer.py` (`PrioritizedReplay` stores “base priorities” and uses `priority**alpha` in the Fenwick tree).
  => Therefore: to implement LAP, change the *base priority* passed to `update_priorities()`.
- CLI hyperparams are defined in `run.py` (PER flags around the existing `--per_alpha`, `--per_beta_*`, `--per_priority_floor`, `--per_priority_clip_pct`).
- `runv48.sh` is the current experiment; create `runv49.sh` by copying it and adjusting only what’s necessary.
</context_you_must_use>

## Goal: LAP (Loss-Adjusted Priorities)

Implement Fujimoto et al. (NeurIPS 2020) style LAP priorities:

- Let TD error per sample be `δ`.
- Define **Huber loss** (threshold `κ`):
  - `L(δ; κ) = 0.5*δ^2` if `|δ| <= κ`
  - `L(δ; κ) = κ*(|δ| - 0.5*κ)` otherwise
- Set **base priority** to `p_base = max(L(δ; κ), floor)`
- Replay already applies exponent: sampling mass ∝ `p_base**alpha`
- Keep the existing IS weights behavior unchanged.

### Why this matches our repo

Because `PrioritizedReplay` applies `**alpha` internally, implementing LAP means we only change `priorities` computed in `src/agent.py` before calling `update_priorities()`.

---

## Deliverable A — Code: LAP toggle + κ parameter

### 1) Add CLI flags in `run.py`

Add:
- `--per_priority_scheme` with choices `standard` (default) and `lap`
- `--per_huber_kappa` float, default `1.0` (only used when scheme=`lap`)

Keep existing flags (already present):
- `--per_priority_floor` (default `1e-6`)
- `--per_priority_clip_pct` (default `0.0`)

Important: preserve current defaults so existing runs are unchanged.

### 2) Plumb new args into `Agent(...)` construction in `run.py`

When building the `Agent` in `run.py`, pass:
- `per_priority_scheme=args.per_priority_scheme`
- `per_huber_kappa=args.per_huber_kappa`

Also ensure the existing `per_priority_clip_pct` is actually used (see below).

### 3) Implement LAP priority transform in `src/agent.py`

In `Agent.__init__`, store:
- `self.per_priority_scheme`
- `self.per_huber_kappa`
- `self.per_priority_clip_pct`

Add ONE small helper (preferred) inside `Agent` (or module-level) used by both `learn_` and `learn_distribution`, e.g.:
- `_compute_base_priorities(td_error, *, floor, clip_pct, scheme, huber_kappa) -> torch.Tensor`

Rules:
- `standard`: `abs(td_error)`
- `lap`: `calculate_huber_loss(td_error, k=huber_kappa)` (this repo already has `calculate_huber_loss` in `src/agent.py`)
- Always detach priorities from the graph.
- Always apply the configured floor via `clamp_min(floor)` (floor is `--per_priority_floor` already wired to `self.memory.min_priority`; but you must also clamp the per-batch tensor before calling `update_priorities()` so the numpy passed is consistent).
- Make priorities finite: use `torch.nan_to_num(priorities, nan=floor, posinf=some_large, neginf=floor)` and then clamp.
- Implement `--per_priority_clip_pct` (currently parsed but unused): if `clip_pct > 0`, do a **per-batch** percentile clip:
  - `thr = torch.quantile(priorities.flatten(), clip_pct/100.0)`
  - `priorities = priorities.clamp_max(thr)`
  - Keep it simple; batch size is small (e.g. 128), so quantile cost is acceptable.
- Keep behavior identical when scheme=`standard` and `clip_pct=0` (default).

### 4) IQN / distributional case

In `learn_distribution()`:
- If `per_priority_scheme == "lap"`:
  - Prefer using **the per-sample quantile loss already computed** (`quantile_loss`, shape `[B]`) as the base priority, because it directly represents the critic’s loss.
  - Apply floor + optional percentile clip the same way.
- If you want the absolute-minimum change, it’s acceptable to apply LAP to the existing scalar TD proxy, but prefer quantile loss because it’s already available and more principled.

---

## Deliverable B — Docs: `HPT.md` v49 entry (LAP)

Append a new section `## v49: LAP (Loss-Adjusted Priorities)` to `HPT.md` that includes, in quant-friendly language:

- What changed in v49 (LAP priorities; how it differs from standard PER).
- What is LAP (definition + Huber loss + priority floor).
- Why we’re trying it in this repo (heavy-tailed TD errors in swing options; PER can oversample outliers; LAP aims to reduce gradient variance / stabilize training).
- What we expect to gain (lower TD tail percentiles, fewer critic loss spikes, lower seed variance in `Pricing/Delta_Percent`, more stable exercise stats).
- Best initial parameters to try (for this repo):
  - `--per_priority_scheme=lap`
  - `--per_huber_kappa=1.0`
  - `--per_priority_floor=1.0` (LAP-style floor; prevents low-loss transitions from being under-sampled)
  - keep v48’s PER schedule initially to isolate the effect (don’t retune everything at once)
  - set `--per_priority_clip_pct=0` initially (Huber already tames tails); only re-enable if you still see spikes
- What tuning might be needed:
  - `κ` too small: priorities flatten early; PER effect vanishes; learning may slow.
  - `κ` too large: behaves closer to squared-loss PER; outlier dominance can return.
  - floor too high: priorities concentrate near the floor; replay becomes near-uniform.
  - floor too low: low-loss transitions can become too rare; can increase overfitting to “hard” samples.
- How to diagnose misadjusted LAP params using existing logs:
  - `TD_Error/p99` and `Critic_loss` spikes
  - `PER/priority_std` rising sharply (outlier dominance)
  - seed divergence in `Pricing/Delta_Percent`
- Performance penalty:
  - extra elementwise ops + a small per-batch `torch.quantile` only when `--per_priority_clip_pct>0`
  - LAP without clipping should be negligible overhead vs baseline
- Include the primary scientific reference:
  - Fujimoto et al., NeurIPS 2020 (Loss-Adjusted Prioritized Experience Replay)

Keep the writing concise and “what-to-do / what-to-look-at” oriented.

---

## Deliverable C — New run script: `runv49.sh`

Create `runv49.sh` by copying `runv48.sh` and making minimal edits:

- Update the header comment: `# v49: v48 + LAP priorities (...)`
- Add LAP flags near the existing PER block with inline comments, e.g.:
  - `--per_priority_scheme=lap      # Use Loss-Adjusted Priorities (Huberized loss-based PER)`
  - `--per_huber_kappa=1.0          # Huber threshold κ (clips loss growth for large TD errors)`
  - `--per_priority_floor=1.0       # LAP floor: ensures p_base >= 1 (prevents under-sampling low-loss transitions)`
  - `--per_priority_clip_pct=0.0    # Disable extra clipping initially; Huber already tames tails`
- Keep the rest of v48 identical to isolate LAP impact.
- Update run names from `v48` to `v49` (e.g., `SwingOption_20_v49_11`, etc.).

---

## Acceptance criteria (must pass)

- With default flags (`--per_priority_scheme=standard`), behavior is unchanged vs current master.
- With LAP enabled:
  - priorities passed to `update_priorities()` are finite, non-negative, and `>= --per_priority_floor`
  - `run.py --help` shows the new flags and they are wired through to training
- `--per_priority_clip_pct` actually affects priorities now (it is currently unused; fix this as part of the change).
- Code remains easy to remove: LAP is behind a single flag and localized to priority computation.

---

## Minimal validation

Run:
- `python -m py_compile run.py src/agent.py src/replay_buffer.py`
- A short smoke run (tiny episodes) with LAP enabled to confirm no crashes and priorities update:
  - `python run.py ... --per_priority_scheme=lap --per_huber_kappa=1.0 --per_priority_floor=1.0 --per_priority_clip_pct=0`

---

## Output format

After implementing, respond with:
- bullets: “What changed”, “Where”, “How to run v49”
- no long explanations, no pasted full files.

