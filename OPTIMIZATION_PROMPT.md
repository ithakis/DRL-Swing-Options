# Optimization Prompt for Codex

**Role**

You are an experienced ML / RL engineer tasked with making a PyTorch-based swing option pricing training script significantly faster, without changing its numerical behavior or user-facing outputs.

**Non‑negotiable goal**

- For the command `bash runv26Profile.sh`, the **inputs, outputs, logs, and learned policy/metrics must remain effectively the same** (within normal floating‑point noise), but the **wall‑clock runtime should be reduced**.
- You may refactor and optimize code, but **do not change the swing contract definition, stochastic process, reward function, training schedule, or random seeding logic** in ways that alter the learned solution.

**Environment / how to run**

- Repo root: `DRL-Swing-Options`
- Main entry shell script: `runv26Profile.sh` (calls `python run.py "${args[@]}" -name "SwingOption_20_v26_11" -seed 11`)
- Conda env to use for any tests: `EP11`
- Always assume commands are run from the repo root:
  ```bash
  conda activate EP11
  bash runv26Profile.sh
  ```
- You are allowed to:
  - Run small, focused benchmarks in the terminal.
  - Add temporary micro‑benchmarks or timers and remove them after you’ve learned what you need.

**Constraints / preferences**

- Batch size:
  - I’ve experimented with `-bs` and found 64 is faster, but 128 gives slightly better training. **Prefer to keep `-bs=128` unless you can justify a change that maintains performance.**
- Network architecture:
  - Please **do not aggressively shrink network width/depth** just to gain speed; only consider safe micro‑architecture changes that preserve capacity.
- Algorithmic behavior:
  - Keep the D4PG/DDPG‑style training logic, PER, replay buffer behavior, and exploration schedule intact. You may **optimize implementations**, but **do not change algorithms** (e.g., don’t switch optimizers or remove PER).
- Acceptable changes:
  - Vectorization, using more efficient PyTorch APIs, avoiding redundant work, better use of batched operations, caching, fewer Python loops, less overhead inside the hot training loop, etc.
  - Safer library / PyTorch configuration changes (e.g. enabling/disabling specific features) that improve speed without changing results.

**Profiling summary (runv26.prof)**

- `bash runv26Profile.sh` with `cProfile` produced `runv26.prof`. A pstats summary (`runv26_profile_summary.txt`) shows:

Top 40 cumulative time:
```text
2594633961 function calls (2277966005 primitive calls) in 1657.263 seconds

Ordered by: cumulative time

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
5032/1    0.116    0.000 1657.269 1657.269 {built-in method builtins.exec}
     1    0.001    0.001 1657.269 1657.269 run.py:1(<module>)
     1    0.012    0.012 1652.967 1652.967 run.py:1353(main)
     1    6.378    6.378 1643.682 1643.682 run.py:1172(run_training)
348623    4.724    0.000  789.780    0.002 agent.py:292(step)
    17   17.307    1.018  725.744   42.691 run.py:846(evaluate_swing_option)
3271578   19.409    0.000  649.656    0.000 agent.py:274(act)
165312   12.574    0.000  497.552    0.003 agent.py:315(learn_)
71974716/6543156   68.519    0.000  383.997    0.000 module.py:2855(train)
41014460/4101446   21.466    0.000  324.807    0.000 module.py:1767(_wrapped_call_impl)
41014460/4101446   36.818    0.000  322.093    0.000 module.py:1775(_call_impl)
3603856   13.298    0.000  249.349    0.000 networks.py:210(forward)
8202892   14.077    0.000  239.243    0.000 container.py:243(forward)
165312    8.535    0.000  226.664    0.001 replay_buffer.py:417(sample)
71974840   90.945    0.000  221.359    0.000 module.py:1964(__setattr__)
3271578    1.104    0.000  194.196    0.000 module.py:2877(eval)
10476115    6.176    0.000  150.970    0.000 fromnumeric.py:51(_wrapfunc)
460885100/316935503   55.122    0.000  125.604    0.000 {built-in method builtins.isinstance}
330624    0.492    0.000  113.508    0.000 lr_scheduler.py:128(wrapper)
330624    2.321    0.000  113.016    0.000 optimizer.py:486(wrapper)
330624    1.509    0.000  100.025    0.000 optimizer.py:60(_use_grad)
330624    2.463    0.000   97.855    0.000 adam.py:213(step)
165312   94.475    0.001   94.475    0.001 replay_buffer.py:462(_update_probability_cache)
165312    0.480    0.000   93.261    0.001 replay_buffer.py:474(_update_cumsum_cache)
165312    0.153    0.000   92.755    0.001 fromnumeric.py:2879(cumsum)
165312   92.415    0.001   92.415    0.001 {method 'cumsum' of 'numpy.ndarray' objects}
12304338   15.357    0.000   92.379    0.000 linear.py:124(forward)
137406444   36.789    0.000   90.318    0.000 module.py:2750(children)
8202892   12.233    0.000   89.200    0.000 normalization.py:216(forward)
330624    0.569    0.000   88.858    0.000 _tensor.py:592(backward)
330624    1.637    0.000   88.260    0.000 __init__.py:243(backward)
661248    0.868    0.000   84.932    0.000 optimizer.py:131(maybe_fallback)
661248    2.099    0.000   83.793    0.000 adam.py:872(adam)
330624    0.763    0.000   82.016    0.000 graph.py:820(_engine_run_backward)
330624   80.987    0.000   80.987    0.000 {method 'run_backward' of 'torch._C._EngineBase' objects}
661248   30.957    0.000   76.843    0.000 adam.py:345(_single_tensor_adam)
8202892   11.339    0.000   72.770    0.000 functional.py:2898(layer_norm)
35075498   72.333    0.000   72.333    0.000 {built-in method builtins.round}
12304338   70.909    0.000   70.909    0.000 {built-in method torch._C._nn.linear}
3271578   22.255    0.000   68.605    0.000 swing_env.py:138(step)
```

Top 40 self time:
```text
Ordered by: internal time

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
165312   94.475    0.001   94.475    0.001 replay_buffer.py:462(_update_probability_cache)
165312   92.415    0.001   92.415    0.001 {method 'cumsum' of 'numpy.ndarray' objects}
71974840   90.945    0.000  221.359    0.000 module.py:1964(__setattr__)
330624   80.987    0.000   80.987    0.000 {method 'run_backward' of 'torch._C._EngineBase' objects}
35075498   72.333    0.000   72.333    0.000 {built-in method builtins.round}
12304338   70.909    0.000   70.909    0.000 {built-in method torch._C._nn.linear}
71974716/6543156   68.519    0.000  383.997    0.000 module.py:2855(train)
8202892   57.393    0.000   57.393    0.000 {built-in method torch.layer_norm}
460885100/316935503   55.122    0.000  125.604    0.000 {built-in method builtins.isinstance}
... (rest omitted for brevity)
```

Key hotspots:
- `agent.py:274(act)` and `agent.py:292(step)` + `agent.py:315(learn_)`
- Replay buffer operations, especially `replay_buffer.py:462(_update_probability_cache)` and `replay_buffer.py:474(_update_cumsum_cache)` (PER cumsum and `searchsorted`)
- Neural network forward/backward: `networks.py`, `linear.py:124`, `functional.layer_norm`, PyTorch `backward` and Adam optimizer
- `swing_env.py:138(step)` and related observation/action processing

**Machine / environment info**

Running:
```bash
python -V
python -c "import torch; print(torch.__version__)"
sysctl -n machdep.cpu.brand_string
```

Output:
```text
Python 3.11.13
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
OMP: Hint: multiple OpenMP runtimes detected (libomp.dylib). This can degrade performance or cause incorrect results.
[1]    5630 abort      python -c "import torch; print(torch.__version__)"
Apple M1
```

- CPU: Apple M1 (ARM64, macOS)
- PyTorch version currently causes an OpenMP duplicate runtime error when imported this way; part of your task is to diagnose whether OpenMP / BLAS configuration or specific library versions are hurting performance and propose safe fixes.

**What you should do**

1. **Analyze** the provided profile summary and project structure (especially `run.py`, `src/agent.py`, `src/replay_buffer.py`, `src/swing_env.py`, `src/networks.py`).
2. **Identify concrete, code‑level optimizations** that:
   - Reduce time spent in the identified hotspots (act/step/learn, replay buffer PER updates, NN forward/backward, layer norm, Adam, etc.).
   - Prefer algorithmically equivalent transforms (vectorization, caching, reduced allocations, better data layout, minimizing Python overhead).
3. **Suggest environment/library optimizations**:
   - Fix or work around the OpenMP duplicate runtime issue on Apple M1 in a way that **improves performance but does not compromise correctness**.
   - Consider whether a specific PyTorch version (for M1/Metal or CPU) would be faster and safe to adopt.
4. **Propose and implement code changes**:
   - Show diffs / patches for the key files.
   - For each change, explain **why it should speed things up** and **why it preserves behavior**.
5. **Design small, fast validation checks**:
   - Commands using `conda activate EP11` and short runs (e.g., reduced `-n_paths`) to verify:
     - Same reward distribution / option price and exercise statistics (within noise).
     - Reduced runtime compared to baseline.
   - Provide exact shell commands I can run to reproduce your tests.

Focus on a small number of **high‑impact, well‑reasoned optimizations** rather than micro‑tweaks everywhere.

---

You are now focusing specifically on using **Numba** to accelerate the **prioritized replay buffer** implementation, without bloating the code or breaking anything.

Context:
- The main file to touch is `src/replay_buffer.py`, especially the `PrioritizedReplay` class.
- The replay buffer already uses a **Fenwick tree (Binary Indexed Tree)** with:
  - `_fenwick_update`
  - `_fenwick_prefix_sum`
  - `_fenwick_find_prefix_index`
- Sampling (`sample`) currently:
  - Draws `random_vals` via NumPy.
  - Loops in Python over `random_vals` and calls `_fenwick_find_prefix_index` for each sample.
- The older bottleneck based on `np.cumsum`/`np.searchsorted` has been removed; the remaining hot work is the Fenwick tree logic and its loops.

Your job:

1. **Apply Numba only where it is clearly safe and beneficial.**
   - Restrict Numba to small, pure numerical helper functions operating on NumPy arrays and scalars.
   - Do not decorate any function that touches PyTorch tensors, Python objects, or has complex side effects; keep Numba strictly for the Fenwick tree and related math.

2. **Refactor Fenwick helpers into Numba-friendly functions.**
   - Extract the core Fenwick operations into standalone helpers that operate purely on arrays and scalars, for example:
     - A function that, given a Fenwick tree array, capacity, current size, and a mass value, returns the corresponding index (0-based) via prefix-sum search.
     - A function that rebuilds the Fenwick tree from priorities, `alpha`, `size`, and `capacity`.
     - Optionally, a function that performs a single-index Fenwick update (updating both the tree and the cached `priority**alpha` array), as long as it remains Numba-safe.
   - In `PrioritizedReplay.sample`, replace the Python loop over `random_vals` with a Numba-accelerated helper that:
     - Takes the tree and a NumPy array of random masses.
     - Returns a NumPy array of indices.
   - Keep the **public API and method signatures of `PrioritizedReplay` unchanged**.

3. **Numba decorators and options to use.**
   - Use `@numba.njit(cache=True, fastmath=False)` as the default decorator for these numerical helpers:
     - `cache=True` to avoid re-JITting on every run.
     - `fastmath=False` to preserve numerical behavior unless you can prove it is safe.
   - Do **not** use `parallel=True` / `prange` unless the loop is embarrassingly parallel and there is no cross-iteration dependency (and even then, keep it minimal).
   - Do not use `nogil` or other advanced options.

4. **Guarded import and fallbacks.**
   - At the top of `replay_buffer.py`, add a guarded import:
     ```python
     try:
         import numba
     except ImportError:
         numba = None
     ```
   - For each Numba helper, ensure there is a **pure-Python fallback** so the code still works if Numba is not installed:
     - Define the helper in plain Python first.
     - If `numba` is available, wrap or replace it with an `@numba.njit`-compiled version.
   - The replay buffer must behave correctly, and with the same API, whether or not Numba is present.

5. **Correctness and determinism.**
   - The sampling distribution and priority updates must remain effectively identical:
     - Same indices returned for the same priorities and RNG seeds (within floating-point noise).
     - Same behavior of `alpha`, `min_priority`, `priority_clip_pct`, and `beta` scheduling.
   - Do **not** change the RNG or seeding logic.
   - Preserve operation order where it can affect floating-point rounding.

6. **No file bloat, minimal changes.**
   - Do **not** blanket-apply Numba to many methods.
   - Only add a **small number of focused `@njit` helpers** around the hottest Fenwick-related code.
   - Avoid touching unrelated parts of `CircularReplayBuffer` and the n-step logic unless absolutely necessary.
   - Keep `replay_buffer.py` clean and readable; do not introduce large new abstractions or complex Numba plumbing.

7. **Verification mindset (no new test code required).**
   - Think about how to confirm, conceptually, that the Numba-accelerated helpers are drop-in replacements for the existing Python versions (same inputs, same outputs).
   - You do **not** need to add tests or scripts, but your changes should be easy to validate by running the existing training command and comparing behavior and runtime.

Overall goal:
- Introduce a **small, surgical set of Numba-`njit` helpers** to accelerate the Fenwick tree operations used by `PrioritizedReplay` and its sampling, without changing the external behavior or bloating the file. Only use Numba where you are confident it works in nopython mode and genuinely reduces CPU time.

---

## Phase 3: Post-Optimization Analysis & Continued Speed-Up

You have been given two cProfile outputs from the same training command (`bash runv26Profile.sh`):

| Profile file | Description |
|--------------|-------------|
| `runv26_profile_summary.txt` | **v1 baseline** – before optimizations |
| `runv26_v2_profile_summary.txt` | **v2** – after the Fenwick-tree PER rewrite and other changes |

### Task 1 – Comparative Analysis (report in chat)

Compare the two profiles and produce a concise analysis covering:

1. **Overall runtime**: v1 total vs v2 total; absolute seconds saved and percentage speedup.
2. **Per-hotspot deltas**: For the top ~10 functions by cumulative or self time in v1, show:
   - v1 time → v2 time
   - Δ seconds and Δ %
   - Brief explanation of *why* the change helped (or didn't).
3. **New hotspots**: Identify any functions that are now proportionally *more* expensive in v2 (even if faster in absolute terms) — these are your next targets.
4. **Summary table**: A markdown table with columns `Function | v1 cumtime | v2 cumtime | Δ s | Δ %`.

Present this analysis **in chat** before moving on.

---

### Task 2 – Continue Speeding Up the Algorithm

After completing Task 1, proceed with further optimizations. The non-negotiable constraint remains:

> **Same inputs → same outputs** (within floating-point noise). Do not alter the learned policy, reward structure, or RNG seeding.

#### 2.1 Methodology

For every remaining hotspot, systematically ask:

| Question | Action if "yes" |
|----------|-----------------|
| Can I use **Numba `@njit`** here? | Extract pure-numerical logic into a standalone helper; use `@njit(cache=True, fastmath=False)` (or `fastmath=True` if provably safe). Provide a pure-Python fallback. |
| Is there a **faster algorithm or data structure**? | Consider segment trees, vectorized NumPy, pre-allocated buffers, caching, lazy recomputation, etc. |
| Can I **batch / vectorize** instead of looping? | Replace Python `for` loops with NumPy or PyTorch batch ops. |
| Can I **parallelize** with multiple cores? | Use `joblib.Parallel`, `multiprocessing.Pool`, or `concurrent.futures` where the workload is embarrassingly parallel and GIL-free. On Apple M1 (8 cores), use `n_jobs=-1` or explicit core count. |
| Is there **redundant work** I can eliminate? | Cache intermediate results, skip unnecessary recomputation, hoist invariants out of loops. |

#### 2.2 Specific Targets (not exhaustive)

1. **Agent evaluation (`agent_evaluation.py`)**
   - `_evaluate_swing_batch` is now the single largest self-time function (~68 s).
   - `_build_state_batch` and `_feasible_actions` together add ~53 s.
   - Investigate:
     - Vectorizing state construction with pre-allocated arrays.
     - Using Numba for feasibility checks if they are pure NumPy.
     - Running evaluation batches **in parallel** across paths (e.g., `joblib.Parallel(n_jobs=-1)(delayed(...))`), provided RNG is handled correctly per worker.
   - If evaluation is called 17 times and takes ~367 s total, reducing per-call cost or parallelizing across paths can yield large wins.

2. **Replay buffer (`replay_buffer.py`)**
   - `_fenwick_update` is called ~19 M times (15.5 s self).
   - `sample` and `update_priorities` still show meaningful time (~18 s and ~29 s).
   - Ensure Numba helpers are actually being used (check import fallback).
   - Consider batching priority updates instead of per-index loops.

3. **Neural network forward/backward**
   - `torch._C._nn.linear` (66 s), `torch.layer_norm` (43 s), `run_backward` (83 s) are PyTorch internals; limited direct control.
   - Potential micro-optimizations:
     - Fuse LayerNorm + activation if PyTorch supports it.
     - Use `torch.compile` (PyTorch 2.x) with `mode="reduce-overhead"` for the actor/critic forward if safe.
     - Ensure `torch.no_grad()` wraps all inference-only paths.

4. **Soft update (`agent.py:soft_update`)**
   - 38 s cumulative; largely parameter iteration.
   - Replace Python loop with a single fused in-place op:
     ```python
     with torch.no_grad():
         for p, tp in zip(net.parameters(), target.parameters()):
             tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
     ```
     (may already be done; verify).

5. **Gradient clipping (`clip_grad.py`)**
   - ~45 s cumulative.
   - Ensure you're using `torch.nn.utils.clip_grad_norm_` (C++ fast path) and not a Python re-implementation.

6. **Miscellaneous**
   - `torch.quantile` is called ~445 K times (14 s). If used for priority clipping, consider caching or reducing call frequency.
   - `np.clip` / `_methods._clip` (~14 s). Batch where possible; consider Numba if inside tight loops.

#### 2.3 Numba Guidelines (reiterated)

- **Decorator**: `@numba.njit(cache=True, fastmath=False)` by default.
- **Parallelism**: Use `parallel=True` + `prange` only for independent iterations with no shared mutation.
- **Fallback**: Always keep a pure-Python version; check `if numba is None`.
- **Avoid**: Decorating anything that touches PyTorch tensors or complex Python objects.

#### 2.4 Parallelism Guidelines

- **joblib**: `from joblib import Parallel, delayed`
  ```python
  results = Parallel(n_jobs=-1, backend="loky")(
      delayed(func)(arg) for arg in args
  )
  ```
- **multiprocessing**: Use `Pool` with `fork` start method (careful on macOS; prefer `spawn` for safety but accept overhead).
- **Thread pool**: Only effective for IO-bound or GIL-releasing C extensions.
- **RNG safety**: When parallelizing Monte-Carlo paths, pass a unique seed or `np.random.SeedSequence` to each worker to avoid correlated streams.

#### 2.5 Deliverables

1. **Analysis report** (Task 1) in chat.
2. **Code patches** for each optimization, with:
   - Explanation of what it does and expected speedup.
   - Confirmation it preserves outputs.
3. **Validation commands**: short runs to verify correctness and measure new runtime.

---

### Reference: v1 vs v2 Profile Summaries

<details>
<summary><b>v1 (`runv26_profile_summary.txt`) – Top 40 cumulative</b></summary>

```
2594633961 function calls in 1657.263 seconds

   ncalls  tottime  cumtime  filename:lineno(function)
   ...
   165312   94.475   94.475  replay_buffer.py:462(_update_probability_cache)
   165312   92.415   92.415  {method 'cumsum' of 'numpy.ndarray' objects}
   ...
```
</details>

<details>
<summary><b>v2 (`runv26_v2_profile_summary.txt`) – Top 40 cumulative</b></summary>

```
1073054172 function calls in 981.694 seconds

   ncalls  tottime  cumtime  filename:lineno(function)
   ...
   139264   68.091  361.714  agent_evaluation.py:158(_evaluate_swing_batch)
  2510502   28.643   50.097  agent_evaluation.py:93(_build_state_batch)
  2510502   24.750   45.401  agent_evaluation.py:126(_feasible_actions)
   ...
```
</details>

Use these summaries to drive your analysis and next optimizations.

---


## New Codex-Max Task: Episode Termination & Net Payoff Feature

You now have two additional, targeted refactoring tasks. Apply them carefully so that training remains numerically consistent (up to normal stochastic noise) and all existing scripts (`run.py`, `runv*.sh`, `evaluate_agent.py`) keep working.

### 1) Remove Early Episode Termination When Quantity Is Exhausted

**Current behavior (what to locate and understand)**

- In the swing option RL environment (`src/swing_env.py`, plus any wrappers in `run.py` / `agent_evaluation.py`), an episode is terminated **early** once the agent has exercised the maximum total quantity `Q_max`.
- Concretely, there is logic equivalent to:
   - `terminated = (current_step >= contract.n_rights) or (q_exercised >= contract.Q_max - eps)`
   - or any other condition that ends the episode as soon as the global quantity constraint is fully used.

**Required new behavior**

- **Remove** any logic that terminates the episode **just because** `Q_max` is reached.
- Episodes must now **always run until the final time step** (e.g., `current_step >= contract.n_rights` or the equivalent horizon condition), regardless of how quickly the agent spends its quantity.
- After the remaining quantity is zero (up to a small numerical tolerance), the agent must effectively be forced to exercise **zero quantity** at every subsequent step, but still receive observations until the end of the horizon.

**Implementation details and constraints**

- Preserve all quantity constraints:
   - The environment must never allow `q_actual` to exceed `Q_max - q_exercised` at any step.
   - Once `Q_remaining <= 0`, any positive proposed action must be **clipped/masked to 0**, and the realized `q_actual` must be exactly `0.0`.
- Make the behavior explicit in the environment:
   - Centralize feasibility in `_get_feasible_action` (or the equivalent helper in `src/swing_env.py`):
      - If `Q_remaining <= 0`, enforce `q_feasible = 0.0` regardless of the proposed action.
- Termination:
   - The **only** time-related termination condition should be the horizon, e.g. `current_step >= contract.n_rights`.
   - Do **not** reintroduce early stopping based on global quantity anywhere else (environment, agent, training loop, or evaluation code).

**Checks / invariants to maintain**

- For any simulated path where quantity is exhausted before maturity:
   - `env.step(...)` continues to be called up to the final time index.
   - For all steps after exhaustion: `q_actual = 0.0` and `Q_remaining = 0.0` (within tolerance).
- The cumulative exercised quantity at episode end is still `<= Q_max` (within tolerance).
- Public interfaces (`env.step`, `env.reset`, calls from `run.py`, `evaluate_agent.py`) should not need signature changes; only the termination logic and feasible-action enforcement change.

### 2) Add Explicit Gross vs Net Payoff Features for the RL Agent

**Current behavior (what exists already)**

- The environment already computes a per-step payoff decomposition in `src/swing_env.py` via `calculate_standardized_reward(...)`, which returns:
   - discounted reward,
   - **gross payoff** (e.g. `q_t * max(S_t - K, 0)`),
   - **exercise cost** (e.g. convex cost `c * q_t^gamma`),
   - **net payoff** (`gross_payoff - exercise_cost`).
- These values are used for reward calculation and logging, but the **state vector** given to the RL agent does not explicitly expose **both** gross and net payoff as distinct state features.

**New requirement**

- Extend the RL observation / feature space so that **two separate features** are always present in the state:
   1. A **gross payoff feature** (per-step payoff without costs).
   2. A **net payoff feature** (per-step payoff including costs).
- Make this distinction clear and consistent across the codebase (variable names, comments, and documentation).

**Implementation steps**

1. **Use `calculate_standardized_reward` as the single source of truth**
    - Keep `calculate_standardized_reward(...)` in `src/swing_env.py` as the canonical place where:
       - `gross_payoff`, `exercise_cost`, and `net_payoff` are computed.
    - Do **not** duplicate payoff logic elsewhere; reuse these outputs when building state features.

2. **Extend the observation vector in `SwingOptionEnv`**

    In `src/swing_env.py`:

    - Currently, `_get_observation()` builds a state vector like:

       ```python
       state = np.array([
             spot_price - contract.strike,
             q_exercised / contract.Q_max,
             q_remaining / contract.Q_max,
             time_to_maturity / contract.maturity,
             normalized_time,
             spot_price,
             X_t,
             Y_t,
             days_since_exercise / contract.n_rights,
       ], dtype=np.float32)
       ```

    - Extend this state to include **two additional entries**:
       - `last_gross_payoff` (per-step, without costs).
       - `last_net_payoff` (per-step, with costs).
    - Maintain these as environment attributes, for example:
       - Initialize `self.last_gross_payoff = 0.0` and `self.last_net_payoff = 0.0` in `reset`.
       - After each `step`, when `calculate_standardized_reward` is called, update:
          - `self.last_gross_payoff = gross_payoff`
          - `self.last_net_payoff = net_payoff`
    - On the first step after `reset`, these values should be `0.0`.
    - Update `state_dim` and `observation_space` accordingly to reflect the two new features.

3. **Propagate the new state dimension**

    - Wherever `state_size` is inferred or passed (e.g. in `run.py`, `agent_evaluation.py`, or other helpers creating the `Agent` and networks), update the logic to pull the state dimension directly from the environment:
       - e.g., `state_size = env.observation_space.shape[0]`.
    - Ensure that the `Actor` and `Critic` in `src/networks.py` are instantiated with the new `state_size` without needing further structural change in their architectures.

4. **Clarify gross vs net usage in the codebase**

    - Search for ambiguous use of “payoff” or “reward” in state features and logs.
    - Where applicable:
       - Rename variables to `gross_payoff` / `net_payoff` instead of generic `payoff` when they refer specifically to one or the other.
       - Update comments and docstrings so it is always clear whether a quantity is gross (no costs) or net (with costs).
    - Do not change the economic meaning of any payoff or cost; just make the naming explicit.

5. **Update `README.md`**

    - In `README.md`, update or add a section describing the RL state / feature space.
    - Explicitly list all features, including the new ones, in order. For example (adapt to the actual ordering you implement):

       - `S_t - K` (spot minus strike),
       - normalized cumulative exercise,
       - normalized remaining capacity,
       - normalized time to maturity,
       - normalized current time index,
       - `S_t` (spot price),
       - `X_t` (mean-reverting factor),
       - `Y_t` (jump factor),
       - normalized `days_since_last_exercise`,
       - **gross payoff** at the last step (0 at the first step),
       - **net payoff** at the last step (0 at the first step).

    - Add a short sentence explicitly stating:
       - The RL agent uses **one feature for the gross payoff (without costs) and one feature for the net payoff (after costs)**, both computed from the same payoff/cost formula used by the pricing logic.

**Validation expectations**

- Training and evaluation commands (e.g. `bash runv26Profile.sh`, `python run.py ...`, `python evaluate_agent.py ...`) still run end-to-end without changes to their CLI arguments.
- For episodes where the agent exhausts quantity early:
   - Episode length is now determined solely by the time horizon, not by quantity exhaustion.
   - Post-exhaustion steps have `q_actual = 0.0`, `gross_payoff = 0.0`, `net_payoff = 0.0`, but still produce valid observations until the last step.
- The new payoff features:
   - Are zero on steps with `q_t = 0`.
   - Reflect the correct gross and net payoffs on steps where `q_t > 0`.
- Overall pricing outputs (Monte Carlo value estimates, distribution of total discounted payoffs) remain consistent with the previous implementation up to normal Monte Carlo / floating-point noise.

Apply these two tasks as **small, surgical refactors** focused on `src/swing_env.py`, any state-building helpers in evaluation/training code, and `README.md`. Do not introduce new conceptual algorithms or change the economic meaning of the swing option – just adjust termination logic and expose the gross/net payoffs as explicit features.

