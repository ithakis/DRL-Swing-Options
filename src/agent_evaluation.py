from __future__ import annotations

import cProfile
import os
import pstats
import time
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None  # torch is expected to be available in normal runs

from .swing_env import SwingOptionEnv


@dataclass
class EvaluationSummary:
    """Container for evaluation results."""

    returns: List[float]
    exercise_stats: List[Dict[str, Any]]
    option_price: float
    price_std: float
    confidence_95: float
    avg_total_exercised: float
    avg_exercise_count: float
    min_return: float
    max_return: float
    n_paths: int
    csv_filename: Optional[str] = None


def signed_zero_aware_pct_change(initial: float, new: float) -> float:
    """
    Compute the signed, zero-aware percentage change between an initial and new value.
    """
    a = np.asarray(initial, dtype=float)
    b = np.asarray(new, dtype=float)
    a, b = np.broadcast_arrays(a, b)

    result = np.empty_like(a, dtype=float)

    zero_mask = a == 0.0
    both_zero_mask = zero_mask & (b == 0.0)
    nonzero_mask = ~zero_mask
    new_zero_mask = nonzero_mask & (b == 0.0)
    same_sign_mask = nonzero_mask & (np.sign(a) == np.sign(b)) & ~new_zero_mask

    result[both_zero_mask] = 0.0

    inf_mask = zero_mask & ~both_zero_mask
    if np.any(inf_mask):
        result[inf_mask] = np.sign(b[inf_mask]) * np.inf

    if np.any(new_zero_mask):
        result[new_zero_mask] = -np.sign(a[new_zero_mask]) * 100.0

    if np.any(same_sign_mask):
        result[same_sign_mask] = ((b - a) / a * 100.0)[same_sign_mask]

    cross_mask = nonzero_mask & ~same_sign_mask & ~new_zero_mask
    if np.any(cross_mask):
        result[cross_mask] = (
            np.sign(b[cross_mask])
            * (np.abs(b[cross_mask]) + np.abs(a[cross_mask]))
            / np.abs(a[cross_mask])
            * 100.0
        )

    rounded = np.round(result, 1)
    return float(rounded) if rounded.shape == () else rounded


def _round_eval_rows(rows: List[List[float]], decimals: int = 6) -> List[List[float]]:
    """Vectorized rounding for evaluation CSV rows."""
    if not rows:
        return rows
    arr = np.asarray(rows, dtype=np.float64)
    float_start = 2  # path and time_step stay integers
    np.around(arr[:, float_start:], decimals=decimals, out=arr[:, float_start:])
    rounded_rows: List[List[float]] = []
    for r in arr:
        rounded_rows.append([int(r[0]), int(r[1]), *r[float_start:].tolist()])
    return rounded_rows


def _build_state_batch(
    contract,
    S: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    path_ids: np.ndarray,
    steps: np.ndarray,
    q_exercised: np.ndarray,
    last_exercise_step: np.ndarray,
) -> np.ndarray:
    """Construct state observations matching SwingOptionEnv._get_observation for a batch."""
    batch_size = len(path_ids)
    spot_price = S[path_ids, steps]
    q_remaining = contract.Q_max - q_exercised
    time_to_maturity = (contract.n_rights - steps) * contract.dt
    normalized_time = steps / contract.n_rights
    days_since_exercise = np.where(last_exercise_step >= 0, steps - last_exercise_step, steps)

    state = np.empty((batch_size, 9), dtype=S.dtype)
    state[:, 0] = spot_price - contract.strike
    state[:, 1] = q_exercised / contract.Q_max
    state[:, 2] = q_remaining / contract.Q_max
    state[:, 3] = time_to_maturity / contract.maturity
    state[:, 4] = normalized_time
    state[:, 5] = spot_price
    state[:, 6] = X[path_ids, steps]
    state[:, 7] = Y[path_ids, steps]
    state[:, 8] = days_since_exercise / contract.n_rights
    return state


def _feasible_actions(
    contract,
    current_step: np.ndarray,
    q_exercised: np.ndarray,
    q_proposed: np.ndarray,
    last_exercise_step: np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of SwingOptionEnv._get_feasible_action."""
    q_feasible = np.clip(q_proposed, contract.q_min, contract.q_max)
    max_allowed = contract.Q_max - q_exercised
    q_feasible = np.minimum(q_feasible, max_allowed)

    if contract.min_refraction_periods > 0:
        refraction_mask = (last_exercise_step >= 0) & (
            current_step - last_exercise_step <= contract.min_refraction_periods
        )
        q_feasible[refraction_mask] = 0.0

    remaining_steps = contract.n_rights - current_step - 1
    if np.any(remaining_steps > 0):
        min_needed_later = np.maximum(0.0, contract.Q_min - q_exercised - q_feasible)
        max_possible_later = contract.q_max * np.maximum(remaining_steps, 0)
        need_more_mask = (remaining_steps > 0) & (min_needed_later > max_possible_later)
        if np.any(need_more_mask):
            required_now = min_needed_later - max_possible_later
            adjusted = np.maximum(q_feasible, required_now)
            adjusted = np.minimum(adjusted, contract.q_max)
            q_feasible = np.where(need_more_mask, adjusted, q_feasible)

    return np.maximum(q_feasible, 0.0)


def _evaluate_swing_batch(
    agent,
    contract,
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_indices: Sequence[int],
    *,
    collect_path_data: bool,
) -> Tuple[List[float], List[Dict[str, Any]], List[List[float]]]:
    """Evaluate a batch of paths with batched policy inference."""
    _, S, X, Y = dataset
    batch_indices_arr = np.asarray(batch_indices, dtype=np.int64)
    batch_size = len(batch_indices_arr)

    current_step = np.zeros(batch_size, dtype=np.int64)
    step_counts = np.zeros(batch_size, dtype=np.int64)
    q_exercised = np.zeros(batch_size, dtype=np.float64)
    last_exercise_step = np.full(batch_size, -1, dtype=np.int64)
    episode_return = np.zeros(batch_size, dtype=np.float64)
    total_exercised = np.zeros(batch_size, dtype=np.float64)
    exercise_count = np.zeros(batch_size, dtype=np.int64)
    done = np.zeros(batch_size, dtype=bool)

    log_rows: List[np.ndarray] = [] if collect_path_data else []

    while not np.all(done):
        active_idx = np.nonzero(~done)[0]
        if active_idx.size == 0:
            break

        active_paths = batch_indices_arr[active_idx]
        active_steps = current_step[active_idx]
        active_q_exercised = q_exercised[active_idx]
        active_last_ex = last_exercise_step[active_idx]

        state_batch = _build_state_batch(
            contract, S, X, Y, active_paths, active_steps, active_q_exercised, active_last_ex
        )

        # Batched policy inference (adds no_grad guard even though Agent.act uses inference_mode)
        if torch is not None:
            with torch.no_grad():
                actions = agent.act(state_batch, add_noise=False)
        else:
            actions = agent.act(state_batch, add_noise=False)

        actions = np.clip(np.asarray(actions).reshape(len(active_idx), -1), 0.0, 1.0).squeeze(-1)

        q_proposed = contract.q_min + actions * (contract.q_max - contract.q_min)
        q_actual = _feasible_actions(contract, active_steps, active_q_exercised, q_proposed, active_last_ex)

        spot_price = state_batch[:, 5].astype(np.float64)
        payoff_per_unit = np.maximum(spot_price - contract.strike, 0.0)
        gross_payoff = q_actual * payoff_per_unit
        exercise_cost = contract.c_cost * np.power(q_actual, contract.gamma_cost)
        net_payoff = gross_payoff - exercise_cost
        discounted_reward = np.power(contract.discount_factor, active_steps) * net_payoff

        non_positive_mask = net_payoff <= 0.0
        if np.any(non_positive_mask):
            discounted_reward[non_positive_mask] = 0.0
            gross_payoff[non_positive_mask] = 0.0
            exercise_cost[non_positive_mask] = 0.0
            q_actual[non_positive_mask] = 0.0

        exercised_mask = q_actual > 1e-6
        last_exercise_step[active_idx[exercised_mask]] = active_steps[exercised_mask]

        if collect_path_data:
            # Log step data per path before state advances
            step_numbers = step_counts[active_idx]
            rows = np.column_stack(
                (
                    active_paths,
                    step_numbers,
                    state_batch,
                    q_actual,
                    exercise_cost,
                    discounted_reward,
                )
            )
            log_rows.append(rows)

        # Apply environment transitions
        q_exercised[active_idx] += q_actual
        episode_return[active_idx] += discounted_reward
        total_exercised[active_idx] += q_actual
        exercise_count[active_idx] += exercised_mask.astype(np.int64)
        step_counts[active_idx] += 1
        current_step[active_idx] += 1

        done_now = (current_step[active_idx] >= contract.n_rights) | (
            q_exercised[active_idx] >= contract.Q_max - 1e-6
        )
        done[active_idx] = done_now

    discounted_returns: List[float] = []
    exercise_stats: List[Dict[str, Any]] = []
    all_path_data: List[List[float]] = []

    if collect_path_data and log_rows:
        stacked_rows = np.concatenate(log_rows, axis=0)
        order = np.lexsort((stacked_rows[:, 1], stacked_rows[:, 0]))
        all_path_data.extend(stacked_rows[order].tolist())

    for local_idx in range(batch_size):
        discounted_returns.append(float(episode_return[local_idx]))
        exercise_stats.append({
            "total_exercised": float(total_exercised[local_idx]),
            "exercise_count": int(exercise_count[local_idx]),
            "steps": int(step_counts[local_idx]),
        })

    return discounted_returns, exercise_stats, all_path_data


def _write_eval_parquet(
    parquet_filepath: str,
    columns: Sequence[str],
    rows: List[List[float]],
    parquet_writer: Optional[Any],
) -> None:
    """Write evaluation Parquet either asynchronously (if provided) or synchronously."""
    rounded_data = _round_eval_rows(rows)
    if not rounded_data:
        return

    if parquet_writer:
        parquet_writer.write_parquet(parquet_filepath, rounded_data, columns)
        return

    try:
        df = pd.DataFrame(rounded_data, columns=columns)
        df.to_parquet(
            parquet_filepath,
            index=False,
            engine="pyarrow",
            compression="zstd",
            compression_level=22,
        )
    except Exception as e:
        print(f"Warning: Failed to write Parquet file {parquet_filepath}: {e}")


def evaluate_agent(
    agent,
    eval_env,
    writer,
    path: Optional[int],
    evaluations_dir: Optional[str],
    lsm_price: Optional[float],
    parquet_writer: Optional[Any] = None,
    eval_batch_size: int = 1,
    profile: bool = False,
    profile_dir: str = "profilelogs",
    n_episodes: Optional[int] = None,
) -> EvaluationSummary:
    """
    Unified evaluation entry point.

    If eval_env is a SwingOptionEnv, the optimized batched evaluator is used.
    Otherwise, a generic Gym-style rollout is performed.
    lsm_price is expected to be computed out-of-sample using frozen estimators.
    """

    def _run():
        if isinstance(eval_env, SwingOptionEnv):
            return _evaluate_swing_agent(
                agent=agent,
                eval_env=eval_env,
                writer=writer,
                path=path,
                evaluations_dir=evaluations_dir,
                lsm_price=lsm_price,
                parquet_writer=parquet_writer,
                eval_batch_size=eval_batch_size,
            )
        return _evaluate_generic_env(agent, eval_env, writer, path, evaluations_dir, n_episodes=n_episodes)

    if not profile:
        return _run()

    os.makedirs(profile_dir, exist_ok=True)
    prof = cProfile.Profile()
    summary: EvaluationSummary = prof.runcall(_run)

    profile_file = os.path.join(profile_dir, f"eval_profile_{path if path is not None else 'standalone'}.prof")
    prof.dump_stats(profile_file)
    stream = StringIO()
    pstats.Stats(prof, stream=stream).sort_stats("cumtime").print_stats(25)
    print("Evaluation profile (top 25 by cumtime):")
    print(stream.getvalue())
    print(f"Profile saved to {profile_file}")
    return summary


def _evaluate_swing_agent(
    agent,
    eval_env: SwingOptionEnv,
    writer,
    path: Optional[int],
    evaluations_dir: Optional[str],
    lsm_price: Optional[float],
    parquet_writer: Optional[Any],
    eval_batch_size: int,
) -> EvaluationSummary:
    """Batched evaluation for SwingOptionEnv."""
    # Perf note (eval_benchmark, default contract, 20 eval paths on CPU):
    #   eval_batch_size=1  -> ~0.50s total (0.025s/path)
    #   eval_batch_size=4  -> ~0.24s total (0.012s/path) ~2.1x faster
    parquet_columns = [
        "path",
        "time_step",
        "spot_minus_strike",
        "q_exercised_norm",
        "q_remaining_norm",
        "time_to_maturity_norm",
        "normalized_time",
        "spot",
        "X_t",
        "Y_t",
        "days_since_exercise_norm",
        "q_t",
        "exercise_cost",
        "reward",
    ]

    dataset = (eval_env.t, eval_env.S, eval_env.X, eval_env.Y)
    n_paths = eval_env.S.shape[0]
    batch_size = max(1, min(int(eval_batch_size), n_paths))

    discounted_returns: List[float] = []
    exercise_stats: List[Dict[str, Any]] = []
    collect_path_data = bool(evaluations_dir)
    all_path_data: List[List[float]] = [] if collect_path_data and not parquet_writer else []
    csv_filename = None
    parquet_filepath = None
    if evaluations_dir and collect_path_data:
        os.makedirs(evaluations_dir, exist_ok=True)
        csv_filename = f"rl_episode_{path if path is not None else 0}.parquet"
        parquet_filepath = os.path.join(evaluations_dir, csv_filename)
        if parquet_writer and os.path.exists(parquet_filepath):
            os.remove(parquet_filepath)

    for start in range(0, n_paths, batch_size):
        end = min(start + batch_size, n_paths)
        batch_indices = list(range(start, end))
        batch_returns, batch_stats, batch_rows = _evaluate_swing_batch(
            agent,
            eval_env.contract,
            dataset,
            batch_indices,
            collect_path_data=collect_path_data,
        )
        discounted_returns.extend(batch_returns)
        exercise_stats.extend(batch_stats)
        if collect_path_data:
            if parquet_writer and parquet_filepath:
                if batch_rows:
                    parquet_writer.write_parquet(parquet_filepath, batch_rows, parquet_columns)
            else:
                all_path_data.extend(batch_rows)

    eval_env._episode_counter = -1

    option_price = float(np.mean(discounted_returns))
    price_std = float(np.std(discounted_returns))
    avg_exercised = float(np.mean([s["total_exercised"] for s in exercise_stats]))
    avg_exercises = float(np.mean([s["exercise_count"] for s in exercise_stats]))
    confidence_95 = float(1.96 * price_std / np.sqrt(n_paths))
    if evaluations_dir and collect_path_data:
        if parquet_writer and parquet_filepath:
            parquet_writer.write_parquet(parquet_filepath, None, None)
        elif parquet_filepath:
            _write_eval_parquet(parquet_filepath, parquet_columns, all_path_data, parquet_writer)

    summary = EvaluationSummary(
        returns=discounted_returns,
        exercise_stats=exercise_stats,
        option_price=option_price,
        price_std=price_std,
        confidence_95=confidence_95,
        avg_total_exercised=avg_exercised,
        avg_exercise_count=avg_exercises,
        min_return=float(np.min(discounted_returns)),
        max_return=float(np.max(discounted_returns)),
        n_paths=n_paths,
        csv_filename=csv_filename,
    )

    if writer is not None and path is not None:
        writer.add_scalar("Swing_Option_Price", option_price, path)
        writer.add_scalar("Price_Std", price_std, path)
        writer.add_scalar("Avg_Total_Exercised", avg_exercised, path)
        writer.add_scalar("Avg_Exercise_Count", avg_exercises, path)

        if lsm_price is not None:
            price_delta = option_price - float(lsm_price)
            price_delta_pct = signed_zero_aware_pct_change(float(lsm_price), option_price)
            writer.add_scalar("Pricing/RL_Price", option_price, path)
            writer.add_scalar("Pricing/LSM_Price", float(lsm_price), path)
            writer.add_scalar("Pricing/Delta_Price", price_delta, path)
            writer.add_scalar("Pricing/Delta_Percent", price_delta_pct, path)

    if path is not None:
        print(f"\n{'=' * 50}")
        print(f"EVALUATION RESULTS (Episode {path})")
        print(f"{'=' * 50}")
        print(f"Option Price: {option_price:.3f} ± {confidence_95:.3f}")
        print(f"Price Std Dev: {price_std:.3f}")
        print(f"Avg Total Exercised: {avg_exercised:.3f}")
        print(f"Avg Exercise Count: {avg_exercises:.1f}")
        print(f"Evaluation Runs: {n_paths}")
        print(f"Min Return: {summary.min_return:.3f}")
        print(f"Max Return: {summary.max_return:.3f}")
        if csv_filename:
            suffix = " (async)" if parquet_writer else ""
            print(f"Parquet saved: {csv_filename}{suffix}")
        print(f"{'=' * 50}")

    return summary


def _evaluate_generic_env(
    agent, eval_env, writer, path: Optional[int], evaluations_dir: Optional[str], n_episodes: Optional[int]
) -> EvaluationSummary:
    """
    Generic fallback evaluator for non-swing environments.
    """
    discounted_returns: List[float] = []
    exercise_stats: List[Dict[str, Any]] = []

    n_paths = n_episodes if n_episodes is not None else getattr(eval_env, "n_paths_eval", None) or 0
    n_paths = n_paths if n_paths > 0 else 1

    action_low = None
    action_high = None
    try:
        if hasattr(eval_env, "action_space"):
            action_low = float(np.asarray(eval_env.action_space.low).min())
            action_high = float(np.asarray(eval_env.action_space.high).max())
    except Exception:
        action_low = None
        action_high = None
    for _ in range(n_paths):
        state, _ = eval_env.reset()
        rewards = 0.0
        steps = 0
        exercise_count = 0

        while True:
            action = agent.act(np.expand_dims(state, axis=0))
            if action_low is not None and action_high is not None:
                action_v = np.clip(action, action_low, action_high)
            else:
                action_v = np.clip(action, -1.0, 1.0)
            state, reward, terminated, truncated, _ = eval_env.step(action_v[0])
            done = terminated or truncated
            rewards += float(reward)
            steps += 1
            if done:
                break

        discounted_returns.append(rewards)
        exercise_stats.append({"total_exercised": 0.0, "exercise_count": exercise_count, "steps": steps})

    option_price = float(np.mean(discounted_returns))
    price_std = float(np.std(discounted_returns))
    confidence_95 = float(1.96 * price_std / np.sqrt(len(discounted_returns)))

    return EvaluationSummary(
        returns=discounted_returns,
        exercise_stats=exercise_stats,
        option_price=option_price,
        price_std=price_std,
        confidence_95=confidence_95,
        avg_total_exercised=float(np.mean([s["total_exercised"] for s in exercise_stats])),
        avg_exercise_count=float(np.mean([s["exercise_count"] for s in exercise_stats])),
        min_return=float(np.min(discounted_returns)),
        max_return=float(np.max(discounted_returns)),
        n_paths=len(discounted_returns),
        csv_filename=None,
    )


def benchmark_evaluation(
    agent,
    eval_env,
    writer,
    evaluations_dir: Optional[str],
    lsm_price: Optional[float],
    parquet_writer: Optional[Any],
    eval_batch_size: int = 1,
    profile: bool = False,
    profile_dir: str = "profilelogs",
) -> Tuple[EvaluationSummary, float]:
    """
    Run a standalone evaluation benchmark and return timing.

    Timing excludes model loading; caller should construct the agent and environment first.
    """
    start = time.perf_counter()
    summary = evaluate_agent(
        agent=agent,
        eval_env=eval_env,
        writer=writer,
        path=0,
        evaluations_dir=evaluations_dir,
        lsm_price=lsm_price,
        parquet_writer=parquet_writer,
        eval_batch_size=eval_batch_size,
        profile=profile,
        profile_dir=profile_dir,
    )
    duration = time.perf_counter() - start
    per_path = duration / max(1, summary.n_paths)
    print(f"\n⏱️ Evaluation benchmark: {duration:.4f}s total | {per_path:.6f}s per path")
    return summary, duration
