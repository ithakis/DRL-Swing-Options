# Profile Report: runv42_11_16k.prof

This report is derived from `cProfile` stats.

## Summary

- Total runtime (profiled): **562.918s**
- Function calls: **357,956,014** (primitive: **334,100,095**)

## Time By Component

Two complementary views are shown:
- **Self time** (`tottime`): CPU time spent *inside* functions in that component.
- **Cumulative time** (`cumtime`): includes time inside callees (can double-count across groups).

| Component | Self time (s) | Self % | Cum time (s) |
|---|---:|---:|---:|
| PyTorch kernels (builtins) | 299.697 | 53.2% | 299.754 |
| PyTorch (python overhead) | 109.680 | 19.5% | 1623.668 |
| Networks (src/networks.py) | 31.259 | 5.6% | 393.308 |
| Agent (src/agent.py) | 22.063 | 3.9% | 974.683 |
| Replay buffer (src/replay_buffer.py) | 20.431 | 3.6% | 43.497 |
| Environment (src/swing_env.py) | 4.452 | 0.8% | 14.081 |
| Evaluation (src/agent_evaluation.py) | 2.481 | 0.4% | 51.329 |
| Training loop (run.py) | 4.217 | 0.7% | 1670.917 |
| NumPy builtins | 9.496 | 1.7% | 15.033 |
| NumPy (python overhead) | 4.159 | 0.7% | 6.220 |
| Python/C builtins | 14.435 | 2.6% | 636.411 |
| Other | 40.550 | 7.2% | 601.218 |

## Key Hotspots (You’ll Feel These)

| Hotspot | Calls | Self (s) | Cum (s) | Why it matters |
|---|---:|---:|---:|---|
| `<method 'run_backward' of 'torch._C._EngineBase' objects>` | 332,906 | 116.002 | 116.002 | Backprop dominates total compute |
| `<built-in method torch._C._nn.linear>` | 3,662,988 | 45.531 | 45.531 | Matmul/linear kernels |
| `<built-in method torch.layer_norm>` | 2,441,992 | 36.828 | 36.828 | Normalization kernel + wrapper |
| `<built-in method torch._C._nn.silu_>` | 2,441,992 | 24.502 | 24.502 | Activation kernel |
| `adam.py:344(_single_tensor_adam)` | 665,812 | 29.244 | 69.958 | AdamW update math |
| `functional.py:2889(layer_norm)` | 2,441,992 | 2.682 | 40.690 | Normalization kernel + wrapper |
| `replay_buffer.py:549(sample)` | 166,453 | 7.113 | 16.760 | PER sampling + device copies |
| `replay_buffer.py:45(_py_fenwick_find_prefix_indices)` | 166,453 | 4.972 | 5.096 | PER prefix-search (Python loop) |
| `networks.py:303(apply_profitability_gate)` | 719,972 | 16.863 | 21.752 | Extra per-action math each step |
| `agent_evaluation.py:294(evaluate_agent)` | 17 | 0.000 | 12.939 | Periodic evaluation overhead |
| `swing_env.py:140(step)` | 373,433 | 1.928 | 8.381 | Environment transition cost |
| `<method 'tolist' of 'numpy.ndarray' objects>` | 1,088 | 5.214 | 5.214 | Materializing Python lists (often avoidable) |

## Top Hotspots (Cumulative Time)

| Rank | Calls | Self (s) | Cum (s) | Function |
|---:|---:|---:|---:|---|
| 1 | 6,088 | 0.135 | 562.930 | `<built-in method builtins.exec>` |
| 2 | 1 | 0.001 | 562.930 | `run.py:1(<module>)` |
| 3 | 1 | 0.027 | 558.513 | `run.py:1326(main)` |
| 4 | 1 | 4.020 | 546.435 | `run.py:1124(run_training)` |
| 5 | 350,905 | 3.841 | 464.004 | `agent.py:322(step)` |
| 6 | 166,453 | 7.494 | 436.043 | `agent.py:347(learn_)` |
| 7 | 11,824,559 | 7.002 | 184.042 | `module.py:1747(_wrapped_call_impl)` |
| 8 | 11,824,559 | 11.859 | 182.683 | `module.py:1755(_call_impl)` |
| 9 | 2,441,992 | 4.767 | 136.822 | `container.py:238(forward)` |
| 10 | 332,906 | 0.522 | 123.142 | `_tensor.py:592(backward)` |
| 11 | 332,906 | 1.619 | 122.590 | `__init__.py:243(backward)` |
| 12 | 332,906 | 0.621 | 116.858 | `graph.py:815(_engine_run_backward)` |
| 13 | 332,906 | 116.002 | 116.002 | `<method 'run_backward' of 'torch._C._EngineBase' objects>` |
| 14 | 332,906 | 0.608 | 105.223 | `lr_scheduler.py:120(wrapper)` |
| 15 | 332,906 | 2.203 | 104.615 | `optimizer.py:465(wrapper)` |
| 16 | 332,906 | 1.393 | 92.985 | `optimizer.py:60(_use_grad)` |
| 17 | 332,906 | 2.226 | 90.763 | `adam.py:212(step)` |
| 18 | 501,024 | 6.944 | 84.583 | `networks.py:454(forward)` |
| 19 | 719,972 | 2.371 | 81.938 | `networks.py:262(forward_preact)` |
| 20 | 665,812 | 0.794 | 78.168 | `optimizer.py:130(maybe_fallback)` |
| 21 | 665,812 | 1.846 | 77.092 | `adam.py:865(adam)` |
| 22 | 334,571 | 0.227 | 71.273 | `networks.py:249(forward)` |
| 23 | 334,571 | 1.268 | 71.046 | `networks.py:297(forward_raw_and_gated)` |
| 24 | 665,812 | 29.244 | 69.958 | `adam.py:344(_single_tensor_adam)` |
| 25 | 334,571 | 0.509 | 58.678 | `networks.py:269(forward_raw)` |
| 26 | 385,401 | 5.714 | 55.685 | `agent.py:303(act)` |
| 27 | 3,662,988 | 4.472 | 52.012 | `linear.py:124(forward)` |
| 28 | 2,441,992 | 3.874 | 45.947 | `normalization.py:216(forward)` |
| 29 | 3,662,988 | 45.531 | 45.531 | `<built-in method torch._C._nn.linear>` |
| 30 | 2,441,992 | 2.682 | 40.690 | `functional.py:2889(layer_norm)` |

## Top Hotspots (Self Time)

| Rank | Calls | Self (s) | Cum (s) | Function |
|---:|---:|---:|---:|---|
| 1 | 332,906 | 116.002 | 116.002 | `<method 'run_backward' of 'torch._C._EngineBase' objects>` |
| 2 | 3,662,988 | 45.531 | 45.531 | `<built-in method torch._C._nn.linear>` |
| 3 | 2,441,992 | 36.828 | 36.828 | `<built-in method torch.layer_norm>` |
| 4 | 665,812 | 29.244 | 69.958 | `adam.py:344(_single_tensor_adam)` |
| 5 | 2,441,992 | 24.502 | 24.502 | `<built-in method torch._C._nn.silu_>` |
| 6 | 719,972 | 16.863 | 21.752 | `networks.py:303(apply_profitability_gate)` |
| 7 | 11,824,559 | 11.859 | 182.683 | `module.py:1755(_call_impl)` |
| 8 | 4,327,789 | 9.221 | 9.221 | `<method 'mul_' of 'torch._C.TensorBase' objects>` |
| 9 | 166,453 | 7.494 | 436.043 | `agent.py:347(learn_)` |
| 10 | 166,453 | 7.113 | 16.760 | `replay_buffer.py:549(sample)` |
| 11 | 11,824,559 | 7.002 | 184.042 | `module.py:1747(_wrapped_call_impl)` |
| 12 | 501,024 | 6.944 | 84.583 | `networks.py:454(forward)` |
| 13 | 665,812 | 6.872 | 9.574 | `adam.py:137(_init_group)` |
| 14 | 3,329,062 | 6.442 | 6.442 | `<method 'add_' of 'torch._C.TensorBase' objects>` |
| 15 | 3,329,060 | 5.831 | 5.831 | `<method 'sqrt' of 'torch._C.TensorBase' objects>` |
| 16 | 385,401 | 5.714 | 55.685 | `agent.py:303(act)` |
| 17 | 332,906 | 5.653 | 5.653 | `<built-in method torch._foreach_mul_>` |
| 18 | 15,654,184 | 5.646 | 5.646 | `module.py:1927(__getattr__)` |
| 19 | 1,088 | 5.214 | 5.214 | `<method 'tolist' of 'numpy.ndarray' objects>` |
| 20 | 166,453 | 4.972 | 5.096 | `replay_buffer.py:45(_py_fenwick_find_prefix_indices)` |
| 21 | 2,441,992 | 4.767 | 136.822 | `container.py:238(forward)` |
| 22 | 3,662,988 | 4.472 | 52.012 | `linear.py:124(forward)` |
| 23 | 3,329,060 | 4.093 | 4.093 | `<method 'addcdiv_' of 'torch._C.TensorBase' objects>` |
| 24 | 1 | 4.020 | 546.435 | `run.py:1124(run_training)` |
| 25 | 3,329,060 | 3.953 | 3.953 | `<method 'lerp_' of 'torch._C.TensorBase' objects>` |
| 26 | 2,441,992 | 3.874 | 45.947 | `normalization.py:216(forward)` |
| 27 | 350,905 | 3.841 | 464.004 | `agent.py:322(step)` |
| 28 | 1,511,641 | 3.827 | 5.002 | `_methods.py:99(_clip)` |
| 29 | 3,329,060 | 3.798 | 3.798 | `<method 'addcmul_' of 'torch._C.TensorBase' objects>` |
| 30 | 665,812 | 3.640 | 3.640 | `<built-in method torch._ops.profiler._record_function_enter_new>` |

## What’s Essential vs. Non-Essential

### Essential (core training compute)

- Backprop + forward kernels + optimizer math dominate runtime on CPU.

### Non-essential / tunable overhead (can hold things up)

- **Evaluation**: `agent_evaluation.py:294(evaluate_agent)` totals **12.939s** cumulative time in this profile.
- **Evaluation row materialization**: `numpy.ndarray.tolist` totals **5.214s** self time (pure overhead when CSVs are disabled).
- **Profitability gate**: `src/networks.py:303(apply_profitability_gate)` totals **21.752s** cumulative time.
- **PER overhead**: `src/replay_buffer.py:549(sample)` totals **16.760s** cumulative time and the Fenwick prefix-search loop is **5.096s** cumulative time.