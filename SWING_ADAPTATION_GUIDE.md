# Integrating Swing Option Pricing with D4PG-QR-FRM

This guide explains how to adapt the repository to price swing options using the existing D4PG framework.

## 1. Create a `SwingOptionEnv`

Implement a Gymnasium environment (e.g. `swing_env.py`) that models a swing option contract.
The environment must expose continuous actions and follow the standard Gymnasium API:

```python
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

class SwingOptionEnv(gym.Env):
    def __init__(self, contract_params=None):
        super().__init__()
        self.contract = contract_params
        self.action_space = Box(low=self.contract.q_min,
                                high=self.contract.q_max,
                                shape=(1,), dtype=np.float32)
        obs_dim = ...  # e.g. len(state vector)
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._seed = seed
        self.spot_path = simulate_hhk_spot(seed=seed)
        ...  # initialize episode state
        return observation, {}

    def step(self, action):
        ...  # update state, compute reward
        return obs, reward, terminated, truncated, {}
```

State representation should include the underlying price, cumulative volumes and time features:
`[S_t, Q_exercised, Q_remaining, time_to_maturity, time_to_next_decision, K, underlying_features]`.
The reward at each step is `q_t * (S_t - K)` (no transaction cost). Terminate when maturity
is reached or the contract rights are exhausted.

## 2. Integrate the Environment Permanently

This codebase is now dedicated to swing option pricing, so registration via `gym.make` is unnecessary.
Modify `run.py` to instantiate `SwingOptionEnv` directly:

```python
from swing_env import SwingOptionEnv

train_env = SwingOptionEnv()
eval_env = SwingOptionEnv()
temp_env = SwingOptionEnv()
```

Remove the `-env` command line option and replace all calls to `gym.make(args.env)` with the
instantiation shown above. The custom environment becomes the default for training and evaluation.

## 3. Monte Carlo Pricing Evaluation

The value of the swing option is obtained by averaging the discounted payoff over many
simulated price paths. Evaluation can be performed sequentially, but the neural
networks inside the agent can process batched inputs efficiently. Instead of stepping
one path at a time, replicate the environment for multiple Monte Carlo paths and feed
a **batch of states** to the *same* agent at each time step. The agent outputs a batch
of actions, and the discounted rewards are accumulated for each path.

Below is a sequential implementation followed by a vectorized variant.

```python
def evaluate_price(agent, runs=100):
    discounted_returns = []
    for i in range(runs):
        path_seed = base_seed + i + 1
        state, _ = eval_env.reset(seed=path_seed)
        disc_return = 0.0
        step = 0
        while True:
            action = agent.act(state[None, :])
            state, reward, terminated, truncated, _ = eval_env.step(action[0])
            disc_return += (discount ** step) * reward
            step += 1
            if terminated or truncated:
                break
        discounted_returns.append(disc_return)
    return np.mean(discounted_returns)
```

`discount` should equal `exp(-r * dt)`. Logging this value during training allows you to
monitor convergence and yields the final option price once training is complete.

## 4. Unified Seed Handling

Use one base seed for the entire experiment. Increment the seed whenever a new Monte Carlo path
is simulated:

```python
base_seed = args.seed
episode_seed = base_seed
state, _ = train_env.reset(seed=episode_seed)
for frame in range(frames):
    ...  # interact with environment
    if done:
        episode_seed += 1
        state, _ = train_env.reset(seed=episode_seed)
```

Evaluation follows the same pattern using `base_seed + i + 1` for each run. This approach keeps paths independent yet reproducible.

## 5. Supporting Files

- `swing_contract.py` — defines contract parameters (`q_min`, `q_max`, `Q_max`, `K`, etc.)
- `simulate_hhk_spot.py` — generates HHK spot price paths
- `swing_env.py` — environment implementation using the contract and simulator

## 6. Training

Run `python run.py` to start training. All existing D4PG options (PER, Munchausen, IQN, etc.) remain available. The critic\x27s estimate `Q(s_t, a_t)` approximates the swing option value, and `evaluate_price` can be used during and after training to compute the option premium.

