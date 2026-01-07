import sys
import os
import torch
import numpy as np
import time

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.swing_contract import SwingContract
from src.swing_env import SwingOptionEnv
from src.agent import Agent
from src.utils_calibration import calibrate_actor_output
from run import generate_datasets, no_seasonal_function

def run_tuning():
    print("🚀 Starting Parameter Tuning...")
    
    epsilons = [0.01, 0.05, 0.10, 0.20]
    path_counts = [4096]
    
    # Test cases: (name, c, gamma)
    cases = [
        ("Hard", 0.15, 2.0),
        ("Normal", 0.01, 1.0)
    ]
    
    results = []
    
    # Standard Params
    strike = 1.0
    maturity = 0.0833
    n_rights = 22
    S0 = 1.0
    hhk_params = {
        "S0": S0, "alpha": 12.0, "sigma": 1.2, "beta": 150.0, 
        "lam": 6.0, "mu_J": 0.3, "f": no_seasonal_function,
        "T": maturity, "n_steps": n_rights - 1
    }

    for name, c, gamma in cases:
        print(f"\n--- Scenario: {name} (c={c}, gamma={gamma}) ---")
        contract = SwingContract(
            q_min=0.0, q_max=2.0, Q_min=0.0, Q_max=20.0,
            strike=strike, maturity=maturity, n_rights=n_rights, r=0.05,
            c_cost=c, gamma_cost=gamma
        )
        
        for n_paths in path_counts:
            # Generate dataset once per path count to control variance across epsilons
            seed = 42
            torch.manual_seed(seed)
            np.random.seed(seed)
            train_ds, _ = generate_datasets(hhk_params, n_paths, 10, seed)
            env = SwingOptionEnv(contract, hhk_params, train_ds)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            for eps in epsilons:
                print(f"   Testing N={n_paths}, Eps={eps}...", end="", flush=True)
                
                # Reset agent
                agent = Agent(
                    state_size=state_dim, action_size=action_dim,
                    n_step=1, per=False, munchausen=False, distributional=False,
                    random_seed=seed, hidden_size=64, device="cpu"
                )
                
                start_time = time.time()
                # Suppress stdout to keep tuning log clean? 
                # We can't easily suppress without redirecting fd. 
                # Let's just print.
                print("") 
                history = calibrate_actor_output(
                    agent=agent, env=env, n_episodes=n_paths,
                    max_iterations=10, target_std=0.005, epsilon_scale=eps
                )
                elapsed = time.time() - start_time
                
                final_price = history[-1]['price']
                steps = len(history)
                converged = (steps < 10)
                
                print(f"      -> Price={final_price:.4f}, Steps={steps}, Time={elapsed:.1f}s")
                
                results.append({
                    "case": name, "n_paths": n_paths, "eps": eps,
                    "price": final_price, "steps": steps, "time": elapsed
                })

    print("\n\n====== TUNING RESULTS ======")
    print(f"{'Case':<8} {'N_Paths':<8} {'Eps':<6} {'Price':<10} {'Steps':<6} {'Time(s)':<8}")
    for r in results:
        print(f"{r['case']:<8} {r['n_paths']:<8} {r['eps']:<6.2f} {r['price']:<10.4f} {r['steps']:<6} {r['time']:<8.1f}")
        
if __name__ == "__main__":
    run_tuning()
