#!/bin/bash

# Longstaff-Schwartz Monte Carlo Swing Option Pricing
# This script runs the LSM algorithm with the same parameters as run.sh

#  Usage: 
#  ./run_longstaff_schwartz.sh
#  or for background execution:
#  nohup "./run_longstaff_schwartz.sh" > .lsm_output.txt 2>&1 &
#  tail -f .lsm_output.txt

echo "=== Longstaff-Schwartz Swing Option Pricing ==="
echo "Starting LSM Monte Carlo pricing with parameters matching run.sh..."

# Monthly Swing Option Parameters (matching run.sh exactly):
args=( 
    # Simulation parameters
    -n_paths=16384                # Number of Monte Carlo paths (4096 or 16384)
    -seed=11                      # Random seed for reproducibility
    -name="MonthlySwing_LSM"      # Run name for output files
    
    # Monthly Swing Option Contract Parameters
    --strike=100.0              # At-the-money strike
    --maturity=0.0833           # 1 month = 1/12 year
    --n_rights=22               # ~22 trading days in a month
    --q_min=0.0                 # No minimum exercise requirement per day
    --q_max=2.0                 # Max 2 units per day (reasonable daily limit)
    --Q_min=0.0                 # No minimum total exercise requirement
    --Q_max=20.0                # Max 20 units total over the month (10 days worth)
    --risk_free_rate=0.05       # 5% annual risk-free rate
    --min_refraction_days=0     # No refraction period (can exercise daily)
    
    # Market Process Parameters (monthly calibration)
    --S0=100.0                  # Initial spot price
    --alpha=12.0                # Higher mean reversion for monthly timeframe
    --sigma=1.2                 # Moderate volatility for monthly period
    --beta=150.0                # Jump decay rate
    --lam=6.0                   # Jump intensity (6 jumps per year average)
    --mu_J=0.3                  # Mean jump size (30%)
    
    # LSM specific parameters
    --basis_type=polynomial       # Use polynomial basis functions
    --polynomial_degree=3         # Cubic polynomials for regression
)

echo "Running Longstaff-Schwartz pricing with ${args[0]#*=} Monte Carlo paths..."

# Run the LSM pricing algorithm
python src/longstaff_schwartz_pricer.py "${args[@]}"

echo "=== LSM Pricing Complete ==="
echo "Results saved to 'Longstaff Schwartz Pricer/MonthlySwing_LSM/' directory"
echo ""
echo "Files generated:"
echo "- MonthlySwing_LSM_solution.csv (detailed path-by-path solutions)"
echo "- MonthlySwing_LSM_results.txt (summary results and analysis)"
echo ""
echo "To compare with RL results:"
echo "- LSM option value is the theoretical benchmark"
echo "- RL agent should converge towards this LSM value"
echo "- CSV format matches RL evaluation output for easy comparison"
