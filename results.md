## RL vs LSM convex-cost comparison

| c | gamma | seed | step | RL price +/- SE | LSM price +/- SE | Delta (RL-LSM) | %Delta vs LSM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 1.0 | 11 | 29696 | 2.615 +/- 0.030 | 2.647 +/- 0.031 | -0.032 | -1.21% |
| 0.05 | 1.0 | 11 | 26624 | 2.101 +/- 0.029 | 2.114 +/- 0.029 | -0.013 | -0.60% |
| 0.05 | 1.5 | 14 | 27648 | 1.971 +/- 0.028 | 1.941 +/- 0.028 | 0.030 | 1.54% |
| 0.05 | 2.0 | 14 | 26624 | 1.832 +/- 0.026 | 1.696 +/- 0.027 | 0.137 | 8.06% |

### Key conclusions
- RL-LSM price delta at c=0.05 moves from -0.013 (g=1) to 0.030 (g=1.5) and 0.137 (g=2), with per-path SE <= 0.0034; baseline c=0 delta is -0.032.
- At g=2 the RL pricer beats LSM on 82.4% of paths and delivers an +8.06% price lift while lowering discounted costs by -0.175.
- For moderate convexity (g=1.5), RL edges LSM by 0.030 with cost savings of -0.063 and a slight quantity increase (Delta Q=0.087).
- The g=1 convex run still underprices LSM by -0.013 despite RL winning 53.6% of paths, reflecting a heavier left tail; quantity is -0.185 lower and costs -0.009 lower than LSM.
- Baseline c=0 run shows a small -0.032 delta (-1.21%) with RL and LSM quantities aligned (~11.5 units), confirming the convexity-driven crossover rather than a persistent bias.
- Discounted cost deltas are negative for every convex setting (-0.009, -0.063, -0.175 from g=1 to g=2), so RL achieves price gains via smarter allocation rather than higher spending.
- Validation: each run uses 8,192 eval paths over 22 steps; reward-vs-cost reconstruction error <5e-3 and merges have no NaNs (aside from expected missing first-exercise times).
- Notebook 5 reported Delta PV ~0.108 for an earlier gamma=2 convex run; the new g=2 result is higher (0.137) with twice the eval paths and later checkpoint, consistent in direction and improved magnitude.

### Data sources (RL / LSM)
- c=0, g=1, seed=11, step=29696: logs/SwingOption_20_v26_11/evaluations/rl_episode_29696.csv | logs/SwingOption_20_v26_11/evaluations/lsm.csv
- c=0.05, g=1, seed=11, step=26624: logs/SwingOption_20_v26_g1_11/evaluations/rl_episode_26624.csv | logs/SwingOption_20_v26_g1_11/evaluations/lsm.csv
- c=0.05, g=1.5, seed=14, step=27648: logs/SwingOption_20_v26_g1.5_14/evaluations/rl_episode_27648.csv | logs/SwingOption_20_v26_g1.5_14/evaluations/lsm.csv
- c=0.05, g=2, seed=14, step=26624: logs/SwingOption_20_v26_g2_14/evaluations/rl_episode_26624.csv | logs/SwingOption_20_v26_g2_14/evaluations/lsm.csv