.PHONY: help paper clean-paper sweep single-exp train eval compare-lsm-state-modes tensorboard v64-sweep-4k v64-sweep-32k hedging-agents

PYTHON ?= python
RUN_NAME ?=
RUNS ?= 100
EXP_SCRIPT ?= Convex Cost Experiments/SwingOption_20_c0.05_gamma2.sh
COMPARE_OUTPUT ?= logs/lsm_state_mode_comparison.csv
COMPARE_PARQUET_DIR ?= logs/lsm_full_state
TENSORBOARD_ARGS ?= --load_fast=true --samples_per_plugin=scalars=500 --reload_interval=30 --max_reload_threads=4

help:
	@printf '%s\n' \
	  'Available targets:' \
	  '  make paper                         Build the manuscript into Paper/build/' \
	  '  make clean-paper                   Remove generated manuscript artifacts' \
	  '  make sweep                         Run the convex-cost experiment sweep' \
	  '  make v64-sweep-4k                  Restartable v64 re-baseline @ 4096 episodes' \
	  '  make v64-sweep-32k                 Restartable v64 re-baseline @ 32768 episodes' \
	  '  make hedging-agents                Train the kernel-off focal agent for Hedging.ipynb' \
	  '  make single-exp EXP_SCRIPT="..."    Run one experiment shell script' \
	  '  make train ARGS="..."              Run run.py with custom arguments' \
	  '  make eval RUN_NAME=name [RUNS=100] Evaluate a saved agent' \
	  '  make compare-lsm-state-modes       Compare reduced-state vs full-state LSM' \
	  '  make tensorboard                   Launch TensorBoard on runs/'

paper:
	./tools/build_latex.sh "$(PWD)/Paper" DRL_Swing_Options.tex

clean-paper:
	./tools/clean_latex.sh

sweep:
	bash conv_cost_exps.sh

# Restartable v64 re-baseline of the convex-cost grid (4 concurrent for 4k, 3 for 32k).
# Skips runs whose runs/<name>.{pth,json} already exist, so it is safe to re-run.
v64-sweep-4k:
	$(PYTHON) tools/run_v64_sweep.py --budget 4k --concurrency 4

v64-sweep-32k:
	$(PYTHON) tools/run_v64_sweep.py --budget 32k --concurrency 3

# Focal kernel-off agent for the Hedging notebook's LSM / RL / RL-kernel comparison.
# Both variant configs are defined in tools/train_hedging_agents.py; restartable.
hedging-agents:
	$(PYTHON) tools/train_hedging_agents.py --variant nokernel --episodes 32768 --seeds 11

single-exp:
	bash "$(EXP_SCRIPT)"

train:
	$(PYTHON) run.py $(ARGS)

eval:
	@test -n "$(RUN_NAME)" || (echo "RUN_NAME is required, e.g. make eval RUN_NAME=MyRun" >&2; exit 1)
	$(PYTHON) evaluate_saved_agent.py --run_name "$(RUN_NAME)" --runs "$(RUNS)"

compare-lsm-state-modes:
	$(PYTHON) tools/compare_lsm_state_modes.py --output_csv "$(COMPARE_OUTPUT)" --full_parquet_dir "$(COMPARE_PARQUET_DIR)"

tensorboard:
	tensorboard --logdir=runs $(TENSORBOARD_ARGS)
