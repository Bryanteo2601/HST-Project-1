# Cardinality-Constrained Portfolio Optimization via Metaheuristics

Course project for **40.018 Heuristics and Systems Theory**. Implements and
benchmarks four metaheuristics (Simulated Annealing, Tabu Search, Genetic
Algorithm, and an Ant Colony Optimization + inner-QP hybrid, our main method)
against an exact Gurobi MIQP for a cardinality-constrained mean-variance
portfolio problem, with a careful in-sample vs out-of-sample overfitting study.
The cardinality limit is K = 10 (Evans & Archer, 1968).

## Problem

Maximize on TRAINING data:

    sum_i mu_i w_i  -  lambda * w^T Sigma w  -  gamma * sum_i c_i |w_i - w_i0|

subject to fully invested (`sum w_i = 1`), `0 <= w_i <= u_i z_i`, cardinality
(`sum z_i <= K`), sector caps (`sum_{i in s} w_i <= L_s`), `z_i in {0,1}`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Gurobi (academic license) is used for the exact MIQP reference when available;
otherwise the code falls back to a cvxpy MIQP backend and degrades gracefully.

## Reproduce

```bash
python run_all.py                       # full suite (Ledoit-Wolf, variant A)
python run_all.py --fast                # small budgets, ~1-2 min smoke run
python run_all.py --only core gap       # run a subset of stages
python run_all.py --override config/objective_B.yaml   # swap objective variant
pytest -q                               # tests
```

`run_all.py` builds the data once, then runs the stages below; each is also a
standalone script. Outputs land in `results/` (CSV tables) and `plots/` (PNG).

| Stage | Command | Produces |
|-------|---------|----------|
| core   | `python run_all.py --only core` | `results/master_table.csv`; `plots/convergence.png`, `oos_equity.png`, `risk_return.png`, `is_oos_gap.png`, `aco_pheromone.png`, `aco_selection.png` |
| gap    | `python -m experiments.exp_gap_to_optimal` | `results/gap_to_optimal.csv` (exact MIQP vs methods) |
| seed   | `python -m experiments.exp_seed_robustness` | `results/seed_robustness.csv`, `plots/seed_robustness.png` |
| sens   | `python -m experiments.exp_sensitivity` | `results/sensitivity.csv`, `plots/sensitivity_val_sharpe.png` |
| obj    | `python -m experiments.exp_objective_variants` | `results/objective_variants.csv`, `variant_C_sweep.csv` |
| wf     | `python -m experiments.exp_walkforward` | `results/walkforward.csv`, `plots/walkforward_equity.png` |
| shrink | `python -m experiments.exp_shrinkage_gap` | `results/shrinkage_gap.csv`, `plots/shrinkage_gap.png` |

Data artifacts (`python -c "from src.config import load_config; from src.data import
build_dataset; build_dataset(load_config('config/default.yaml'))"`):
`data/raw/*.parquet` (cache), `data/processed/{prices,returns,mu_sigma_sector,covariance}.csv`,
`data/portfolio_data.xlsx`.

The covariance estimator (`covariance.estimator: sample | ledoit_wolf`) and the
train/validation/test split live in `config/default.yaml`. Add `--fast` to any
`run_all.py` invocation for quick budgets; standalone experiment scripts accept
`--fast` too. Exact MIQP uses Gurobi if licensed, else a cvxpy MIQP backend.

## Layout

```
config/         YAML configs (default + objective variants A/B/C)
data/           cached prices (raw/), processed mu/Sigma, portfolio_data.xlsx
src/            data, covariance, model, qp, metaheuristics/, benchmarks,
                backtest, metrics, plotting
experiments/    standalone runnable studies
results/  plots/  generated tables and figures
tests/          pytest suite
```

## Leakage policy

`mu`/`Sigma` are estimated within each split. Optimizers read only TRAIN (and
VALIDATION for hyperparameter tuning). The TEST window is read only by
`src/backtest.py` evaluation functions, applied to already-frozen portfolios —
test set is look-once. See `src/config.py` and `src/data.py`.
