"""Typed configuration for the portfolio-optimization suite.

YAML files in ``config/`` are loaded and validated into the nested dataclasses
below. Use :func:`load_config` as the single entry point; pass override files
(e.g. an objective variant) to merge on top of ``default.yaml``. The resolved
config can be serialized back to YAML via :meth:`Config.to_dict` so every run in
``results/`` records the exact parameters that produced it.

Leakage note: the split is a first-class config object. Code that optimizes must
only ever read ``split.train`` (and ``split.validation`` for tuning); the test
window is consumed solely by the evaluation layer. See ``src/data.py``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardConfig:
    train_months: int = 24
    test_months: int = 3
    step_months: int = 3


@dataclass
class SplitConfig:
    train: list[str] = field(default_factory=lambda: ["2021-06-30", "2023-06-30"])
    validation: list[str] = field(default_factory=lambda: ["2023-06-30", "2023-12-31"])
    test: list[str] = field(default_factory=lambda: ["2023-12-31", "2024-06-30"])
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)


@dataclass
class DataConfig:
    tickers: Optional[list[str]] = None
    universe_size: Optional[int] = None   # None -> built-in 50; else top-N S&P500 by liquidity
    start: str = "2021-06-30"
    end: str = "2024-06-30"
    price_field: str = "Adj Close"
    cache_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    excel_path: str = "data/portfolio_data.xlsx"
    annualization: int = 252


@dataclass
class CovarianceConfig:
    estimator: str = "ledoit_wolf"  # "sample" | "ledoit_wolf"


@dataclass
class ConstraintsConfig:
    K: int = 10
    u_i: float = 0.20
    L_s: float = 0.30
    fully_invested: bool = True


@dataclass
class ObjectiveConfig:
    variant: str = "A"  # "A" | "B" | "C"
    lambda_risk: float = 5.0
    gamma_cost: float = 0.001
    c_i: float = 0.001
    w0: str = "equal"  # "equal" | "zero"
    lambda_L1: float = 0.01
    risk_free: float = 0.0


@dataclass
class InnerQPConfig:
    use_for_sa: bool = True
    use_for_ga: bool = True
    use_for_tabu: bool = True
    solver: str = "auto"


@dataclass
class SAConfig:
    t_init: float = 1.0
    t_min: float = 1.0e-4
    cooling: float = 0.95
    iters_per_temp: int = 50
    schedule: str = "geometric"
    move_weights: dict[str, float] = field(
        default_factory=lambda: {"add": 0.25, "remove": 0.25, "swap": 0.25, "perturb": 0.25}
    )


@dataclass
class GAConfig:
    pop: int = 80
    generations: int = 200
    cx_rate: float = 0.8
    mut_rate: float = 0.1
    tournament: int = 3
    elitism: int = 2


@dataclass
class ACOConfig:
    ants: int = 30
    iters: int = 100
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    tau_min: float = 0.01
    tau_max: float = 1.0


@dataclass
class TabuConfig:
    iters: int = 200                 # tabu iterations
    tenure: int = 12                 # how long a move stays forbidden
    neighbors: int = 150             # max swap-neighbors evaluated per iteration
    patience: int = 40               # stop after this many non-improving iterations


@dataclass
class PenaltyConfig:
    weight: float = 100.0
    mode: str = "repair_then_penalty"


@dataclass
class BacktestConfig:
    rebalance_cost_bps: float = 10.0
    charge_on_rebalance: bool = True


@dataclass
class ExperimentConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    out_dir: str = "results"


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    inner_qp: InnerQPConfig = field(default_factory=InnerQPConfig)
    sa: SAConfig = field(default_factory=SAConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
    tabu: TabuConfig = field(default_factory=TabuConfig)
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # ----- (de)serialization ------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = d or {}
        return cls(
            seed=d.get("seed", 42),
            data=DataConfig(**d.get("data", {})),
            split=_split_from_dict(d.get("split", {})),
            covariance=CovarianceConfig(**d.get("covariance", {})),
            constraints=ConstraintsConfig(**d.get("constraints", {})),
            objective=ObjectiveConfig(**d.get("objective", {})),
            inner_qp=InnerQPConfig(**d.get("inner_qp", {})),
            sa=SAConfig(**d.get("sa", {})),
            ga=GAConfig(**d.get("ga", {})),
            aco=ACOConfig(**d.get("aco", {})),
            tabu=TabuConfig(**d.get("tabu", {})),
            penalty=PenaltyConfig(**d.get("penalty", {})),
            backtest=BacktestConfig(**d.get("backtest", {})),
            experiment=ExperimentConfig(**d.get("experiment", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    # ----- validation ------------------------------------------------------- #
    def validate(self) -> "Config":
        c = self
        assert c.covariance.estimator in {"sample", "ledoit_wolf"}, \
            f"unknown covariance estimator: {c.covariance.estimator}"
        assert c.objective.variant in {"A", "B", "C"}, \
            f"unknown objective variant: {c.objective.variant}"
        assert c.objective.w0 in {"equal", "zero"}
        assert c.penalty.mode in {"repair_then_penalty", "penalty_only"}
        assert c.constraints.K >= 1
        assert 0 < c.constraints.u_i <= 1
        assert 0 < c.constraints.L_s <= 1
        assert c.constraints.K * c.constraints.u_i >= 1.0, \
            "K * u_i < 1 makes the fully-invested constraint infeasible"
        # split ordering: train.start <= val.start <= test.start, no overlap
        order = [c.split.train, c.split.validation, c.split.test]
        for (a_lo, a_hi), (b_lo, b_hi) in zip(order, order[1:]):
            assert a_hi <= b_lo, f"split windows overlap or are out of order: {a_hi} > {b_lo}"
        return self


def _split_from_dict(d: dict[str, Any]) -> SplitConfig:
    d = dict(d or {})
    wf = d.pop("walkforward", {})
    return SplitConfig(walkforward=WalkForwardConfig(**(wf or {})), **d)


# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(
    path: str | Path = "config/default.yaml",
    overrides: list[str | Path] | None = None,
) -> Config:
    """Load ``path`` and deep-merge any ``overrides`` (e.g. an objective variant) on top."""
    with open(path) as f:
        merged = yaml.safe_load(f) or {}
    for ov in overrides or []:
        with open(ov) as f:
            merged = _deep_merge(merged, yaml.safe_load(f) or {})
    return Config.from_dict(merged).validate()
