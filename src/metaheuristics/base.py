"""Common optimizer interface, subset evaluation, and convergence logging.

All three metaheuristics return an :class:`OptimizerResult` carrying the best
portfolio, a best-so-far ``history`` and a per-iteration ``curve`` for the
convergence plots, plus ``meta`` for method-specific artifacts (e.g. ACO
pheromones).

The apples-to-apples flag lives here: :class:`SubsetEvaluator` maps a stock
subset to ``(weights, fitness)`` either through the shared inner QP (optimal
weights) or a fast heuristic weighting. ACO always uses the QP; SA/GA use it iff
``inner_qp.use_for_{sa,ga}`` is set, so the methods differ only in how they
search the discrete subset space.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import Config
from src.model import ProblemData, fitness, repair
from src.qp import solve_weights


@dataclass
class OptimizerResult:
    name: str
    best_w: np.ndarray
    best_fitness: float
    best_subset: list[int]
    history: list[float]            # best-so-far fitness per iteration
    curve: list[float]              # current/iteration fitness per iteration
    n_evals: int
    meta: dict = field(default_factory=dict)


def heuristic_weights(subset, p: ProblemData) -> np.ndarray:
    """Fast non-QP weighting: seed proportional to mu/sigma, then repair feasible."""
    idx = list(subset)
    d = np.sqrt(np.clip(np.diag(p.sigma), 1e-12, None))
    seed = np.zeros(p.n)
    seed[idx] = np.clip(p.mu[idx], 1e-6, None) / d[idx]
    return repair(seed, p, subset=idx)


class SubsetEvaluator:
    """Subset -> (weights, fitness), cached. QP weights when ``use_qp`` else heuristic."""

    def __init__(self, p: ProblemData, cfg: Config, use_qp: bool):
        self.p = p
        self.cfg = cfg
        self.use_qp = use_qp
        self._cache: dict[frozenset, tuple[np.ndarray, float]] = {}
        self.n_evals = 0

    def eval(self, subset) -> tuple[np.ndarray, float]:
        key = frozenset(int(i) for i in subset)
        if key in self._cache:
            return self._cache[key]
        self.n_evals += 1
        if self.use_qp:
            res = solve_weights(sorted(key), self.p, self.cfg)
            w, f = res.w, res.fitness
        else:
            w = heuristic_weights(sorted(key), self.p)
            f = fitness(w, self.p)
        self._cache[key] = (w, f)
        return w, f


def random_subset(p: ProblemData, rng: np.random.Generator, size: int | None = None) -> list[int]:
    size = min(size or p.K, p.n)
    return sorted(rng.choice(p.n, size=size, replace=False).tolist())


def make_rng(cfg: Config, seed: int | None) -> np.random.Generator:
    return np.random.default_rng(cfg.seed if seed is None else seed)
