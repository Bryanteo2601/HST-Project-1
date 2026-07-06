"""Ant Colony Optimization (HYBRID method).

Each ant builds a K-stock subset, drawing stocks without replacement with
probability proportional to tau_i^alpha * eta_i^beta, where eta_i = mu_i / sigma_i.
For each ant's fixed subset the weights are solved EXACTLY by the shared inner QP,
so ACO only searches the discrete subset space. Pheromones evaporate by rho and
are deposited on the global-best subset, with Max-Min bounds [tau_min, tau_max] to
resist premature convergence.

``meta`` carries the final pheromone vector, the per-iteration pheromone history
(for the heatmap), and per-stock selection counts (most-persistently-selected).
"""
from __future__ import annotations

import numpy as np

from src.config import Config
from src.model import ProblemData
from src.metaheuristics.base import OptimizerResult, SubsetEvaluator, make_rng


class AntColony:
    def __init__(self, p: ProblemData, cfg: Config, seed: int | None = None):
        self.p = p
        self.cfg = cfg
        self.rng = make_rng(cfg, seed)
        self.evaluator = SubsetEvaluator(p, cfg, use_qp=True)  # ACO always uses the QP
        d = np.sqrt(np.clip(np.diag(p.sigma), 1e-12, None))
        self.eta = np.clip(p.mu, 1e-6, None) / d              # heuristic desirability

    def _construct(self, tau: np.ndarray) -> list[int]:
        a, b = self.cfg.aco.alpha, self.cfg.aco.beta
        weight = (tau ** a) * (self.eta ** b)
        avail = list(range(self.p.n))
        chosen: list[int] = []
        for _ in range(min(self.p.K, self.p.n)):
            wv = weight[avail]
            s = wv.sum()
            probs = (wv / s) if s > 0 else None
            j = int(self.rng.choice(avail, p=probs))
            chosen.append(j)
            avail.remove(j)
        return sorted(chosen)

    def optimize(self) -> OptimizerResult:
        p, cfg, rng = self.p, self.cfg, self.rng
        tau = np.full(p.n, cfg.aco.tau_max)
        select_counts = np.zeros(p.n)
        pher_history = []
        history, curve = [], []
        best_w, best_f, best_subset = None, -np.inf, []

        for _ in range(cfg.aco.iters):
            iter_best_f = -np.inf
            for _ in range(cfg.aco.ants):
                subset = self._construct(tau)
                w, f = self.evaluator.eval(subset)
                select_counts[subset] += 1
                if f > iter_best_f:
                    iter_best_f = f
                if f > best_f:
                    best_w, best_f, best_subset = w.copy(), f, subset

            tau *= (1.0 - cfg.aco.rho)                    # evaporate
            if best_subset:
                tau[best_subset] += cfg.aco.rho           # deposit on global best
            tau = np.clip(tau, cfg.aco.tau_min, cfg.aco.tau_max)  # Max-Min bounds

            pher_history.append(tau.copy())
            history.append(best_f)
            curve.append(iter_best_f)

        if best_w is None:
            best_w = np.zeros(p.n)
        return OptimizerResult(
            "ACO", best_w, float(best_f), sorted(best_subset), history, curve,
            self.evaluator.n_evals,
            meta={"pheromone": tau, "pheromone_history": np.array(pher_history),
                  "selection_counts": select_counts, "eta": self.eta},
        )
