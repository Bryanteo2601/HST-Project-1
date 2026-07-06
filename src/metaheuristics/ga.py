"""Genetic Algorithm (comparison method).

Chromosome = a stock subset (the selection vector z with at most K ones). Weights
are assigned by the shared inner QP when ``inner_qp.use_for_ga`` is set, else by a
fast heuristic weighting -- so GA and SA/ACO stay apples-to-apples. Operators:
tournament selection, uniform-style subset crossover, swap mutation, repair to
respect cardinality, and elitism.
"""
from __future__ import annotations

import numpy as np

from src.config import Config
from src.model import ProblemData
from src.metaheuristics.base import (
    OptimizerResult, SubsetEvaluator, make_rng, random_subset,
)


class GeneticAlgorithm:
    def __init__(self, p: ProblemData, cfg: Config, seed: int | None = None):
        self.p = p
        self.cfg = cfg
        self.rng = make_rng(cfg, seed)
        self.evaluator = SubsetEvaluator(p, cfg, use_qp=cfg.inner_qp.use_for_ga)

    def _trim(self, genes: set[int]) -> list[int]:
        """Coerce a gene set to a valid subset of size in [1, K]."""
        genes = set(int(g) for g in genes)
        if len(genes) > self.p.K:
            genes = set(self.rng.choice(sorted(genes), size=self.p.K, replace=False).tolist())
        if not genes:
            genes = {int(self.rng.integers(self.p.n))}
        return sorted(genes)

    def _crossover(self, a: list[int], b: list[int]) -> list[int]:
        common = set(a) & set(b)
        rest = list((set(a) | set(b)) - common)
        self.rng.shuffle(rest)
        room = max(0, self.p.K - len(common))
        return self._trim(common | set(rest[: self.rng.integers(0, room + 1)]))

    def _mutate(self, genes: list[int]) -> list[int]:
        g = set(genes)
        unsel = [i for i in range(self.p.n) if i not in g]
        if unsel and g:
            g.discard(int(self.rng.choice(sorted(g))))
            g.add(int(self.rng.choice(unsel)))
        return self._trim(g)

    def _tournament(self, pop, fits):
        idx = self.rng.choice(len(pop), size=min(self.cfg.ga.tournament, len(pop)), replace=False)
        best = max(idx, key=lambda i: fits[i])
        return pop[best]

    def optimize(self) -> OptimizerResult:
        p, cfg, rng = self.p, self.cfg, self.rng
        pop = [random_subset(p, rng) for _ in range(cfg.ga.pop)]
        evals = [self.evaluator.eval(s) for s in pop]
        fits = [f for _, f in evals]

        best_i = int(np.argmax(fits))
        best_w, best_f, best_subset = evals[best_i][0].copy(), fits[best_i], pop[best_i]
        history, curve = [], []

        for _ in range(cfg.ga.generations):
            order = np.argsort(fits)[::-1]
            new_pop = [pop[i] for i in order[: cfg.ga.elitism]]
            while len(new_pop) < cfg.ga.pop:
                a, b = self._tournament(pop, fits), self._tournament(pop, fits)
                child = self._crossover(a, b) if rng.random() < cfg.ga.cx_rate else list(a)
                if rng.random() < cfg.ga.mut_rate:
                    child = self._mutate(child)
                new_pop.append(child)

            pop = new_pop
            evals = [self.evaluator.eval(s) for s in pop]
            fits = [f for _, f in evals]
            gen_best = int(np.argmax(fits))
            if fits[gen_best] > best_f:
                best_w, best_f, best_subset = evals[gen_best][0].copy(), fits[gen_best], pop[gen_best]
            history.append(best_f)
            curve.append(fits[gen_best])

        return OptimizerResult("GA", best_w, float(best_f), sorted(best_subset),
                               history, curve, self.evaluator.n_evals,
                               meta={"mode": "qp" if cfg.inner_qp.use_for_ga else "native"})
