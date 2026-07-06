"""Simulated Annealing (MAIN method).

Two genuine modes selected by ``inner_qp.use_for_sa``:

- QP mode (apples-to-apples): the state is a stock SUBSET; neighborhood moves are
  add / remove / swap; weights for the subset are solved exactly by the inner QP.
- Native mode: the state is a full weight vector; moves are add / remove / swap /
  perturb-weights; infeasibility is handled by repair followed by a penalty.

Metropolis acceptance maximizes the fitness; the temperature follows a
configurable geometric or linear schedule.
"""
from __future__ import annotations

import numpy as np

from src.config import Config
from src.model import ProblemData, fitness, penalized_fitness, repair
from src.metaheuristics.base import (
    OptimizerResult, SubsetEvaluator, heuristic_weights, make_rng, random_subset,
)

EPS = 1e-9


class SimulatedAnnealing:
    def __init__(self, p: ProblemData, cfg: Config, seed: int | None = None):
        self.p = p
        self.cfg = cfg
        self.rng = make_rng(cfg, seed)
        self.use_qp = cfg.inner_qp.use_for_sa
        self.evaluator = SubsetEvaluator(p, cfg, use_qp=True) if self.use_qp else None

    # ----- evaluation -------------------------------------------------------- #
    def _eval_subset(self, subset):
        return self.evaluator.eval(subset)

    def _eval_w(self, w):
        return penalized_fitness(w, self.p, self.cfg.penalty.weight)

    # ----- moves ------------------------------------------------------------- #
    def _move_subset(self, subset: list[int]) -> list[int]:
        p, rng = self.p, self.rng
        sel = set(subset)
        unsel = [i for i in range(p.n) if i not in sel]
        moves = []
        if len(sel) < p.K and unsel:
            moves.append("add")
        if len(sel) > 1:
            moves.append("remove")
        if unsel:
            moves.append("swap")
        move = rng.choice(moves)
        if move == "add":
            sel.add(int(rng.choice(unsel)))
        elif move == "remove":
            sel.discard(int(rng.choice(list(sel))))
        else:  # swap
            sel.discard(int(rng.choice(list(sel))))
            sel.add(int(rng.choice(unsel)))
        return sorted(sel)

    def _move_w(self, w: np.ndarray) -> np.ndarray:
        p, rng = self.p, self.rng
        sel = [i for i in range(p.n) if w[i] > EPS]
        unsel = [i for i in range(p.n) if w[i] <= EPS]
        mw = self.cfg.sa.move_weights
        names = ["add", "remove", "swap", "perturb"]
        probs = np.array([mw.get(n, 0.0) for n in names], dtype=float)
        # mask out infeasible moves
        if len(sel) >= p.K or not unsel:
            probs[0] = 0.0
        if len(sel) <= 1:
            probs[1] = 0.0
        if not unsel:
            probs[2] = 0.0
        if probs.sum() == 0:
            probs[3] = 1.0
        move = names[rng.choice(len(names), p=probs / probs.sum())]

        w2 = w.copy()
        if move == "add":
            j = int(rng.choice(unsel)); w2[j] = w2[w2 > EPS].mean() if (w2 > EPS).any() else 1.0
            sub = [i for i in range(p.n) if w2[i] > EPS]
        elif move == "remove":
            j = int(rng.choice(sel)); w2[j] = 0.0
            sub = [i for i in range(p.n) if w2[i] > EPS]
        elif move == "swap":
            i = int(rng.choice(sel)); j = int(rng.choice(unsel))
            w2[j] = w2[i]; w2[i] = 0.0
            sub = [k for k in range(p.n) if w2[k] > EPS]
        else:  # perturb
            noise = 1.0 + rng.normal(0.0, 0.15, size=len(sel))
            w2[sel] = np.clip(w2[sel] * noise, 0.0, None)
            sub = sel
        return repair(w2, self.p, subset=sub)

    # ----- main loop --------------------------------------------------------- #
    def optimize(self) -> OptimizerResult:
        p, cfg, rng = self.p, self.cfg, self.rng

        if self.use_qp:
            state = random_subset(p, rng)
            w_cur, f_cur = self._eval_subset(state)
        else:
            state = heuristic_weights(random_subset(p, rng), p)
            w_cur, f_cur = state, self._eval_w(state)

        best_w, best_f = w_cur.copy(), f_cur
        history, curve = [], []
        T = cfg.sa.t_init
        n_levels = max(1, int(np.ceil(np.log(cfg.sa.t_min / cfg.sa.t_init) / np.log(cfg.sa.cooling))))

        while T > cfg.sa.t_min:
            for _ in range(cfg.sa.iters_per_temp):
                if self.use_qp:
                    cand = self._move_subset(state)
                    w_new, f_new = self._eval_subset(cand)
                else:
                    w_new = self._move_w(state)
                    f_new = self._eval_w(w_new)
                    cand = w_new

                delta = f_new - f_cur
                if delta >= 0 or (np.isfinite(f_new) and rng.random() < np.exp(delta / T)):
                    state, w_cur, f_cur = cand, w_new, f_new
                    if f_cur > best_f:
                        best_w, best_f = w_cur.copy(), f_cur
                history.append(best_f)
                curve.append(f_cur)
            if cfg.sa.schedule == "linear":
                T -= (cfg.sa.t_init - cfg.sa.t_min) / n_levels
            else:  # geometric
                T *= cfg.sa.cooling

        n_evals = self.evaluator.n_evals if self.evaluator else len(curve)
        best_subset = sorted(int(i) for i in np.where(best_w > EPS)[0])
        return OptimizerResult("SA", best_w, float(best_f), best_subset, history, curve, n_evals,
                               meta={"mode": "qp" if self.use_qp else "native"})
