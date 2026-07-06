"""Tabu Search (a refinement of the main SA method).

Same fixed-cardinality subset encoding and the same swap neighborhood as SA, but
with MEMORY instead of randomness: at each step it scans the neighborhood and
takes the best non-tabu swap, where a recently added stock cannot be removed and a
recently removed stock cannot be re-added for ``tenure`` iterations. An aspiration
criterion overrides the tabu status if a move yields a new global best. This
removes SA's cycling/random-walk inefficiency, so Tabu typically reaches the same
optimum in far fewer evaluations -- the headline SA-vs-Tabu comparison.

Weights are assigned by the shared inner QP (apples-to-apples with SA/GA/ACO).
"""
from __future__ import annotations

import numpy as np

from src.config import Config
from src.model import ProblemData
from src.metaheuristics.base import (
    OptimizerResult, SubsetEvaluator, make_rng, random_subset,
)

EPS = 1e-9


class TabuSearch:
    def __init__(self, p: ProblemData, cfg: Config, seed: int | None = None):
        self.p = p
        self.cfg = cfg
        self.rng = make_rng(cfg, seed)
        self.use_qp = cfg.inner_qp.use_for_tabu
        self.evaluator = SubsetEvaluator(p, cfg, use_qp=self.use_qp)

    def _eval(self, subset):
        return self.evaluator.eval(subset)

    def _neighbors(self, subset: list[int]):
        """Sample up to ``neighbors`` swaps; yield (new_subset, removed, added)."""
        p, rng = self.p, self.rng
        sel = list(subset)
        unsel = [i for i in range(p.n) if i not in set(sel)]
        if not unsel or len(sel) < 1:
            return
        pairs = [(a, b) for a in sel for b in unsel]
        cap = self.cfg.tabu.neighbors
        if len(pairs) > cap:
            idx = rng.choice(len(pairs), size=cap, replace=False)
            pairs = [pairs[i] for i in idx]
        for a, b in pairs:
            new = sorted(set(sel) - {a} | {b})
            yield new, a, b

    def optimize(self) -> OptimizerResult:
        p, cfg, rng = self.p, self.cfg, self.rng
        state = random_subset(p, rng, size=p.K)
        w_cur, f_cur = self._eval(state)
        best_w, best_f, best_subset = w_cur.copy(), f_cur, list(state)

        tabu_add: dict[int, int] = {}     # stock -> iteration until re-adding is forbidden
        tabu_remove: dict[int, int] = {}  # stock -> iteration until removing is forbidden
        history, curve = [], []
        no_improve = 0

        for it in range(cfg.tabu.iters):
            best_move = None  # (f, subset, w, removed, added)
            best_admissible = None
            for new, a, b in self._neighbors(state):
                w, f = self._eval(new)
                if best_move is None or f > best_move[0]:
                    best_move = (f, new, w, a, b)
                is_tabu = (it < tabu_add.get(b, -1)) or (it < tabu_remove.get(a, -1))
                aspires = f > best_f
                if (not is_tabu) or aspires:
                    if best_admissible is None or f > best_admissible[0]:
                        best_admissible = (f, new, w, a, b)

            chosen = best_admissible or best_move   # fall back to best if all tabu
            if chosen is None:
                break
            f, new, w, a, b = chosen
            state, w_cur, f_cur = new, w, f
            tabu_add[a] = it + cfg.tabu.tenure       # don't re-add the removed stock
            tabu_remove[b] = it + cfg.tabu.tenure    # don't remove the just-added stock

            if f_cur > best_f + EPS:
                best_w, best_f, best_subset = w_cur.copy(), f_cur, list(state)
                no_improve = 0
            else:
                no_improve += 1
            history.append(best_f)
            curve.append(f_cur)
            if no_improve >= cfg.tabu.patience:
                break

        best_subset = sorted(int(i) for i in np.where(best_w > EPS)[0]) or best_subset
        return OptimizerResult("Tabu", best_w, float(best_f), best_subset, history, curve,
                               self.evaluator.n_evals,
                               meta={"mode": "qp" if self.use_qp else "native"})
