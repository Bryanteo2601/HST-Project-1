"""Universe-scaling study: gap-to-optimal and runtime vs universe size N.

For N in {50, 100, 150} (nested, most-liquid first) it solves the exact MIQP
(Gurobi) for the optimal reference and runs each metaheuristic over several seeds.
As N grows the discrete subset space explodes: the exact solver slows (and may hit
its time limit) while the metaheuristics stay fast but start showing a real
gap-to-optimal -- the canonical justification for using metaheuristics.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.model import ProblemData  # noqa: E402
from src.benchmarks import exact_miqp, gap_to_optimal  # noqa: E402
from src.experiment import METHODS, save_table  # noqa: E402


def run(cfg, bundle, results_dir: Path, plots_dir: Path,
        sizes=(50, 100, 150), miqp_time_limit: int = 120) -> pd.DataFrame:
    p_full = ProblemData.from_segment(bundle.train, cfg)
    sizes = [n for n in sizes if n <= p_full.n]
    rows = []

    for N in sizes:
        p = p_full.restrict(range(N))                 # top-N most liquid (nested)
        miqp = exact_miqp(p, cfg, time_limit=miqp_time_limit)
        opt = miqp.fitness if miqp.available else None
        rows.append({"N": N, "method": "MIQP", "fitness_mean": round(opt, 5) if opt else np.nan,
                     "fitness_std": 0.0, "gap_mean_%": 0.0, "gap_std_%": 0.0,
                     "runtime_mean_s": round(miqp.runtime_s, 2), "n_evals_mean": np.nan,
                     "proven_optimal": miqp.proven_optimal if miqp.available else False})
        tag = "optimal" if miqp.proven_optimal else "INCUMBENT (time-limited)"
        print(f"[N={N}] MIQP {tag}: fit={opt} in {miqp.runtime_s:.1f}s")

        for name, Opt in METHODS.items():
            fits, gaps, rts, nevs = [], [], [], []
            for seed in cfg.experiment.seeds:
                t0 = time.time()
                r = Opt(p, cfg, seed=seed).optimize()
                rts.append(time.time() - t0)
                fits.append(r.best_fitness); nevs.append(r.n_evals)
                if opt is not None:
                    gaps.append(gap_to_optimal(r.best_fitness, opt) * 100)
            rows.append({"N": N, "method": name,
                         "fitness_mean": round(float(np.mean(fits)), 5),
                         "fitness_std": round(float(np.std(fits)), 5),
                         "gap_mean_%": round(float(np.mean(gaps)), 3) if gaps else np.nan,
                         "gap_std_%": round(float(np.std(gaps)), 3) if gaps else np.nan,
                         "runtime_mean_s": round(float(np.mean(rts)), 2),
                         "n_evals_mean": round(float(np.mean(nevs)), 1), "proven_optimal": np.nan})
            print(f"        {name:5s} gap={np.mean(gaps) if gaps else float('nan'):6.2f}%  "
                  f"runtime={np.mean(rts):5.1f}s  evals={np.mean(nevs):.0f}")

    df = pd.DataFrame(rows)
    save_table(df, results_dir / "scaling.csv")
    _plot(df, sizes, plots_dir / "scaling.png")
    print(f"\n=== Universe scaling ===\n{df.to_string(index=False)}")
    return df


def _plot(df: pd.DataFrame, sizes, path: Path) -> None:
    methods = [m for m in df["method"].unique() if m != "MIQP"]
    colors = {"SA": "C0", "Tabu": "C1", "GA": "C2", "ACO": "C3", "MIQP": "C4"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # LEFT: solution quality vs universe size
    for m in methods:
        d = df[df["method"] == m].sort_values("N")
        ax1.errorbar(d["N"], d["gap_mean_%"], yerr=d["gap_std_%"], marker="o", capsize=4,
                     color=colors.get(m), label=m)
    ax1.set_xlabel("universe size N"); ax1.set_ylabel("gap-to-optimal (%)")
    ax1.set_title("Solution quality vs universe size"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_xticks(list(sizes))

    # RIGHT: quality vs runtime trade-off (best = bottom-left). A scatter shows
    # which method is both accurate and fast, which a runtime-vs-N line hides.
    for m in list(methods) + ["MIQP"]:
        d = df[df["method"] == m]
        ax2.scatter(d["runtime_mean_s"], d["gap_mean_%"], s=70, edgecolor="k", lw=0.4, zorder=3,
                    color=colors.get(m), marker="X" if m == "MIQP" else "o",
                    label=m + (" (exact)" if m == "MIQP" else ""))
    ax2.set_xscale("log")
    ax2.set_xlabel("runtime (s, log scale)"); ax2.set_ylabel("gap-to-optimal (%)")
    ax2.set_title("Quality vs runtime  (best = bottom-left)")
    ax2.grid(alpha=0.3, which="both"); ax2.legend(fontsize=8)
    ax2.annotate("benchmark", xy=(0.03, 1.5), xytext=(0.5, 10), fontsize=10, color="green",
                 arrowprops=dict(arrowstyle="->", color="green"))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    from experiments._common import parse_and_load, ROOT
    cfg, bundle = parse_and_load()
    results = ROOT / cfg.experiment.out_dir
    plots = ROOT / cfg.experiment.out_dir.replace("results", "plots")
    run(cfg, bundle, results, plots)
