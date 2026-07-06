"""Generate the project report as a Word document (.docx) via Pandoc.

We emit a Markdown file with LaTeX math (`$...$` / `$$...$$`), Markdown tables, and
image references, then let Pandoc convert it to .docx. Pandoc turns the LaTeX into
NATIVE, editable Word equations (OMML) — no image hacks — and Markdown tables into
real Word tables. Results tables come from results*/ CSVs; figures from plots*/.

Run:  python build_report.py      (requires: pandoc on PATH, pip install tabulate)
"""
from __future__ import annotations

import subprocess
from math import comb
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
RES_H = ROOT / "results_hard"
PLT_H = ROOT / "plots_hard"
RES_S = ROOT / "results_scaling"
PLT_S = ROOT / "plots_scaling"


HEADER_RENAME = {
    "method": "Method", "mu_annual": "Expected Return (annual)", "vol_annual": "Volatility (annual)",
    "ticker": "Ticker", "weight": "Weight", "sector": "Sector", "solver": "Solver",
    "objective": "Objective", "fitness": "Fitness", "N": "Universe Size (N)", "K": "Cardinality (K)",
    "best_fitness_mean": "Best Fitness (mean)", "best_fitness_std": "Best Fitness (std)",
    "oos_sharpe": "OOS Sharpe", "oos_ann_vol": "OOS Volatility", "oos_ann_return": "OOS Return (annual)",
    "oos_max_drawdown": "OOS Max Drawdown", "gap_sharpe": "Sharpe Gap (IS minus OOS)",
    "n_holdings": "Number of Holdings", "runtime_mean_s": "Runtime (s, mean)", "runtime_s": "Runtime (s)",
    "n_evals_mean": "Evaluations (mean)", "gap_%": "Gap to Optimal (%)",
    "gap_mean_%": "Gap to Optimal (mean %)", "gap_std_%": "Gap to Optimal (std %)",
    "fitness_mean": "Fitness (mean)", "fitness_std": "Fitness (std)", "fitness_max": "Fitness (max)",
    "oos_sharpe_mean": "OOS Sharpe (mean)", "oos_sharpe_std": "OOS Sharpe (std)",
    "is_sharpe": "In-Sample Sharpe", "is_sharpe_mean": "In-Sample Sharpe (mean)",
    "is_sharpe_std": "In-Sample Sharpe (std)", "lambda_L1": "L1 Penalty",
    "eff_cardinality": "Effective Cardinality", "val_sharpe": "Validation Sharpe",
    "lambda_risk": "Risk Aversion", "gamma_cost": "Cost Weight", "effective_cost_bps": "Effective Cost (bps)",
    "val_ann_vol": "Validation Volatility", "ledoit_wolf": "Ledoit-Wolf", "sample": "Sample Covariance",
    "narrowed_by_shrinkage": "Gap Narrowed by Shrinkage", "n_rebalances": "Number of Rebalances",
    "avg_turnover": "Average Turnover", "total_cost": "Total Transaction Cost",
    "cardinality_control": "Cardinality Control", "long_only": "Long-Only",
    "active_stocks": "Number of Holdings", "solve_time_s": "Solve Time (s)", "formulation": "Formulation",
}
ROW_RENAME = {
    "A: MIQP (hard cardinality)": "Exact cardinality MIQP", "A: MIQP (exact cardinality)": "Exact cardinality MIQP",
    "C: L1 penalty (lam1=0.17)": "L1-regularised (λ₁=0.17)", "C: L1 penalty (λ₁ = 0.17)": "L1-regularised (λ₁=0.17)",
}


def _md(df: pd.DataFrame, index: bool = False) -> str:
    df = df.rename(columns=HEADER_RENAME)
    if df.index.name in HEADER_RENAME:
        df.index.name = HEADER_RENAME[df.index.name]
    return df.to_markdown(index=index, floatfmt=".4g") + "\n"


def table(path: Path, cols=None, index_col=None) -> str:
    if not path.exists():
        return f"*[missing table: {path.name}]*\n"
    df = pd.read_csv(path, index_col=index_col)
    if index_col is not None:
        df.index = [ROW_RENAME.get(str(i), i) for i in df.index]
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return _md(df, index=index_col is not None)


def fig(path: Path, caption: str, width: float = 5.6) -> str:
    if not path.exists():
        return f"*[missing figure: {path.name}]*\n"
    rel = path.relative_to(ROOT)
    return f"![{caption}]({rel}){{width={width}in}}\n"


def build_markdown() -> str:
    md: list[str] = []
    A = md.append

    # ---- metadata / title ---- #
    A("---")
    A('title: "Cardinality-Constrained Portfolio Optimization via Metaheuristics"')
    A('subtitle: "40.018 Heuristics and Systems Theory --- Course Project I"')
    A('author: "Group X --- [Member 1, ID] , [Member 2, ID] , [Member 3, ID]"')
    A('date: "July 2026"')
    A("geometry: margin=2.2cm")
    A("fontsize: 11pt")
    A("---\n")

    A("## Abstract\n")
    A("We solve the cardinality-constrained mean-variance portfolio selection problem, a "
      "mixed-integer quadratic program (MIQP) that is NP-hard owing to the combinatorial choice of "
      "which assets to hold. We implement and compare four metaheuristics --- Simulated Annealing (SA), "
      "Tabu Search, a Genetic Algorithm (GA), and an Ant Colony Optimization (ACO) hybrid that pairs "
      "the discrete subset search with an exact convex quadratic program (QP) for the weights --- to "
      "determine which best solves the cardinality-constrained problem. We benchmark them against "
      "naive heuristics and an exact Gurobi MIQP that certifies the optimum on small instances. On "
      "real S&P 500 data we show (i) ACO and Tabu match the certified optimum where SA falls short "
      "by 13--24%, (ii) the methods separate as the universe scales to 150 assets, (iii) a strict "
      "in-sample/out-of-sample protocol with Ledoit--Wolf shrinkage quantifies overfitting, and "
      "(iv) a comparison of the exact cardinality MIQP against an L1-regularized convex "
      "formulation.\n")

    # ================================================================= #
    A("# 1. Problem Background and Formulation\n")
    A("## 1.1 Background\n")
    A("Choosing a portfolio to maximise risk-adjusted return is a classic problem (Markowitz, "
      "1952). Real investors add a *cardinality* constraint: hold at most $K$ names to limit "
      "monitoring and transaction overhead, with per-asset and per-sector caps for "
      "diversification. The binary hold/do-not-hold decision turns the convex mean-variance QP "
      "into a mixed-integer quadratic program, which is NP-hard.\n")

    A("## 1.2 Mathematical formulation (the mean-variance objective)\n")
    A("Decision variables: continuous weights $w_i$ and binary selectors $z_i$ ($z_i=1$ iff asset "
      "$i$ is held). The objective maximises expected return, penalises variance via risk-aversion "
      "$\\lambda$, and penalises rebalancing cost via $\\gamma$:\n")
    A("$$\\max_{w,z}\\ \\sum_{i=1}^{N}\\mu_i w_i \\;-\\; \\lambda\\, w^{\\top}\\Sigma w "
      "\\;-\\; \\gamma\\sum_{i=1}^{N} c_i\\,\\lvert w_i - w_i^{0}\\rvert$$\n")
    A("subject to\n")
    A("$$\\begin{aligned}"
      "&\\sum_{i=1}^{N} w_i = 1 && \\text{(fully invested)}\\\\"
      "&0 \\le w_i \\le u_i\\, z_i && \\forall i \\quad\\text{(box / linking)}\\\\"
      "&\\sum_{i=1}^{N} z_i \\le K && \\text{(cardinality)}\\\\"
      "&\\sum_{i \\in s} w_i \\le L_s && \\forall\\, s \\quad\\text{(sector caps)}\\\\"
      "&z_i \\in \\{0,1\\} && \\forall i"
      "\\end{aligned}$$\n")
    A("where $\\mu$ is the annualised expected-return vector, $\\Sigma$ the annualised covariance, "
      "$u_i$ the per-name cap (0.20), $L_s$ the per-sector cap (0.30), $K$ the maximum holdings "
      "(10), $c_i$ unit transaction costs, and $w^{0}$ the incumbent portfolio. The box-linking "
      "constraint couples $w$ and $z$: an asset can carry weight only if selected. Note that "
      "$\\lambda$ trades return against *risk*; it does not control the number of holdings --- "
      "that is the role of the cardinality constraint.\n")

    A("## 1.3 Computational complexity\n")
    A("For a fixed subset the weights are a convex QP, solvable in polynomial time; the hardness "
      "lies entirely in the discrete choice of subset. The number of candidate subsets of size $K$ "
      "grows combinatorially, so enumeration is hopeless even at modest sizes:\n")
    comb_df = pd.DataFrame([{"N (universe)": N, "K": 12, "candidate subsets $\\binom{N}{K}$":
                             f"{comb(N,12):.2e}"} for N in (25, 50, 100, 150, 500)])
    A(comb_df.to_markdown(index=False) + "\n")
    A("This combinatorial explosion is why metaheuristics are appropriate. We use the exact MIQP "
      "solver (Gurobi) as a ground-truth benchmark on small instances, as permitted by the brief.\n")

    A("## 1.4 Objective formulations\n")
    A("The objective above (the mean-variance objective) is our main formulation. Alongside it we "
      "examine two alternatives that probe different modelling choices. The first replaces the fixed "
      "return-versus-risk trade-off with the **Sharpe ratio**, maximising excess return per unit of "
      "volatility directly. Because it divides by the square root of a quadratic it is non-convex "
      "over the choice of subset and has no mixed-integer quadratic form, so the exact solver cannot "
      "be applied --- this is the one setting where metaheuristics are necessary, not merely "
      "convenient:\n")
    A("$$\\max_{w}\\ \\frac{\\mu^{\\top}w - r_f}{\\sqrt{w^{\\top}\\Sigma w}}$$\n")
    A("The second alternative, suggested during our consultation, removes the binary cardinality "
      "constraint entirely and instead adds an **L1 (sparsity) penalty** on the weights. Because the "
      "L1 norm of a fully-invested long-only portfolio is constant, this penalty only induces "
      "sparsity once short positions are permitted; it then drives small holdings to zero, so tuning "
      "the penalty strength controls the effective number of names held. The result is a fully "
      "convex problem that approximates cardinality control without any integer variables (compared "
      "against the mean-variance objective in Section 5.5):\n")
    A("$$\\max_{w}\\ \\mu^{\\top}w - \\lambda\\, w^{\\top}\\Sigma w "
      "- \\lambda_{1}\\lVert w\\rVert_{1}$$\n")

    # ================================================================= #
    A("# 2. Data and Preprocessing\n")
    A("We pull daily adjusted-close prices for liquid S&P 500 large-caps via the yfinance API "
      "(3 years, 2021-06-30 to 2024-06-28; 754 trading days). For the scaling study we fetch the "
      "full S&P 500 constituent list with GICS sectors from Wikipedia and keep the top-$N$ names by "
      "median dollar volume (liquidity). Raw prices are cached to Parquet for reproducible offline "
      "runs and exported to an Excel workbook and CSVs (sheets: prices, returns, mu/sigma/sector, "
      "covariance).\n")
    A("From prices we compute daily simple returns, annualised expected returns "
      "$\\mu = 252\\cdot\\overline{r}$, and the annualised covariance $\\Sigma = 252\\cdot\\mathrm{cov}$. "
      "Two covariance estimators sit behind a flag: the plain sample covariance and Ledoit--Wolf "
      "shrinkage (Ledoit & Wolf, 2004), our main overfitting mitigation. A slice of the estimated "
      "inputs (top names by $\\mu$):\n")
    mss = RES.parent / "data" / "processed" / "mu_sigma_sector.csv"
    if mss.exists():
        d = pd.read_csv(mss).sort_values("mu_annual", ascending=False).head(10).round(3)
        A(_md(d))
    A("**Leakage control is enforced structurally.** The pipeline produces train / validation / "
      "test splits: $\\mu$ and $\\Sigma$ are estimated on training only; hyperparameters "
      "($\\lambda,\\gamma$, algorithm settings) are tuned on validation; the test window is read "
      "exactly once, for final evaluation. In code the optimiser cannot receive the test segment, "
      "so look-ahead bias is impossible by construction.\n")

    # ================================================================= #
    A("# 3. Methods\n")
    A("We implement four metaheuristics --- Simulated Annealing, Tabu Search, a Genetic Algorithm, "
      "and an Ant Colony Optimization hybrid --- and compare them head-to-head to determine which "
      "searches the cardinality-constrained space most effectively. To keep the comparison fair, "
      "all four share the same solution encoding, feasibility-repair operator and inner weight "
      "solver (Section 3.4); they differ only in how they explore the discrete choice of which "
      "stocks to hold.\n")
    A("## 3.1 Encoding, feasibility and repair\n")
    A("A solution is the pair $(z,w)$, stored as a full-length weight vector whose support is the "
      "selection. Constraint violations are measured and a repair operator projects any candidate "
      "back to feasibility: keep the top-$K$ names, clip to the box, scale down over-cap sectors, "
      "and renormalise to full investment. Residual infeasibility is discouraged by a penalty term.\n")

    A("## 3.2 Simulated Annealing\n")
    A("SA explores the $(z,w)$ space with add / remove / swap / perturb moves. A candidate with "
      "fitness change $\\Delta f$ is accepted with the Metropolis probability, under geometric "
      "cooling:\n")
    A("$$P(\\text{accept}) = \\min\\!\\left(1,\\ \\exp(\\Delta f / T)\\right), "
      "\\qquad T_{k+1} = \\rho_{\\text{cool}}\\, T_k$$\n")
    A("```")
    A("Algorithm 1: Simulated Annealing")
    A("create a random feasible portfolio; call it the current and the best portfolio")
    A("set the temperature to its high starting value")
    A("while the temperature is above the minimum:")
    A("    repeat several times at this temperature:")
    A("        propose a neighbour by adding, removing or swapping a stock")
    A("            (or nudging the weights), then repair it")
    A("        if the neighbour is better, accept it")
    A("        if it is worse, still accept it with a probability that")
    A("            shrinks as the temperature falls")
    A("        record it if it is the best portfolio seen so far")
    A("    lower the temperature by the cooling factor")
    A("return the best portfolio found")
    A("```\n")

    A("## 3.3 Tabu Search\n")
    A("Tabu Search is a memory-based local search. It uses the same swap neighbourhood but, instead "
      "of accepting moves at random, always takes the best available non-tabu move, forbids "
      "reversing a recent add or remove for a fixed tenure, and applies an aspiration rule that "
      "overrides the tabu status whenever a move yields a new global best. The memory stops the "
      "search from cycling back over recently visited portfolios.\n")
    A("```")
    A("Algorithm 2: Tabu Search")
    A("start from a random feasible portfolio of K stocks; record it as the best")
    A("keep a short tabu list (a memory of recently reversed moves)")
    A("repeat for a fixed number of iterations:")
    A("    list many neighbours formed by swapping one held stock for one unheld stock")
    A("    pick the best neighbour that is not forbidden by the tabu list")
    A("        (but always allow a move that beats the best portfolio so far)")
    A("    move to it, and forbid reversing this swap for a few iterations")
    A("    record it if it is the best portfolio so far")
    A("    stop early if there has been no improvement for a while")
    A("return the best portfolio found")
    A("```\n")

    A("## 3.4 Inner QP and the buy/sell linearisation\n")
    A("Every metaheuristic in this project shares the same weight solver, which is what keeps the "
      "comparison fair: the algorithms differ only in which stocks they choose to hold, not in how "
      "the money is divided among those stocks. Once a subset of stocks is fixed, finding the "
      "weights that maximise the objective is a convex quadratic programme --- a well-behaved "
      "problem with a single best answer. We solve it exactly using **cvxpy, an open-source Python "
      "library for convex optimisation** that lets us write the problem in mathematical form and "
      "passes it to a numerical solver.\n")
    A("The objective charges a transaction cost based on how far the new portfolio moves from the "
      "one we currently hold, written with an absolute value that is not smooth. We remove it with a "
      "buy/sell split. For each stock, let $w^{0}$ be its current (incumbent) weight --- what we "
      "already own. The new weight is written $w = w^{0} + w^{+} - w^{-}$, where $w^{+}$ is the "
      "amount we **buy** (how much we increase the holding) and $w^{-}$ is the amount we **sell**; "
      "both are non-negative, and at the optimum only one of them is ever positive. The total amount "
      "traded, the absolute change $\\lvert w - w^{0}\\rvert$, then equals $w^{+} + w^{-}$ --- a "
      "simple linear expression --- so the weight problem stays a convex QP:\n")
    A("$$w = w^{0} + w^{+} - w^{-}, \\quad w^{+},w^{-}\\ge 0, \\quad "
      "\\lvert w_i - w_i^{0}\\rvert = w_i^{+} + w_i^{-}$$\n")

    A("## 3.5 Ant Colony Optimization\n")
    A("Each ant builds a $K$-asset subset, selecting asset $i$ with probability proportional to "
      "pheromone $\\tau_i$ and heuristic desirability $\\eta_i = \\mu_i/\\sigma_i$; the inner QP "
      "then sets weights. Pheromone evaporates and is deposited on the best subset, with Max--Min "
      "bounds to resist premature convergence:\n")
    A("$$p_i = \\frac{\\tau_i^{\\alpha}\\,\\eta_i^{\\beta}}{\\sum_{j}\\tau_j^{\\alpha}\\,\\eta_j^{\\beta}}, "
      "\\qquad \\tau_i \\leftarrow (1-\\rho)\\,\\tau_i + \\rho\\,\\Delta\\tau_i^{\\text{best}}, "
      "\\qquad \\tau_{\\min} \\le \\tau_i \\le \\tau_{\\max}$$\n")
    A("```")
    A("Algorithm 3: Ant Colony Optimization (QP hybrid)")
    A("give every stock an equal starting pheromone level")
    A("give every stock a desirability score equal to its return divided by its risk")
    A("repeat for a fixed number of iterations:")
    A("    for each ant:")
    A("        build a portfolio of K stocks, picking them one at a time,")
    A("            favouring stocks with more pheromone and higher desirability")
    A("        use the inner QP to set the best weights for that subset")
    A("        remember it if it is the best portfolio so far")
    A("    reduce every pheromone level slightly (evaporation)")
    A("    add pheromone to the stocks in the best portfolio found (reinforcement)")
    A("    keep every pheromone level within a fixed minimum and maximum")
    A("return the best portfolio found")
    A("```\n")

    A("## 3.6 Genetic Algorithm\n")
    A("The GA encodes the subset as a chromosome and evolves a population with tournament "
      "selection, subset crossover, swap mutation, repair to enforce cardinality, and elitism. "
      "Weights come from the same inner QP, keeping it comparable.\n")

    A("## 3.7 Benchmarks\n")
    A("**Exact benchmark:** the full MIQP is modelled in Gurobi (academic licence) with binary "
      "$z$, the linking and sector constraints, and the buy/sell linearisation --- giving the "
      "proven optimum on small instances. **Naive heuristic baselines** are equal-weight ($1/N$), "
      "random subset sampling, and a greedy return/risk rule; these are *heuristics, not "
      "metaheuristics*, and serve as a floor the metaheuristics should beat.\n")

    # ================================================================= #
    A("# 4. Exact Benchmark and Gap-to-Optimal\n")
    A("On a small instance ($N=15$, $K=5$) Gurobi returns the proven optimum in milliseconds, "
      "giving a gap-to-optimal reference. All four metaheuristics reach it (0% gap), validating "
      "their correctness; the naive baselines do not:\n")
    A(table(RES_H / "gap_to_optimal.csv"))
    A("\nThis table is where the metaheuristics **demonstrably add value over the naive "
      "heuristics**. Against the Gurobi optimum, equal-weight leaves 22.3% of the objective on the "
      "table, the greedy return/risk rule 9.8%, and random sampling 2.3% --- whereas all four "
      "metaheuristics close the gap to 0%. Intelligently searching the subset space recovers the "
      "full 10--22% that simple rules forgo, and the advantage widens out-of-sample and at scale.\n")

    # ================================================================= #
    A("# 5. Results and Insights\n")

    A("## 5.1 Method comparison\n")
    A(table(RES_H / "master_table.csv",
            cols=["method", "best_fitness_mean", "best_fitness_std", "oos_sharpe", "oos_ann_vol",
                  "gap_sharpe", "n_holdings", "runtime_mean_s", "n_evals_mean"]))
    A("\nOn the harder instance (50 assets, $\\lambda=0.5$, $K=12$, five seeds) Tabu, GA and ACO "
      "all reach the Gurobi optimum with zero variance, while SA lands ~13% short and is noisy "
      "($\\pm0.022$). The convergence curves show SA plateauing below the others.\n")
    A(fig(PLT_H / "convergence.png",
          "Figure 1. Convergence (best-so-far fitness, mean +/- std over seeds): SA plateaus."))

    A("\n## 5.2 Scaling: robustness as the universe grows\n")
    A(table(RES_S / "scaling.csv",
            cols=["N", "method", "gap_mean_%", "gap_std_%", "runtime_mean_s", "n_evals_mean"]))
    A("\nScaling to $N=50,100,150$ most-liquid names yields a clear ranking: ACO and Tabu stay "
      "optimal, GA degrades (5--8% gap at $N\\ge100$), and SA fails throughout. The exact solver "
      "remains an exact benchmark at every size; this study shows which metaheuristics stay "
      "reliable as the search space explodes.\n")
    A(fig(PLT_S / "scaling.png",
          "Figure 2. Gap-to-optimal (left) and runtime, log scale (right) vs universe size.", 6.2))

    A("\n## 5.3 Algorithm stochasticity (seed robustness)\n")
    A("Metaheuristics are randomised, so a single run could be lucky or unlucky. Re-running each "
      "method under five different random seeds separates this algorithm stochasticity from "
      "estimation overfitting: the data and the problem are fixed, and only the random numbers "
      "change.\n")
    A(table(RES_H / "seed_robustness.csv"))
    A(fig(PLT_H / "seed_robustness.png", "Figure 3. Best fitness mean +/- std over five seeds.", 4.8))
    A("\n**Result found.** Tabu, GA and ACO are perfectly stable --- every seed reaches the same "
      "optimum (fitness 0.343 with zero standard deviation), so their quality does not depend on "
      "luck. SA is the outlier: its best fitness varies from run to run (0.298 ± 0.022) and its "
      "out-of-sample Sharpe swings widely (± 0.82). In practice this means a single SA run cannot be "
      "trusted --- it would need several restarts and a best-of selection to be reliable --- whereas "
      "the other three are dependable in one run. So SA's weakness is not only that it can fall "
      "short of the optimum, but that its outcome is unpredictable.\n")

    A("\n## 5.4 The Sharpe-ratio objective --- beyond the solver's reach\n")
    A("The Sharpe-ratio objective maximises return per unit of risk directly. Because of the ratio "
      "and square root it is non-convex over subsets and has **no exact MIQP form** --- so, unlike "
      "the mean-variance objective, there is no Gurobi benchmark to fall back on. This is precisely "
      "the regime where metaheuristics are indispensable rather than optional. All four still "
      "produce strong feasible portfolios:\n")
    A(table(RES / "variant_B.csv"))
    A("\n**Result found.** All four metaheuristics converge to the same portfolio on the Sharpe "
      "objective --- an identical in-sample Sharpe of 1.86 and out-of-sample Sharpe of 3.88. "
      "Because this objective has no exact MIQP form, no solver can certify the optimum here; but "
      "four independent search strategies agreeing on exactly the same selection is itself strong "
      "evidence that they have all reached the (near-)optimal subset, corroborating one another in "
      "the solver's absence. Where they differ is purely in speed: ACO finds the answer in about 4 "
      "seconds, GA in 9, Tabu in 24, and SA in 46 --- roughly ten times slower than ACO. On this "
      "harder, non-convex objective the practical question is therefore not *which* method finds the "
      "best portfolio (they tie) but which finds it most efficiently, and the ACO+QP hybrid is the "
      "clear winner --- the same efficiency ranking seen on the mean-variance objective.\n")

    A("\n## 5.5 Cardinality MIQP versus the L1-regularized formulation\n")
    A("Our professor suggested an alternative to the hard cardinality constraint: an L1 penalty "
      "$\\lambda_1\\lVert w\\rVert_1$ that induces sparsity *without* binary variables, tuned to "
      "the desired number of active stocks. We compare the two formulations directly. One subtlety "
      "drives the design: on a long-only, fully-invested book $\\lVert w\\rVert_1 = \\sum_i w_i = 1$ "
      "is **constant**, so the L1 term is inert --- it can only induce sparsity if short positions "
      "are permitted. The L1 formulation is therefore a *long-short* convex QP; we sweep "
      "$\\lambda_1$ to match roughly $K$ active names.\n")
    A(table(RES / "formulation_comparison.csv", index_col=0))
    A("\nThe sparsity of the L1 formulation is controlled smoothly by $\\lambda_1$:\n")
    A(table(RES_H / "variant_C_sweep.csv"))
    A("\n**Findings.** (i) *Near-identical selection:* the convex L1 relaxation recovers 9 of the "
      "exact MIQP's 9 names (adding only DIS) --- it finds essentially the same portfolio the "
      "NP-hard formulation does. (ii) *Tractability:* the L1 formulation is convex and solved "
      "globally in milliseconds, whereas the exact cardinality MIQP is NP-hard and needs Gurobi or a "
      "metaheuristic. (iii) *Trade-offs of the L1 approach:* it requires allowing short sales, only "
      "*approximately* controls cardinality (no exact $K$), and here gave a slightly lower "
      "out-of-sample Sharpe (3.53 vs 3.98). **Conclusion:** the L1 formulation is an excellent fast "
      "heuristic for *stock selection*, but the exact cardinality MIQP gives precise control and, on "
      "this data, better risk-adjusted performance --- the two are complementary, and the "
      "combinatorial MIQP remains the formulation that justifies our metaheuristics.\n")

    A("\n## 5.6 Hyperparameter sensitivity (validation-tuned)\n")
    sens = RES_H / "sensitivity.csv"
    if sens.exists():
        d = pd.read_csv(sens)
        d = d[[c for c in d.columns if not c.startswith("Unnamed")]]
        A(_md(d))
    A("\nWe sweep the two objective weights --- the risk-aversion $\\lambda$ and the "
      "transaction-cost multiplier $\\gamma$ --- and score each setting on the validation split "
      "only, never on test, so the choice cannot leak look-ahead information into the final result.\n")
    A(fig(PLT_H / "sensitivity_val_sharpe.png",
          "Figure 4. Validation Sharpe over the lambda x gamma grid (tuned before any test look).", 4.8))
    A("\n**Result found.** $\\lambda$ is the dominant knob: raising it from 1 to 10 steadily lowers "
      "both volatility (0.143 to 0.116) and validation Sharpe (1.89 to 1.00) --- the expected "
      "return-for-risk trade-off. $\\gamma$, by contrast, barely moves anything at realistic cost "
      "levels; only at a stressed 50 bps ($\\gamma=5$) combined with an aggressive $\\lambda=1$ does "
      "it begin to bite. Validation prefers the low-$\\lambda$, low-$\\gamma$ corner (Sharpe 1.89), "
      "which is the single configuration we then evaluate on test. The practical lesson is that risk "
      "aversion, not transaction cost, drives the portfolio at the cost levels a real large-cap book "
      "would face.\n")

    A("\n## 5.7 Overfitting: in-sample vs out-of-sample\n")
    A("A portfolio can look excellent on the data it was built from simply by fitting noise. To "
      "expose this, we freeze each train-optimised portfolio and evaluate it on the untouched test "
      "window, then report the in-sample versus out-of-sample Sharpe gap explicitly --- making "
      "overfitting a headline result rather than a footnote.\n")
    A(fig(PLT_H / "is_oos_gap.png",
          "Figure 5. In-sample vs out-of-sample Sharpe per method (the overfitting gap)."))
    A(fig(PLT_H / "oos_equity.png",
          "Figure 6. Out-of-sample equity curves: metaheuristics vs benchmarks."))
    A("\n**Result found.** The outcome is the opposite of the usual overfitting warning. Instead of "
      "a high in-sample Sharpe collapsing out-of-sample, every method's out-of-sample Sharpe is "
      "*higher* than its in-sample Sharpe (Figure 5). This does not prove overfitting is impossible; "
      "it means the held-out 2024 window simply happened to be kinder to these portfolios than the "
      "training period, so they generalised well instead of breaking down. The equity curves "
      "(Figure 6) confirm it --- the metaheuristic portfolios compound smoothly through the test "
      "window while equal-weight lags well behind. Because a single favourable window can flatter "
      "any method, we do not lean on this result alone: the scaling, seed-robustness and "
      "walk-forward tests are what actually establish the method ranking.\n")

    A("\n## 5.8 Overfitting mitigation: sample vs Ledoit--Wolf covariance\n")
    A("Estimation error in the covariance matrix is a classic source of overfitting. Ledoit--Wolf "
      "shrinkage pulls the noisy sample covariance towards a stable, well-conditioned target, and we "
      "test whether it narrows the in-sample/out-of-sample gap by running the entire pipeline both "
      "ways.\n")
    A(table(RES_H / "shrinkage_gap.csv", index_col=0))
    A(fig(PLT_H / "shrinkage_gap.png", "Figure 7. Does shrinkage narrow the overfitting gap?", 4.8))
    A("\n**Result found.** The effect is small and mixed here. Shrinkage clearly helps SA --- the "
      "magnitude of its gap shrinks from 2.95 to 2.39 --- but barely changes GA or ACO. The reason "
      "is that our estimation window is already well-conditioned: 502 daily observations for 50 "
      "assets means the sample covariance is not especially noisy, so there is little for shrinkage "
      "to fix. Shrinkage earns its keep when data is scarce relative to the number of assets, which "
      "is exactly the regime of the walk-forward backtest's short trailing windows; on the "
      "comfortable main split it acts as a sensible safeguard rather than a decisive lever.\n")

    A("\n## 5.9 Walk-forward backtest\n")
    A("Every result so far freezes a single portfolio and evaluates it once. A walk-forward backtest "
      "is more demanding and more realistic: it repeatedly re-estimates the expected returns and "
      "covariance on a trailing window, re-optimises the portfolio, holds it for the next period, "
      "then rolls forward and rebalances --- paying a transaction cost each time it trades. This "
      "mimics how a portfolio would actually be run over time and tests whether an edge *persists* "
      "across many out-of-sample windows, rather than surviving in a single lucky test period.\n")
    A(table(RES_H / "walkforward.csv"))
    A(fig(PLT_H / "walkforward_equity.png",
          "Figure 8. Walk-forward equity curves, net of transaction cost."))
    A("\n**Result found.** The edge persists across the rolling windows: every method grows $1 into "
      "roughly $1.5--1.8 over the three-year walk, rising steadily through all four rebalances "
      "rather than depending on a single window. The method ranking from our earlier experiments "
      "also reappears even after transaction costs are charged. Tabu, GA and ACO are "
      "indistinguishable --- they re-select the same near-optimal subset each window and reach an "
      "out-of-sample Sharpe of 3.26 (annual return 0.81, max drawdown -0.10). SA trails at a Sharpe "
      "of 2.77 (return 0.57) and, tellingly, trades more --- an average turnover of 0.80 against "
      "0.65 for the others. Its weaker search settles on different subsets from window to window, so "
      "it rebalances more heavily yet ends up with less. The finding is therefore that stronger "
      "subset search gives higher and more stable out-of-sample performance even after costs, and "
      "that this advantage holds up under repeated re-estimation and rebalancing --- not just in a "
      "single frozen test.\n")
    A("One qualification when reading the numbers: 2021--2024 was an unusually strong period for "
      "these large-cap names, so the absolute Sharpe ratios near 3 are flattering and reflect the "
      "market regime, not a forecast. The reliable signal here is the *relative* result --- the "
      "metaheuristic ranking survives out-of-sample and after costs --- rather than the headline "
      "return.\n")

    A("\n## 5.10 ACO interpretability\n")
    A("Unlike the other methods, which return only a final portfolio, ACO leaves a trail: the "
      "pheromone level on each stock records how often strong portfolios have used it. Reading that "
      "trail turns a black-box answer into an interpretable ranking of which assets the search "
      "actually values.\n")
    A(fig(PLT_H / "aco_pheromone.png",
          "Figure 9. ACO pheromone evolution over iterations (most-selected assets)."))
    A(fig(PLT_H / "aco_selection.png",
          "Figure 10. Most-persistently-selected stocks across ACO ants.", 4.8))
    A("\n**Result found.** As the iterations progress, pheromone concentrates on a small, stable set "
      "of names while the rest decay towards the floor (Figure 9). The selection-frequency chart "
      "(Figure 10) makes this concrete: a handful of stocks are chosen by almost every ant, and "
      "these are precisely the names that make up the final optimal portfolio. This gives ACO a "
      "diagnostic the other methods lack --- we can see not just *what* the recommended portfolio is "
      "but *which* stocks the search consistently judged worth holding, and how strongly. For a "
      "portfolio manager, that transparency is valuable in its own right.\n")

    A("\n## 5.11 Recommended portfolio\n")
    A("On the main problem (50 liquid large-caps, $K=10$, $\\lambda=5$, Ledoit--Wolf), ACO returns "
      "the portfolio below, which Gurobi certifies as globally optimal (gap $<10^{-11}$). It "
      "respects all constraints (per-sector $\\le 30\\%$, per-name $\\le 20\\%$) and achieves an "
      "out-of-sample Sharpe of 3.98.\n")
    A(table(RES / "recommended_portfolio.csv"))
    A(fig(PLT_H / "risk_return.png", "Figure 11. Out-of-sample risk-return of candidate portfolios.", 4.8))

    # ================================================================= #
    A("\n# 6. Conclusions\n")
    A("We formulated cardinality-constrained portfolio selection as an NP-hard MIQP and compared "
      "four metaheuristics on it --- Simulated Annealing, Tabu Search, a Genetic Algorithm, and an "
      "Ant Colony Optimization hybrid --- all benchmarked against an exact Gurobi MIQP and naive "
      "heuristics. The solver certifies the "
      "optimum on small instances, where all metaheuristics achieve 0% gap; on harder and larger "
      "instances a clear robustness ranking emerges --- ACO $\\approx$ Tabu > GA > SA. A "
      "disciplined train/validation/test protocol with Ledoit--Wolf shrinkage quantified "
      "overfitting, the non-convex Sharpe objective demonstrated a regime beyond the solver's "
      "reach, and the L1 comparison showed a convex relaxation recovers nearly the same selection "
      "as the exact formulation. Of the four, ACO emerges as the strongest: it matches the optimum, is the most "
      "evaluation-efficient, and yields interpretable pheromone insights, with Tabu and GA close "
      "behind and SA the least reliable.\n")

    A("# References\n")
    for r_ in [
        "Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77--91.",
        "Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional "
        "covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365--411.",
        "Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated "
        "Annealing. *Science*, 220(4598), 671--680.",
        "Glover, F. (1989). Tabu Search --- Part I. *ORSA Journal on Computing*, 1(3), 190--206.",
        "Dorigo, M., & Stutzle, T. (2004). *Ant Colony Optimization*. MIT Press.",
        "Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. Univ. of Michigan Press.",
        "Chang, T.-J., Meade, N., Beasley, J. E., & Sharaiha, Y. M. (2000). Heuristics for "
        "cardinality constrained portfolio optimisation. *Computers & Operations Research*, 27, 1271--1302.",
        "Brodie, J., Daubechies, I., De Mol, C., Giannone, D., & Loris, I. (2009). Sparse and "
        "stable Markowitz portfolios. *PNAS*, 106(30), 12267--12272.",
        "Diamond, S., & Boyd, S. (2016). CVXPY: A Python-embedded modeling language for convex "
        "optimization. *JMLR*, 17(83), 1--5.",
        "Gurobi Optimization, LLC (2024). *Gurobi Optimizer Reference Manual*.",
    ]:
        A(f"- {r_}")
    A("")

    A("# Contribution Summary\n")
    A("[Member 1]: problem formulation, SA and Tabu implementation, report writing. "
      "[Member 2]: data pipeline, ACO + inner QP, Gurobi model and benchmarking. "
      "[Member 3]: GA, backtesting/overfitting analysis, scaling and L1 comparison, figures.\n")

    A("# Appendix A. Reproducibility\n")
    A("```")
    A("pip install -r requirements.txt")
    A("python run_all.py                                  # full suite (easy config)")
    A("python run_all.py --override config/hard.yaml      # hard config, 5 seeds")
    A("python -m experiments.exp_scaling --override config/scaling.yaml")
    A("python build_report.py                             # regenerate this report")
    A("pytest -q                                          # 54 unit tests")
    A("```")

    return "\n".join(md)


def build() -> Path:
    md_path = ROOT / "report" / "report.md"
    out = ROOT / "report" / "HST_Project_Report.docx"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_markdown())
    subprocess.run(
        ["pandoc", str(md_path.relative_to(ROOT)), "-o", str(out.relative_to(ROOT)),
         "--from", "markdown+tex_math_dollars+pipe_tables", "--resource-path", str(ROOT)],
        cwd=ROOT, check=True,
    )
    return out


if __name__ == "__main__":
    p = build()
    print(f"Report written to {p}")
