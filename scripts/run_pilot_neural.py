"""
run_pilot_neural.py
-------------------
Second kill-criterion pilot: does the portfolio predict when the neural
hurdle beats the classical hurdle?

For each dataset: fit classical models and one neural hurdle on the same
train/test split, compare log scores, record margin.
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data.synthetic import generate_one, build_lagged_design
from src.diagnostics.portfolio import compute_portfolio
from src.models.classical import CLASSICAL_MODELS
from src.models.neural import NeuralHurdle
from src.evaluation.metrics import evaluate_all


GRID = {
    "gamma0":       [0.0, 0.3, 0.6],
    "dependence":   ["none", "occurrence", "magnitude"],
    "nonlinearity": ["linear", "quadratic", "threshold"],
}
N_REPS = 3
T = 500


def one(cfg, seed):
    data = generate_one(
        T=T, gamma0=cfg["gamma0"],
        dependence=cfg["dependence"],
        nonlinearity=cfg["nonlinearity"], seed=seed,
    )
    portfolio = compute_portfolio(data["Y"][: int(T * 0.6)])
    D, y = build_lagged_design(data["X"], data["Y"], p_lag=1)
    n_tr = int(0.7 * len(y))
    D_tr, y_tr = D[:n_tr], y[:n_tr]
    D_te, y_te = D[n_tr:], y[n_tr:]

    logs = {}
    for name, cls in CLASSICAL_MODELS.items():
        try:
            m = cls().fit(D_tr, y_tr)
            logs[name] = evaluate_all(m, D_te, y_te, y_max=200)["log_score"]
        except Exception:
            logs[name] = np.nan

    try:
        m = NeuralHurdle(epochs=150, patience=15, seed=seed).fit(D_tr, y_tr)
        logs["neural_hurdle"] = evaluate_all(m, D_te, y_te, y_max=200)["log_score"]
    except Exception as e:
        logs["neural_hurdle"] = np.nan

    best_classical = min((v for k, v in logs.items() if k != "neural_hurdle"
                          and np.isfinite(v)), default=np.nan)
    margin = best_classical - logs["neural_hurdle"] if np.isfinite(best_classical) else np.nan

    return {
        **cfg, "seed": seed,
        **{f"logS_{k}": v for k, v in logs.items()},
        "best_classical": best_classical,
        "neural_margin": margin,
        **portfolio,
    }


def main():
    rows = []
    seed = config.MASTER_SEED
    n_tasks = (len(GRID["gamma0"]) * len(GRID["dependence"])
               * len(GRID["nonlinearity"]) * N_REPS)
    print(f"Running {n_tasks} datasets")
    print()
    t0 = time.time()
    for g in GRID["gamma0"]:
        for dep in GRID["dependence"]:
            for nl in GRID["nonlinearity"]:
                for rep in range(N_REPS):
                    seed += 1
                    row = one({"gamma0": g, "dependence": dep,
                               "nonlinearity": nl, "rep": rep}, seed)
                    rows.append(row)
                    tag = f"g={g:.1f} dep={dep:<11} nl={nl:<10} rep={rep}"
                    print(f"  {tag}  neural_margin={row['neural_margin']:+.3f}")
    df = pd.DataFrame(rows)
    out = config.CSV_DIR / "pilot_neural.csv"
    df.to_csv(out, index=False)
    print()
    print(f"Wrote {out}  ({len(df)} rows, {time.time()-t0:.0f}s)")
    print()

    # Analysis
    print("Neural vs classical margin, by regime")
    print("-" * 70)
    summary = df.groupby(["gamma0", "dependence", "nonlinearity"])[
        "neural_margin"].mean().round(3)
    print(summary.to_string())
    print()

    print("Correlation of portfolio statistics with neural_margin")
    print("-" * 70)
    cols = ["zero_frac", "mean_pos", "od_pos", "acf1_zero",
            "acf1_pos", "run_length_mean", "run_length_cv"]
    corrs = df[cols + ["neural_margin"]].corr()["neural_margin"].drop("neural_margin")
    print(corrs.sort_values(key=abs, ascending=False).round(3).to_string())
    print()

    wins = (df["neural_margin"] > 0).sum()
    print(f"Neural wins in {wins}/{len(df)} datasets "
          f"({100 * wins / len(df):.1f}%)")


if __name__ == "__main__":
    main()