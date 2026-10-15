"""
run_pilot.py
------------
Kill-criterion pilot: does the diagnostic portfolio predict model performance?

For each dataset in a small grid:
    1. Compute portfolio statistics on the training window.
    2. Run rolling-origin evaluation for all classical models.
    3. Record mean log_score per model.
    4. Determine the winning model by mean log_score.
    5. Save everything to CSV.

After the loop, we compute the correlation between portfolio statistics
and the winning model, and print a summary of winners by regime.
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data.synthetic import generate_one
from src.diagnostics.portfolio import compute_portfolio
from src.models.classical import CLASSICAL_MODELS
from src.experiment.rolling_origin import run_rolling_origin


GRID = {
    "gamma0":       [0.0, 0.3, 0.6],
    "dependence":   ["none", "occurrence", "magnitude"],
    "nonlinearity": ["linear", "quadratic"],
}
N_REPS = 2
T = 500
STEP = 20


def pilot_one(cfg, seed):
    data = generate_one(
        T=T,
        gamma0=cfg["gamma0"],
        dependence=cfg["dependence"],
        nonlinearity=cfg["nonlinearity"],
        seed=seed,
    )
    portfolio = compute_portfolio(data["Y"][: int(T * 0.6)])

    results = run_rolling_origin(
        data["Y"], data["X"],
        model_dict=CLASSICAL_MODELS,
        initial_train_frac=0.6,
        horizon=1,
        step=STEP,
        p_lag=1,
    )

    per_model = {}
    for r in results:
        per_model.setdefault(r["model"], []).append(r)
    summary = {}
    for name, rows in per_model.items():
        summary[name] = float(np.nanmean([r["log_score"] for r in rows]))

    winner = min(summary, key=summary.get)
    row = {
        **cfg,
        "seed": seed,
        **{f"logS_{k}": v for k, v in summary.items()},
        "winner": winner,
        "best_logS": summary[winner],
        **portfolio,
    }
    return row


def main():
    rows = []
    t0 = time.time()
    seed = config.MASTER_SEED

    for g in GRID["gamma0"]:
        for dep in GRID["dependence"]:
            for nl in GRID["nonlinearity"]:
                for rep in range(N_REPS):
                    seed += 1
                    cfg = {"gamma0": g, "dependence": dep,
                           "nonlinearity": nl, "rep": rep}
                    tag = f"g={g:.1f} dep={dep:<11} nl={nl:<10} rep={rep}"
                    t1 = time.time()
                    row = pilot_one(cfg, seed)
                    dt = time.time() - t1
                    rows.append(row)
                    print(f"  {tag}  winner={row['winner']:<18} "
                          f"logS={row['best_logS']:.3f}  ({dt:.1f}s)")

    df = pd.DataFrame(rows)
    out = config.CSV_DIR / "pilot_classical.csv"
    df.to_csv(out, index=False)
    print()
    print(f"Wrote {out}  ({len(df)} rows, {time.time()-t0:.0f}s total)")
    print()

    print("Winners by regime")
    print("-" * 70)
    print(df.groupby(["gamma0", "dependence", "nonlinearity"])["winner"]
            .agg(lambda x: x.value_counts().index[0])
            .to_string())

    print()
    print("Overall winner counts")
    print("-" * 70)
    print(df["winner"].value_counts().to_string())

    print()
    print("Portfolio means by winner")
    print("-" * 70)
    cols = ["zero_frac", "mean_pos", "od_pos", "acf1_zero", "acf1_pos",
            "run_length_mean", "run_length_cv"]
    print(df.groupby("winner")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()