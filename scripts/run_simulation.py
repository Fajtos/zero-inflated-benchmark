"""
run_simulation.py
-----------------
Parallel rolling-origin simulation over a factorial grid.

Each dataset is processed in its own worker process. Results are written
to a single CSV. Progress and ETA are printed to stdout.
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def _one_task(args):
    cfg, seed = args
    try:
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
        summary = {
            name: float(np.nanmean([r["log_score"] for r in rows]))
            for name, rows in per_model.items()
        }
        winner = min(summary, key=summary.get)
        return {
            **cfg,
            "seed": seed,
            **{f"logS_{k}": v for k, v in summary.items()},
            "winner": winner,
            "best_logS": summary[winner],
            **portfolio,
        }
    except Exception as e:
        return {
            **cfg,
            "seed": seed,
            "winner": "ERROR",
            "error": f"{type(e).__name__}: {e}",
        }


def main(n_workers=None):
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)
    print(f"Detected {os.cpu_count()} CPUs; using {n_workers} workers")
    print()

    tasks = []
    seed = config.MASTER_SEED
    for g in GRID["gamma0"]:
        for dep in GRID["dependence"]:
            for nl in GRID["nonlinearity"]:
                for rep in range(N_REPS):
                    seed += 1
                    tasks.append(({"gamma0": g, "dependence": dep,
                                   "nonlinearity": nl, "rep": rep}, seed))

    print(f"Running {len(tasks)} datasets")
    print()
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_one_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            if i % 5 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(tasks) - i) / max(rate, 1e-9)
                print(f"  {i:>3}/{len(tasks)}  "
                      f"elapsed={elapsed:>5.0f}s  eta={eta:>5.0f}s  "
                      f"winner={row.get('winner', '?'):<18}")

    df = pd.DataFrame(rows)
    out = config.CSV_DIR / "simulation_classical.csv"
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()
    main(n_workers=args.workers)