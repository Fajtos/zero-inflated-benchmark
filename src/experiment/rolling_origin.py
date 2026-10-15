"""
src/experiment/rolling_origin.py
--------------------------------
Rolling-origin evaluation engine.

At each forecast origin k, every model is refit on rows [0, k) and asked
to predict rows [k, k + horizon). Metrics are computed per (model, origin).
No information from future data ever enters a fit.

Returns a list of dicts: one row per (model, origin), with all metrics.
"""
from __future__ import annotations
import numpy as np

from ..data.synthetic import build_lagged_design
from ..evaluation.metrics import evaluate_all


def run_rolling_origin(
    Y,
    X,
    model_dict,
    *,
    initial_train_frac=0.6,
    horizon=1,
    step=1,
    p_lag=1,
    y_max=None,
    verbose=False,
):
    """
    Parameters
    ----------
    Y : (T,) integer counts
    X : (T, p) covariates
    model_dict : dict name -> class with fit/predict/evaluate interface
    initial_train_frac : fraction of series used for the first training window
    horizon : forecast horizon (>=1)
    step : refit every `step` origins; step=1 means refit at every time point
    p_lag : number of lags of Y in the design
    y_max : truncation level for the PMF; if None, use max(50, 2*Y.max())
    verbose : print progress

    Returns
    -------
    list of dicts, each with keys 'model', 'origin', and metric names.
    """
    T = len(Y)
    D, y = build_lagged_design(X, Y, p_lag=p_lag)
    n_rows = len(y)

    if y_max is None:
        y_max = int(max(50, 2 * int(Y.max())))

    first_origin = int(T * initial_train_frac) - p_lag
    if first_origin < 10:
        first_origin = min(10, n_rows - horizon - 1)
    last_origin = n_rows - horizon

    results = []
    origins = list(range(first_origin, last_origin + 1, step))

    for k, origin in enumerate(origins):
        if verbose and (k % 10 == 0):
            print(f"  origin {origin}/{last_origin}  ({k+1}/{len(origins)})")

        D_train = D[:origin]
        y_train = y[:origin]
        D_test = D[origin:origin + horizon]
        y_test = y[origin:origin + horizon]

        for name, cls in model_dict.items():
            try:
                model = cls().fit(D_train, y_train)
                metrics = evaluate_all(model, D_test, y_test, y_max=y_max)
                row = {
                    "model": name,
                    "origin": int(origin),
                    "y_true_first": int(y_test[0]),
                    "y_true_mean": float(y_test.mean()),
                }
                row.update(metrics)
                results.append(row)
            except Exception as e:
                row = {
                    "model": name,
                    "origin": int(origin),
                    "y_true_first": int(y_test[0]),
                    "y_true_mean": float(y_test.mean()),
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "mspe": float("nan"),
                    "log_score": float("nan"),
                    "crps": float("nan"),
                    "coverage_90": float("nan"),
                    "cond_mae": float("nan"),
                    "cond_rmse": float("nan"),
                    "brier": float("nan"),
                    "error": f"{type(e).__name__}: {e}",
                }
                results.append(row)

    return results