"""
src/diagnostics/portfolio.py
----------------------------
Model-free diagnostic portfolio for zero-inflated count time series.

Eight statistics computed from a training window:
    zero_frac          fraction of zeros
    mean_pos           mean of positive counts
    var_pos            variance of positive counts
    od_pos             var_pos / mean_pos
    acf1_zero          1-lag autocorrelation of the zero indicator
    acf1_pos           1-lag autocorrelation of positive counts
    run_length_mean    mean length of same-regime runs
    run_length_cv      coefficient of variation of run lengths
"""
from __future__ import annotations
import numpy as np


def _acf1(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-12:
        return 0.0
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(x[:-1] * x[1:]) / denom)


def _run_lengths(binary):
    if len(binary) == 0:
        return np.array([])
    changes = np.where(np.diff(binary) != 0)[0] + 1
    segments = np.split(binary, changes)
    return np.array([len(s) for s in segments], dtype=np.float64)


def compute_portfolio(y_train):
    y = np.asarray(y_train, dtype=np.float64)
    n = len(y)

    zero_mask = (y == 0)
    pos = y[~zero_mask]
    zero_frac = float(zero_mask.mean())

    if pos.size >= 2:
        mean_pos = float(pos.mean())
        var_pos = float(pos.var(ddof=1))
        od_pos = float(var_pos / max(mean_pos, 1e-8))
    elif pos.size == 1:
        mean_pos = float(pos[0])
        var_pos = 0.0
        od_pos = 0.0
    else:
        mean_pos = 0.0
        var_pos = 0.0
        od_pos = 0.0

    acf1_zero = _acf1(zero_mask.astype(np.float64))

    if pos.size >= 3:
        pos_lag = y[:-1][~zero_mask[1:]]
        pos_cur = y[1:][~zero_mask[1:]]
        if len(pos_cur) >= 3 and np.std(pos_cur) > 1e-12 and np.std(pos_lag) > 1e-12:
            acf1_pos = float(np.corrcoef(pos_lag, pos_cur)[0, 1])
        else:
            acf1_pos = 0.0
    else:
        acf1_pos = 0.0

    runs = _run_lengths(zero_mask.astype(int))
    run_length_mean = float(runs.mean()) if runs.size else 0.0
    run_length_cv = float(runs.std(ddof=1) / max(runs.mean(), 1e-8)) if runs.size >= 2 else 0.0

    return {
        "zero_frac":       zero_frac,
        "mean_pos":        mean_pos,
        "var_pos":         var_pos,
        "od_pos":          od_pos,
        "acf1_zero":       acf1_zero,
        "acf1_pos":        acf1_pos,
        "run_length_mean": run_length_mean,
        "run_length_cv":   run_length_cv,
        "n":               int(n),
        "n_pos":           int(pos.size),
    }
