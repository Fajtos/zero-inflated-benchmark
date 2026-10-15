"""
src/evaluation/metrics.py
-------------------------
Evaluation metrics for zero-inflated count forecasting.
"""
from __future__ import annotations
import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff * diff)))


def mspe(y_true, y_pred):
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(diff * diff))


def log_score(pmf, y_true):
    pmf = np.asarray(pmf, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    n, K = pmf.shape
    y_clipped = np.clip(y_true, 0, K - 1)
    p_obs = pmf[np.arange(n), y_clipped]
    return float(-np.mean(np.log(np.clip(p_obs, 1e-300, None))))


def crps(pmf, y_true):
    pmf = np.asarray(pmf, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    n, Kp1 = pmf.shape
    K = Kp1 - 1
    cdf = np.cumsum(pmf, axis=1)
    k_grid = np.arange(Kp1)[None, :]
    indicator = (y_true[:, None] <= k_grid).astype(np.float64)
    sq = (cdf - indicator) ** 2
    return float(np.mean(sq.sum(axis=1)))


def coverage_90(pmf, y_true):
    pmf = np.asarray(pmf, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    n, Kp1 = pmf.shape
    cdf = np.cumsum(pmf, axis=1)
    lo = np.array([np.searchsorted(cdf[i], 0.05) for i in range(n)])
    hi = np.array([np.searchsorted(cdf[i], 0.95) for i in range(n)])
    inside = (y_true >= lo) & (y_true <= hi)
    return float(inside.mean())


def brier(y_true, pi_hat):
    S = (np.asarray(y_true) > 0).astype(np.float64)
    p = np.asarray(pi_hat, dtype=np.float64)
    return float(np.mean((S - p) ** 2))


def calibration(y_true, pi_hat, n_bins=10):
    S = (np.asarray(y_true) > 0).astype(np.float64)
    p = np.asarray(pi_hat, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pred = np.full(n_bins, np.nan)
    obs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        mask = (idx == b)
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            pred[b] = float(p[mask].mean())
            obs[b] = float(S[mask].mean())
    return centers, pred, obs, counts


def cond_mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def cond_rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    diff = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def evaluate_all(model, D_test, y_test, y_max=200):
    y_mean = model.predict_mean(D_test)
    pmf = model.predict_pmf(D_test, y_max=y_max)
    out = {
        "mae":         mae(y_test, y_mean),
        "rmse":        rmse(y_test, y_mean),
        "mspe":        mspe(y_test, y_mean),
        "log_score":   log_score(pmf, y_test),
        "crps":        crps(pmf, y_test),
        "coverage_90": coverage_90(pmf, y_test),
        "cond_mae":    cond_mae(y_test, y_mean),
        "cond_rmse":   cond_rmse(y_test, y_mean),
    }
    if getattr(model, "is_hurdle", False) and hasattr(model, "_components"):
        try:
            pi_hat = model._components(D_test)
            if isinstance(pi_hat, tuple):
                pi_hat = pi_hat[0]
            out["brier"] = brier(y_test, pi_hat)
        except Exception:
            out["brier"] = float("nan")
    else:
        out["brier"] = float("nan")
    return out