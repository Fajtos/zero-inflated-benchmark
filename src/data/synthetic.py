"""
src/data/synthetic.py
---------------------
Controllable zero-inflated hurdle data generator.

DGP for t = 1, ..., T:
    logit(pi_t) = alpha0 + alpha_lag * S_{t-1} + alpha_x * x1_t
    log(mu_t)   = beta0 + beta_lag * log1p(Y_{t-1}) + beta_x * x1_t + f(x1_t)

    With probability gamma0:  Y_t = 0,  S_t = 0   (extra structural zero)
    Otherwise:                S_t ~ Bernoulli(pi_t)
                              Y_t ~ ZTP(mu_t) if S_t = 1 else 0

Controls:
    gamma0       fraction of extra structural zeros
    dependence   where lagged dependence enters:
                   'none'       alpha_lag=0.0, beta_lag=0.0
                   'occurrence' alpha_lag=1.0, beta_lag=0.0
                   'magnitude'  alpha_lag=0.0, beta_lag=0.6
                   'both'       alpha_lag=1.0, beta_lag=0.6
    nonlinearity shape of f(x1):
                   'linear'     f = 0
                   'quadratic'  f = 0.8 * x1^2
                   'threshold'  f = 0.8 * I(x1 > 0.3)
"""
from __future__ import annotations
import numpy as np


DEPENDENCE_MAP = {
    "none":       (0.0, 0.0),
    "occurrence": (1.0, 0.0),
    "magnitude":  (0.0, 0.6),
    "both":       (1.0, 0.6),
}


def _nonlinear_effect(x, kind, strength=0.8):
    if kind == "linear":
        return 0.0
    if kind == "quadratic":
        return strength * x * x
    if kind == "threshold":
        return strength * float(x > 0.3)
    raise ValueError(f"Unknown nonlinearity: {kind}")


def _sample_ztp(rng, mu, max_tries=100):
    """One draw from a zero-truncated Poisson with mean parameter mu."""
    if mu < 1e-8:
        return 1
    for _ in range(max_tries):
        y = rng.poisson(mu)
        if y > 0:
            return int(y)
    return 1


def generate_one(
    T,
    gamma0=0.0,
    dependence="none",
    nonlinearity="linear",
    alpha0=0.2,
    alpha_x=0.4,
    beta0=1.0,
    beta_x=0.3,
    burn_in=200,
    seed=0,
):
    """Generate one zero-inflated hurdle time series."""
    rng = np.random.default_rng(seed)
    alpha_lag, beta_lag = DEPENDENCE_MAP[dependence]

    total = T + burn_in
    t = np.arange(total)
    x1 = np.sin(2 * np.pi * t / 52.0).astype(np.float32)
    x2 = np.cos(2 * np.pi * t / 52.0).astype(np.float32)

    Y = np.zeros(total, dtype=np.int64)
    S = np.zeros(total, dtype=np.int64)
    pi_true = np.zeros(total, dtype=np.float64)
    mu_true = np.zeros(total, dtype=np.float64)

    for i in range(1, total):
        logit_pi = alpha0 + alpha_lag * S[i - 1] + alpha_x * x1[i]
        pi_i = 1.0 / (1.0 + np.exp(-np.clip(logit_pi, -30.0, 30.0)))
        pi_true[i] = pi_i

        f = _nonlinear_effect(float(x1[i]), nonlinearity)
        log_mu = beta0 + beta_lag * np.log1p(Y[i - 1]) + beta_x * x1[i] + f
        mu_i = float(np.exp(np.clip(log_mu, -5.0, 5.0)))
        mu_true[i] = mu_i

        if rng.random() < gamma0:
            Y[i] = 0
            S[i] = 0
            continue

        if rng.random() < pi_i:
            Y[i] = _sample_ztp(rng, mu_i)
            S[i] = 1
        else:
            Y[i] = 0
            S[i] = 0

    Y_obs = Y[burn_in:]
    X_obs = np.column_stack([x1[burn_in:], x2[burn_in:]]).astype(np.float32)
    pos = Y_obs[Y_obs > 0]

    return {
        "Y": Y_obs,
        "X": X_obs,
        "pi_true": pi_true[burn_in:],
        "mu_true": mu_true[burn_in:],
        "cov_names": ["x1", "x2"],
        "meta": {
            "T": int(T),
            "gamma0": float(gamma0),
            "dependence": dependence,
            "nonlinearity": nonlinearity,
            "alpha_lag": float(alpha_lag),
            "beta_lag": float(beta_lag),
            "zero_frac": float((Y_obs == 0).mean()),
            "mean_pos": float(pos.mean()) if pos.size else 0.0,
            "n_pos": int(pos.size),
            "seed": int(seed),
        },
    }


def generate_grid(sizes, gammas, dependences, nonlinearities, n_reps, base_seed):
    """Yield (config_dict, data_dict) over the full factorial grid."""
    seed = base_seed
    for T in sizes:
        for g in gammas:
            for dep in dependences:
                for nl in nonlinearities:
                    for rep in range(n_reps):
                        seed += 1
                        cfg = {
                            "T": T,
                            "gamma0": g,
                            "dependence": dep,
                            "nonlinearity": nl,
                            "rep": rep,
                            "seed": seed,
                        }
                        data = generate_one(
                            T,
                            gamma0=g,
                            dependence=dep,
                            nonlinearity=nl,
                            seed=seed,
                        )
                        yield cfg, data


def build_lagged_design(X, Y, p_lag=1):
    T = len(Y)
    Y_lag = np.zeros((T, p_lag), dtype=np.float32)
    for j in range(p_lag):
        Y_lag[j + 1:, j] = Y[: T - j - 1]
    S_lag = (Y_lag > 0).astype(np.float32)
    D_full = np.concatenate([Y_lag, S_lag, X], axis=1).astype(np.float32)
    return D_full[p_lag:], Y[p_lag:].astype(np.int64)
    """
    Align (X, Y) into a time-series regression design with p_lag lags of Y.

    Returns
    -------
    D : (T - p_lag, p_lag + p) float32 design matrix
        columns = [Y_{t-1}, ..., Y_{t-p}, X_t]
    y : (T - p_lag,) int64 target vector
    """
    T = len(Y)
    Y_lag = np.zeros((T, p_lag), dtype=np.float32)
    for j in range(p_lag):
        Y_lag[j + 1:, j] = Y[: T - j - 1]
    D_full = np.concatenate([Y_lag, X], axis=1).astype(np.float32)
    return D_full[p_lag:], Y[p_lag:].astype(np.int64)