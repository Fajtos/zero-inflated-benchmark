"""
src/models/classical.py
-----------------------
Six classical count models: Poisson, NB, ZIP, ZINB, Hurdle-Poisson, Hurdle-NB.

All models implement:
    fit(D, y)                -> self
    predict_mean(D)          -> (n,) array of E[Y | X]
    predict_pmf(D, y_max)    -> (n, y_max+1) array of P(Y = y | X)

PMFs are truncated at y_max and renormalized so each row sums to 1.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import statsmodels.api as sm

from .base import BaseModel


# numerical helpers

def _log1mexp(x: np.ndarray) -> np.ndarray:
    """Stable log(1 - exp(-x)) for x > 0."""
    x = np.clip(x, 1e-12, None)
    return np.log(-np.expm1(-x))


def _renormalize_pmf(pmf: np.ndarray) -> np.ndarray:
    """Divide each row by its sum so rows sum to 1. Keeps nonnegative."""
    pmf = np.clip(pmf, 0.0, None)
    row_sums = pmf.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return pmf / row_sums


def _poisson_logpmf(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-12, None)
    return y * np.log(mu) - mu - gammaln(y + 1.0)


def _poisson_pmf(mu, y_max: int) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1, 1)
    mu = np.clip(mu, 1e-12, None)
    grid = np.arange(y_max + 1)[None, :].astype(np.float64)
    log_p = grid * np.log(mu) - mu - gammaln(grid + 1.0)
    return _renormalize_pmf(np.exp(log_p))


def _nbinom_logpmf(y: np.ndarray, mu: np.ndarray, phi: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-12, None)
    phi = max(float(phi), 1e-6)
    log_ratio = np.log(phi) - np.log(phi + mu)
    return (gammaln(y + phi) - gammaln(y + 1.0) - gammaln(phi)
            + phi * log_ratio + y * (np.log(mu) - np.log(phi + mu)))


def _nbinom_pmf(mu, phi: float, y_max: int) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1, 1)
    mu = np.clip(mu, 1e-12, None)
    phi = max(float(phi), 1e-6)
    grid = np.arange(y_max + 1)[None, :].astype(np.float64)
    log_p = (gammaln(grid + phi) - gammaln(grid + 1.0) - gammaln(phi)
             + phi * (np.log(phi) - np.log(phi + mu))
             + grid * (np.log(mu) - np.log(phi + mu)))
    return _renormalize_pmf(np.exp(log_p))


def _ztp_pmf(mu, y_max: int) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1, 1)
    mu = np.clip(mu, 1e-12, None)
    grid = np.arange(y_max + 1)[None, :].astype(np.float64)
    log_p = grid * np.log(mu) - mu - gammaln(grid + 1.0)
    log_norm = _log1mexp(mu)
    pmf = np.exp(log_p - log_norm)
    pmf[:, 0] = 0.0
    return _renormalize_pmf(pmf)


def _ztnb_pmf(mu, phi: float, y_max: int) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1, 1)
    mu = np.clip(mu, 1e-12, None)
    phi = max(float(phi), 1e-6)
    grid = np.arange(y_max + 1)[None, :].astype(np.float64)
    log_p = (gammaln(grid + phi) - gammaln(grid + 1.0) - gammaln(phi)
             + phi * (np.log(phi) - np.log(phi + mu))
             + grid * (np.log(mu) - np.log(phi + mu)))
    log_p0 = phi * (np.log(phi) - np.log(phi + mu))
    log_norm = np.log(-np.expm1(log_p0) + 1e-300)
    pmf = np.exp(log_p - log_norm)
    pmf[:, 0] = 0.0
    return _renormalize_pmf(pmf)


def _fit_mle(nll_fn, init: np.ndarray, args: tuple,
             n_restarts: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_restarts):
        x0 = init + (rng.normal(0.0, 0.1, size=init.shape) if k > 0 else 0.0)
        try:
            res = minimize(nll_fn, x0, args=args, method="L-BFGS-B",
                           options={"maxiter": 500})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    return None if best is None else best.x


def _add_intercept(D: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(D.shape[0]), D])


# Poisson GLM

class PoissonGLM(BaseModel):
    name = "Poisson GLM"

    def fit(self, D, y):
        self._fallback = float(np.log(max(y.mean(), 1e-3)))
        try:
            Dc = sm.add_constant(D, has_constant="add")
            self.model_ = sm.GLM(y, Dc, family=sm.families.Poisson()).fit()
            self._ok = True
        except Exception:
            self._ok = False
        return self

    def predict_mean(self, D):
        if not self._ok:
            return np.full(D.shape[0], float(np.exp(self._fallback)))
        Dc = sm.add_constant(D, has_constant="add")
        return np.clip(np.asarray(self.model_.predict(Dc)), 1e-8, None)

    def predict_pmf(self, D, y_max=200):
        return _poisson_pmf(self.predict_mean(D), y_max)


# Negative Binomial GLM

class NegBinGLM(BaseModel):
    name = "NB GLM"

    def fit(self, D, y):
        self._fallback = float(np.log(max(y.mean(), 1e-3)))
        try:
            from statsmodels.discrete.discrete_model import NegativeBinomial
            Dc = sm.add_constant(D, has_constant="add")
            res = NegativeBinomial(y, Dc).fit(disp=False, maxiter=200)
            self.beta_ = np.asarray(res.params[:-1], dtype=np.float64)
            self.alpha_ = float(res.params[-1])
            self._ok = True
        except Exception:
            try:
                Dc = sm.add_constant(D, has_constant="add")
                self.model_ = sm.GLM(y, Dc, family=sm.families.Poisson()).fit()
                self.alpha_ = 1e-3
                self._ok = True
            except Exception:
                self._ok = False
        return self

    def predict_mean(self, D):
        if not self._ok:
            return np.full(D.shape[0], float(np.exp(self._fallback)))
        if hasattr(self, "beta_"):
            return np.clip(np.exp(_add_intercept(D) @ self.beta_), 1e-8, None)
        Dc = sm.add_constant(D, has_constant="add")
        return np.clip(np.asarray(self.model_.predict(Dc)), 1e-8, None)

    def predict_pmf(self, D, y_max=200):
        phi = 1.0 / max(self.alpha_, 1e-4)
        return _nbinom_pmf(self.predict_mean(D), phi, y_max)


# ZIP

def _zip_nll(params, y, D):
    p = D.shape[1]
    alpha = params[: p + 1]
    beta = params[p + 1:]
    Dc = _add_intercept(D)
    mu = np.exp(np.clip(Dc @ beta, -10, 10))
    w = 1.0 / (1.0 + np.exp(-np.clip(Dc @ alpha, -30, 30)))
    log_p0 = np.log(w + (1.0 - w) * np.exp(-mu) + 1e-12)
    log_pp = np.log(1.0 - w + 1e-12) + _poisson_logpmf(y, mu)
    return -np.where(y == 0, log_p0, log_pp).sum()


class ZIP(BaseModel):
    name = "ZIP"

    def fit(self, D, y):
        p = D.shape[1]
        self._fallback = float(np.log(max(y.mean(), 1e-3)))
        init = np.zeros(2 * (p + 1))
        init[p + 1] = self._fallback
        self.params_ = _fit_mle(_zip_nll, init, (y, D), n_restarts=3)
        self._ok = self.params_ is not None
        return self

    def _components(self, D):
        p = D.shape[1]
        alpha = self.params_[: p + 1]
        beta = self.params_[p + 1:]
        Dc = _add_intercept(D)
        mu = np.exp(np.clip(Dc @ beta, -10, 10))
        w = 1.0 / (1.0 + np.exp(-np.clip(Dc @ alpha, -30, 30)))
        return w, mu

    def predict_mean(self, D):
        if not self._ok:
            return np.full(D.shape[0], float(np.exp(self._fallback)))
        w, mu = self._components(D)
        return (1.0 - w) * mu

    def predict_pmf(self, D, y_max=200):
        if not self._ok:
            return _poisson_pmf(np.full(D.shape[0], np.exp(self._fallback)), y_max)
        w, mu = self._components(D)
        base = _poisson_pmf(mu, y_max)
        out = (1.0 - w)[:, None] * base
        out[:, 0] += w
        return _renormalize_pmf(out)


# ZINB

def _zinb_nll(params, y, D):
    p = D.shape[1]
    alpha = params[: p + 1]
    beta = params[p + 1: 2 * p + 2]
    phi = float(np.exp(np.clip(params[-1], -5, 5)))
    Dc = _add_intercept(D)
    mu = np.exp(np.clip(Dc @ beta, -10, 10))
    w = 1.0 / (1.0 + np.exp(-np.clip(Dc @ alpha, -30, 30)))
    log_p0_nb = phi * (np.log(phi) - np.log(phi + mu))
    log_ppy = _nbinom_logpmf(y, mu, phi)
    log_p0 = np.log(w + (1.0 - w) * np.exp(log_p0_nb) + 1e-12)
    log_pp = np.log(1.0 - w + 1e-12) + log_ppy
    return -np.where(y == 0, log_p0, log_pp).sum()


class ZINB(BaseModel):
    name = "ZINB"

    def fit(self, D, y):
        p = D.shape[1]
        self._fallback = float(np.log(max(y.mean(), 1e-3)))
        init = np.zeros(2 * (p + 1) + 1)
        init[p + 1] = self._fallback
        init[-1] = np.log(2.0)
        self.params_ = _fit_mle(_zinb_nll, init, (y, D), n_restarts=3)
        self._ok = self.params_ is not None
        return self

    def _components(self, D):
        p = D.shape[1]
        alpha = self.params_[: p + 1]
        beta = self.params_[p + 1: 2 * p + 2]
        phi = float(np.exp(np.clip(self.params_[-1], -5, 5)))
        Dc = _add_intercept(D)
        mu = np.exp(np.clip(Dc @ beta, -10, 10))
        w = 1.0 / (1.0 + np.exp(-np.clip(Dc @ alpha, -30, 30)))
        return w, mu, phi

    def predict_mean(self, D):
        if not self._ok:
            return np.full(D.shape[0], float(np.exp(self._fallback)))
        w, mu, _ = self._components(D)
        return (1.0 - w) * mu

    def predict_pmf(self, D, y_max=200):
        if not self._ok:
            return _poisson_pmf(np.full(D.shape[0], np.exp(self._fallback)), y_max)
        w, mu, phi = self._components(D)
        base = _nbinom_pmf(mu, phi, y_max)
        out = (1.0 - w)[:, None] * base
        out[:, 0] += w
        return _renormalize_pmf(out)


# Hurdle-Poisson

def _hurdle_poisson_nll(params, y, D):
    p = D.shape[1]
    alpha = params[: p + 1]
    beta = params[p + 1:]
    Dc = _add_intercept(D)
    pi = 1.0 / (1.0 + np.exp(-np.clip(Dc @ alpha, -30, 30)))
    mu = np.exp(np.clip(Dc @ beta, -10, 10))
    mu = np.clip(mu, 1e-8, None)

    S = (y > 0).astype(np.float64)
    ll = S * np.log(pi + 1e-12) + (1.0 - S) * np.log(1.0 - pi + 1e-12)

    pos = S > 0.5
    if pos.any():
        yp = y[pos].astype(np.float64)
        mp = mu[pos]
        ll_p = yp * np.log(mp) - mp - gammaln(yp + 1.0) - _log1mexp(mp)
        ll[pos] += ll_p
    return -ll.sum()


class HurdlePoisson(BaseModel):
    name = "Hurdle-Poisson"
    is_hurdle = True

    def fit(self, D, y):
        p = D.shape[1]
        self._fallback = float(np.log(max(y.mean(), 1e-3)))
        init = np.zeros(2 * (p + 1))
        init[p + 1] = self._fallback
        self.params_ = _fit_mle(_hurdle_poisson_nll, init, (y, D), n_restarts=3)
        self._ok = self.params_ is not None
        return self

    def _components(self, D):
        p = D.shape[1]
        alpha = self.params_[: p + 1]
        beta = self.params_[p + 1:]
        Dc = _add_intercept(D)
        pi = 1.0 / (1.0 + np.exp(-np.clip(Dc @ alpha, -30, 30)))
        mu = np.clip(np.exp(np.clip(Dc @ beta, -10, 10)), 1e-8, None)
        return pi, mu

    def predict_mean(self, D):
        if not self._ok:
            return np.full(D.shape[0], float(np.exp(self._fallback)))
        pi, mu = self._components(D)
        return pi * mu / (1.0 - np.exp(-mu) + 1e-12)

    def predict_pmf(self, D, y_max=200):
        if not self._ok:
            return _poisson_pmf(np.full(D.shape[0], np.exp(self._fallback)), y_max)
        pi, mu = self._components(D)
        base = _ztp_pmf(mu, y_max)
        out = pi[:, None] * base
        out[:, 0] += (1.0 - pi)
        return _renormalize_pmf(out)


# Hurdle-NB

def _hurdle_nb_nll(params, y, D):
    p = D.shape[1]
    alpha = params[: p + 1]
    beta = params[p + 1: 2 * p + 2]
    phi = float(np.exp(np.clip(params[-1], -5, 5)))
    Dc = _add_intercept(D)
    pi = 1.0 / (1.0 + np.exp(-np.clip(Dc @ alpha, -30, 30)))
    mu = np.clip(np.exp(np.clip(Dc @ beta, -10, 10)), 1e-8, None)

    S = (y > 0).astype(np.float64)
    ll = S * np.log(pi + 1e-12) + (1.0 - S) * np.log(1.0 - pi + 1e-12)

    pos = S > 0.5
    if pos.any():
        yp = y[pos].astype(np.float64)
        mp = mu[pos]
        log_p0_nb = phi * (np.log(phi) - np.log(phi + mp))
        log_norm = np.log(-np.expm1(log_p0_nb) + 1e-300)
        ll_p = _nbinom_logpmf(yp, mp, phi) - log_norm
        ll[pos] += ll_p
    return -ll.sum()


class HurdleNB(BaseModel):
    name = "Hurdle-NB"
    is_hurdle = True

    def fit(self, D, y):
        p = D.shape[1]
        self._fallback = float(np.log(max(y.mean(), 1e-3)))
        init = np.zeros(2 * (p + 1) + 1)
        init[p + 1] = self._fallback
        init[-1] = np.log(2.0)
        self.params_ = _fit_mle(_hurdle_nb_nll, init, (y, D), n_restarts=3)
        self._ok = self.params_ is not None
        return self

    def _components(self, D):
        p = D.shape[1]
        alpha = self.params_[: p + 1]
        beta = self.params_[p + 1: 2 * p + 2]
        phi = float(np.exp(np.clip(self.params_[-1], -5, 5)))
        Dc = _add_intercept(D)
        pi = 1.0 / (1.0 + np.exp(-np.clip(Dc @ alpha, -30, 30)))
        mu = np.clip(np.exp(np.clip(Dc @ beta, -10, 10)), 1e-8, None)
        return pi, mu, phi

    def predict_mean(self, D):
        if not self._ok:
            return np.full(D.shape[0], float(np.exp(self._fallback)))
        pi, mu, phi = self._components(D)
        log_p0 = phi * (np.log(phi) - np.log(phi + mu))
        return pi * mu / (1.0 - np.exp(log_p0) + 1e-12)

    def predict_pmf(self, D, y_max=200):
        if not self._ok:
            return _poisson_pmf(np.full(D.shape[0], np.exp(self._fallback)), y_max)
        pi, mu, phi = self._components(D)
        base = _ztnb_pmf(mu, phi, y_max)
        out = pi[:, None] * base
        out[:, 0] += (1.0 - pi)
        return _renormalize_pmf(out)


# registry

CLASSICAL_MODELS = {
    "poisson_glm":    PoissonGLM,
    "negbin_glm":     NegBinGLM,
    "zip":            ZIP,
    "zinb":           ZINB,
    "hurdle_poisson": HurdlePoisson,
    "hurdle_nb":      HurdleNB,
}