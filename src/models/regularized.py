"""
src/models/regularized.py
-------------------------
Regularized hurdle models: LASSO, Ridge, Elastic Net.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from .base import BaseModel
from .classical import _ztp_pmf


def _make_logistic(penalty):
    """sklearn >= 1.8 uses l1_ratio, not penalty. Handle both."""
    if penalty == "l1":
        return LogisticRegression(C=1.0, solver="liblinear", max_iter=2000)
    if penalty == "l2":
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    return LogisticRegression(C=1.0, solver="saga", l1_ratio=0.5, max_iter=2000)


class _RegularizedHurdle(BaseModel):
    name = "RegularizedHurdle"
    is_hurdle = True
    penalty = "l2"

    def fit(self, D, y):
        D = np.asarray(D, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        S = (y > 0).astype(int)

        if len(np.unique(S)) < 2:
            self.clf_ = None
            self.pi_const_ = float(S.mean()) if len(S) else 0.5
        else:
            self.clf_ = _make_logistic(self.penalty)
            self.clf_.fit(D, S)

        pos = S == 1
        if pos.sum() >= 2:
            self.reg_ = Ridge(alpha=1.0).fit(D[pos], np.log1p(y[pos]))
        else:
            self.reg_ = None
            self.mu_const_ = float(np.log1p(y[y > 0]).mean()) if (y > 0).any() else 1.0
        return self

    def _components(self, D):
        D = np.asarray(D, dtype=np.float32)
        if self.clf_ is None:
            pi = np.full(len(D), self.pi_const_)
        else:
            pi = self.clf_.predict_proba(D)[:, 1]
        if self.reg_ is None:
            mu_pos = np.full(len(D), float(np.expm1(self.mu_const_)))
        else:
            mu_pos = np.clip(np.expm1(self.reg_.predict(D)), 1e-6, None)
        return pi, mu_pos

    def predict_mean(self, D):
        pi, mu_pos = self._components(D)
        return pi * mu_pos

    def predict_pmf(self, D, y_max=200):
        pi, mu_pos = self._components(D)
        # ZTP parameter mu such that its mean is mu_pos. Solve numerically:
        # mu / (1 - exp(-mu)) = mu_pos. Approximate by mu ≈ mu_pos for mu_pos > 2.
        # For small mu_pos, iterate.
        mu = mu_pos.copy()
        for _ in range(20):
            denom = 1.0 - np.exp(-mu)
            denom = np.where(denom < 1e-10, 1e-10, denom)
            mu = mu_pos * denom
        mu = np.clip(mu, 1e-6, 50.0)

        base = _ztp_pmf(mu, y_max)
        out = pi[:, None] * base
        out[:, 0] += (1.0 - pi)
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return out / row_sums


class LASSOHurdle(_RegularizedHurdle):
    name = "LASSO hurdle"
    penalty = "l1"


class RidgeHurdle(_RegularizedHurdle):
    name = "Ridge hurdle"
    penalty = "l2"


class ElasticNetHurdle(_RegularizedHurdle):
    name = "EN hurdle"
    penalty = "elasticnet"


REGULARIZED_MODELS = {
    "lasso_hurdle": LASSOHurdle,
    "ridge_hurdle": RidgeHurdle,
    "en_hurdle":    ElasticNetHurdle,
}