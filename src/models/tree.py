"""
src/models/tree.py
------------------
Tree-based baselines: single-stage and hurdle versions of XGBoost and LightGBM.

Single-stage: fits a Poisson-objective regressor directly on y.
Hurdle:       classifier for P(Y>0) times a regressor on positive counts.
"""
from __future__ import annotations
import numpy as np

from .base import BaseModel
from .classical import _ztp_pmf


class XGBoostModel(BaseModel):
    name = "XGBoost"

    def fit(self, D, y):
        import xgboost as xgb
        self.model_ = xgb.XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            objective="count:poisson", n_jobs=-1, random_state=0,
            verbosity=0).fit(D, y)
        return self

    def predict_mean(self, D):
        return np.clip(self.model_.predict(D), 1e-6, None)


class XGBoostHurdle(BaseModel):
    name = "XGBoost hurdle"
    is_hurdle = True

    def fit(self, D, y):
        import xgboost as xgb
        S = (y > 0).astype(int)
        if len(np.unique(S)) < 2:
            self.clf_ = None
            self.pi_const_ = float(S.mean()) if len(S) else 0.5
        else:
            self.clf_ = xgb.XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                n_jobs=-1, random_state=0, verbosity=0,
                eval_metric="logloss").fit(D, S)
        pos = S == 1
        if pos.sum() >= 2:
            self.reg_ = xgb.XGBRegressor(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                objective="count:poisson", n_jobs=-1, random_state=0,
                verbosity=0).fit(D[pos], y[pos])
        else:
            self.reg_ = None
            self.mu_const_ = float(y[y > 0].mean()) if (y > 0).any() else 1.0
        return self

    def predict_mean(self, D):
        if self.clf_ is None:
            pi = np.full(len(D), self.pi_const_)
        else:
            pi = self.clf_.predict_proba(D)[:, 1]
        if self.reg_ is None:
            mu = np.full(len(D), self.mu_const_)
        else:
            mu = np.clip(self.reg_.predict(D), 1e-6, None)
        return pi * mu


class LightGBMModel(BaseModel):
    name = "LightGBM"

    def fit(self, D, y):
        import lightgbm as lgb
        self.model_ = lgb.LGBMRegressor(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            objective="poisson", n_jobs=-1, random_state=0,
            verbose=-1).fit(D, y)
        return self

    def predict_mean(self, D):
        return np.clip(self.model_.predict(D), 1e-6, None)


class LightGBMHurdle(BaseModel):
    name = "LightGBM hurdle"
    is_hurdle = True

    def fit(self, D, y):
        import lightgbm as lgb
        S = (y > 0).astype(int)
        if len(np.unique(S)) < 2:
            self.clf_ = None
            self.pi_const_ = float(S.mean()) if len(S) else 0.5
        else:
            self.clf_ = lgb.LGBMClassifier(
                n_estimators=400, num_leaves=31, learning_rate=0.05,
                n_jobs=-1, random_state=0, verbose=-1).fit(D, S)
        pos = S == 1
        if pos.sum() >= 2:
            self.reg_ = lgb.LGBMRegressor(
                n_estimators=400, num_leaves=31, learning_rate=0.05,
                objective="poisson", n_jobs=-1, random_state=0,
                verbose=-1).fit(D[pos], y[pos])
        else:
            self.reg_ = None
            self.mu_const_ = float(y[y > 0].mean()) if (y > 0).any() else 1.0
        return self

    def predict_mean(self, D):
        if self.clf_ is None:
            pi = np.full(len(D), self.pi_const_)
        else:
            pi = self.clf_.predict_proba(D)[:, 1]
        if self.reg_ is None:
            mu = np.full(len(D), self.mu_const_)
        else:
            mu = np.clip(self.reg_.predict(D), 1e-6, None)
        return pi * mu


TREE_MODELS = {
    "xgboost":         XGBoostModel,
    "xgboost_hurdle":  XGBoostHurdle,
    "lightgbm":        LightGBMModel,
    "lightgbm_hurdle": LightGBMHurdle,
}