"""
src/models/neural.py
--------------------
Neural hurdle model with MLP backbone and multi-seed ensemble.

Stability measures:
    log_mu clamped to [-6, 4]      caps mu at ~55
    gradient clipping at norm 5    prevents divergence on steep likelihoods
    ensemble over 3 seeds          outvotes single-seed training failures
"""
from __future__ import annotations
import numpy as np

from .base import BaseModel
from .classical import _ztp_pmf


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception as e:
        raise ImportError(
            "PyTorch is required for neural models. Install with: pip install torch"
        ) from e


LOG_MU_MIN = -6.0
LOG_MU_MAX = 4.0
GRAD_CLIP = 5.0


class _MLPHurdleModule:
    def __init__(self, p_in, hidden):
        torch, nn = _require_torch()
        self.net = nn.Sequential(
            nn.Linear(p_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_pi = nn.Linear(hidden, 1)
        self.head_mu = nn.Linear(hidden, 1)

    def parameters(self):
        return (list(self.net.parameters())
                + list(self.head_pi.parameters())
                + list(self.head_mu.parameters()))

    def __call__(self, x):
        import torch
        h = self.net(x)
        logit_pi = self.head_pi(h).squeeze(-1)
        log_mu = self.head_mu(h).squeeze(-1)
        return logit_pi, log_mu


class NeuralHurdle(BaseModel):
    name = "NeuralHurdle"
    is_hurdle = True

    def __init__(self, hidden=64, epochs=200, patience=20,
                 lr=1e-3, batch=64, val_frac=0.2, seed=0,
                 n_ensemble=3):
        self.hidden = hidden
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.batch = batch
        self.val_frac = val_frac
        self.seed = seed
        self.n_ensemble = int(max(1, n_ensemble))

    def _prepare(self, D):
        D = np.asarray(D, dtype=np.float32)
        if not hasattr(self, "_mu_D"):
            self._mu_D = D.mean(axis=0, keepdims=True)
            self._sd_D = D.std(axis=0, keepdims=True)
            self._sd_D[self._sd_D < 1e-6] = 1.0
        return (D - self._mu_D) / self._sd_D

    def _train_one(self, Dn, y, seed):
        torch, _ = _require_torch()
        torch.manual_seed(seed)
        np.random.seed(seed)

        n, p_in = Dn.shape
        n_val = max(int(n * self.val_frac), 5)
        n_tr = n - n_val
        if n_tr < 20:
            raise RuntimeError("Not enough rows to train neural model")

        Xtr = torch.from_numpy(Dn[:n_tr])
        ytr = torch.from_numpy(y[:n_tr])
        Xva = torch.from_numpy(Dn[n_tr:])
        yva = torch.from_numpy(y[n_tr:])

        module = _MLPHurdleModule(p_in, self.hidden)
        opt = torch.optim.Adam(module.parameters(), lr=self.lr)

        def hurdle_nll(logit_pi, log_mu, y_b):
            pi = torch.sigmoid(logit_pi).clamp(1e-6, 1 - 1e-6)
            mu = torch.exp(log_mu.clamp(LOG_MU_MIN, LOG_MU_MAX))
            S = (y_b > 0).float()
            ll_bin = S * torch.log(pi) + (1 - S) * torch.log(1 - pi)
            log_norm = torch.log1p(-torch.exp(-mu).clamp(max=1 - 1e-6))
            log_ztp = (y_b * torch.log(mu + 1e-12) - mu
                       - log_norm - torch.lgamma(y_b + 1))
            return -(ll_bin + S * log_ztp).mean()

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            module.net.train()
            perm = torch.randperm(n_tr)
            for i in range(0, n_tr, self.batch):
                idx = perm[i:i + self.batch]
                xb, yb = Xtr[idx], ytr[idx]
                logit_pi, log_mu = module(xb)
                loss = hurdle_nll(logit_pi, log_mu, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), GRAD_CLIP)
                opt.step()

            module.net.eval()
            with torch.no_grad():
                logit_pi, log_mu = module(Xva)
                val_loss = float(hurdle_nll(logit_pi, log_mu, yva).item())

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                epochs_no_improve = 0
                best_state = {
                    "net":     {k: v.clone() for k, v in module.net.state_dict().items()},
                    "head_pi": {k: v.clone() for k, v in module.head_pi.state_dict().items()},
                    "head_mu": {k: v.clone() for k, v in module.head_mu.state_dict().items()},
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            module.net.load_state_dict(best_state["net"])
            module.head_pi.load_state_dict(best_state["head_pi"])
            module.head_mu.load_state_dict(best_state["head_mu"])
        return module

    def fit(self, D, y):
        Dn = self._prepare(D)
        y = np.asarray(y, dtype=np.float32)
        self._modules = []
        for k in range(self.n_ensemble):
            m = self._train_one(Dn, y, seed=self.seed + 100 * k)
            self._modules.append(m)
        return self

    def _predict_components(self, D):
        torch, _ = _require_torch()
        Dn = self._prepare(D)
        X = torch.from_numpy(Dn)
        pis, mus = [], []
        for m in self._modules:
            m.net.eval()
            with torch.no_grad():
                logit_pi, log_mu = m(X)
                pi = torch.sigmoid(logit_pi).numpy()
                mu = np.exp(np.clip(log_mu.numpy(), LOG_MU_MIN, LOG_MU_MAX))
            pis.append(pi)
            mus.append(mu)
        return np.mean(pis, axis=0), np.mean(mus, axis=0)

    def predict_mean(self, D):
        pi, mu = self._predict_components(D)
        return pi * mu / (1.0 - np.exp(-mu) + 1e-12)

    def predict_pmf(self, D, y_max=200):
        pi, mu = self._predict_components(D)
        base = _ztp_pmf(mu, y_max)
        out = pi[:, None] * base
        out[:, 0] += (1.0 - pi)
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return out / row_sums


NEURAL_MODELS = {
    "neural_hurdle": NeuralHurdle,
}