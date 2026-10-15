"""
src/models/neural.py
--------------------
Neural hurdle model with MLP backbone.

Architecture:
    trunk:   Linear(p, h) -> ReLU -> Linear(h, h) -> ReLU
    head_pi: Linear(h, 1) -> sigmoid     => pi_hat
    head_mu: Linear(h, 1) -> exp(clip)   => mu_hat

Training objective (hurdle NLL):
    L = -sum_t [ S_t log(pi_t) + (1-S_t) log(1-pi_t)
                 + S_t ( y_t log(mu_t) - mu_t
                        - log(1 - exp(-mu_t))
                        - lgamma(y_t + 1) ) ]

predicted mean: pi_hat * mu_hat / (1 - exp(-mu_hat))
predicted pmf : pi_hat * ZTP(mu_hat) with pmf[:, 0] += (1 - pi_hat)
"""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln

from .base import BaseModel
from .classical import _ztp_pmf, _poisson_pmf


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception as e:
        raise ImportError(
            "PyTorch is required for neural models. "
            "Install with: pip install torch"
        ) from e


class _MLPHurdleModule:
    """Lazy-constructed PyTorch module. Instantiated inside fit."""

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
                 lr=1e-3, batch=64, val_frac=0.2, seed=0):
        self.hidden = hidden
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.batch = batch
        self.val_frac = val_frac
        self.seed = seed

    def _prepare(self, D):
        D = np.asarray(D, dtype=np.float32)
        if not hasattr(self, "_mu_D"):
            self._mu_D = D.mean(axis=0, keepdims=True)
            self._sd_D = D.std(axis=0, keepdims=True)
            self._sd_D[self._sd_D < 1e-6] = 1.0
        return (D - self._mu_D) / self._sd_D

    def fit(self, D, y):
        torch, nn = _require_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        Dn = self._prepare(D)
        y = np.asarray(y, dtype=np.float32)
        n, p_in = Dn.shape

        n_val = max(int(n * self.val_frac), 5)
        n_tr = n - n_val
        if n_tr < 20:
            raise RuntimeError("Not enough rows to train neural model")

        Xtr = torch.from_numpy(Dn[:n_tr])
        ytr = torch.from_numpy(y[:n_tr])
        Xva = torch.from_numpy(Dn[n_tr:])
        yva = torch.from_numpy(y[n_tr:])

        self._module = _MLPHurdleModule(p_in, self.hidden)
        opt = torch.optim.Adam(self._module.parameters(), lr=self.lr)

        def hurdle_nll(logit_pi, log_mu, y_b):
            pi = torch.sigmoid(logit_pi).clamp(1e-6, 1 - 1e-6)
            mu = torch.exp(log_mu.clamp(-8.0, 8.0))
            S = (y_b > 0).float()

            ll_bin = S * torch.log(pi) + (1 - S) * torch.log(1 - pi)

            # Zero-truncated Poisson log-pmf, computed for every row.
            log_norm = torch.log1p(-torch.exp(-mu).clamp(max=1 - 1e-6))
            log_ztp = (y_b * torch.log(mu + 1e-12) - mu
                       - log_norm - torch.lgamma(y_b + 1))

            # Mask: positive rows contribute log_ztp, zero rows contribute 0.
            ll_pos = S * log_ztp

            return -(ll_bin + ll_pos).mean()

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self._module.net.train()
            perm = torch.randperm(n_tr)
            for i in range(0, n_tr, self.batch):
                idx = perm[i:i + self.batch]
                xb, yb = Xtr[idx], ytr[idx]
                logit_pi, log_mu = self._module(xb)
                loss = hurdle_nll(logit_pi, log_mu, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            self._module.net.eval()
            with torch.no_grad():
                logit_pi, log_mu = self._module(Xva)
                val_loss = float(hurdle_nll(logit_pi, log_mu, yva).item())

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                epochs_no_improve = 0
                best_state = {
                    "net": {k: v.clone() for k, v in self._module.net.state_dict().items()},
                    "head_pi": {k: v.clone() for k, v in self._module.head_pi.state_dict().items()},
                    "head_mu": {k: v.clone() for k, v in self._module.head_mu.state_dict().items()},
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            self._module.net.load_state_dict(best_state["net"])
            self._module.head_pi.load_state_dict(best_state["head_pi"])
            self._module.head_mu.load_state_dict(best_state["head_mu"])

        return self

    def _predict_components(self, D):
        torch, _ = _require_torch()
        Dn = self._prepare(D)
        self._module.net.eval()
        with torch.no_grad():
            X = torch.from_numpy(Dn)
            logit_pi, log_mu = self._module(X)
            pi = torch.sigmoid(logit_pi).numpy()
            mu = np.exp(np.clip(log_mu.numpy(), -8.0, 8.0))
        return pi, mu

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