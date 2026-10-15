"""
src/models/base.py
------------------
Abstract interface for every model in the benchmark.

Every model must implement:
    fit(D, y)               -> self
    predict_mean(D)         -> (n,) array of E[Y | X]

Models with a proper distribution (ZIP, ZINB, hurdle, neural hurdle)
should override predict_pmf. The default assumes Poisson.

Conventions
-----------
D : (n, p) float32 design matrix
y : (n,)   int64 count response

All predict_* methods return arrays of length n = D.shape[0].
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import poisson


class BaseModel(ABC):
    """Abstract base for all models in the benchmark."""

    # Human-readable name, set by each subclass.
    name: str = "base"

    # True for hurdle-type models, False otherwise.
    is_hurdle: bool = False

    @abstractmethod
    def fit(self, D: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Fit the model. Returns self to allow chaining."""
        ...

    @abstractmethod
    def predict_mean(self, D: np.ndarray) -> np.ndarray:
        """Return E[Y | X] for each row of D, shape (n,)."""
        ...

    def predict_pmf(self, D: np.ndarray, y_max: int = 200) -> np.ndarray:
        """
        Return predictive PMF, shape (n, y_max + 1).
        Default: Poisson centred on predict_mean. Subclasses override.
        """
        mean = np.clip(self.predict_mean(D), 1e-8, None)
        grid = np.arange(y_max + 1)
        return poisson.pmf(grid[None, :], mean[:, None])