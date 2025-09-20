# src/derivkit/forecast_kit.py
"""Provides the ForecastKit class.

A light wrapper around :class:`LikelihoodExpansion` that exposes a simple
API for Fisher and DALI tensors.

Typical usage example:

>>> forecaster = ForecastKit(function=model, theta0=theta0, cov=cov)
>>> F = forecaster.fisher()
>>> G, H = forecaster.dali()
"""

from typing import Callable, Sequence

import numpy as np

from derivkit.likelihood_expansion import LikelihoodExpansion


class ForecastKit:
    """Provides access to Fisher and DALI likelihood-expansion tensors.

    Attributes:
        fisher: Method returning the Fisher information matrix (shape (P, P)).
        dali: Method returning the doublet-DALI tensors G and H (shapes (P,P,P) and (P,P,P,P)).
    """

    def __init__(
        self,
        function: Callable[[Sequence[float] | np.ndarray], np.ndarray],
        theta0: Sequence[float] | np.ndarray,
        cov: np.ndarray,
    ):
        """Initialises the forecaster with model, fiducials, and covariance.

        Args:
            function: Model mapping parameters -> observables (1D array-like in, 1D array out).
            theta0: Fiducial parameter values (shape (P,)).
            cov: Observables covariance (shape (N, N)).
        """
        self._lx = LikelihoodExpansion(function, theta0, cov)

    def fisher(self, *, n_workers: int = 1) -> np.ndarray:
        """Return the Fisher information matrix with shape (P, P)."""
        return self._lx.fisher(n_workers=n_workers)

    def dali(self, *, n_workers: int = 1):
        """Return the doublet-DALI tensors (G, H) with shapes (P,P,P) and (P,P,P,P)."""
        return self._lx.dali(n_workers=n_workers)
