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

    Methods:
        fisher: Method returning the Fisher information matrix (shape (P, P)) with
        P being the number of model parameters.
        dali: Method returning the doublet-DALI tensors G and H (shapes (P,P,P) and (P,P,P,P)).

    Notes:
        This class is a light wrapper around :class:`LikelihoodExpansion`
        that exposes a simple API for Fisher and DALI tensors.
        P = n_parameters = len(theta0)
        N = n_observables = cov.shape[0]
    """

    def __init__(
        self,
        function: Callable[[Sequence[float] | np.ndarray], np.ndarray],
        theta0: Sequence[float] | np.ndarray,
        cov: np.ndarray,
    ):
        """Initialises the forecaster with model, fiducials, and covariance.

        Args:
            function: Model mapping parameters to observables (1D array-like in, 1D array out).
            theta0: Fiducial parameter values (shape (P,)).
            cov: Observables covariance (shape (N, N)).

        Notes:
            The model function should accept a 1D array-like input of shape (P,)
            and return a 1D NumPy array of shape (N,). Here, P is the number of
            model parameters (P = len(theta0)) and N is the number of observables (N = cov.shape[0]).
        """
        self._lx = LikelihoodExpansion(function, theta0, cov)

    def fisher(self, *, n_workers: int = 1):
        """Return the Fisher information matrix with shape (P, P) with P being the number of model parameters."""
        return self._lx.get_forecast_tensors(forecast_order=1, n_workers=n_workers)

    def dali(self, *, n_workers: int = 1):
        """Return (G, H): third- and fourth-order DALI tensors—the cubic and quartic
        terms of the log-likelihood expansion—shapes (P,P,P) and (P,P,P,P), where
        P is the number of model parameters."""
        return self._lx.get_forecast_tensors(forecast_order=2, n_workers=n_workers)
