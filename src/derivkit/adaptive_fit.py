"""Public API: AdaptiveFitDerivative (thin orchestrator)."""

from __future__ import annotations

import warnings

import numpy as np

from .adaptive_internal import (
    build_x_offsets,
    estimate_component,
    fit_once_impl,
    get_adaptive_offsets_impl,
    validate_inputs,
)
from .diagnostics import DiagnosticsRecorder
from .finite_difference import FiniteDifferenceDerivative
from .utils import eval_function_batch, inverse_distance_weights

warnings.simplefilter("once", category=RuntimeWarning)


class AdaptiveFitDerivative:
    """Adaptive polynomial-fit derivative estimator."""

    def __init__(self, function, x0):
        """Initialize the estimator.

        Args:
            function (callable): Target function mapping float -> scalar/1D array.
            x0 (float): Expansion point at which to evaluate the derivative.
        """
        self.function = function
        self.x0 = x0
        self.diagnostics_data = None
        self.min_used_points = 5

    def differentiate(
        self,
        order: int = 1,
        min_samples: int = 7,
        diagnostics: bool = False,
        fallback_mode: str = "finite_difference",
        fit_tolerance: float = 0.05,
        include_zero: bool = True,
        floor_accept_multiplier: float = 2.0,
        n_workers: int = 1,
    ):
        """Estimate the derivative by adaptive polynomial fitting.

        See class docstring for behavior and the meaning of the parameters.

        Returns:
            float | np.ndarray | tuple: Derivative estimate, or (estimate, diagnostics)
                when ``diagnostics=True``.
        """
        validate_inputs(order, min_samples, self.min_used_points)

        # 1) grid: build x offsets and values
        x_offsets, required_points = build_x_offsets(
            x0=self.x0,
            order=order,
            include_zero=include_zero,
            min_samples=min_samples,
            min_used_points=self.min_used_points,
            get_adaptive_offsets=self.get_adaptive_offsets,
        )
        x_values = self.x0 + x_offsets

        # 2) evaluate f: batch eval
        Y = eval_function_batch(self.function, x_values, n_workers)
        n_components = Y.shape[1]
        derivs = np.zeros(n_components, dtype=float)

        # 3) diagnostics recorder: set up
        rec = DiagnosticsRecorder(enabled=diagnostics, x_all=x_values, y_all=Y)

        # 4) per-component: loop over components
        for i in range(n_components):
            outcome = estimate_component(
                x0=self.x0,
                x_values=x_values,
                y_values=Y[:, i],
                order=order,
                required_points=required_points,
                fit_tolerance=fit_tolerance,
                fallback_mode=fallback_mode,
                floor_accept_multiplier=floor_accept_multiplier,
                n_workers=n_workers,
                weight_fn=lambda xs: inverse_distance_weights(xs, self.x0, eps_frac=1e-3),
                fit_once_fn=lambda xv, yv: self._fit_once(xv, yv, order),
                fallback_fn=lambda: self._fallback_derivative(order, n_workers),
                store_diag=lambda *args, **kw: self._store_diagnostics_entry(*args, **kw),
            )
            derivs[i] = outcome.value
            rec.add(outcome)

        return (
            (derivs.item() if derivs.size == 1 else derivs, rec.build())
            if diagnostics
            else (derivs.item() if derivs.size == 1 else derivs)
        )

    def get_adaptive_offsets(
        self,
        x0=None,
        base_rel=0.01,
        base_abs=1e-6,
        factor=1.5,
        num_offsets=10,
        max_rel=0.05,
        max_abs=1e-2,
        step_mode="auto",
        x_small_threshold=1e-3,
    ):
        """Compatibility wrapper retained for tests; delegates to internal impl."""
        return get_adaptive_offsets_impl(
            self.x0 if x0 is None else float(x0),
            base_rel=base_rel,
            base_abs=base_abs,
            factor=factor,
            num_offsets=num_offsets,
            max_rel=max_rel,
            max_abs=max_abs,
            step_mode=step_mode,
            x_small_threshold=x_small_threshold,
        )

    def _fit_once(self, x_vals: np.ndarray, y_vals: np.ndarray, order: int):
        """Compatibility wrapper retained for tests; delegates to internal impl."""
        return fit_once_impl(self.x0, x_vals, y_vals, order, weight_eps_frac=1e-3)

    def _fallback_derivative(self, order: int, n_workers: int = 1):
        """Finite-difference fallback used when fitting cannot be accepted."""
        warnings.warn(
            "Falling back to finite difference derivative.", RuntimeWarning
        )
        fd = FiniteDifferenceDerivative(function=self.function, x0=self.x0)
        result = fd.differentiate(order=order, n_workers=n_workers)
        return np.atleast_1d(result)

    # Private no-op (kept only so the lambda resolves in all code paths)
    def _store_diagnostics_entry(self, *args, **kwargs):
        return
