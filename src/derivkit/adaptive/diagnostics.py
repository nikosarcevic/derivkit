"""Diagnostics helpers for the adaptive-fit derivative estimator.

This module records per-component information (points used, fitted values,
residuals, and status) during adaptive polynomial fitting so tests and
downstream tools can inspect the fitting process.
"""

from __future__ import annotations

import numpy as np


class DiagnosticsRecorder:
    """Record per-component diagnostics and build a test-friendly dict.

    When disabled, the recorder is not populated and ``build()`` returns an empty dict.
    """

    def __init__(self, *, enabled: bool, x_all: np.ndarray, y_all: np.ndarray):
        """Initialize the recorder.

        Args:
          enabled: If False, this recorder is inert.
          x_all: All abscissae used for evaluation (1D array).
          y_all: All function values evaluated at ``x_all``. Shape is
            ``(n_points,)`` or ``(n_points, n_components)``.
        """
        self.enabled = bool(enabled)
        if not self.enabled:
            self._data = None
            return
        self._x_all = np.asarray(x_all, float)
        self._y_all = np.asarray(y_all, float)
        self._x_used = []
        self._y_used = []
        self._y_fit = []
        self._residuals = []
        self._used_mask = []
        self._status = []

    def add(self, outcome) -> None:
        """Append diagnostics for one component outcome.

        The ``outcome`` object is expected to expose attributes:
        ``x_used``, ``y_used``, ``y_fit``, ``residuals``, and ``status``.
        Missing attributes or ``None`` values are handled gracefully.

        Args:
          outcome: Per-component fitting result with the attributes listed
            above.
        """
        if not self.enabled:
            return
        xu = None if outcome.x_used is None else np.asarray(outcome.x_used).copy()
        yu = None if outcome.y_used is None else np.asarray(outcome.y_used).copy()
        yf = None if outcome.y_fit is None else np.asarray(outcome.y_fit).copy()
        rr = None if outcome.residuals is None else np.asarray(outcome.residuals).copy()

        self._x_used.append(xu)
        self._y_used.append(yu)
        self._y_fit.append(yf)
        self._residuals.append(rr)

        if xu is None:
            mask = np.zeros_like(self._x_all, dtype=bool)
        else:
            mask = np.isclose(self._x_all[:, None], xu, rtol=1e-12, atol=1e-15).any(axis=1)
        self._used_mask.append(mask)

        self._status.append(dict(outcome.status))

    def build(self) -> dict:
        """Build the diagnostics dictionary in the structure tests expect.

        Returns:
          A dictionary with keys:
          ``x_all``, ``y_all``, ``x_used``, ``y_used``, ``y_fit``,
          ``residuals``, ``used_mask``, and ``status``. Returns ``{}`` if the
          recorder is disabled.
        """
        if not self.enabled:
            return {}
        return {
            "x_all": self._x_all.copy(),
            "y_all": self._y_all.copy(),
            "x_used": self._x_used,
            "y_used": self._y_used,
            "y_fit": self._y_fit,
            "residuals": self._residuals,
            "used_mask": self._used_mask,
            "status": self._status,
        }
