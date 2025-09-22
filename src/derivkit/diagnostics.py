"""Diagnostics helpers for the adaptive-fit derivative estimator."""

from __future__ import annotations

import numpy as np


class DiagnosticsRecorder:
    """Collects per-component diagnostics and builds the dict tests expect."""

    def __init__(self, *, enabled: bool, x_all: np.ndarray, y_all: np.ndarray):
        """Initialize the recorder.

        Args:
            enabled (bool): If False, this recorder is inert.
            x_all (np.ndarray): All abscissae used for evaluation.
            y_all (np.ndarray): All function values evaluated at ``x_all``.
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
        """Append diagnostics for one component outcome."""
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
        """Return the diagnostics dictionary in the structure expected by tests."""
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
