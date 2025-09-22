"""Fallback-path tests for AdaptiveFitDerivative.

These tests force the adaptive polynomial fit to "fail" and assert that the
implementation cleanly falls back to finite differences.
"""

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit


def _fail_fit(x0, x_vals, y_vals, order, **kw):
    """Stub fit function that always reports failure."""
    return {
        "ok": False,
        "reason": "singular_normal_equations",
        "h": 1.0,
        "poly_u": None,
        "y_fit": None,
        "residuals": None,
        "rel_error": np.inf,
    }


def test_fallback_used(monkeypatch):
    """Fallback to finite differences when the poly fit fails."""
    calc = DerivativeKit(lambda x: np.exp(x), x0=0.2).adaptive

    # Patch the instance's injected fit function hook
    monkeypatch.setattr(calc, "_fit_once_fn", _fail_fit, raising=True)

    with pytest.warns(RuntimeWarning, match=r"Falling back to finite difference"):
        val = calc.differentiate()

    assert np.isfinite(val)
    assert np.isclose(val, np.exp(0.2), rtol=1e-4, atol=1e-8)


def test_fallback_triggers_when_fit_unavailable(monkeypatch):
    """If the internal fit cannot be performed, code must fall back to FD."""
    calc = DerivativeKit(lambda x: np.exp(x), x0=0.0).adaptive

    monkeypatch.setattr(calc, "_fit_once_fn", _fail_fit, raising=True)

    with pytest.warns(RuntimeWarning, match=r"Falling back to finite difference"):
        val = calc.differentiate()

    assert np.isfinite(val)
    assert np.isclose(val, 1.0, rtol=1e-4, atol=1e-8)


def test_fallback_returns_finite_value_when_fit_fails(monkeypatch):
    """Return finite FD value when the fit cannot meet tolerance."""
    calc = DerivativeKit(lambda x: 1e-10 * x**3, x0=1.0).adaptive

    monkeypatch.setattr(calc, "_fit_once_fn", _fail_fit, raising=True)

    with pytest.warns(RuntimeWarning, match=r"Falling back to finite difference"):
        result = calc.differentiate(order=2)

    # Analytic d2/dx2 of 1e-10 * x^3 at x=1 is 6e-10
    assert np.isfinite(result)
    assert np.isclose(result, 6e-10, rtol=0.2)
