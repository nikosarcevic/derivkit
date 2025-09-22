"""Focused tests for the finite-difference backend."""

import numpy as np
import pytest

from derivkit.derivative_kit import DerivativeKit


def test_stencil_matches_analytic():
    """Finite differences match the analytic derivative for sin(x)."""
    x0 = np.pi / 4
    exact = np.cos(x0)
    result = DerivativeKit(lambda x: np.sin(x), x0).finite.differentiate(order=1)
    assert np.isclose(result, exact, rtol=1e-2)


def test_invalid_order_finite():
    """Unsupported derivative order raises ValueError."""
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x, 1.0).finite.differentiate(order=5)
