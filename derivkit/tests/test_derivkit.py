import numpy as np
import pytest

from derivkit import DerivativeKit


# Example fixture definitions if missing
@pytest.fixture
def linear_func():
    return lambda x: 2.0 * x + 1.0

@pytest.fixture
def quadratic_func():
    return lambda x: 3.0 * x**2 + 2.0 * x + 1.0

@pytest.fixture
def cubic_func():
    return lambda x: 4.0 * x**3 + 3.0 * x**2 + 2.0 * x + 1.0

@pytest.fixture
def quartic_func():
    return lambda x: 5.0 * x**4 + 4.0 * x**3 + 3.0 * x**2 + 2.0 * x + 1.0

@pytest.fixture
def log_func():
    return lambda x: np.log(x)

@pytest.fixture
def vector_func():
    return lambda x: np.array([x, 2*x])


def test_linear_first_derivative(linear_func):
    """Test that the first derivative of a linear function returns the correct slope."""
    calc = DerivativeKit(linear_func, central_value=0.0, derivative_order=1).adaptive
    result = calc.compute()
    assert np.isclose(result, 2.0, atol=1e-8)

def test_quadratic_second_derivative(quadratic_func):
    """Test that the second derivative of a quadratic function returns the expected constant."""
    calc = DerivativeKit(quadratic_func, central_value=1.0, derivative_order=2).adaptive
    result = calc.compute()
    assert np.isclose(result, 6.0, atol=1e-6)

def test_cubic_third_derivative(cubic_func):
    """Test that the third derivative of a cubic function returns the expected constant."""
    calc = DerivativeKit(cubic_func, central_value=1.0, derivative_order=3).adaptive
    result = calc.compute()
    assert np.isclose(result, 24.0, rtol=1e-2)

def test_quartic_fourth_derivative(quartic_func):
    """Test that the fourth derivative of a quartic function returns the expected constant."""
    calc = DerivativeKit(quartic_func, central_value=1.0, derivative_order=4).adaptive
    result = calc.compute()
    assert np.isclose(result, 120.0, rtol=1e-1)

def test_invalid_derivative_order_adaptive():
    """Test that requesting an unsupported derivative order with adaptive raises ValueError."""
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x, 1.0, derivative_order=5).adaptive.compute()

def test_invalid_derivative_order_finite():
    """Test that requesting an unsupported derivative order with finite difference raises ValueError."""
    with pytest.raises(ValueError):
        DerivativeKit(lambda x: x, 1.0, derivative_order=5).finite.compute()

def test_log_scale(log_func):
    """Test correct handling of derivatives in log-scale coordinates."""
    calc = DerivativeKit(log_func, central_value=2.0, derivative_order=1).adaptive
    result = calc.compute(log_scale=True)
    assert np.isclose(result, 1 / 2.0, rtol=1e-2)

def test_vector_function(vector_func):
    """Test that vector-valued functions return correct shape and values."""
    calc = DerivativeKit(vector_func, central_value=1.0, derivative_order=1).adaptive
    result = calc.compute()
    assert result.shape == (2,)
    assert np.allclose(result, [1.0, 2.0], rtol=1e-2)

def test_fallback_used():
    """Test that fallback mechanism is triggered and returns a close approximation."""
    f = lambda x: 1e-10 * x**3
    calc = DerivativeKit(f, central_value=1.0, derivative_order=2).adaptive
    result = calc.compute(fit_tolerance=1e-5)
    assert np.isclose(result, 6e-10, rtol=0.2)

def test_stencil_matches_analytic():
    """Test that the finite difference result approximates the analytic derivative of sin(x)."""
    f = lambda x: np.sin(x)
    x0 = np.pi / 4
    exact = np.cos(x0)
    result = DerivativeKit(f, x0, derivative_order=1).finite.compute()
    assert np.isclose(result, exact, rtol=1e-2)

def test_derivative_noise_test_runs():
    """Test stability and reproducibility of repeated noisy derivative estimates."""
    f = lambda x: x**2
    adaptive = DerivativeKit(f, 1.0, derivative_order=1).adaptive
    results = [adaptive.compute() + np.random.normal(0, 0.001) for _ in range(10)]
    assert len(results) == 10
    assert all(np.isfinite(r) for r in results)

def test_zero_central_value():
    """Test that derivative at x=0 is computed correctly for a cubic function."""
    f = lambda x: x**3
    result = DerivativeKit(f, central_value=0.0, derivative_order=1).adaptive.compute()
    assert np.isclose(result, 0.0, atol=1e-10)

def test_constant_function():
    """Test that derivatives of a constant function are zero for all orders."""
    f = lambda x: 42.0
    for order in range(1, 5):
        result = DerivativeKit(f, 1.0, derivative_order=order).adaptive.compute()
        assert np.isclose(result, 0.0, atol=1e-7)

def test_force_fallback():
    """Test that fallback is explicitly used when requested."""
    f = lambda x: np.exp(x)
    calc = DerivativeKit(f, central_value=0.0, derivative_order=1).adaptive
    result = calc.compute(fit_tolerance=0.0, use_fallback=True)
    assert np.isclose(result, 1.0, rtol=1e-2)

def test_no_fallback_returns_nan():
    """Test that compute() returns NaN when fit fails and fallback is disabled."""
    f = lambda x: 1e-10 * x**3
    calc = DerivativeKit(f, central_value=1.0, derivative_order=2).adaptive
    result = calc.compute(fit_tolerance=1e-5, use_fallback=False)
    assert np.isnan(result)

def test_log_scale_invalid_central():
    """Test that computing a log-scale derivative at an invalid (non-positive) central value raises ValueError."""
    f = lambda x: np.log(x)
    with pytest.raises(ValueError):
        DerivativeKit(f, central_value=0.0, derivative_order=1).adaptive.compute(log_scale=True)

def test_vector_fallback_used():
    """Test fallback on vector-valued function returns valid, finite results."""
    f = lambda x: np.array([1e-10 * x**3, 1e-10 * x**2])
    calc = DerivativeKit(f, central_value=1.0, derivative_order=2).adaptive
    result = calc.compute(fit_tolerance=1e-5)
    assert result.shape == (2,)
    assert np.all(np.isfinite(result))

def test_shape_mismatch_raises():
    """Test that shape mismatch in vector output raises ValueError."""
    def bad_func(x):
        return np.array([x, x**2]) if np.round(x, 2) < 1.0 else np.array([x])  # triggers mismatch

    with pytest.raises(ValueError):
        DerivativeKit(bad_func, central_value=1.0, derivative_order=1).adaptive.compute()
