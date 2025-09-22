"""Generic, non-adaptive utilities (safe to import anywhere)."""

from __future__ import annotations  # (harmless) keeps import groups tidy

import numpy as np

__all__ = [
    "log_debug_message",
    "is_finite_and_differentiable",
    "normalize_derivative",
    "central_difference_error_estimate",
    "is_symmetric_grid",
    "generate_test_function",
]


def log_debug_message(
    message: str,
    debug: bool = False,
    log_file: str | None = None,
    log_to_file: bool | None = None,
) -> None:
    """Optionally print and/or append a debug message.

    Args:
        message: Text to log.
        debug: If True, print to stdout.
        log_file: Path to a log file (used if ``log_to_file`` is True).
        log_to_file: If True, append to ``log_file`` when provided.
    """
    if not debug and not log_to_file:
        return
    if debug:
        print(message)
    if log_to_file and log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(message + "\n")
        except Exception as e:  # pragma: no cover - defensive
            print(f"[log_debug_message] Failed to write to log file: {e}")


def is_finite_and_differentiable(function, x, delta: float = 1e-5, tol: float = 1e-2):
    """Quick check that ``function`` is finite near ``x`` and well-behaved.

    Evaluates at ``x`` and ``x + delta`` and ensures finite values; returns
    False on exceptions or non-finite results.

    Args:
        function: Callable f(x).
        x: Point at which to probe.
        delta: Small step for a forward check.
        tol: Unused (kept for back-compat).

    Returns:
        bool: True if finite at both points, else False.
    """
    try:
        f0 = np.asarray(function(x))
        f1 = np.asarray(function(x + delta))
        return np.isfinite(f0).all() and np.isfinite(f1).all()
    except Exception:
        return False


def normalize_derivative(derivative, reference):
    """Normalize a derivative against a reference scale.

    Args:
        derivative: Value to normalize.
        reference: Reference scale.

    Returns:
        Normalized value using ``|reference| + 1e-12`` as denominator.
    """
    return (derivative - reference) / (np.abs(reference) + 1e-12)


def central_difference_error_estimate(step_size, order: int = 1):
    """Heuristic truncation error estimate for central differences.

    Args:
        step_size: Grid spacing.
        order: Derivative order (1–4 supported).

    Returns:
        Estimated truncation error scale.

    Raises:
        ValueError: If ``order`` is not supported.
    """
    if order == 1:
        return step_size**2 / 6
    if order == 2:
        return step_size**2 / 12
    if order == 3:
        return step_size**2 / 20
    if order == 4:
        return step_size**2 / 30
    raise ValueError("Only derivative orders 1–4 are supported.")


def is_symmetric_grid(x_vals):
    """Return True if ``x_vals`` are symmetric about zero (within tolerance)."""
    x_vals = np.sort(np.asarray(x_vals))
    n = len(x_vals)
    mid = n // 2
    return np.allclose(x_vals[:mid], -x_vals[: mid : -1])


def generate_test_function(name: str = "sin"):
    """Return (f, f', f'') tuple for a named test function.

    Args:
        name: One of {"sin"}; more may be added.

    Returns:
        Tuple of callables (f, df, d2f) for testing.
    """
    if name == "sin":
        return (lambda x: np.sin(x), lambda x: np.cos(x), lambda x: -np.sin(x))
    raise ValueError(f"Unknown test function: {name!r}")
