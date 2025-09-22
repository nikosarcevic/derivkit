"""Provides assorted utility functions."""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def log_debug_message(message, debug=False, log_file=None, log_to_file=None):
    """Logs a debug message to stdout and optionally to a file.

    Args:
        message (str): The debug message to print/log.
        debug (bool): Whether to print the message.
        log_file (str or None): Path to the log file.
        log_to_file (bool or None): Whether to write the message to the file.
    """
    if not debug:
        return

    print(message)

    if log_to_file and log_file:
        try:
            with open(log_file, "a") as f:
                f.write(message + "\n")
        except (IOError, OSError) as e:
            print(f"[log_debug_message] Failed to write to log file: {e}")


def is_finite_and_differentiable(function, x, delta=1e-5, tol=1e-2):
    """Checks if a function is finite and numerically differentiable at a point.

    Args:
        function (callable): The function to test.
        x (float or np.ndarray): The input value(s).
        delta (float): Step size for finite difference.
        tol (float): Tolerance for differentiability check.

    Returns:
        bool: True if function is finite and differentiable at x, False
            otherwise.
    """
    try:
        f0 = np.asarray(function(x))
        f_minus = np.asarray(function(x - delta))
        f_plus = np.asarray(function(x + delta))

        # Check for finiteness
        if (
            not np.isfinite(f0).all()
            or not np.isfinite(f_minus).all()
            or not np.isfinite(f_plus).all()
        ):
            return False

        # Numerical derivative comparison (left and right)
        left = (f0 - f_minus) / delta
        right = (f_plus - f0) / delta

        return np.all(np.abs(left - right) < tol)

    except (ValueError, TypeError, ArithmeticError):
        return False


def normalize_derivative(derivative, reference):
    """Computes the relative error between estimated and reference derivative.

    Args:
        derivative (float or np.ndarray): Estimated derivative.
        reference (float or np.ndarray): True/reference derivative.

    Returns:
        float or np.ndarray: Normalized relative error.
    """
    return (derivative - reference) / (np.abs(reference) + 1e-12)


def central_difference_error_estimate(step_size, order=1):
    """Provides a rough truncation error estimate for central differences.

    Args:
        step_size (float): Finite difference step size.
        order (int): Order of the derivative (1 to 4).

    Returns:
        float: Estimated error magnitude.
    """
    if order == 1:
        return step_size**2 / 6
    elif order == 2:
        return step_size**2 / 12
    elif order == 3:
        return step_size**2 / 8
    elif order == 4:
        return step_size**2 / 6
    else:
        raise ValueError("Only derivative orders 1–4 are supported.")


def is_symmetric_grid(x_vals):
    """Checks if evaluation grid is symmetric around 0.

    Args:
        x_vals (:class:`np.ndarray`): Evaluation points (1D).

    Returns:
        bool: True if grid is symmetric, False otherwise.
    """
    x_vals = np.sort(np.asarray(x_vals))
    n = len(x_vals)
    if n % 2 == 0:
        return False
    mid = n // 2
    return np.allclose(x_vals[:mid], -x_vals[:mid:-1])


def generate_test_function(name="sin"):
    """Returns a known test function and its first/second derivatives.

    Args:
        name (str): One of 'sin', 'exp', 'polynomial', 'gaussian'.

    Returns:
        tuple: (f(x), df/dx, d2f/dx2)
    """
    if name == "sin":
        return (lambda x: np.sin(x), lambda x: np.cos(x), lambda x: -np.sin(x))
    elif name == "exp":
        return (lambda x: np.exp(x), lambda x: np.exp(x), lambda x: np.exp(x))
    elif name == "polynomial":
        return (
            lambda x: x**3 + 2 * x**2 - x + 5,
            lambda x: 3 * x**2 + 4 * x - 1,
            lambda x: 6 * x + 4,
        )
    elif name == "gaussian":
        return (
            lambda x: np.exp(-(x**2)),
            lambda x: -2 * x * np.exp(-(x**2)),
            lambda x: (4 * x**2 - 2) * np.exp(-(x**2)),
        )
    else:
        raise ValueError(
            "Supported names: 'sin', 'exp', 'polynomial', 'gaussian'"
        )


def eval_function_batch(function, xs: np.ndarray, n_workers: int) -> np.ndarray:
    """Evaluate a function on a 1D batch of inputs.

    Shapes are normalized so the return is always 2D: (n_points, n_components).

    Args:
      function: Callable mapping float -> scalar or 1D array-like.
      xs: 1D array of abscissae where `function` will be evaluated.
      n_workers: Number of worker processes. If <= 1, runs serially.

    Returns:
      np.ndarray: Array of shape (n_points, n_components).
    """
    xs = np.asarray(xs, dtype=float)
    if n_workers and n_workers > 1:
        # Import here to avoid making utils depend on multiprocess at import time.
        from multiprocess import Pool  # type: ignore
        n_workers = min(int(n_workers), int(xs.size))
        with Pool(n_workers) as pool:
            ys = [np.atleast_1d(y) for y in pool.map(function, xs.tolist())]
    else:
        ys = [np.atleast_1d(function(float(x))) for x in xs]

    Y = np.vstack(ys)
    if Y.ndim == 1:
        Y = Y[:, None]
    return Y


def normalize_coords(x_vals: np.ndarray, x0: float) -> Tuple[np.ndarray, float]:
    """Normalize coordinates around a center.

    Computes u = (x - x0) / h with h = max|x - x0|, guarded away from zero.

    Args:
      x_vals: 1D sample locations.
      x0: Center point.

    Returns:
      Tuple[np.ndarray, float]: (u, h) where u has same length as x_vals and h > 0.
    """
    t = np.asarray(x_vals, dtype=float) - float(x0)
    h = float(np.max(np.abs(t))) if t.size else 0.0
    h = max(h, 1e-12)
    u = t / h
    return u, h


def inverse_distance_weights(x_vals, x0, eps_frac=1e-3, eps_floor=1e-9):
    """Scale-aware inverse-distance weights centered at ``x0``.

    This emphasizes samples closest to the expansion point while staying
    numerically stable across *very small* and *very large* scales.

    Formulation:
      Let ``d_i = |x_i - x0|`` and ``D = max_i d_i`` (current sampling span).
      Define ``eps = max(eps_frac * D, eps_floor)`` and
      ``w_i = 1 / (d_i + eps)``.

    Intuition:
      With ``eps = 1e-3 * D``, the center-to-edge weight ratio is roughly
      ``w(0) / w(D) ≈ (D + eps)/eps ≈ 1/eps_frac ≈ 10^3``. Increase
      ``eps_frac`` (e.g., 1e-2) for milder emphasis (~100×) or decrease
      (e.g., 1e-4) for sharper emphasis (~10,000×).

    Why not a fixed epsilon?
      A constant epsilon misbehaves across scales: it can flatten weights for
      tiny spans or make the center weight dominate for large spans. Tying
      epsilon to the current span adapts automatically.

    Notes:
      * Complexity is O(n).
      * ``eps_floor`` guards degenerate spans (e.g., all points ~ x0).
      * Weights are **relative** (not normalized). Normalize with
        ``w / w.sum()`` if needed.

    Args:
      x_vals (np.ndarray): Sample locations, shape (n_points,).
      x0 (float): Expansion point.
      eps_frac (float): Softening as a fraction of span ``D``. Default 1e-3.
      eps_floor (float): Absolute minimum softening. Default 1e-9.

    Returns:
      np.ndarray: Weights of shape (n_points,).
    """
    d = np.abs(np.asarray(x_vals) - float(x0))
    D = np.max(d) if d.size else 0.0
    eps = max(eps_frac * D, eps_floor)
    return 1.0 / (d + eps)


def polyfit_u(u_vals: np.ndarray, y_vals: np.ndarray, order: int, weights: np.ndarray) -> Optional[np.poly1d]:
    """Fit a polynomial in the power basis to y ≈ P(u).

    This mirrors the original `np.polyfit(..., w=...)` behavior.

    Args:
      u_vals: 1D normalized coordinates.
      y_vals: 1D function values at u_vals.
      order: Degree of the polynomial.
      weights: 1D weights for weighted least squares.

    Returns:
      np.poly1d or None: Fitted polynomial, or None if the system is singular.
    """
    try:
        coeffs = np.polyfit(np.asarray(u_vals, float), np.asarray(y_vals, float), deg=order, w=np.asarray(weights, float))
        return np.poly1d(coeffs)
    except np.linalg.LinAlgError:
        return None


def residuals_relative(y_fit: np.ndarray, y_true: np.ndarray, floor: float = 1e-8) -> Tuple[np.ndarray, float]:
    """Compute relative residuals and their maximum.

    Args:
      y_fit: Fitted values.
      y_true: True values.
      floor: Denominator guard when |y_true| is tiny.

    Returns:
      Tuple[np.ndarray, float]: (residuals, rel_error=max residual).
    """
    y_true = np.asarray(y_true, float)
    y_fit = np.asarray(y_fit, float)
    safe = np.maximum(np.abs(y_true), floor)
    resid = np.abs(y_fit - y_true) / safe
    rel_error = float(np.max(resid)) if resid.size else 0.0
    return resid, rel_error


def symmetric_offsets(offsets: np.ndarray, include_zero: bool) -> np.ndarray:
    """Build a symmetric set of offsets around zero.

    Args:
      offsets: Positive offsets (1D, strictly > 0 recommended).
      include_zero: Whether to include 0 in the final grid.

    Returns:
      np.ndarray: Symmetric array of offsets (including negatives, and maybe zero).
    """
    steps = np.insert(offsets, 0, 0.0) if include_zero else np.asarray(offsets, float)
    return np.unique(np.concatenate([steps, -steps]))


def extend_offsets_to_required(
    offsets: np.ndarray,
    include_zero: bool,
    factor: float,
    growth_limit: float,
    required_points: int,
) -> np.ndarray:
    """Extend offsets geometrically until enough symmetric points exist.

    Args:
      offsets: Initial positive offsets (1D).
      include_zero: Whether zero is part of the sampling grid.
      factor: Geometric growth factor for extending offsets.
      growth_limit: Upper bound for the next offset (stop growing if exceeded).
      required_points: Minimum number of points in the final symmetric grid.

    Returns:
      np.ndarray: Symmetric offsets meeting the required count or the best effort.
    """
    cur = np.array(offsets, float)
    while True:
        x_off = symmetric_offsets(cur, include_zero)
        if x_off.size >= int(required_points):
            return x_off
        next_off = cur[-1] * float(factor)
        if next_off > float(growth_limit):
            return x_off
        cur = np.append(cur, next_off)


def prune_by_residuals(
    x0: float,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    residuals: np.ndarray,
    fit_tolerance: float,
    required_points: int,
    *,
    max_remove: int = 2,
    keep_center: bool = True,
    keep_symmetric: bool = True,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Remove worst residual points (and optional mirrors) without dropping below a floor.

    Behavior matches the in-class version used by `AdaptiveFitDerivative`.

    Args:
      x0: Center point.
      x_vals: 1D sample locations.
      y_vals: 1D values corresponding to x_vals.
      residuals: 1D relative residuals at x_vals.
      fit_tolerance: Acceptable residual threshold.
      required_points: Minimum number of points to retain.
      max_remove: Max number of points to remove in one call.
      keep_center: Whether to keep the point closest to x0.
      keep_symmetric: Whether to remove the mirror point as well.

    Returns:
      Tuple[np.ndarray, np.ndarray, bool]: (x_vals_pruned, y_vals_pruned, removed_any).
    """
    x_vals = np.asarray(x_vals, float)
    y_vals = np.asarray(y_vals, float)
    residuals = np.asarray(residuals, float)
    assert x_vals.size == y_vals.size == residuals.size

    order_idx = np.argsort(residuals)[::-1]
    center_idx = int(np.argmin(np.abs(x_vals - float(x0)))) if x_vals.size else -1
    keep = np.ones(x_vals.size, dtype=bool)
    removed = 0

    for j in order_idx:
        if removed >= max_remove:
            break
        if residuals[j] <= fit_tolerance:
            break
        if keep_center and j == center_idx:
            continue
        if keep.sum() - 1 < required_points:
            break

        keep[j] = False
        removed += 1

        if keep_symmetric and removed < max_remove and (keep.sum() - 1) >= required_points:
            target = 2.0 * float(x0) - x_vals[j]
            k = int(np.argmin(np.abs(x_vals - target)))
            if k != j and (not keep_center or k != center_idx) and keep[k]:
                keep[k] = False
                removed += 1

    if removed == 0:
        return x_vals, y_vals, False
    return x_vals[keep], y_vals[keep], True
