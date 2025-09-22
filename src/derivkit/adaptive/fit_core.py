"""Core polynomial-fitting utilities used by the adaptive estimator."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .weights import inverse_distance_weights

__all__ = [
    "normalize_coords",
    "polyfit_u",
    "residuals_relative",
    "fit_once_impl",
]


def normalize_coords(x_vals: np.ndarray, x0: float) -> Tuple[np.ndarray, float]:
    """Normalize coordinates around ``x0`` to the range [-1, 1] approximately.

    The routine shifts by ``x0`` and scales by the maximum absolute deviation.

    Args:
        x_vals: Sample abscissae.
        x0: Expansion point.

    Returns:
        Tuple of ``(u_vals, h)`` where ``u_vals`` are normalized coordinates and
        ``h`` is the scaling factor (>= 1e-12).
    """
    t = np.asarray(x_vals, dtype=float) - float(x0)
    h = float(np.max(np.abs(t))) if t.size else 0.0
    h = max(h, 1e-12)
    return t / h, h


def polyfit_u(
    u_vals: np.ndarray,
    y_vals: np.ndarray,
    order: int,
    weights: np.ndarray,
) -> Optional[np.poly1d]:
    """Weighted polynomial fit in normalized coordinates.

    Args:
        u_vals: Normalized abscissae.
        y_vals: Ordinates.
        order: Polynomial degree.
        weights: Per-sample weights.

    Returns:
        A ``np.poly1d`` object on success, otherwise ``None`` if the system is
        singular or the fit fails.
    """
    try:
        coeffs = np.polyfit(
            np.asarray(u_vals, float),
            np.asarray(y_vals, float),
            deg=order,
            w=np.asarray(weights, float),
        )
        return np.poly1d(coeffs)
    except np.linalg.LinAlgError:
        return None


def residuals_relative(
    y_fit: np.ndarray,
    y_true: np.ndarray,
    floor: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Compute per-sample relative residuals and their maximum.

    Args:
        y_fit: Model predictions at the sample points.
        y_true: True observed values at the sample points.
        floor: Small positive floor for the denominator to avoid division by
            small/zero values.

    Returns:
        Tuple of ``(residuals, rel_error)`` where ``residuals`` are the
        elementwise relative residuals, and ``rel_error`` is their maximum
        (or 0.0 if empty).
    """
    y_true = np.asarray(y_true, float)
    y_fit = np.asarray(y_fit, float)
    safe = np.maximum(np.abs(y_true), floor)
    resid = np.abs(y_fit - y_true) / safe
    rel_error = float(np.max(resid)) if resid.size else 0.0
    return resid, rel_error


def fit_once_impl(
    x0: float,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    order: int,
    *,
    weight_eps_frac: float = 1e-3,
) -> Dict[str, Any]:
    """Perform one weighted polynomial fit in normalized coordinates.

    Args:
        x0: Expansion point used for normalization.
        x_vals: Sample abscissae.
        y_vals: Sample ordinates.
        order: Polynomial degree (and derivative order to be extracted later).
        weight_eps_frac: Epsilon fraction used by the inverse-distance weights.

    Returns:
        A dictionary with keys:
            - ``ok`` (bool): Fit succeeded.
            - ``reason`` (str | None): Failure reason if any.
            - ``h`` (float): Normalization scale.
            - ``poly_u`` (np.poly1d | None): Polynomial model in normalized coords.
            - ``y_fit`` (np.ndarray | None): Fitted values.
            - ``residuals`` (np.ndarray | None): Per-point relative residuals.
            - ``rel_error`` (float): Maximum relative residual.
    """
    u_vals, h = normalize_coords(x_vals, x0)
    weights = inverse_distance_weights(x_vals, x0, eps_frac=weight_eps_frac)
    poly_u = polyfit_u(u_vals, y_vals, order, weights)
    if poly_u is None:
        return {"ok": False, "reason": "singular_normal_equations"}
    y_fit = poly_u(u_vals)
    resid, rel_error = residuals_relative(y_fit, y_vals, floor=1e-8)
    return {
        "ok": True,
        "reason": None,
        "h": h,
        "poly_u": poly_u,
        "y_fit": y_fit,
        "residuals": resid,
        "rel_error": rel_error,
    }
