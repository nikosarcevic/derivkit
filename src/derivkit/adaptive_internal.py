"""Internal helpers for adaptive_fit: core loop & small utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np

from .utils import (
    extend_offsets_to_required,
    inverse_distance_weights,
    normalize_coords,
    polyfit_u,
    prune_by_residuals,
    residuals_relative,
)

Mode = Literal["poly", "auto_accept_at_floor", "poly_at_floor", "finite_difference"]


@dataclass
class ComponentOutcome:
    """Container for a single component’s result and diagnostics."""

    value: float
    mode: Mode
    x_used: Optional[np.ndarray]
    y_used: Optional[np.ndarray]
    y_fit: Optional[np.ndarray]
    residuals: Optional[np.ndarray]
    status: dict


def validate_inputs(order: int, min_samples: int, min_used_points: int) -> None:
    """Validate core arguments for the adaptive fit."""
    if order not in (1, 2, 3, 4):
        raise ValueError(f"Invalid order={order}. Only orders 1–4 are supported.")
    if min_samples - order < 2:
        warnings.warn(
            "min_samples must be at least max(2 + order, min_used_points) to support fit/fallback strategies.",
            RuntimeWarning,
        )


def build_x_offsets(
    *,
    x0: float,
    order: int,
    include_zero: bool,
    min_samples: int,
    min_used_points: int,
    get_adaptive_offsets: Callable[..., np.ndarray],
) -> Tuple[np.ndarray, int]:
    """Build symmetric offsets and return (offsets, required_points)."""
    order_floor = order + 2
    required = max(min_samples, max(min_used_points, order_floor))
    offsets = get_adaptive_offsets(x0=x0)
    growth_limit = offsets[-1] * (1.5**3)
    x_offsets = extend_offsets_to_required(
        offsets=offsets,
        include_zero=include_zero,
        factor=1.5,
        growth_limit=growth_limit,
        required_points=required,
    )
    return x_offsets, required


def estimate_component(
    *,
    x0: float,
    x_values: np.ndarray,
    y_values: np.ndarray,
    order: int,
    required_points: int,
    fit_tolerance: float,
    fallback_mode: str,
    floor_accept_multiplier: float,
    n_workers: int,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    fit_once_fn: Callable[[np.ndarray, np.ndarray], dict],
    fallback_fn: Callable[[], np.ndarray],
    store_diag: Callable[..., None],
) -> ComponentOutcome:
    """Run the adaptive loop for a single output component and return its outcome."""
    x_vals = x_values.copy()
    y_vals = y_values.copy()

    last = {"x": None, "y": None, "yfit": None, "resid": None, "fit": None}

    while x_vals.size >= required_points:
        fit = fit_once_fn(x_vals, y_vals)
        if not fit["ok"]:
            break

        last.update(x=x_vals.copy(), y=y_vals.copy(),
                    yfit=fit["y_fit"].copy(), resid=fit["residuals"].copy(), fit=fit)

        # accept
        if fit["rel_error"] < fit_tolerance:
            m = order
            val = fit["poly_u"].deriv(m=m)(0.0) / (fit["h"] ** m)
            return ComponentOutcome(
                value=float(val),
                mode="poly",
                x_used=x_vals, y_used=y_vals,
                y_fit=fit["y_fit"], residuals=fit["residuals"],
                status={"mode": "poly", "rel_error": float(fit["rel_error"]), "accepted": True},
            )

        # prune + retry
        x_vals, y_vals, removed = prune_by_residuals(
            x0, x_vals, y_vals, fit["residuals"], fit_tolerance, required_points,
            max_remove=2, keep_center=True, keep_symmetric=True,
        )
        if removed:
            continue

        # at floor: decide accept or fallback
        at_floor = (last["x"] is not None) and (last["x"].size == required_points)
        accept, tag = _maybe_accept_at_floor(
            last["resid"], at_floor, fit_tolerance, fallback_mode, floor_accept_multiplier
        )
        if accept:
            m = order
            val = last["fit"]["poly_u"].deriv(m=m)(0.0) / (last["fit"]["h"] ** m)
            warnings.warn(
                f"[AdaptiveFitDerivative] Accepted polynomial at minimum points ({required_points}) "
                f"with residuals above tolerance (max={np.max(last['resid']):.3g}, tol={fit_tolerance:.3g}) using mode='{tag}'.",
                RuntimeWarning,
            )
            return ComponentOutcome(
                value=float(val),
                mode=tag, x_used=last["x"], y_used=last["y"],
                y_fit=last["yfit"], residuals=last["resid"],
                status={"mode": tag, "accepted": True},
            )
        break

    # fallback
    fd = float(fallback_fn()[0])
    warnings.warn(
        "[AdaptiveFitDerivative] Falling back to finite differences because polynomial fit did not meet tolerance.",
        RuntimeWarning,
    )
    return ComponentOutcome(
        value=fd, mode="finite_difference",
        x_used=last["x"], y_used=last["y"],
        y_fit=last["yfit"], residuals=last["resid"],
        status={"mode": "finite_difference", "accepted": True,
                "reason": "fit_not_within_tolerance_or_insufficient_points"},
    )


def _maybe_accept_at_floor(
    last_residuals: Optional[np.ndarray],
    at_floor: bool,
    fit_tolerance: float,
    fallback_mode: str,
    floor_accept_multiplier: float,
):
    """Decision rule for accepting a floor-size fit that missed tolerance."""
    if not at_floor:
        return False, "not_at_floor"
    if fallback_mode == "poly_at_floor":
        return True, "poly_at_floor"
    if fallback_mode == "auto":
        if last_residuals is None:
            return False, "auto_no_residuals"
        max_r = float(np.max(last_residuals))
        med_r = float(np.median(last_residuals))
        close_enough = (max_r < floor_accept_multiplier * fit_tolerance) and (med_r < fit_tolerance)
        return (True, "auto_accept_at_floor") if close_enough else (False, "auto_reject")
    return False, "finite_difference"


def get_adaptive_offsets_impl(
    x0: float,
    *,
    base_rel: float = 0.01,
    base_abs: float = 1e-6,
    factor: float = 1.5,
    num_offsets: int = 10,
    max_rel: float = 0.05,
    max_abs: float = 1e-2,
    step_mode: str = "auto",
    x_small_threshold: float = 1e-3,
) -> np.ndarray:
    """Return absolute, positive step sizes (no zero) tailored to x0 scale."""
    x0 = float(x0)
    use_abs = (step_mode == "absolute") or (
        step_mode == "auto" and abs(x0) < x_small_threshold
    )
    if use_abs:
        bases = [min(base_abs * (factor**i), max_abs) for i in range(num_offsets)]
    else:
        scale = max(abs(x0), x_small_threshold)
        bases = [min(base_rel * (factor**i), max_rel) * scale for i in range(num_offsets)]
    offs = np.unique([b for b in bases if b > 0.0])
    if offs.size == 0:
        raise ValueError("No valid offsets generated.")
    return offs


def fit_once_impl(
    x0: float,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    order: int,
    *,
    weight_eps_frac: float = 1e-3,
) -> dict:
    """One weighted poly fit in normalized coords; mirrors class `_fit_once`."""
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
