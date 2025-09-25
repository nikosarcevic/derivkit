"""Adaptive component estimation utilities (fit/prune/fallback loop).

Runs the per-component loop: fit → optionally prune → accept or fall back.
The outcome is recorded as a *path* in ``ComponentOutcome.mode``. Supported
paths are:
- "poly": fit accepted within tolerance
- "poly_at_floor": forced accept at minimum points by policy
- "auto_accept_at_floor": auto policy accepted at the floor
- "finite_difference": polynomial path rejected; finite-difference fallback

Note: internal decision tags ("not_at_floor", "auto_no_residuals",
"auto_reject") are only for diagnostics/messages and are never stored as the
path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np

Mode = Literal["poly", "auto_accept_at_floor", "poly_at_floor", "finite_difference"]


@dataclass
class ComponentOutcome:
    """Container for a single component’s result and diagnostics.

    Attributes:
      value: Estimated derivative value for this component.
      mode: Path that produced the result (e.g., "poly", "finite_difference").
      x_used: Abscissae used in the final decision.
      y_used: Ordinates used in the final decision.
      y_fit: Fitted values corresponding to ``x_used`` (if available).
      residuals: Per-point relative residuals (if available).
      status: Free-form status/details for logging or debugging.
    """

    value: float
    mode: Mode
    x_used: Optional[np.ndarray]
    y_used: Optional[np.ndarray]
    y_fit: Optional[np.ndarray]
    residuals: Optional[np.ndarray]
    status: dict


def _maybe_accept_at_floor(
    last_residuals: Optional[np.ndarray],
    at_floor: bool,
    fit_tolerance: float,
    fallback_mode: str,
    floor_accept_multiplier: float,
) -> Tuple[bool, str]:
    """Decide whether to accept a fit at the minimum sample count ("floor").

    Invoked when pruning cannot further reduce residuals and the algorithm is
    at the minimum number of points required for the fit.

    Args:
      last_residuals: Residuals from the last successful fit, or ``None``.
      at_floor: Whether the algorithm is at the minimum point count.
      fit_tolerance: Maximum allowed relative residual (acceptance threshold).
      fallback_mode: Policy string; accepted values include "poly_at_floor",
        "auto", or a mode that triggers fallback (e.g., "finite_difference").
      floor_accept_multiplier: Extra headroom for the maximum residual when
        deciding acceptance under "auto".

    Returns:
      Tuple[bool, str]: ``(accept, tag)`` where ``accept`` indicates
      acceptance and ``tag`` is the chosen mode label for diagnostics.
    """
    if not at_floor:
        return False, "not_at_floor"
    if fallback_mode == "poly_at_floor":
        return True, "poly_at_floor"
    elif fallback_mode == "auto":
        if last_residuals is None:
            return False, "auto_no_residuals"
        max_r = float(np.max(last_residuals))
        med_r = float(np.median(last_residuals))
        close_enough = (max_r < floor_accept_multiplier * fit_tolerance) and (
            med_r < fit_tolerance
        )
        return (True, "auto_accept_at_floor") if close_enough else (False, "auto_reject")
    return False, "finite_difference"


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
    """Remove worst residuals (and optional mirrors) above tolerance.

    Removes up to ``max_remove`` points whose residuals exceed ``fit_tolerance``,
    optionally preserving the sample closest to ``x0`` and pruning in symmetric
    pairs about ``x0``. Never prunes below ``required_points``.

    Args:
      x0: Expansion point around which symmetry is defined.
      x_vals: 1D sample abscissae.
      y_vals: 1D sample ordinates aligned with ``x_vals``.
      residuals: Per-point relative residuals, aligned with ``x_vals``.
      fit_tolerance: Residual threshold used to decide pruning.
      required_points: Minimum number of points allowed after pruning.
      max_remove: Maximum number of points to remove in one pass.
      keep_center: Whether to preserve the sample closest to ``x0``.
      keep_symmetric: Whether to remove symmetric mirrors when possible.

    Returns:
      Tuple[np.ndarray, np.ndarray, bool]: ``(x_kept, y_kept, removed_any)``,
      where ``removed_any`` indicates whether any point was pruned.

    Raises:
      AssertionError: If input arrays do not share the same length.
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

        if (
            keep_symmetric
            and removed < max_remove
            and (keep.sum() - 1) >= required_points
        ):
            target = 2.0 * float(x0) - x_vals[j]
            k = int(np.argmin(np.abs(x_vals - target)))
            if k != j and (not keep_center or k != center_idx) and keep[k]:
                keep[k] = False
                removed += 1

    if removed == 0:
        return x_vals, y_vals, False
    return x_vals[keep], y_vals[keep], True


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
    fit_once_fn: Callable[[np.ndarray, np.ndarray], dict],
    fallback_fn: Callable[[], np.ndarray],
) -> ComponentOutcome:
    """Run the adaptive fit/prune/fallback loop for a single component.

    The loop performs a weighted polynomial fit, checks a relative-error
    criterion, prunes worst residuals if needed, and either accepts the fit,
    accepts at the floor according to a policy, or falls back to finite
    differences.

    Args:
      x0: Expansion point used for symmetry and normalization.
      x_values: 1D abscissae for this component’s evaluations.
      y_values: 1D ordinates for this component’s evaluations.
      order: Derivative order to extract from the polynomial fit.
      required_points: Minimum number of points to accept a fit.
      fit_tolerance: Residual threshold for fit acceptance.
      fallback_mode: Strategy string for floor handling ("auto",
        "poly_at_floor", etc.); non-accepting values trigger fallback.
      floor_accept_multiplier: Extra headroom for the maximum residual when
        deciding floor acceptance in "auto" mode.
      fit_once_fn: Callable performing one weighted fit over the provided
        samples; must return a dict with keys:
        ``ok``, ``h``, ``poly_u``, ``y_fit``, ``residuals``, ``rel_error``.
      fallback_fn: Callable returning a 1D array-like with a finite-diff
        estimate in position 0; used when the poly path is rejected.

    Returns:
      ComponentOutcome: Final value, path used, and diagnostics.

    Warns:
      RuntimeWarning: When accepting at the floor or when falling back to
        finite differences.
    """
    x_vals = x_values.copy()
    y_vals = y_values.copy()
    last = {"x": None, "y": None, "yfit": None, "resid": None, "fit": None}

    while x_vals.size >= required_points:
        fit = fit_once_fn(x_vals, y_vals)
        if not fit["ok"]:
            break

        last.update(
            x=x_vals.copy(),
            y=y_vals.copy(),
            yfit=fit["y_fit"].copy(),
            resid=fit["residuals"].copy(),
            fit=fit,
        )

        # Accept if beneath tolerance.
        if fit["rel_error"] < fit_tolerance:
            m = order
            val = fit["poly_u"].deriv(m=m)(0.0) / (fit["h"] ** m)
            return ComponentOutcome(
                value=float(val),
                mode="poly",
                x_used=x_vals,
                y_used=y_vals,
                y_fit=fit["y_fit"],
                residuals=fit["residuals"],
                status={
                    "mode": "poly",
                    "rel_error": float(fit["rel_error"]),
                    "accepted": True,
                },
            )

        # Prune and retry.
        x_vals, y_vals, removed = prune_by_residuals(
            x0,
            x_vals,
            y_vals,
            fit["residuals"],
            fit_tolerance,
            required_points,
            max_remove=2,
            keep_center=True,
            keep_symmetric=True,
        )
        if removed:
            continue

        # At floor: decide accept or fallback.
        at_floor = (last["x"] is not None) and (last["x"].size == required_points)
        accept, tag = _maybe_accept_at_floor(
            last["resid"], at_floor, fit_tolerance, fallback_mode, floor_accept_multiplier
        )
        if accept:
            m = order
            val = last["fit"]["poly_u"].deriv(m=m)(0.0) / (last["fit"]["h"] ** m)
            warnings.warn(
                (
                    "[AdaptiveFitDerivative] Accepted polynomial at minimum points "
                    f"({required_points}) with residuals above tolerance "
                    f"(max={np.max(last['resid']):.3g}, tol={fit_tolerance:.3g}) "
                    f"using mode='{tag}'."
                ),
                RuntimeWarning,
            )
            return ComponentOutcome(
                value=float(val),
                mode=tag,
                x_used=last["x"],
                y_used=last["y"],
                y_fit=last["yfit"],
                residuals=last["resid"],
                status={"mode": tag, "accepted": True},
            )
        break

    # Fallback.
    fd = float(fallback_fn()[0])
    warnings.warn(
        (
            "[AdaptiveFitDerivative] Falling back to finite differences because "
            "polynomial fit did not meet tolerance."
        ),
        RuntimeWarning,
    )
    return ComponentOutcome(
        value=fd,
        mode="finite_difference",
        x_used=last["x"],
        y_used=last["y"],
        y_fit=last["yfit"],
        residuals=last["resid"],
        status={
            "mode": "finite_difference",
            "accepted": True,
            "reason": "fit_not_within_tolerance_or_insufficient_points",
        },
    )
