"""Weighting utilities for adaptive polynomial fits."""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = ["inverse_distance_weights"]


def inverse_distance_weights(
    x_vals: Iterable[float] | np.ndarray,
    x0: float,
    eps_frac: float = 1e-3,
    eps_floor: float = 1e-9,
) -> np.ndarray:
    r"""Scale-aware inverse-distance weights centered at ``x0``.

    Let :math:`d_i = |x_i - x0|` and :math:`D = \max_i d_i` (span). Define
    :math:`\epsilon = \max(\text{eps\_frac} \cdot D, \text{eps\_floor})`
    and compute :math:`w_i = 1 / (d_i + \epsilon)`.

    Args:
        x_vals: Sample locations.
        x0: Center location.
        eps_frac: Fraction of the span used to set the distance floor.
        eps_floor: Absolute lower bound for the distance floor.

    Returns:
        np.ndarray: Unnormalized weights with shape ``(n_points,)``.
    """
    d = np.abs(np.asarray(x_vals, dtype=float) - float(x0))
    D = float(np.max(d)) if d.size else 0.0
    eps = max(eps_frac * D, eps_floor)
    return 1.0 / (d + eps)
