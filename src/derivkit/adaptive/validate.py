"""Input validation utilities for the adaptive fit pipeline."""

from __future__ import annotations

import warnings


def validate_inputs(order: int, min_samples: int, min_used_points: int) -> None:
    """Validate core arguments for the adaptive fit.

    Args:
        order: Derivative order. Allowed values are 1–4 (inclusive).
        min_samples: Requested minimum number of samples.
        min_used_points: Minimum usable points in the fit loop (not enforced here,
            but referenced in the warning message for context).

    Raises:
        ValueError: If ``order`` is not one of {1, 2, 3, 4}.

    Warns:
        RuntimeWarning: If ``min_samples`` is less than ``order + 2``, which is
            required to support fit/fallback strategies robustly.
    """
    if order not in (1, 2, 3, 4):
        raise ValueError(f"Invalid order={order}. Only orders 1–4 are supported.")
    if min_samples - order < 2:
        warnings.warn(
            "min_samples must be at least max(2 + order, min_used_points) "
            "to support fit/fallback strategies.",
            RuntimeWarning,
        )
