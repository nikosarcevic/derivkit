"""Offset grid builders for adaptive sampling around ``x0``."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .offsets import get_adaptive_offsets as _default_get_adaptive_offsets

__all__ = [
    "build_x_offsets",
    "symmetric_offsets",
    "extend_offsets_to_required",
    "get_adaptive_offsets_impl",
]


def symmetric_offsets(offsets: np.ndarray, include_zero: bool) -> np.ndarray:
    """Create a symmetric set of offsets around zero.

    Turns strictly positive offsets into a symmetric grid about 0. Optionally
    includes 0 itself.

    Args:
        offsets: Strictly positive offsets (1D).
        include_zero: If True, include 0 in the final grid.

    Returns:
        A sorted 1D array of symmetric offsets about 0.

    Raises:
        ValueError: If any input offset is non-positive.
    """
    pos = np.asarray(offsets, float)
    if np.any(pos <= 0):
        raise ValueError("symmetric_offsets expects strictly positive offsets.")
    steps = np.insert(pos, 0, 0.0) if include_zero else pos
    out = np.unique(np.concatenate([steps, -steps]))
    return np.sort(out)


def extend_offsets_to_required(
    offsets: np.ndarray,
    include_zero: bool,
    factor: float,
    growth_limit: float,
    required_points: int,
) -> np.ndarray:
    """Grow offsets geometrically until the symmetric grid meets a size target.

    Offsets are extended by multiplying the last element by ``factor`` until the
    symmetric grid size (after mirroring and optional zero) reaches
    ``required_points`` or the next candidate would exceed ``growth_limit``.

    Args:
        offsets: Initial strictly positive offsets (1D).
        include_zero: Whether to include zero in the symmetric grid.
        factor: Geometric growth factor for the last offset.
        growth_limit: Maximum allowed offset value.
        required_points: Target size for the final symmetric grid.

    Returns:
        The final symmetric grid (1D) meeting the size/limit criteria.
    """
    cur = np.array(offsets, float)
    while True:
        grid = symmetric_offsets(cur, include_zero)
        if grid.size >= int(required_points):
            return grid
        nxt = cur[-1] * float(factor)
        if nxt > float(growth_limit):
            return grid
        cur = np.append(cur, nxt)


def build_x_offsets(
    *,
    x0: float,
    order: int,
    include_zero: bool,
    min_samples: int,
    min_used_points: int,
    get_adaptive_offsets: Callable[..., np.ndarray] = _default_get_adaptive_offsets,
) -> Tuple[np.ndarray, int]:
    """Construct the symmetric evaluation grid and the required point count.

    Args:
        x0: Expansion point (passed to ``get_adaptive_offsets``).
        order: Derivative order (controls the minimum floor).
        include_zero: Whether to include zero in the symmetric grid.
        min_samples: Requested minimum number of samples.
        min_used_points: Hard floor on usable sample count in the fit loop.
        get_adaptive_offsets: Factory returning strictly positive offsets near ``x0``.

    Returns:
        Tuple ``(x_offsets, required_points)`` where ``x_offsets`` are symmetric
        relative offsets around 0 (to be shifted by ``x0``), and
        ``required_points`` is the effective minimum sample count.
    """
    order_floor = order + 2
    required = max(min_samples, max(min_used_points, order_floor))
    pos = get_adaptive_offsets(x0=x0)
    growth_limit = pos[-1] * (1.5**3)
    x_offsets = extend_offsets_to_required(
        offsets=pos,
        include_zero=include_zero,
        factor=1.5,
        growth_limit=growth_limit,
        required_points=required,
    )
    return x_offsets, required


def get_adaptive_offsets_impl(*args, **kwargs):
    """Back-compat alias to ``get_adaptive_offsets`` (thin pass-through)."""
    from .offsets import get_adaptive_offsets

    return get_adaptive_offsets(*args, **kwargs)
