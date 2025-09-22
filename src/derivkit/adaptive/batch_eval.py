"""Batch evaluation utilities for derivative estimation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__ = ["eval_function_batch"]


def _eval_serial(function: Callable[[float], Any], xs: np.ndarray) -> list[np.ndarray]:
    """Evaluate `function` over `xs` serially, returning list of 1D arrays."""
    return [np.atleast_1d(function(float(x))) for x in xs]


def _eval_parallel(
    function: Callable[[float], Any],
    xs: np.ndarray,
    n_workers: int,
) -> list[np.ndarray]:
    """Evaluate `function` over `xs` using a worker pool when available."""
    # import inside to avoid hard dependency
    try:
        from multiprocess import Pool  # type: ignore
    except Exception:
        # graceful fallback
        return _eval_serial(function, xs)

    n_workers = max(1, min(int(n_workers), int(xs.size)))
    with Pool(n_workers) as pool:
        return [np.atleast_1d(y) for y in pool.map(function, xs.tolist())]


def eval_function_batch(
    function: Callable[[float], Any],
    xs: np.ndarray,
    n_workers: int = 1,
) -> np.ndarray:
    """Evaluate `function` over 1D inputs `xs` and return shape (n_points, n_components).

    Accepts scalar or 1D-vector outputs and enforces consistent output shape.

    Parameters
    ----------
    function
        Maps float -> scalar or 1D array-like.
    xs
        1D abscissae.
    n_workers
        If > 1, try to parallelize with `multiprocess`; otherwise run serially.

    Returns:
        np.ndarray: Array with shape (n_points, n_components).
    """
    xs = np.asarray(xs, dtype=float)
    if xs.ndim != 1:
        raise ValueError("eval_function_batch: xs must be 1D.")

    if n_workers and n_workers > 1:
        ys = _eval_parallel(function, xs, n_workers)
    else:
        ys = _eval_serial(function, xs)

    # Check output consistency
    lengths = [np.asarray(y).size for y in ys]
    if len(set(lengths)) != 1:
        raise ValueError(
            "eval_function_batch: function output dimension changed across xs; "
            f"sizes={lengths}"
        )

    y = np.vstack([np.atleast_1d(y) for y in ys])
    if y.ndim == 1:
        y = y[:, None]
    return y
