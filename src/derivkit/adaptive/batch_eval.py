"""Batch evaluation utilities for derivative estimation.

Evaluate a user function over a 1D grid with optional parallelism and return
a 2D array with consistent shape suitable for downstream polynomial fitting
and diagnostics (e.g., in AdaptiveFitDerivative).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__ = ["eval_function_batch"]


def _eval_serial(function: Callable[[float], Any], xs: np.ndarray) -> list[np.ndarray]:
    """Evaluate a function over points serially.

    Args:
      function: Callable mapping a float to a scalar or 1D array-like.
      xs: 1D array of abscissae.

    Returns:
      List[np.ndarray]: One array per input x, each at least 1D.
    """
    return [np.atleast_1d(function(float(x))) for x in xs]


def _eval_parallel(
    function: Callable[[float], Any],
    xs: np.ndarray,
    n_workers: int,
) -> list[np.ndarray]:
    """Evaluate a function over points using a worker pool when available.

    Tries to use the optional ``multiprocess`` package. If unavailable, falls
    back to serial evaluation.

    Args:
      function: Callable mapping a float to a scalar or 1D array-like. Must be
        picklable for process-based parallelism.
      xs: 1D array of abscissae.
      n_workers: Desired number of worker processes; clipped to [1, len(xs)].

    Returns:
      List[np.ndarray]: One array per input x, each at least 1D.

    Notes:
      Input order is preserved. Errors raised by ``function`` propagate.
    """
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
    """Evaluate a function over 1D inputs and return a 2D array.

    Evaluates ``function(x)`` for each ``x`` in ``xs``. If ``n_workers > 1``
    and the optional ``multiprocess`` package is available, uses a process
    pool; otherwise runs serially. Scalar outputs are promoted to shape
    ``(n_points, 1)``. 1D outputs must have constant length across ``xs``.

    Args:
      function: Callable mapping a float to a scalar or 1D array-like.
      xs: 1D array of abscissae.
      n_workers: If > 1, attempt parallel evaluation with ``multiprocess``.

    Returns:
      np.ndarray: Array of shape ``(n_points, n_components)``.

    Raises:
      ValueError: If ``xs`` is not 1D or output lengths vary across inputs.

    Examples:
      >>> import numpy as np
      >>> f = lambda x: np.array([x, x**2])
      >>> xs = np.linspace(-1.0, 1.0, 5)
      >>> Y = eval_function_batch(f, xs)
      >>> Y.shape
      (5, 2)
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
