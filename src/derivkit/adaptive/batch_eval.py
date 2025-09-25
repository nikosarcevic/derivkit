"""Batch evaluation utilities for derivative estimation.

Evaluate a user function over a 1D grid with optional parallelism and return
a 2D array with consistent shape suitable for downstream polynomial fitting
and diagnostics (e.g., in `AdaptiveFitDerivative`).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from multiprocess import Pool

__all__ = ["eval_function_batch"]


def _eval_serial(function: Callable[[float], Any], xs: np.ndarray) -> list[np.ndarray]:
    """Evaluate a function over points serially.

    Args:
      function: Callable mapping a float to a scalar or 1D array-like.
      xs: 1D array of abscissae.

    Returns:
      list[np.ndarray]: One array per input x, each at least 1D.
    """
    return [np.atleast_1d(function(float(x))) for x in xs]


def _eval_parallel(
    function: Callable[[float], Any],
    xs: np.ndarray,
    n_workers: int,
) -> list[np.ndarray]:
    """Evaluate a function over points in parallel.

    Uses ``multiprocess.Pool`` with ``n_workers`` processes. Falls back to the
    serial path if the workload is too small or if pool creation/execution fails.

    Args:
      function: Maps a float to a scalar or 1D array-like.
      xs: 1D abscissae to evaluate.
      n_workers: Desired number of processes.

    Returns:
      list[np.ndarray]: One 1D array per input, order-preserving.
    """
    if n_workers <= 1:
        return _eval_serial(function, xs)

    # Light heuristic: avoid pool overhead for tiny workloads.
    n = max(1, min(int(n_workers), int(xs.size)))
    if xs.size < max(8, 2 * n):
        return _eval_serial(function, xs)

    try:
        with Pool(n) as pool:
            ys = pool.map(function, xs.tolist())
    except Exception:
        # Spawn/pickle/start-method issues → graceful serial fallback.
        return _eval_serial(function, xs)

    return [np.atleast_1d(y) for y in ys]


def eval_function_batch(
    function: Callable[[float], Any],
    xs: np.ndarray,
    n_workers: int = 1,
) -> np.ndarray:
    """Evaluate a function over 1D inputs and return a 2D array.

    Evaluates ``function(x)`` for each ``x`` in ``xs``. If ``n_workers > 1``,
    uses a ``multiprocess.Pool``; otherwise runs serially. Scalar outputs are
    promoted to shape ``(n_points, 1)``. 1D outputs must have constant length
    across ``xs``.

    Args:
      function: Callable mapping a float to a scalar or 1D array-like.
      xs: 1D array of abscissae.
      n_workers: If > 1, evaluate in parallel using ``multiprocess``.

    Returns:
      np.ndarray: Array of shape ``(n_points, n_components)``.

    Raises:
      ValueError: If ``xs`` is not 1D or output lengths vary across inputs.

    Examples:
      >>> import numpy as np
      >>> f = lambda x: np.array([x, x**2])
      >>> xs = np.linspace(-1.0, 1.0, 5)
      >>> y = eval_function_batch(f, xs)
      >>> y.shape
      (5, 2)
    """
    xs = np.asarray(xs, dtype=float)
    if xs.ndim != 1:
        raise ValueError(f"eval_function_batch: xs.ndim must be 1 but is {xs.ndim}.")

    ys = _eval_parallel(function, xs, n_workers) if n_workers > 1 else _eval_serial(
        function, xs
    )

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
