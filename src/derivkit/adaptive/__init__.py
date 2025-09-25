"""Adaptive fitting subpackage.

This subpackage provides tools for derivative estimation via adaptive
polynomial fitting. It contains:

- `fit_core`: core adaptive fitting implementation
- `fallback`: fallback strategies when fitting fails
- `grid`, `offsets`, `weights`: utilities for building evaluation grids and weights
- `batch_eval`: functions for parallel evaluation of functions on grids
- `diagnostics` and `validate`: helpers for checking residuals, tolerances, and
  ensuring robust fits
- `estimator`: high-level interfaces for adaptive derivative estimation

These components work together to support `AdaptiveFitDerivative`, which
combines polynomial fits, residual diagnostics, and fallback strategies to
compute stable derivatives.
"""
