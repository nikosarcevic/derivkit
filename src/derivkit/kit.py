from typing import Callable

from derivkit.adaptive_fit import AdaptiveFitDerivative
from derivkit.finite_difference import FiniteDifferenceDerivative


class DerivativeKit:
    """
    Provides unified access to both adaptive and finite difference derivative calculators.

    Attributes
    ----------
    adaptive : AdaptiveFitDerivative
        Adaptive polynomial fit-based derivative method.
    finite : FiniteDifferenceDerivative
        High-order finite difference stencil-based method.

    Parameters
    ----------
    function : callable
        The scalar or vector-valued function to differentiate.
    central_value : float
        The point at which the derivative is evaluated.
    """

    def __init__(
        self,
        function: Callable[[float], float],
        central_value: float,
    ):
        self.adaptive = AdaptiveFitDerivative(function, central_value)
        self.finite = FiniteDifferenceDerivative(function, central_value)

    def get_used_points(self, derivative_order: int = 1, n_workers=1):
        """
        Returns x and y points used in the adaptive fit (for component 0).

        Parameters
        ----------
        derivative_order : int, optional
            Order of the derivative to compute diagnostics for (default is 1).
        n_workers: int, optional
            Number of worker to use in multiprocessing. Default is 1 (no multiprocessing).

        Returns
        -------
        tuple of np.ndarray
            (x_all, y_all, x_used, y_used, used_mask)
        """
        _, diagnostics = self.adaptive.compute(derivative_order=derivative_order, diagnostics=True, 
                                               n_workers=n_workers)

        x_all = diagnostics["x_all"]
        y_all = diagnostics["y_all"][:, 0]  # assuming scalar output or first component
        x_used = diagnostics["x_used"][0]
        y_used = diagnostics["y_used"][0]
        used_mask = diagnostics["used_mask"][0]

        return x_all, y_all, x_used, y_used, used_mask
