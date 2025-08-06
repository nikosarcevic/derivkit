import math
import os

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np

from derivkit.kit import DerivativeKit


class PlotHelpers:
    """
    Utility class to support plotting and evaluation of numerical derivatives under noise.

    Provides helper functions for:
    - Creating noisy function versions
    - Running derivative estimation trials
    - Comparing different derivative methods
    - Computing or approximating reference derivatives
    - Saving consistent figures to disk

    Parameters
    ----------
    function : callable
        Original function to be differentiated.
    x_center : float
        Central evaluation point.
    derivative_order : int, optional
        Derivative order to compute (default is 1).
    fit_tolerance : float, optional
        Tolerance used for adaptive fitting (default is 0.05).
    true_derivative_fn : callable, optional
        Ground-truth derivative function, if available.
    plot_dir : str, optional
        Directory to save output plots (default is "plots").
    """
    def __init__(
        self,
        function,
        x_center,
        derivative_order: int = 1,
        fit_tolerance: float = 0.05,
        true_derivative_fn=None,
        plot_dir: str = "plots",
    ):
        self.function = function
        self.x_center = x_center
        self.plot_dir = plot_dir
        self.derivative_order = derivative_order
        self.fit_tolerance = fit_tolerance
        self.true_derivative_fn = true_derivative_fn
        self.derivs = DerivativeKit(function,
                                    x_center,
                                    derivative_order=self.derivative_order,
                                    fit_tolerance=self.fit_tolerance)
        self.seed = 42

        os.makedirs(self.plot_dir, exist_ok=True)

    def get_noisy_derivatives(self, noise_std=0.01, trials=100):
        """
        Compute derivative estimates across multiple noisy trials using stencil, adaptive, and Numdifftools methods.

        Parameters
        ----------
        noise_std : float
            Standard deviation of the noise to add to the function.
        trials : int
            Number of noisy trials to run.

        Returns
        -------
        tuple of lists
            (finite_differences, adaptive_fits, numdifftools_estimates)
        """
        rng = np.random.default_rng(self.seed)
        stencil_vals, adaptive_vals, nd_vals = [], [], []

        for _ in range(trials):
            # one shared noisy field for this trial
            seed = int(rng.integers(0, 2 ** 31 - 1))
            noisy_f = self.make_shared_noisy_func(noise_std, seed=seed)

            # evaluate both methods on the SAME noisy function
            kit = DerivativeKit(noisy_f,
                                self.x_center,
                                derivative_order=self.derivative_order,
                                fit_tolerance=self.fit_tolerance)
            stencil_vals.append(kit.finite.compute(stencil_stepsize=0.01 * (abs(self.x_center) or 1.0)))
            adaptive_vals.append(kit.adaptive.compute())

            nd_est = nd.Derivative(noisy_f, n=self.derivative_order)(self.x_center)
            nd_vals.append(nd_est)

        return stencil_vals, adaptive_vals, nd_vals

    def run_derivative_trials_with_noise(self, method="finite", order=1, noise_std=0.01, trials=100):
        """
        Run repeated derivative estimation trials with a specified method and added noise.

        Parameters
        ----------
        method : str
            One of {"finite", "adaptive", "numdifftools"}.
        order : int
            Derivative order to compute.
        noise_std : float
            Standard deviation of the noise.
        trials : int
            Number of trials to run.

        Returns
        -------
        list
            Derivative estimates for each trial.
        """
        results = []
        for i in range(trials):
            noisy_f = self.make_additive_noise_function(noise_std=noise_std, seed=i)

            if method == "finite":
                # set h comparable to adaptive's default first offset: 1% of |x0|
                fd_h = 0.01 * abs(self.x_center) if self.x_center != 0 else 0.01
                result = DerivativeKit(noisy_f,
                                       self.x_center,
                                       derivative_order=order,
                                       fit_tolerance=self.fit_tolerance).finite.compute(stencil_stepsize=fd_h)
            elif method == "adaptive":
                result = DerivativeKit(noisy_f,
                                       self.x_center,
                                       derivative_order=order,
                                       fit_tolerance=0.10).adaptive.compute(fallback_mode="poly_at_floor")
            elif method == "numdifftools":
                # Use a real central difference with fixed step, and disable Richardson smoothing.
                fd_h = 0.01 * abs(self.x_center) if self.x_center != 0 else 0.01  # match your FD scale
                result = nd.Derivative(
                    noisy_f,
                    n=order,
                    method="central",  # real, not complex or multicomplex
                    order=2,  # base 2nd-order central difference
                    step=fd_h,  # fixed step comparable to FD
                    richardson_terms=0  # <- important: avoid extrapolation smoothing
                )(self.x_center)

            else:
                raise ValueError(f"Unknown method: {method}")

            results.append(result)
        return results

    def make_noisy_interpolated_function(self, func, x_center, width=0.2, resolution=100, noise_std=0.01, seed=None):
        """
        Create a noisy interpolated version of a function on a local interval around x_center.

        Parameters
        ----------
        func : callable
            Function to be sampled.
        x_center : float
            Central point of the interval.
        width : float
            Half-width of the interval.
        resolution : int
            Number of sample points.
        noise_std : float
            Standard deviation of added Gaussian noise.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        callable
            Interpolated noisy function.
        """
        rng = np.random.default_rng(self.seed if seed is None else seed)
        x_grid = np.linspace(x_center - width, x_center + width, resolution)
        y_grid = np.array([func(x) + rng.normal(0, noise_std) for x in x_grid])

        def noisy_interp(x):
            return np.interp(x, x_grid, y_grid)

        return noisy_interp

    def make_additive_noise_function(self, noise_std=0.01, seed=None):
        """
        Add Gaussian noise to the function directly, no interpolation.

        Parameters
        ----------
        noise_std : float
            Standard deviation of added noise.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        callable
            Noisy function.
        """

        rng = np.random.default_rng(self.seed if seed is None else seed)

        def noisy_f(x):
            return self.function(x) + rng.normal(0, noise_std)

        return noisy_f

    def adaptive_fit_with_outlier_removal(self, x_vals, y_vals, return_inliers=False):
        """
        Fit a line to (x, y) after removing outliers using a 2.5σ residual filter.

        Parameters
        ----------
        x_vals : array-like
            X coordinates.
        y_vals : array-like
            Y coordinates.
        return_inliers : bool
            Whether to return a mask of inlier points.

        Returns
        -------
        slope : float
            Slope of the best-fit line.
        intercept : float
            Intercept of the best-fit line.
        inlier_mask : np.ndarray or None
            Boolean array indicating which points were kept (if return_inliers=True).
        """
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)

        # Simple outlier removal: remove points beyond 2.5 std
        residual = y_vals - np.poly1d(np.polyfit(x_vals, y_vals, deg=1))(x_vals)
        std = np.std(residual)
        inlier_mask = np.abs(residual) < 2.5 * std

        slope, intercept = np.polyfit(x_vals[inlier_mask], y_vals[inlier_mask], 1)

        if return_inliers:
            return slope, intercept, inlier_mask
        return slope, intercept, None

    def nd_derivative(self, x):
        """
        Compute derivative using numdifftools.

        Parameters
        ----------
        x : float
            Point at which to evaluate the derivative.

        Returns
        -------
        float
            Derivative estimate from numdifftools.
        """
        return nd.Derivative(self.function, n=self.derivative_order)(x)

    def reference_derivative(self, x=None, *, degree=None, half_width=None, num=21):
        """
        Estimate the true derivative via polynomial fitting if no analytical derivative is given.

        Parameters
        ----------
        x : float, optional
            Point at which to evaluate (defaults to x_center).
        degree : int, optional
            Degree of the fitting polynomial.
        half_width : float, optional
            Half-width of the fitting interval.
        num : int, optional
            Number of points to use in the fit.

        Returns
        -------
        float
            Estimated reference derivative.
        """
        if x is None: x = self.x_center
        if self.true_derivative_fn is not None:
            return self.true_derivative_fn(x)

        n = self.derivative_order
        degree = degree or max(n + 2, 5)
        base = abs(x) if abs(x) > 0 else 1.0
        half_width = half_width or (0.05 * base)

        # ensure enough points for the polynomial degree
        if num < degree + 1:
            num = degree + 1

        xs = np.linspace(x - half_width, x + half_width, num)
        ys = np.array([self.function(xi) for xi in xs])

        t = (xs - x) / half_width  # scale to [-1,1] for conditioning
        vander = np.vander(t, N=degree + 1, increasing=True)
        coeffs, *_ = np.linalg.lstsq(vander, ys, rcond=None)

        # chain rule for scaled variable: d^n/dx^n = (1/half_width^n) d^n/dt^n
        return math.factorial(n) * coeffs[n] / (half_width ** n)

    def make_shared_noisy_func(self, sigma, *, cover_width=None, resolution=401, seed=None):
        """
        Generate a shared noisy interpolated function with reproducible randomness.

        Parameters
        ----------
        sigma : float
            Standard deviation of the added noise.
        cover_width : float, optional
            Width of the interpolation domain.
        resolution : int, optional
            Number of points in interpolation grid.
        seed : int, optional
            Random seed to use.

        Returns
        -------
        callable
            Noisy interpolated function.
        """
        base = abs(self.x_center) or 1.0
        width = cover_width or (0.5 * base)
        return self.make_noisy_interpolated_function(
            func=self.function,
            x_center=self.x_center,
            width=width,
            resolution=resolution,
            noise_std=sigma,
            seed=self.seed if seed is None else seed,
        )

    def make_less_noisy(self, function, sigma=0.01, seed=42):
        """
        Add signal-scaled noise to a function (noise ∝ |f(x)|).

        Parameters
        ----------
        function : callable
            Base function.
        sigma : float
            Noise scaling factor. Default is 0.01.
        seed : int
            Seed for the noise RNG. Default is 42.

        Returns
        -------
        callable
            Noisy version of the input function.
        """
        rng = np.random.default_rng(seed)

        def noisy(x):
            fx = function(x)
            noise = rng.normal(0.0, sigma * max(abs(fx), 1e-3))  # scale noise to signal
            return fx + noise

        return noisy

    def save_fig(self, filename):
        """
        Save the current matplotlib figure to the configured directory.

        Parameters
        ----------
        filename : str
            File name (with extension) for saving the figure.
        """
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/{filename}", dpi=300)
