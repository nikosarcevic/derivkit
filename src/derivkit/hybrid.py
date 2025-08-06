from datetime import datetime
import os
import warnings

import numpy as np


class HybridDerivativeCalculator:
    """
    A class for estimating first or second derivatives of a scalar or vector-valued
    function using a hybrid approach combining iterative polynomial fitting (STEM)
    and the five-point stencil method.

    Parameters:
        function (callable): The function to differentiate. Must take a scalar float as input
                             and return a 1D NumPy array.
        central_value (float): The x-value at which the derivative is evaluated.
        stem_precision (float): Tolerance for convergence in the STEM method.
        stencil_stepsize (float): Step size used in the stencil method.
        derivative_order (int): Order of the derivative (1 or 2).
        min_samples (int): Minimum number of x-points used in polynomial fitting.
        log_dir (str): Directory to store debug logs.
        log_suffix (str or None): Optional suffix for the log filename.

    Notes:
        If your function returns a 2D array (e.g., image, matrix, or tensor slice),
        make sure to flatten it before returning (e.g., using `.flatten()` or `.ravel()`).
        The internal logic assumes 1D outputs per evaluation point.
    """
    def __init__(self,
                 function,
                 central_value,
                 stem_precision=0.05,
                 stencil_stepsize=0.01,
                 derivative_order=1,
                 min_samples=5,
                 log_dir="logs",
                 log_suffix=None):
        self.function = function
        self.central_value = central_value
        self.min_samples = min_samples
        self.stem_precision = stem_precision
        self.stencil_stepsize = stencil_stepsize
        self.derivative_order = derivative_order
        self.stem_deriv = None
        self.percentages = np.array([0.00625, 0.0125, 0.01875, 0.025, 0.0375, 0.05, 0.1])

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        self.log_file = os.path.join(log_dir, f"stem_debug_{timestamp}_{log_suffix}.log" if log_suffix else f"stem_debug_{timestamp}.log")

    def stem_method(self, include_zero=True, fallback=True, debug=True, to_file=False):
        """
        Estimate the 1st or 2nd order derivative using iterative polynomial fitting.

        Parameters:
            include_zero (bool): Whether to include central value (0% offset) in fitting.
            fallback (bool): Whether to fall back to stencil method on non-convergence.
            debug (bool): Whether to print debug messages to console.
            to_file (bool): Whether to write debug messages to a log file. Defaults to False.

        Returns:
            float or np.ndarray: Estimated derivative at central_value.
        """

        if self.derivative_order not in [1, 2]:
            raise ValueError("Only derivative_order=1 or 2 is supported.")

        percentages = np.insert(self.percentages, 0, 0.0) if include_zero else self.percentages.copy()
        if include_zero and not self._is_finite_and_differentiable():
            if debug:
                msg = "[stem_method] Function not smooth or differentiable at central_value; excluding 0% offset."
                warnings.warn(msg)
                self._log_debug_message(msg)

        # Build x_values
        if self.central_value == 0:
            x_values = np.concatenate((percentages, -percentages))
        else:
            x_values = np.concatenate([self.central_value * (1 + percentages),
                                       self.central_value * (1 - percentages)])

        # Evaluate function
        y_list = [self.function(x) for x in x_values]
        y_values = np.array(y_list)
        if y_values.ndim == 1:
            y_values = y_values.reshape(-1, 1)

        n_points, n_components = y_values.shape
        derivatives = np.zeros(n_components)

        for idx in range(n_components):
            x_vals = x_values.copy()
            y_vals = y_values[:, idx].copy()

            while len(x_vals) >= max(self.min_samples, self.derivative_order + 2):
                coeffs = np.polyfit(x_vals, y_vals, deg=self.derivative_order)
                poly = np.poly1d(coeffs)
                y_fit = poly(x_vals)

                safe_y = np.where(y_vals == 0, 1e-10, y_vals)
                spread = np.abs((y_fit - y_vals) / safe_y)
                if np.max(spread) < self.stem_precision:
                    poly_deriv = poly.deriv(m=self.derivative_order)
                    derivatives[idx] = poly_deriv(self.central_value)
                    break
                x_vals = x_vals[1:-1]
                y_vals = y_vals[1:-1]
            else:
                if debug:
                    self._log_debug_message(f"[stem_method] Component {idx} did not converge, using fallback.",
                                            to_file=to_file)
                derivatives[idx] = self.five_point_stencil_method_component(idx, fallback)

        return derivatives if n_components > 1 else derivatives[0]

    def five_point_stencil_method_component(self, component_idx, fallback=True):
        """
        Per-component stencil fallback for vector-valued functions.
        """
        h = self.stencil_stepsize
        x0 = self.central_value
        stencil_points = np.array([x0 + i * h for i in range(-2, 3)])
        func_vals = np.array([self.function(x) for x in stencil_points])
        if func_vals.ndim == 1:
            func_vals = func_vals.reshape(-1, 1)

        coeffs_dict = {
            1: np.array([1, -8, 0, 8, -1]) / (12.0 * h),
            2: np.array([-1, 16, -30, 16, -1]) / (12.0 * h ** 2),
        }

        coeffs = coeffs_dict[self.derivative_order]
        return np.dot(coeffs, func_vals[:, component_idx]) if fallback else 0.0

    def five_point_stencil_method(self):
        """
        Calculates the derivative using the five-point stencil method.

        Returns:
            float or np.ndarray: Derivative value(s).
        """
        h = self.stencil_stepsize
        x0 = self.central_value
        stencil_points = np.array([x0 + i * h for i in range(-2, 3)])
        function_values = np.array([self.function(x) for x in stencil_points])

        if function_values.ndim == 1:
            function_values = function_values.reshape(-1, 1)

        n_components = function_values.shape[1]
        coeffs_dict = {
            1: np.array([1, -8, 0, 8, -1]) / (12.0 * h),
            2: np.array([-1, 16, -30, 16, -1]) / (12.0 * h ** 2),
        }
        coeffs = coeffs_dict[self.derivative_order]

        derivatives = np.dot(function_values.T, coeffs)
        return derivatives if n_components > 1 else float(derivatives)

    def _log_debug_message(self, message, debug=True, to_file=None):
        if not debug:
            return

        print(message)
        if to_file is None:
            to_file = debug  # Only save to file if debug is enabled and to_file is not set

        if to_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

    def _is_finite_and_differentiable(self, delta=1e-5, tol=1e-2):
        try:
            f0 = self.function(self.central_value)
            f_minus = self.function(self.central_value - delta)
            f_plus = self.function(self.central_value + delta)
            if not np.isfinite([f0, f_minus, f_plus]).all():
                return False
            left = (f0 - f_minus) / delta
            right = (f_plus - f0) / delta
            return np.all(np.abs(left - right) < tol)
        except Exception:
            return False

    def calculate_derivatives_with_output_noise(self, method='stem', iterations=100, noise_std=0.01):
        """
        Repeats derivative calculation with added Gaussian noise.

        Returns:
            list: List of derivative results for each iteration.
        """
        derivatives = []

        for _ in range(iterations):
            def noisy_func(x, func=self.function):
                return func(x) + np.random.randn() * noise_std

            original_func = self.function
            self.function = noisy_func
            derivative = np.nan

            try:
                if method == 'stem':
                    derivative = self.stem_method()
                elif method == 'five_point_stencil':
                    derivative = self.five_point_stencil_method()
                else:
                    raise ValueError("Invalid method. Choose 'stem' or 'five_point_stencil'.")
            except Exception as e:
                self._log_debug_message(f"[calculate_derivatives_with_output_noise] Derivative failed: {e}")
            finally:
                self.function = original_func

            derivatives.append(derivative)

        return derivatives
