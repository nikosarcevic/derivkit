import numpy as np

from derivkit.finite_difference import FiniteDifferenceDerivative


class AdaptiveFitDerivative:
    """
    Computes derivatives using an adaptive polynomial fitting approach.

    This method evaluates a function at symmetrically spaced points around a central value,
    fits a polynomial of specified degree (equal to the derivative order), and computes the
    derivative by evaluating the derivative of the polynomial at the central point.

    If the polynomial fit fails to meet a specified residual tolerance, the method adaptively
    removes the outermost points to improve fit stability. If all attempts fail, it optionally
    falls back to a finite difference method.

    Attributes
    ----------
    function : callable
        The function to be differentiated. Must return scalar or vector output.
    central_value : float
        The point at which the derivative is evaluated.
    derivative_order : int
        The derivative order (1–4 supported).

    Methods
    -------
    compute(...)
        Runs the adaptive fit procedure to estimate the derivative.
    get_adaptive_offsets(...)
        Returns an array of offset percentages to use for adaptive sampling.
    _fallback_derivative()
        Uses high-order finite differences as a fallback if fit fails.
    recommend_fit_tolerance()
        Suggests a fit tolerance based on local function behavior.
    _compute_weights()
        Provides inverse-distance weights for polynomial fitting.
    """

    def __init__(self, function, central_value, derivative_order=1):
        self.function = function
        self.central_value = central_value
        self.derivative_order = derivative_order
        self.diagnostics_data = None

    def compute(
            self,
            include_zero=True,
            min_samples=7,
            fit_tolerance=0.05,
            use_fallback=True,
            diagnostics=False
    ):
        """
        Compute the derivative using adaptive polynomial fitting.

        Parameters
        ----------
        include_zero : bool
            Whether to include the central x-value in the sample.
        min_samples : int
            Minimum number of samples required for fitting.
        fit_tolerance : float
            Maximum relative residual allowed for a successful fit.
        use_fallback : bool
            Whether to fall back to finite difference if fit fails.
        diagnostics : bool
            Whether to return detailed fit diagnostics.

        Returns
        -------
        derivative : float or np.ndarray
            Estimated derivative(s).
        diagnostics_dict : dict, optional
            Dictionary containing all points, used points, fit, and residuals.
        """
        if self.derivative_order not in [1, 2, 3, 4]:
            raise ValueError("Only derivative orders 1 to 4 are supported.")

        # Start with initial offsets
        offsets = self.get_adaptive_offsets()
        while True:
            percentages = np.insert(offsets, 0, 0.0) if include_zero else offsets
            x_offsets = np.unique(np.concatenate([percentages, -percentages]))
            if len(x_offsets) >= min_samples:
                break
            # Expand adaptively
            next_offset = offsets[-1] * 1.5
            if next_offset > 0.1:
                break  # Avoid runaway spacing
            offsets = np.append(offsets, next_offset)

        x_values = self.central_value + x_offsets
        y_values = np.vstack([np.atleast_1d(self.function(x)) for x in x_values])
        n_points, n_components = y_values.shape
        derivatives = np.zeros(n_components)

        if diagnostics:
            self.diagnostics_data = {
                "x_all": x_values.copy(),
                "y_all": y_values.copy(),
                "x_used": [],
                "y_used": [],
                "y_fit": [],
                "residuals": [],
                "used_mask": []
            }
        else:
            self.diagnostics_data = None

        for idx in range(n_components):
            x_vals = x_values.copy()
            y_vals = y_values[:, idx].copy()
            success = False

            while len(x_vals) >= max(min_samples, self.derivative_order + 2):
                weights = self._compute_weights(x_vals)

                try:
                    coeffs = np.polyfit(x_vals, y_vals, deg=self.derivative_order, w=weights)
                    poly = np.poly1d(coeffs)
                except np.linalg.LinAlgError:
                    break

                y_fit = poly(x_vals)
                safe_y = np.maximum(np.abs(y_vals), 1e-8)
                residuals = np.abs(y_fit - y_vals) / safe_y
                rel_error = np.max(residuals)

                if rel_error < fit_tolerance:
                    derivatives[idx] = poly.deriv(m=self.derivative_order)(self.central_value)
                    if diagnostics:
                        self._store_diagnostics_entry(x_values, x_vals, y_vals, y_fit, residuals)
                    success = True
                    break

                if len(x_vals) > max(min_samples, self.derivative_order + 2) + 2:
                    sort_idx = np.argsort(x_vals)
                    x_vals = x_vals[sort_idx]
                    y_vals = y_vals[sort_idx]
                    mid_idx = np.argmin(np.abs(x_vals - self.central_value))

                    if 0 < mid_idx < len(x_vals) - 1:
                        x_vals = np.concatenate([x_vals[:mid_idx - 1], x_vals[mid_idx + 2:]])
                        y_vals = np.concatenate([y_vals[:mid_idx - 1], y_vals[mid_idx + 2:]])
                    else:
                        break
                else:
                    break

            if not success:
                derivatives[idx] = self._fallback_derivative()[idx] if use_fallback else np.nan
                if diagnostics:
                    self._store_diagnostics_entry(x_values, None, None, None, None)

        if diagnostics:
            return derivatives if n_components > 1 else derivatives[0], self.diagnostics_data
        return derivatives if n_components > 1 else derivatives[0]

    def computeold(
            self,
            include_zero=True,
            min_samples=7,
            fit_tolerance=0.05,
            use_fallback=True,
            diagnostics=False
    ):
        """
        Compute the derivative using adaptive polynomial fitting.

        Parameters
        ----------
        include_zero : bool
            Whether to include the central x-value in the sample.
        min_samples : int
            Minimum number of samples required for fitting.
        fit_tolerance : float
            Maximum relative residual allowed for a successful fit.
        use_fallback : bool
            Whether to fall back to finite difference if fit fails.
        diagnostics : bool
            Whether to return detailed fit diagnostics.

        Returns
        -------
        derivative : float or np.ndarray
            Estimated derivative(s).
        diagnostics_dict : dict, optional
            Dictionary containing all points, used points, fit, and residuals.
        """
        if self.derivative_order not in [1, 2, 3, 4]:
            raise ValueError("Only derivative orders 1 to 4 are supported.")

        offsets = self.get_adaptive_offsets()
        percentages = np.insert(offsets, 0, 0.0) if include_zero else offsets
        x_offsets = np.unique(np.concatenate([percentages, -percentages]))
        x_values = self.central_value + x_offsets
        y_values = np.vstack([np.atleast_1d(self.function(x)) for x in x_values])
        n_points, n_components = y_values.shape
        derivatives = np.zeros(n_components)

        if diagnostics:
            self.diagnostics_data = {
                "x_all": x_values.copy(),
                "y_all": y_values.copy(),
                "x_used": [],
                "y_used": [],
                "y_fit": [],
                "residuals": [],
                "used_mask": []
            }
        else:
            self.diagnostics_data = None

        print("plese fuckign work")

        for idx in range(n_components):
            x_vals = x_values.copy()
            y_vals = y_values[:, idx].copy()
            success = False

            while len(x_vals) >= max(min_samples, self.derivative_order + 2):
                weights = self._compute_weights(x_vals)

                try:
                    coeffs = np.polyfit(x_vals, y_vals, deg=self.derivative_order, w=weights)
                    poly = np.poly1d(coeffs)
                except np.linalg.LinAlgError:
                    break

                y_fit = poly(x_vals)
                safe_y = np.maximum(np.abs(y_vals), 1e-8)
                residuals = np.abs(y_fit - y_vals) / safe_y
                rel_error = np.max(residuals)

                if rel_error < fit_tolerance:
                    derivatives[idx] = poly.deriv(m=self.derivative_order)(self.central_value)
                    if diagnostics:
                        self._store_diagnostics_entry(x_values, x_vals, y_vals, y_fit, residuals)
                    success = True
                    break

                if len(x_vals) > max(min_samples, self.derivative_order + 2) + 2:
                    sort_idx = np.argsort(x_vals)
                    x_vals = x_vals[sort_idx]
                    y_vals = y_vals[sort_idx]
                    mid_idx = np.argmin(np.abs(x_vals - self.central_value))

                    if 0 < mid_idx < len(x_vals) - 1:
                        x_vals = np.concatenate([x_vals[:mid_idx - 1], x_vals[mid_idx + 2:]])
                        y_vals = np.concatenate([y_vals[:mid_idx - 1], y_vals[mid_idx + 2:]])
                    else:
                        break
                else:
                    break

            if not success:
                derivatives[idx] = self._fallback_derivative()[idx] if use_fallback else np.nan
                if diagnostics:
                    self._store_diagnostics_entry(x_values, None, None, None, None)

        if diagnostics:
            return derivatives if n_components > 1 else derivatives[0], self.diagnostics_data
        return derivatives if n_components > 1 else derivatives[0]

    def get_adaptive_offsets(self, base=0.01, factor=1.5, num_offsets=10, max_offset=0.05):
        """
        Generate symmetric adaptive offsets for sampling around central value.

        Parameters
        ----------
        base : float
            Starting offset as fraction of central_value.
        factor : float
            Multiplicative growth factor for subsequent offsets.
        num_offsets : int
            Number of offset steps to generate.
        max_offset : float
            Maximum total offset allowed.

        Returns
        -------
        np.ndarray
            Array of unique offset percentages (positive only).
        """
        offsets = [min(base * (factor ** i), max_offset) for i in range(num_offsets)]
        result = np.unique([o for o in offsets if o <= max_offset])
        if len(result) == 0:
            raise ValueError("No valid offsets generated.")
        return result

    def _fallback_derivative(self):
        """
        Fallback method using high-order finite differences.

        Returns
        -------
        np.ndarray
            Estimated derivative values (always 1D array).
        """
        fd = FiniteDifferenceDerivative(
            function=self.function,
            central_value=self.central_value,
            derivative_order=self.derivative_order,
        )
        result = fd.compute()
        return np.atleast_1d(result)

    def recommend_fit_tolerance(self, dx=1e-3, verbose=True):
        """
        Suggests a fit_tolerance value based on local function variation.

        Parameters
        ----------
        dx : float
            Offset used to probe nearby values.
        verbose : bool
            Whether to print out the suggested value.

        Returns
        -------
        float
            Recommended fit_tolerance.
        """
        try:
            y_plus = np.atleast_1d(self.function(self.central_value + dx))
            y_minus = np.atleast_1d(self.function(self.central_value - dx))
            variation = np.max(np.abs(y_plus - y_minus))
            scale = np.abs(self.central_value)

            if variation > 1e3 or scale < 1e-2:
                tol = 0.1
            elif variation > 10:
                tol = 0.05
            else:
                tol = 0.01

            if verbose:
                print(f"[recommend_fit_tolerance] Δf = {variation:.3g}, x₀ = {scale:.3g} → Suggested tol = {tol}")
            return tol

        except Exception as e:
            if verbose:
                print(f"[recommend_fit_tolerance] Could not evaluate near x₀: {e}")
            return 0.05

    def _compute_weights(self, x_vals):
        """
        Compute inverse-distance weights to emphasize central values.

        Parameters
        ----------
        x_vals : np.ndarray
            Sample locations used in fitting.

        Returns
        -------
        np.ndarray
            Weight array (same length as x_vals).
        """
        return 1 / (np.abs(x_vals - self.central_value) + 1e-4)

    def _store_diagnostics_entry(self, x_all, x_used, y_used, y_fit, residuals):
        """
        Store diagnostic info for a single component.
        """
        if self.diagnostics_data is None:
            return

        if x_used is None:
            self.diagnostics_data["x_used"].append(None)
            self.diagnostics_data["y_used"].append(None)
            self.diagnostics_data["y_fit"].append(None)
            self.diagnostics_data["residuals"].append(None)
            self.diagnostics_data["used_mask"].append(np.zeros_like(x_all, dtype=bool))
        else:
            mask = np.isclose(x_all[:, None], x_used, rtol=1e-12, atol=1e-15).any(axis=1)
            self.diagnostics_data["x_used"].append(x_used.copy())
            self.diagnostics_data["y_used"].append(y_used.copy())
            self.diagnostics_data["y_fit"].append(y_fit.copy())
            self.diagnostics_data["residuals"].append(residuals.copy())
            self.diagnostics_data["used_mask"].append(mask)
