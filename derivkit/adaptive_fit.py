import numpy as np

from derivkit.finite_difference import FiniteDifferenceDerivative


class AdaptiveFitDerivative:
    """
    Computes derivatives using an adaptive polynomial fitting approach.

    This method evaluates a function at symmetrically spaced points around a central value,
    fits a polynomial of specified degree (equal to the derivative order), and computes the
    derivative by evaluating the derivative of the polynomial at the central point.

    If the polynomial fit fails to meet a specified residual tolerance, the method adaptively
    removes points with the largest relative residuals (optionally in symmetric pairs) and refits
    until the tolerance is met or the minimum sample count is reached.

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
        Returns an array of absolute step sizes to use for adaptive sampling.
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
        self.min_used_points = 5  # Minimum number of samples required for fitting

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

        # Enforce a hard floor of >=5 used points
        required_points = max(min_samples, max(self.min_used_points, self.derivative_order + 2))

        # Start with initial (absolute) steps and expand until we have enough symmetric points
        offsets = self.get_adaptive_offsets()
        cap = np.max(offsets)  # honor the effective max_abs from the generator
        while True:
            steps = np.insert(offsets, 0, 0.0) if include_zero else offsets  # renamed for clarity
            x_offsets = np.unique(np.concatenate([steps, -steps]))
            if len(x_offsets) >= required_points:
                break
            # Expand adaptively. Keep a conservative cap to avoid huge nonlocal steps.
            next_offset = offsets[-1] * 1.5
            if next_offset > cap:
                break
            offsets = np.append(offsets, next_offset)

        x_values = self.central_value + x_offsets
        y_values = np.vstack([np.atleast_1d(self.function(x)) for x in x_values])
        _, n_components = y_values.shape
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

            while len(x_vals) >= required_points:
                # Center & normalize coordinates for a well-conditioned fit
                t_vals = x_vals - self.central_value
                h = np.max(np.abs(t_vals))
                h = max(h, 1e-12)  # floor to avoid divide-by-zero
                u_vals = t_vals / h  # now roughly in [-1, 1]

                weights = self._compute_weights(x_vals)

                try:
                    coeffs = np.polyfit(u_vals, y_vals, deg=self.derivative_order, w=weights)
                    poly_u = np.poly1d(coeffs)
                except np.linalg.LinAlgError:
                    break

                # Predicted y on the used points (still a y-fit, just using u as input)
                y_fit = poly_u(u_vals)

                # Relative residuals (guard small y)
                safe_y = np.maximum(np.abs(y_vals), 1e-8)
                residuals = np.abs(y_fit - y_vals) / safe_y
                rel_error = np.max(residuals)

                if rel_error < fit_tolerance:
                    m = self.derivative_order
                    # chain-rule rescaling: d^m f/dx^m at x0 = (1/h^m) * d^m f/du^m at u=0
                    derivatives[idx] = poly_u.deriv(m=m)(0.0) / (h ** m)
                    if diagnostics:
                        self._store_diagnostics_entry(x_values, x_vals, y_vals, y_fit, residuals)
                    success = True
                    break

                # Prune by residuals (remove worst offenders outside tolerance), then refit
                x_vals, y_vals, removed = self._prune_by_residuals(
                    x_vals, y_vals, residuals, fit_tolerance, required_points,
                    max_remove=2, keep_center=True, keep_symmetric=True,
                )
                if not removed:
                    break

            if not success:
                derivatives[idx] = self._fallback_derivative()[idx] if use_fallback else np.nan
                if diagnostics:
                    self._store_diagnostics_entry(x_values, None, None, None, None)

        if diagnostics:
            return derivatives if n_components > 1 else derivatives[0], self.diagnostics_data
        return derivatives if n_components > 1 else derivatives[0]

    def get_adaptive_offsets(
            self,
            base_rel=0.01,  # 1% of |x0| as first relative step
            base_abs=1e-6,  # absolute step if |x0| is small
            factor=1.5,
            num_offsets=10,
            max_rel=0.05,  # cap at 5% of |x0|
            max_abs=1e-2,  # cap absolute step
            step_mode="auto",  # "auto" | "relative" | "absolute"
            x_small_threshold=1e-3,
    ):
        """
         Returns an array of absolute step sizes to use for adaptive sampling.

         Parameters
         ----------
         base_rel : float
            Base relative step size as a fraction of the central value.
         base_abs : float
            Base absolute step size for small central values.
         factor : float
            Factor by which to increase the step size for each offset.
         num_offsets : int
            Number of offsets to generate.
         max_rel : float
            Maximum relative step size as a fraction of the central value.
         max_abs : float
            Maximum absolute step size.
         step_mode : str
            Mode for determining step sizes: "auto", "relative", or "absolute".
         x_small_threshold : float
            Threshold below which the central value is considered small.

        Returns:
            Absolute step sizes (positive only), not including the central 0 step.
        """
        x0 = float(self.central_value)
        use_abs = (step_mode == "absolute") or (step_mode == "auto" and abs(x0) < x_small_threshold)

        if use_abs:
            bases = [min(base_abs * (factor ** i), max_abs) for i in range(num_offsets)]
        else:
            scale = max(abs(x0), x_small_threshold)  # guard near zero
            bases = [min(base_rel * (factor ** i), max_rel) * scale for i in range(num_offsets)]

        offsets = np.unique([b for b in bases if b > 0.0])
        if len(offsets) == 0:
            raise ValueError("No valid offsets generated.")
        return offsets

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
        Compute scale-aware inverse-distance weights centered at ``central_value``.

        This weighting emphasizes samples closest to the expansion point ``x0 = self.central_value``
        while remaining numerically stable across *very small* and *very large* parameter scales.

        **Formulation**
        ----------------
        Let ``d_i = |x_i - x0|`` and ``D = max_i d_i`` (the current sampling span).
        We use
            ``eps = max(1e-3 * D, 1e-9)``,
        and define the weights
            ``w_i = 1 / (d_i + eps)``.

        - The additive softening ``eps`` prevents a singular weight at the center (where ``d_i = 0``).
        - Tying ``eps`` to the span ``D`` makes the weighting *scale-invariant*: the relative
          emphasis near the center is similar whether ``x0`` is ~1e-4 or ~1e4.

        **Intuition**
        --------------
        With ``eps = 1e-3 * D``, the center-to-edge weight ratio is roughly:
            ``w(0) / w(D) ≈ (D + eps) / eps ≈ D / eps ≈ 10^3``.
        So points near ``x0`` can carry ~1000× more weight than the farthest points—strong,
        but not overwhelming. You can tune this by changing the 1e-3 factor:
        - Larger factor (e.g., ``1e-2``) → milder emphasis (~100×).
        - Smaller factor (e.g., ``1e-4``) → sharper emphasis (~10,000×).

        **Why not a fixed epsilon?**
        -----------------------------
        A constant (e.g., ``1e-4``) behaves poorly across scales:
        - If the step span is tiny (<< 1e-4), the constant dominates and flattens the weights
          (little central emphasis).
        - If the span is large (>> 1e-4), the center weight becomes excessively dominant.

        By scaling ``eps`` with the current span, the weighting profile adapts automatically.

        Parameters
        ----------
        x_vals : np.ndarray
            Sample locations used in fitting. Shape ``(n_points,)``.

        Returns
        -------
        np.ndarray
            Weights of shape ``(n_points,)``. These are **not normalized**; downstream code
            (e.g., ``np.polyfit(..., w=weights)``) uses them as relative weights. If a normalized
            weight vector is required, divide by ``weights.sum()``.

        Notes
        -----
        - Complexity is O(n).
        - The absolute floor ``1e-9`` guards degenerate cases (e.g., ``D ≈ 0``) and prevents
          overflow even if multiple samples coincide with ``x0``.
        """
        d = np.abs(x_vals - self.central_value)
        eps = max(np.max(d) * 1e-3, 1e-9)  # 0.1% of span with tiny floor
        return 1.0 / (d + eps)

    def _prune_by_residuals(
            self,
            x_vals: np.ndarray,
            y_vals: np.ndarray,
            residuals: np.ndarray,
            fit_tolerance: float,
            required_points: int,
            *,
            max_remove: int = 2,
            keep_center: bool = True,
            keep_symmetric: bool = True,
    ):
        """
        Remove points whose relative residual exceeds ``fit_tolerance``, then refit.

        Strategy
        --------
        - Sort points by residual (worst-first) and remove up to ``max_remove`` per call.
        - Never drop the center sample (closest to ``central_value``) if ``keep_center`` is True.
        - If ``keep_symmetric`` is True, also remove the mirror of the removed point
          about ``central_value`` when possible (to keep sampling balanced).
        - Never reduce the number of points below ``required_points``.

        Parameters
        ----------
        x_vals, y_vals : np.ndarray
            Current sample abscissae and ordinates (same length).
        residuals : np.ndarray
            Relative residuals for the *current* fit at ``x_vals``.
        fit_tolerance : float
            Acceptable residual threshold (e.g., 0.05 for 5%).
        required_points : int
            Minimum number of points allowed after pruning.
        max_remove : int, optional
            Maximum number of points to remove in this call (default 2).
        keep_center : bool, optional
            If True, never remove the point closest to ``central_value``.
        keep_symmetric : bool, optional
            If True, attempt to remove a mirror point along with the worst point.

        Returns
        -------
        x_new, y_new : np.ndarray
            Pruned arrays (may be unchanged).
        removed_any : bool
            True if at least one point was removed.
        """
        assert len(x_vals) == len(y_vals) == len(residuals)

        # Indices sorted by residual (descending): worst offenders first
        order = np.argsort(residuals)[::-1]

        # Identify the sample closest to the expansion point
        center_idx = int(np.argmin(np.abs(x_vals - self.central_value))) if len(x_vals) else -1

        keep = np.ones(len(x_vals), dtype=bool)
        removed = 0

        for j in order:
            if removed >= max_remove:
                break
            if residuals[j] <= fit_tolerance:
                break  # remaining points are within tolerance

            if keep_center and j == center_idx:
                continue  # don't drop the center point

            # Ensure we don't go under the minimum count
            if keep.sum() - 1 < required_points:
                break

            # Drop the worst point
            keep[j] = False
            removed += 1

            # Try to drop a symmetric mate about x0 to keep balance
            if keep_symmetric and removed < max_remove and keep.sum() - 1 >= required_points:
                target = 2.0 * self.central_value - x_vals[j]  # mirror of x_j about x0
                k = int(np.argmin(np.abs(x_vals - target)))
                if k != j and (not keep_center or k != center_idx) and keep[k]:
                    keep[k] = False
                    removed += 1

        if removed == 0:
            return x_vals, y_vals, False

        return x_vals[keep], y_vals[keep], True

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
