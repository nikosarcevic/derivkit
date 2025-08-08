from typing import Optional
import warnings
warnings.simplefilter("once", category=RuntimeWarning)

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

    def __init__(self, function, central_value, derivative_order=1, fit_tolerance=0.05):
        self.function = function
        self.central_value = central_value
        self.derivative_order = derivative_order
        self.fit_tolerance = fit_tolerance
        self.diagnostics_data = None
        self.min_used_points = 5  # Minimum number of samples required for fitting

    def compute(
            self,
            include_zero=True,
            min_samples=7,
            diagnostics=False,
            fallback_mode: str = "finite_difference",  # "finite_difference" | "poly_at_floor" | "auto"
            floor_accept_multiplier: float = 2.0,  # used when fallback_mode == "auto"
    ):
        """
        Compute the derivative using adaptive polynomial fitting.

        High-level flow
        ---------------
        1) Build a symmetric sampling grid around x0 (>= required_points).
        2) For each output component:
           a) Fit a degree-`derivative_order` polynomial in normalized coords.
           b) If residuals < tolerance -> accept.
           c) Else, prune worst residuals (respecting the minimum count) and repeat.
           d) If pruning hits the minimum and still not within tolerance:
              - Decide according to `fallback_mode`:
                * "poly_at_floor": accept polynomial anyway (warn).
                * "auto": accept if "close enough" (warn), otherwise fall back.
                * "finite_difference": fall back to FD (warn).
        3) Record diagnostics at every exit path so plots are informative.
        """
        if self.derivative_order not in [1, 2, 3, 4]:
            raise ValueError("Only derivative orders 1 to 4 are supported.")

        # Sampling grid
        x_offsets, required_points = self._build_x_offsets(include_zero, min_samples)
        x_values = self.central_value + x_offsets

        # Evaluate the function at those points
        y_values = np.vstack([np.atleast_1d(self.function(x)) for x in x_values])
        _, n_components = y_values.shape
        derivatives = np.zeros(n_components)

        # Init diagnostics container
        if diagnostics:
            self.diagnostics_data = {
                "x_all": x_values.copy(),
                "y_all": y_values.copy(),
                "x_used": [],
                "y_used": [],
                "y_fit": [],
                "residuals": [],
                "used_mask": [],
                "status": [],
            }
        else:
            self.diagnostics_data = None

        # Component-wise fit
        for idx in range(n_components):
            x_vals = x_values.copy()
            y_vals = y_values[:, idx].copy()
            success = False

            # Track last attempt (for floor acceptance / FD fallback)
            last_x = last_y = last_yfit = last_resid = None

            while len(x_vals) >= required_points:
                # Fit once on current set
                fit = self._fit_once(x_vals, y_vals)
                last_fit = fit
                if not fit["ok"]:
                    # singular normal equations; break to FD / floor handling
                    break

                last_x, last_y = x_vals.copy(), y_vals.copy()
                last_yfit, last_resid = fit["y_fit"].copy(), fit["residuals"].copy()

                # Accept if within tolerance
                if fit["rel_error"] < self.fit_tolerance:
                    m = self.derivative_order
                    derivatives[idx] = last_fit["poly_u"].deriv(m=m)(0.0) / (last_fit["h"] ** m)
                    if diagnostics:
                        self._store_diagnostics_entry(x_values, x_vals, y_vals, last_yfit, last_resid)
                        self.diagnostics_data["status"].append({
                            "component": int(idx),
                            "mode": "poly",
                            "rel_error": float(fit["rel_error"]),
                            "accepted": True,
                        })
                    success = True
                    break

                # Prune worst residuals and refit
                x_vals, y_vals, removed = self._prune_by_residuals(
                    x_vals, y_vals, fit["residuals"], self.fit_tolerance, required_points,
                    max_remove=2, keep_center=True, keep_symmetric=True,
                )
                if removed:
                    continue  # loop back with smaller set

                # At floor and still failing tolerance -> decide fallback
                at_floor = (last_x is not None) and (len(last_x) == required_points)
                accept, tag = self._maybe_accept_at_floor(
                    last_resid, at_floor, self.fit_tolerance, fallback_mode, floor_accept_multiplier
                )
                if accept:
                    m = self.derivative_order
                    derivatives[idx] = last_fit["poly_u"].deriv(m=m)(0.0) / (last_fit["h"] ** m)
                    if diagnostics:
                        self._store_diagnostics_entry(x_values, last_x, last_y, last_yfit, last_resid)
                        entry = {
                            "component": int(idx),
                            "mode": tag,
                            "max_resid": float(np.max(last_resid)),
                            "median_resid": float(np.median(last_resid)),
                            "fit_tolerance": float(self.fit_tolerance),
                            "floor_accept_multiplier": float(floor_accept_multiplier),
                            "accepted": True,
                        }
                        self.diagnostics_data["status"].append(entry)
                    warnings.warn(
                        f"[AdaptiveFitDerivative] Accepted polynomial at minimum points ({required_points}) "
                        f"with residuals above tolerance (max={np.max(last_resid):.3g}, "
                        f"tol={self.fit_tolerance:.3g}) using mode='{tag}'.",
                        RuntimeWarning
                    )
                    success = True
                    break

                # couldn't accept at floor -> break to FD
                break

            # FD fallback if still not successful
            if not success:
                fd_val = self._fallback_derivative()[idx]
                derivatives[idx] = fd_val
                if diagnostics:
                    if last_x is not None:
                        self._store_diagnostics_entry(x_values, last_x, last_y, last_yfit, last_resid)
                    else:
                        self._store_diagnostics_entry(x_values, None, None, None, None)
                    self.diagnostics_data["status"].append({
                        "component": int(idx),
                        "mode": "finite_difference",
                        "accepted": True,
                        "reason": "fit_not_within_tolerance_or_insufficient_points",
                    })
                detail = ""
                if last_resid is not None:
                    detail = f" (last max residual {np.max(last_resid):.3g} vs tol {self.fit_tolerance:.3g})"
                warnings.warn(
                    f"[AdaptiveFitDerivative] Falling back to finite differences because polynomial fit "
                    f"did not meet tolerance{detail}.",
                    RuntimeWarning
                )

        if diagnostics:
            return np.atleast_1d(derivatives), self.diagnostics_data
        return np.atleast_1d(derivatives)

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
            Base relative step size as a fraction of the central value. Default is 1% of |x0|.
         base_abs : float
            Base absolute step size for small central values. Default is 1e-6.
         factor : float
            Factor by which to increase the step size for each offset. Default is 1.5.
         num_offsets : int
            Number of offsets to generate. Default is 10.
         max_rel : float
            Maximum relative step size as a fraction of the central value. Default is 5% of |x0|.
         max_abs : float
            Maximum absolute step size. Default is 1e-2.
         step_mode : str
            Mode for determining step sizes: "auto", "relative", or "absolute". Defaults to "auto".
         x_small_threshold : float
            Threshold below which the central value is considered small. Default is 1e-3.

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

    def _build_x_offsets(self, include_zero: bool, min_samples: int):
        """
        Build a symmetric sampling grid around `central_value`, expanding offsets
        until we have at least `required_points` abscissae.

        Returns
        -------
        x_offsets : np.ndarray
            Symmetric offsets (includes +/- and possibly 0).
        required_points : int
            Minimum number of points we must keep in the fit loop.
        """
        # Enforce a hard floor of >=5 used points
        required_points = max(min_samples, max(self.min_used_points, self.derivative_order + 2))

        offsets = self.get_adaptive_offsets()
        growth_limit = offsets[-1] * (1.5 ** 3)

        while True:
            steps = np.insert(offsets, 0, 0.0) if include_zero else offsets
            x_offsets = np.unique(np.concatenate([steps, -steps]))
            if len(x_offsets) >= required_points:
                break
            next_offset = offsets[-1] * 1.5
            if next_offset > growth_limit:
                break
            offsets = np.append(offsets, next_offset)

        return x_offsets, required_points

    def _fit_once(self, x_vals: np.ndarray, y_vals: np.ndarray):
        """
        Perform one normalized weighted polynomial fit on (x_vals, y_vals).

        Normalization: u = (x - x0)/h with h = max|x - x0| so u ~ [-1, 1].
        Returns a dict with objects needed by the main loop.
        """
        # Normalize coordinates around the center for conditioning
        t_vals = x_vals - self.central_value
        h = np.max(np.abs(t_vals))
        h = max(h, 1e-12)  # avoid divide-by-zero
        u_vals = t_vals / h

        # Scale-aware inverse-distance weights
        weights = self._compute_weights(x_vals)

        # Weighted polynomial fit in u-space
        try:
            coeffs = np.polyfit(u_vals, y_vals, deg=self.derivative_order, w=weights)
            poly_u = np.poly1d(coeffs)
        except np.linalg.LinAlgError:
            return {
                "ok": False, "reason": "singular_normal_equations",
                "h": h, "u_vals": u_vals, "poly_u": None,
                "y_fit": None, "residuals": None, "rel_error": np.inf,
            }

        # Residuals (relative, guarded near zero)
        y_fit = poly_u(u_vals)
        safe_y = np.maximum(np.abs(y_vals), 1e-8)
        residuals = np.abs(y_fit - y_vals) / safe_y
        rel_error = float(np.max(residuals))

        return {
            "ok": True, "reason": None,
            "h": h, "u_vals": u_vals, "poly_u": poly_u,
            "y_fit": y_fit, "residuals": residuals, "rel_error": rel_error,
        }

    def _maybe_accept_at_floor(
            self,
            last_residuals: Optional[np.ndarray],
            at_floor: bool,
            fit_tolerance: float,
            fallback_mode: str,
            floor_accept_multiplier: float,
    ):
        """
        Decide whether to accept the polynomial fit at the minimum sample count.

        Returns
        -------
        accept : bool
        tag : str   # human-friendly reason tag for diagnostics/warnings
        """
        if not at_floor:
            return False, "not_at_floor"

        if fallback_mode == "poly_at_floor":
            return True, "poly_at_floor"

        if fallback_mode == "auto":
            if last_residuals is None:
                return False, "auto_no_residuals"
            max_r = float(np.max(last_residuals))
            med_r = float(np.median(last_residuals))
            close_enough = (max_r < floor_accept_multiplier * fit_tolerance) and (med_r < fit_tolerance)
            return (True, "auto_accept_at_floor") if close_enough else (False, "auto_reject")

        # default: finite_difference
        return False, "finite_difference"

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

        Parameters
        ----------
        x_all : np.ndarray
            All x values used in the fit.
        x_used : np.ndarray or None
            x values actually used in the fit (None if no points were used).
        y_used : np.ndarray or None
            y values corresponding to `x_used` (None if no points were used).
        y_fit : np.ndarray or None
            Fitted y values corresponding to `x_used` (None if no points were used).
        residuals : np.ndarray or None
            Residuals of the fit corresponding to `y_used` (None if no points were used).

        Returns
        -------
            Updates the diagnostics_data dictionary with the new entry.
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
