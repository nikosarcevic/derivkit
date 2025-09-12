"""Provides the AdaptiveFitDerivative class.

The user must specify the function to differentiate and the central value
at which the derivative should be evaluated. More details about available
options can be found in the documentation of the methods.

Typical usage example:

>>> derivative = AdaptiveFitDerivative(function_to_differentiate, 1).compute()

derivative is the derivative of function_to_differerentiate at value 1.
"""

import warnings
from typing import Optional

import numpy as np
from multiprocess import Pool

from derivkit.finite_difference import FiniteDifferenceDerivative

warnings.simplefilter("once", category=RuntimeWarning)

class AdaptiveFitDerivative:
    """Computes derivatives using an adaptive polynomial fitting approach.

    This method evaluates a function at symmetrically spaced points around
    a central value, fits a polynomial of specified degree (equal to the
    derivative order), and computes the derivative by evaluating the derivative
    of the polynomial at the central point.

    If the polynomial fit fails to meet a specified residual tolerance, the
    method adaptively removes points with the largest relative residuals
    (optionally in symmetric pairs) and refits until the tolerance is met or the
    minimum sample count is reached.

    Attributes:
        function (callable): The function to be differentiated. Must return
            scalar or vector output.
        central_value (float): The point at which the derivative is evaluated.
        diagnostics_data (dict, optional): The diagnostic data collected during
            the fitting process, if requested. Defaults to `None`.
        min_used_points (int): The default minimum number of samples required
            for fitting. This is used to ensure that the polynomial fit has
            enough data points to be meaningful.
    """

    def __init__(self, function, central_value):
        """Initialises the class based on the function and central value.

        Args:
            function (callable): The function to be differentiated. Must return
                scalar or vector output.
            central_value (float): The point at which the derivative is
                evaluated.
        """
        self.function = function
        self.central_value = central_value
        self.diagnostics_data = None
        self.min_used_points = (
            5  # Minimum number of samples required for fitting
        )

    def compute(
        self,
        include_zero=True,
        derivative_order=1,
        min_samples=7,
        diagnostics=False,
        fallback_mode: str = "finite_difference",
        fit_tolerance=0.05,
        floor_accept_multiplier: float = 2.0,
        n_workers: int = 1,
    ):
        """Computes class derivative using adaptive polynomial fitting.

        This method evaluates the target function at symmetric points around
        a central value, fits a polynomial of the specified order, and computes
        the derivative from that fit. If the fit quality does not meet the
        specified tolerance, it adaptively prunes the worst residual points and
        refits. If all attempts fail, it falls back to a finite difference
        estimate or accepts a fit based on a looser criterion.

        Args:
            derivative_order (int, optional): The order of the derivative to
                compute (default is 1). Must be 1, 2, 3, or 4.
            min_samples (int, optional): Minimum number of total samples to
                start with. Must be large enough to support the fit and any
                fallback strategies. Default is 7.
            fit_tolerance (float, optional): Maximum acceptable relative
                residual for the polynomial fit. Default is 0.05, i.e 5%.
            include_zero (bool, optional): Whether to include the central
                point in the sampling grid. Default is `True`.
            fallback_mode: Strategy if the fit fails to meet
                the tolerance at the minimum sample count. Options are:

                    - `"finite_difference"`: always reject and use finite difference
                      fallback.
                    - `"poly_at_floor"`: always accept the polynomial fit at floor
                      regardless of residuals.
                    - `"auto"`: accept the polynomial at floor if both of the
                      following conditions hold:

                        * (max_residual < floor_accept_multiplier × fit_tolerance)
                        * (median_residual < fit_tolerance);

                      otherwise fall back to finite differences.

                Default is `"finite_difference"`.
            floor_accept_multiplier: Tolerance multiplier
                used in "auto" fallback mode. Default is 2.0.
            diagnostics (bool, optional): Whether to return diagnostic
                information such as residuals, fit values, and pruning status.
                Default is `False`.
            n_workers: Number of worker to use in multiprocessing. Default is 1
                (no multiprocessing).

        Returns:
            :class:`np.ndarray`, (:class:`np.ndarray`, dict): The estimated
                derivative(s). If `diagnostics` is `True`, returns a tuple
                with the derivative array and a dictionary of diagnostic data.

        Raises:
            ValueError: An error occurred attempting to comput derivatives
                of order higher than 4.
        """
        if derivative_order not in [1, 2, 3, 4]:
            raise ValueError(
                f"Invalid derivative_order={derivative_order}. "
                "Only derivative orders 1 to 4 are supported; "
                "higher orders are not currently implemented."
            )

        # Sampling grid
        x_offsets, required_points = self._build_x_offsets(
            central_value=self.central_value,
            derivative_order=derivative_order,
            include_zero=include_zero,
            min_samples=min_samples,
        )
        x_values = self.central_value + x_offsets

        # Evaluate the function at those points
        if n_workers > 1:
            n_workers = np.min((n_workers, len(x_values)))
            with Pool(n_workers) as pool:
                y_values = pool.map(self.function, x_values)
        else:
            y_values = np.vstack(
                [np.atleast_1d(self.function(x)) for x in x_values]
            )

        y_values = np.vstack([np.atleast_1d(y) for y in y_values])
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
                "fit_poly": None,
                "fit_tolerance": None,
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
                fit = self._fit_once(x_vals, y_vals, derivative_order)
                last_fit = fit
                if not fit["ok"]:
                    # singular normal equations; break to FD / floor handling
                    break

                last_x, last_y = x_vals.copy(), y_vals.copy()
                last_yfit, last_resid = (
                    fit["y_fit"].copy(),
                    fit["residuals"].copy(),
                )

                # Accept if within tolerance
                if fit["rel_error"] < fit_tolerance:
                    m = derivative_order
                    derivatives[idx] = last_fit["poly_u"].deriv(m=m)(0.0) / (
                        last_fit["h"] ** m
                    )
                    if diagnostics:
                        self._store_diagnostics_entry(
                            x_values, x_vals, y_vals, last_yfit, last_resid
                        )
                        self.diagnostics_data["status"].append(
                            {
                                "component": int(idx),
                                "mode": "poly",
                                "rel_error": float(fit["rel_error"]),
                                "accepted": True,
                            }
                        )
                    success = True
                    break

                # Prune worst residuals and refit
                x_vals, y_vals, removed = self._prune_by_residuals(
                    x_vals,
                    y_vals,
                    fit["residuals"],
                    fit_tolerance,
                    required_points,
                    max_remove=2,
                    keep_center=True,
                    keep_symmetric=True,
                )
                if removed:
                    continue  # loop back with smaller set

                # At floor and still failing tolerance -> decide fallback
                at_floor = (last_x is not None) and (
                    len(last_x) == required_points
                )
                accept, tag = self._maybe_accept_at_floor(
                    last_resid,
                    at_floor,
                    fit_tolerance,
                    fallback_mode,
                    floor_accept_multiplier,
                )
                if accept:
                    m = derivative_order
                    derivatives[idx] = last_fit["poly_u"].deriv(m=m)(0.0) / (
                        last_fit["h"] ** m
                    )
                    if diagnostics:
                        self._store_diagnostics_entry(
                            x_values, last_x, last_y, last_yfit, last_resid
                        )
                        entry = {
                            "component": int(idx),
                            "mode": tag,
                            "max_resid": float(np.max(last_resid)),
                            "median_resid": float(np.median(last_resid)),
                            "fit_tolerance": float(fit_tolerance),
                            "floor_accept_multiplier": float(
                                floor_accept_multiplier
                            ),
                            "accepted": True,
                        }
                        self.diagnostics_data["status"].append(entry)
                    warnings.warn(
                        f"[AdaptiveFitDerivative] Accepted polynomial at minimum points ({required_points}) "
                        f"with residuals above tolerance (max={np.max(last_resid):.3g}, "
                        f"tol={fit_tolerance:.3g}) using mode='{tag}'.",
                        RuntimeWarning,
                    )
                    success = True
                    break

                # couldn't accept at floor -> break to FD
                break

            # FD fallback if still not successful
            if not success:
                fd_val = self._fallback_derivative(
                    derivative_order, n_workers=n_workers
                )[idx]
                derivatives[idx] = fd_val
                if diagnostics:
                    if last_x is not None:
                        self._store_diagnostics_entry(
                            x_values, last_x, last_y, last_yfit, last_resid
                        )
                    else:
                        self._store_diagnostics_entry(
                            x_values, None, None, None, None
                        )
                    self.diagnostics_data["status"].append(
                        {
                            "component": int(idx),
                            "mode": "finite_difference",
                            "accepted": True,
                            "reason": "fit_not_within_tolerance_or_insufficient_points",
                        }
                    )
                detail = ""
                if last_resid is not None:
                    detail = f" (last max residual {np.max(last_resid):.3g} vs tol {fit_tolerance:.3g})"
                warnings.warn(
                    f"[AdaptiveFitDerivative] Falling back to finite differences because polynomial fit "
                    f"did not meet tolerance{detail}.",
                    RuntimeWarning,
                )

        if diagnostics:
            self.diagnostics_data["fit_poly"] = (
                last_fit["poly_u"] if last_fit is not None else None
            )
            self.diagnostics_data["fit_tolerance"] = fit_tolerance

        val = (
            derivatives.item()
            if derivatives.size == 1
            else np.atleast_1d(derivatives)
        )
        return (val, self.diagnostics_data) if diagnostics else val

    def get_adaptive_offsets(
        self,
        central_value=None,
        base_rel=0.01,  # 1% of central value as first relative step
        base_abs=1e-6,  # absolute step if absolute central value is small
        factor=1.5,
        num_offsets=10,
        max_rel=0.05,  # cap at 5% of absolute central value
        max_abs=1e-2,  # cap absolute step
        step_mode="auto",  # "auto" | "relative" | "absolute"
        x_small_threshold=1e-3,
    ):
        """Returns an array of absolute step sizes to use for adaptive sampling.

        Args:
            central_value (float_optional): The central value around which to
                generate offsets. If None, uses `self.central_value`.
            base_rel (float, optional): Base relative step size as a fraction
                of the central value. Default is 1%.
            base_abs (float, optional): Base absolute step size for small
                central values. Default is 1e-6.
            factor (float, optional): Factor by which to increase the step
                size for each offset. Default is 1.5.
            num_offsets (int, optional): Number of offsets to generate. Default
                is 10.
            max_rel (float, optional): Maximum relative step size as a fraction
                of the central value. Default is 5%.
            max_abs (float, optional): Maximum absolute step size.
                Default is 1e-2.
            step_mode (str, optional): Mode for determining step sizes: "auto",
                "relative", or "absolute". Defaults to "auto".
            x_small_threshold (float, optional): Threshold below which the
                central value is considered small. Default is 1e-3.

        Returns:
            :class:`np.ndarray`: An array with absolute step sizes (positive
                only), not including the central 0 step.
        """
        x0 = (
            self.central_value
            if central_value is None
            else float(central_value)
        )
        use_abs = (step_mode == "absolute") or (
            step_mode == "auto" and abs(x0) < x_small_threshold
        )

        if use_abs:
            bases = [
                min(base_abs * (factor**i), max_abs)
                for i in range(num_offsets)
            ]
        else:
            scale = max(abs(x0), x_small_threshold)  # guard near zero
            bases = [
                min(base_rel * (factor**i), max_rel) * scale
                for i in range(num_offsets)
            ]

        offsets = np.unique([b for b in bases if b > 0.0])
        if len(offsets) == 0:
            raise ValueError("No valid offsets generated.")
        return offsets

    def _build_x_offsets(
        self,
        central_value,
        derivative_order,
        include_zero: bool,
        min_samples: int,
    ):
        """Construct a symmetric array of sampling offsets around central value.

        The method adaptively builds a grid of offsets by mirroring around the
        center, ensuring that the number of total points (including optional
        center) is sufficient for fitting the requested derivative order. It
        expands the offset range as needed until the required minimum number of
        points is met, up to a maximum span.

        Args:
            central_value (float): The central point around which the offsets
                are generated. This is the point at which the derivative will
                be evaluated.
            derivative_order (int): The order of the derivative to be computed.
                Determines the minimum polynomial degree, which affects the
                number of required sample points.
            include_zero: Whether to include the central point (zero
                offset) in the sampling grid.
            min_samples: The minimum number of total sampling points to
                start with before pruning. This interacts with
                `self.min_used_points` (default = 5), which is the minimum
                number of points required to perform a polynomial fit, and with
                the derivative order to determine the final required number of
                usable samples.

        Returns:
            (:class:`np.ndarray`, int): A tuple containing

            - an array of symmetric offsets relative to `self.central_value`,
              optionally including zero. These define where the function will be
              evaluated.
            - The minimum number of samples that must be retained during
              adaptive fitting for the derivative estimate to proceed
              (i.e. the floor used in pruning logic).
        """
        # Enforce a hard floor of >=5 used points
        if self.min_used_points < 5:
            warnings.warn(
                f"[AdaptiveFitDerivative] self.min_used_points={self.min_used_points} is below "
                "the recommended minimum of 5. This may reduce polynomial fit stability.",
                RuntimeWarning,
            )
        order_based_floor = derivative_order + 2
        required_points = max(
            min_samples, max(self.min_used_points, order_based_floor)
        )

        offsets = self.get_adaptive_offsets(central_value=central_value)
        growth_limit = offsets[-1] * (1.5**3)

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

    def _fit_once(
        self, x_vals: np.ndarray, y_vals: np.ndarray, derivative_order: int
    ):
        """Perform one normalized weighted polynomial fit on (x_vals, y_vals).

        Args:
            x_vals: Sample x values used in the fit.
            y_vals: function values corresponsing to `x_vals`.
            derivative_order: Order of the derivative to fit.

        Returns:
            dict: Dictionary containing fit result, polynomial object,
                residuals, and metadata.
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
            coeffs = np.polyfit(
                u_vals, y_vals, deg=derivative_order, w=weights
            )
            poly_u = np.poly1d(coeffs)
        except np.linalg.LinAlgError:
            return {
                "ok": False,
                "reason": "singular_normal_equations",
                "h": h,
                "u_vals": u_vals,
                "poly_u": None,
                "y_fit": None,
                "residuals": None,
                "rel_error": np.inf,
            }

        # Residuals (relative, guarded near zero)
        y_fit = poly_u(u_vals)
        safe_y = np.maximum(np.abs(y_vals), 1e-8)
        residuals = np.abs(y_fit - y_vals) / safe_y
        rel_error = float(np.max(residuals))

        return {
            "ok": True,
            "reason": None,
            "h": h,
            "u_vals": u_vals,
            "poly_u": poly_u,
            "y_fit": y_fit,
            "residuals": residuals,
            "rel_error": rel_error,
        }

    def _maybe_accept_at_floor(
        self,
        last_residuals: Optional[np.ndarray],
        at_floor: bool,
        fit_tolerance: float,
        fallback_mode: str,
        floor_accept_multiplier: float,
    ):
        """Determine whether to accept a polynomial fit.

        The fit is computed at the minimum stencil size ("floor").

        This function is called when the adaptive fitting loop has reached its
        smallest allowable number of sample points (`at_floor=True`) but has
        not yet met the target residual tolerance. Depending on the fallback
        strategy and the quality of the residuals, the method decides whether
        the fit should be accepted anyway or whether to trigger a fallback
        (e.g., finite difference).

        Args:
            last_residuals: Residuals from the last attempted polynomial fit.
                If ``None``, acceptance is based only on fallback_mode and
                `at_floor` status.
            at_floor: Whether the algorithm has reached the minimum allowed
                number of sample points.
            fit_tolerance: The user-specified threshold for acceptablei
                residuals. Typically a small value like 1e-4.
            fallback_mode: Strategy to follow when the fit fails at the floor.
                Options include:

                  - "poly_at_floor: always accept the fit at floor regardless
                    of residuals
                  - "auto": accept if residuals are close enough to tolerance
                  - "finite_difference": always reject and fall back to finite
                    differences

            floor_accept_multiplier: Tolerance multiplier for
                `fallback_mode="auto"`. Allows leniency for accepting slightly
                worse-than-tolerance fits if residuals are close.
                Typically 1.5–2.0.

        Returns:
         (bool, str): a tuple containing

            - A boolean signifying whether the polynomial fit should be accepted
              despite failing to meet tolerance.
            - Human-readable diagnostic tag describing the decision outcome.
              One of:
              - "not_at_floor"
              - "poly_at_floor"
              - "auto_accept_at_floor"
              - "auto_reject"
              - "auto_no_residuals"
              - "finite_difference"
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
            close_enough = (
                max_r < floor_accept_multiplier * fit_tolerance
            ) and (med_r < fit_tolerance)
            return (
                (True, "auto_accept_at_floor")
                if close_enough
                else (False, "auto_reject")
            )

        # default: finite_difference
        return False, "finite_difference"

    def _fallback_derivative(self, derivative_order, n_workers=1):
        """Compute the derivative using a finite difference method.

        Used when adaptive fitting fails.

        This method is only called as a last resort if the adaptive polynomial
        fitting procedure:
          - Cannot achieve the specified residual tolerance, or
          - Encounters numerical issues such as singular matrix errors.

        In such cases, it delegates the computation to
        :class:`derivkit.finite_difference.FiniteDifferenceDerivative`,
        which uses a high-order finite difference scheme to estimate the
        derivative at the same central point.

        Args:
            derivative_order (int):
                The order of the derivative to compute. Must be one of the
                orders supported by :class:`derivkit.finite_difference.FiniteDifferenceDerivative`
                (currently 1–4).
            n_workers (int, optional): Number of worker to use in
                multiprocessing. Default is 1 (no multiprocessing).

        Returns:
            :class:`np.ndarray`: The fallback derivative estimate(s) as a 1D
                array, with one value per output component of the target
                function.
        """
        warnings.warn(
            "Falling back to finite difference derivative.", RuntimeWarning
        )
        fd = FiniteDifferenceDerivative(
            function=self.function,
            central_value=self.central_value,
        )
        result = fd.compute(
            derivative_order=derivative_order, n_workers=n_workers
        )

        return np.atleast_1d(result)

    def recommend_fit_tolerance(self, dx=1e-3, verbose=True):
        """Suggests a fit_tolerance value based on local function variation.

        Args:
            dx (float, optional): Offset used to probe nearby values. Default
                value is `1e-3`.
            verbose (bool, optional): Whether to print out the suggested value.
                Default value is `True`.

        Returns:
            float: Recommended fit_tolerance.
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
                print(
                    f"[recommend_fit_tolerance] Δf = {variation:.3g}, x₀ = {scale:.3g} → Suggested tol = {tol}"
                )
            return tol

        except Exception as e:
            if verbose:
                print(
                    f"[recommend_fit_tolerance] Could not evaluate near x₀: {e}"
                )
            return 0.05

    def _compute_weights(self, x_vals):
        """Compute scale-aware inverse-distance weights around ``central_value``.

        This weighting emphasizes samples closest to the expansion point
        ``x0 = self.central_value`` while remaining numerically stable across
        *very small* and *very large* parameter scales.

        **Formulation**
        ----------------
        Let ``d_i = |x_i - x0|`` and ``D = max_i d_i`` (the current sampling span).
        We use
            ``eps = max(1e-3 * D, 1e-9)``,
        and define the weights
            ``w_i = 1 / (d_i + eps)``.

        - The additive softening ``eps`` prevents a singular weight at the
          center (where ``d_i = 0``).
        - Tying ``eps`` to the span ``D`` makes the weighting *scale-invariant*:
          the relative emphasis near the center is similar whether ``x0``
          is ~1e-4 or ~1e4.

        **Intuition**
        --------------
        With ``eps = 1e-3 * D``, the center-to-edge weight ratio is roughly:
            ``w(0) / w(D) ≈ (D + eps) / eps ≈ D / eps ≈ 10^3``.
        So points near ``x0`` can carry ~1000× more weight than the farthest
        points—strong, but not overwhelming. You can tune this by changing the
        1e-3 factor:
        - Larger factor (e.g., ``1e-2``) → milder emphasis (~100×).
        - Smaller factor (e.g., ``1e-4``) → sharper emphasis (~10,000×).

        **Why not a fixed epsilon?**
        -----------------------------
        A constant (e.g., ``1e-4``) behaves poorly across scales:
        - If the step span is tiny (<< 1e-4), the constant dominates and
          flattens the weights (little central emphasis).
        - If the span is large (>> 1e-4), the center weight becomes excessively
          dominant.

        By scaling ``eps`` with the current span, the weighting profile adapts
        automatically.

        Notes:
        -----
        - Complexity is O(n).
        - The absolute floor ``1e-9`` guards degenerate cases (e.g., ``D ≈ 0``)
          and prevents overflow even if multiple samples coincide with ``x0``.

        Args:
            x_vals (:class:`np.ndarray`): Sample locations used in fitting.
                Shape ``(n_points,)``.

        Returns:
            :class:`np.ndarray`: Weights of shape ``(n_points,)``. These are
                **not normalized**; downstream code (e.g.,
                ``np.polyfit(..., w=weights)``) uses them as relative weights.
                If a normalized weight vector is required, divide by
                ``weights.sum()``.
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
        """Removes points whose relative residual exceeds tolerance.

        The function refits after the residuals have been removed.

        Strategy
        --------
        - Sort points by residual (worst-first) and remove up to ``max_remove``
          per call.
        - Never drop the center sample (closest to ``central_value``) if
          ``keep_center`` is True.
        - If ``keep_symmetric`` is True, also remove the mirror of the removed
          point about ``central_value`` when possible (to keep sampling
          balanced).
        - Never reduce the number of points below ``required_points``.

        Args:
            x_vals: Current sample abscissae.
            y_vals: Current sample ordinates (must be the same length as
                ``x_vals``).
            residuals: Relative residuals for the *current* fit at ``x_vals``.
            fit_tolerance : Acceptable residual threshold (e.g., 0.05 for 5%).
            required_points: Minimum number of points allowed after pruning.
            max_remove: Maximum number of points to remove in this call (default 2).
            keep_center: If True, never remove the point closest to
                ``central_value``. Default is `True`.
            keep_symmetric: If True, attempt to remove a mirror point along with
                the worst point.

        Returns:
            (:class:`np.ndarray`, :class:`np.ndarray`, bool): A tuple containing

                - Pruned arrays x_vals and y_vals (may be unchanged).
                - A boolean which is `True` if at least one point was removed.
        """
        assert len(x_vals) == len(y_vals) == len(residuals)

        # Indices sorted by residual (descending): worst offenders first
        order = np.argsort(residuals)[::-1]

        # Identify the sample closest to the expansion point
        center_idx = (
            int(np.argmin(np.abs(x_vals - self.central_value)))
            if len(x_vals)
            else -1
        )

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
            if (
                keep_symmetric
                and removed < max_remove
                and keep.sum() - 1 >= required_points
            ):
                target = (
                    2.0 * self.central_value - x_vals[j]
                )  # mirror of x_j about x0
                k = int(np.argmin(np.abs(x_vals - target)))
                if k != j and (not keep_center or k != center_idx) and keep[k]:
                    keep[k] = False
                    removed += 1

        if removed == 0:
            return x_vals, y_vals, False

        return x_vals[keep], y_vals[keep], True

    def _store_diagnostics_entry(
        self, x_all, x_used, y_used, y_fit, residuals
    ):
        """Stores diagnostic info for a single component.

        Updates the diagnostics_data dictionary with the new entry.

        Args:
            x_all (:class:`np.ndarray`): All x values used in the fit.
            x_used (:class:`np.ndarray`): x values actually used in the fit
                (``None`` if no points were used).
            y_used (:class:`np.ndarray`): y values corresponding to ``x_used``
                (None if no points were used).
            y_fit (:class:`np.ndarray`): Fitted y values corresponding to
                `x_used` (None if no points were used).
            residuals (:class:`np.ndarray`): Residuals of the fit corresponding
                to `y_used` (None if no points were used).
        """
        if self.diagnostics_data is None:
            return

        if x_used is None:
            self.diagnostics_data["x_used"].append(None)
            self.diagnostics_data["y_used"].append(None)
            self.diagnostics_data["y_fit"].append(None)
            self.diagnostics_data["residuals"].append(None)
            self.diagnostics_data["used_mask"].append(
                np.zeros_like(x_all, dtype=bool)
            )
        else:
            mask = np.isclose(
                x_all[:, None], x_used, rtol=1e-12, atol=1e-15
            ).any(axis=1)
            self.diagnostics_data["x_used"].append(x_used.copy())
            self.diagnostics_data["y_used"].append(y_used.copy())
            self.diagnostics_data["y_fit"].append(y_fit.copy())
            self.diagnostics_data["residuals"].append(residuals.copy())
            self.diagnostics_data["used_mask"].append(mask)
