import os
from time import perf_counter
import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from derivkit.kit import DerivativeKit
from derivkit.plotutils.plot_helpers import PlotHelpers
from derivkit.plotutils.plot_style import (
    apply_plot_style, DEFAULT_LINEWIDTH, DEFAULT_COLORS, GRADIENT_COLORS
)

# Apply my style globally
apply_plot_style()

class PlotKit:
    """
    A high-level plotting utility for evaluating and comparing numerical derivative methods.

    This class provides multiple plotting routines to assess the accuracy and behavior of
    finite difference and adaptive polynomial fit derivative estimates under noisy conditions.

    Parameters
    ----------
    function : callable
        The target function for which derivatives will be evaluated.
    central_value : float
        The point at which the derivative is computed.
    derivative_order : int, optional
        The order of the derivative to compute (default is 1).
    fit_tolerance : float, optional
        Residual tolerance used in adaptive fitting (default is 0.05).
    plot_dir : str, optional
        Directory in which to save generated plots (default is "plots").
    linewidth : float, optional
        Line width to use for plots. Uses default style if not specified.
    colors : dict, optional
        A dictionary of color overrides keyed by method name (e.g., 'finite', 'adaptive').
    gradient_colors : dict, optional
        Optional color map or overrides for gradients or tolerance bands.
    true_derivative_fn : callable, optional
        Optional function to compute the true derivative for reference.

    Attributes
    ----------
    derivs : DerivativeKit
        Internal instance used to compute derivatives using both finite and adaptive methods.
    h : PlotHelpers
        Helper instance used for diagnostics, noise injection, and figure saving.
    seed : int
        Random seed for reproducibility in plots involving noise.
    """

    def __init__(
        self,
        function,
        central_value: float,
        derivative_order: int = 1,
        fit_tolerance=0.05,
        plot_dir: str = "plots",
        linewidth: Optional[float] = None,
        colors: Optional[dict] = None,
        gradient_colors: Optional[dict] = None,
        true_derivative_fn=None
    ):
        self.function = function
        self.central_value = central_value
        self.plot_dir = plot_dir
        self.derivative_order = derivative_order
        self.true_derivative_fn = true_derivative_fn
        self.fit_tolerance = fit_tolerance
        self.derivs = DerivativeKit(self.function,
                                    self.central_value)
        self.seed = 42

        os.makedirs(self.plot_dir, exist_ok=True)

        # style
        self.lw = DEFAULT_LINEWIDTH if linewidth is None else float(linewidth)
        self.colors = {**DEFAULT_COLORS, **(colors or {})}
        self.gradient_colors = {**GRADIENT_COLORS, **(gradient_colors or {})}

        self.h = PlotHelpers(function=self.function,
                             central_value=self.central_value,
                             true_derivative_fn=self.true_derivative_fn,
                             plot_dir=self.plot_dir)

    def color(self, key: str) -> str:
        """
        Retrieve the color assigned to a specific plot element key.

        Parameters
        ----------
        key : str
            Identifier for the color key (e.g., 'finite', 'adaptive', 'central', etc.).

        Returns
        -------
        str
            The hex code or named color string associated with the key.

        Notes
        -----
        - Falls back to the default color mapping if the key was not explicitly overridden.
        - Used internally for consistent styling across plots.
        """
        return self.colors[key]

    def plot_overlaid_histograms(self, derivative_order, noise_std=0.01, trials=100, bins=20, title=None, extra_info=None):
        """
        Plot overlaid histograms of derivative estimates from stencil and adaptive methods
        under repeated noisy evaluations.

        This method runs multiple trials of noisy function evaluations, computes the derivative
        using both the finite difference and adaptive fitting methods, and plots their empirical
        distributions overlaid as histograms.

        Parameters
        ----------
        derivative_order: int
            Order of the derivative to compute (e.g., 1 for first derivative).
        noise_std : float, optional
            Standard deviation of the Gaussian noise added to the function (default is 0.01).
        trials : int, optional
            Number of repeated noisy evaluations to perform (default is 100).
        bins : int, optional
            Number of bins to use in the histogram (default is 20).
        title : str, optional
            Optional title for the plot.
        extra_info : str, optional
            Additional string to append to the saved filename.

        Notes
        -----
        - Excludes outliers based on 1st and 99th percentiles to improve plot readability.
        - Displays method-specific variance in the legend for comparison.
        """

        plt.figure(figsize=(7.2, 5.2))
        plt.rcParams.update({
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'legend.fontsize': 15, 'axes.labelsize': 18,
            'xtick.labelsize': 15, 'ytick.labelsize': 15
        })

        # Run trials for each method
        stencil_vals, adaptive_vals, _ = self.h.get_noisy_derivatives(derivative_order, noise_std, trials)

        stencil_vals = np.array(stencil_vals)
        adaptive_vals = np.array(adaptive_vals)

        def clip_outliers(data, q_low=1, q_high=99):
            lower = np.percentile(data, q_low)
            upper = np.percentile(data, q_high)
            return data[(data >= lower) & (data <= upper)]

        stencil_vals = clip_outliers(stencil_vals)
        adaptive_vals = clip_outliers(adaptive_vals)

        # Plot all three histograms
        mu_s, var_s = np.mean(stencil_vals), np.var(stencil_vals, ddof=1)
        mu_a, var_a = np.mean(adaptive_vals), np.var(adaptive_vals, ddof=1)

        label_stencil = f"finite: $s^2$={var_s:.2f}"
        label_adaptive = f"adaptive: $s^2$={var_a:.2f}"

        labels = [label_stencil, label_adaptive]
        hists = [stencil_vals, adaptive_vals]
        # ghost scatter for legend
        plt.scatter([], [], c="white", label=f"$\\sigma={noise_std}$, trials={trials}",
                    edgecolor="none", s=0)  # invisible scatter for legend
        # Plot each histogram
        for vals, label in zip(hists, labels):
            plt.hist(vals, bins=bins, histtype="stepfilled", density=True, linewidth=self.lw,
                     color=self.color(label.split(':')[0].lower()), alpha=0.8,
                     label=label)

        plt.xlabel(rf"$\hat{{f}}^{({self.derivative_order})}(x_0)$", fontsize=15)
        plt.ylabel("density", fontsize=15)
        plt.legend(frameon=True, loc='upper left', framealpha=0.6)
        if title is not None:
            plt.title(title, fontsize=17)
        extra = f"_{extra_info}" if extra_info else ""
        self.h.save_fig(f"overlaid_histograms_with_trials_order{self.derivative_order}{extra}.pdf")
        self.h.save_fig(f"overlaid_histograms_with_trials_order{self.derivative_order}{extra}.png")
        plt.show()

    def adaptive_fit_demo(self, derivative_order=1, noise_std=0.01, width=0.2, title=None, extra_info=None):
        """
        Visualize the adaptive polynomial fit on a noisy function segment.

        This method creates a reproducible noisy realization of the input function
        over a local region around `central_value` and shows which points are used or excluded
        by the adaptive fitting method. It also overlays the fitted polynomial and its
        tolerance band.

        Parameters
        ----------
        derivative_order : int
            The order of the derivative to compute (e.g., 1 for first derivative).
            Defaults to 1.
        noise_std : float, optional
            Standard deviation of the Gaussian noise added to the function (default is 0.01).
        width : float, optional
            Width of the interval around `central_value` for generating the noisy segment (default is 0.2).
        title : str, optional
            Optional title for the plot.
        extra_info : str, optional
            Additional string to append to the saved filename.

        Notes
        -----
        - Uses a fixed random seed for reproducibility.
        - Highlights the internal logic of the adaptive derivative fitting mechanism visually.
        """

        noisy_func = self.h.make_noisy_interpolated_function(
            function=self.function,
            central_value=self.central_value,
            width=width,
            resolution=401,
            noise_std=noise_std,
            seed=self.seed
        )

        # Run the adaptive method using the updated API
        fitter = DerivativeKit(noisy_func, self.central_value)
        val, diagnostics = fitter.adaptive.compute(
            derivative_order=derivative_order,
            fit_tolerance=self.fit_tolerance,
            diagnostics=True
        )

        x_all = diagnostics["x_all"]
        y_all = diagnostics["y_all"].flatten()
        used_mask = diagnostics["used_mask"][0] if diagnostics["used_mask"] else np.zeros_like(x_all, dtype=bool)
        x_used = x_all[used_mask]
        y_used = y_all[used_mask]

        y_center = noisy_func(self.central_value)

        plt.figure(figsize=(7, 5))
        markersize = 100

        plt.scatter([self.central_value], [y_center], color=self.colors["excluded"],
                    s=markersize, label='central value', zorder=4)

        excluded_mask = (~used_mask) & (x_all != self.central_value)
        plt.scatter(x_all[excluded_mask], y_all[excluded_mask],
                    label='excluded from fit', color=self.colors["finite"],
                    s=markersize, zorder=3)

        included_mask = used_mask & (x_all != self.central_value)
        plt.scatter(x_all[included_mask], y_all[included_mask],
                    color=self.colors["adaptive"], label='used in fit',
                    s=markersize, zorder=3)

        if len(x_used) >= 2:
            slope, intercept, _ = self.h.adaptive_fit_with_outlier_removal(x_used, y_used, return_inliers=True)
            x_fit = np.linspace(min(x_all), max(x_all), 100)
            y_fit = slope * x_fit + intercept
            y_upper = y_fit + self.fit_tolerance
            y_lower = y_fit - self.fit_tolerance

            plt.fill_between(x_fit, y_lower, y_upper, color=self.colors["adaptive_lite"],
                             alpha=0.3, label=rf"tolerance band ($\pm${self.fit_tolerance * 100:.0f}%)", zorder=1)

            plt.plot(x_fit, y_fit, label='fit (noisy data)',
                     color=self.color("adaptive"), lw=3, zorder=2)

        plt.xlabel('evaluation point $x$', fontsize=17)
        plt.ylabel('$f(x) + \\epsilon$', fontsize=17)
        if title:
            plt.title(title, fontsize=17)
        plt.legend(frameon=True, fontsize=14, loc="best", framealpha=0.5)
        extra = f"_{extra_info}" if extra_info else ""
        self.h.save_fig(f"adaptive_fit_with_noise_order{derivative_order}{extra}.pdf")
        self.h.save_fig(f"adaptive_fit_with_noise_order{derivative_order}{extra}.png")
        plt.show()

    def plot_error_vs_noise(self, derivative_order, noise_levels, trials=50, title=None, extra_info=None):
        """
        Plot mean squared error (MSE) of derivative estimates vs. noise level.

        This method evaluates both finite difference and adaptive derivative estimators
        across a range of Gaussian noise standard deviations. It plots:
        - MSE vs. noise level (log scale)
        - Relative difference between adaptive and finite MSEs

        Parameters
        ----------
        derivative_order : int
            The order of the derivative to compute (e.g., 1 for first derivative).
        noise_levels : array-like
            List or array of noise standard deviation values to test.
        trials : int, optional
            Number of Monte Carlo trials per noise level (default is 50).
        title : str, optional
            Optional title for the plot.
        extra_info : str, optional
            Additional string to append to the saved filename.

        Notes
        -----
        - Finite difference uses a fixed step size relative to `central_value`.
        - Adaptive fit uses polynomial regression with tolerance-controlled filtering.
        - Useful for comparing robustness of methods under varying noise.
        """

        true_val = self.h.reference_derivative(self.central_value)
        rng = np.random.default_rng(42)  # master seed for this figure

        stencil_mse, adaptive_mse = [], []
        for sigma in noise_levels:
            st_errs, ad_errs = [], []
            for _ in range(trials):
                seed = int(rng.integers(0, 2 ** 31 - 1))
                noisy_f = self.h.make_shared_noisy_func(sigma, seed=seed)  # shared across methods

                kit = DerivativeKit(noisy_f,
                                    self.central_value)
                st_est = kit.finite.compute(derivative_order=derivative_order,
                                            stepsize=0.01 * (abs(self.central_value) or 1.0))
                ad_est = kit.adaptive.compute(derivative_order=derivative_order)

                st_errs.append((st_est - true_val) ** 2)
                ad_errs.append((ad_est - true_val) ** 2)

            stencil_mse.append(np.mean(st_errs))
            adaptive_mse.append(np.mean(ad_errs))

        # need to convert to float for plotting
        stencil_mse = np.array(stencil_mse, dtype=float)
        adaptive_mse = np.array(adaptive_mse, dtype=float)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(7, 5), sharex=True,
            gridspec_kw={'height_ratios': [2.0, 1.0], 'hspace': 0.0}, constrained_layout=True
        )

        ax1.plot(noise_levels, stencil_mse,
                 label='finite', marker='o', color=self.colors['finite'],
                 ms=10)
        ax1.plot(noise_levels, adaptive_mse,
                 label='adaptive', marker='o', color=self.colors['adaptive'], ms=10,)
        ax1.set_ylabel("MSE", fontsize=15)
        ax1.set_yscale("log")
        ax1.legend(loc="lower right", frameon=True, fontsize=15)

        eps = 1e-300  # setting a small epsilon to avoid division by zero
        ratio_minus_1 = adaptive_mse / np.maximum(stencil_mse, eps) - 1.0

        ax2.plot(noise_levels, ratio_minus_1, marker='o', linestyle='-', color=self.colors['central'], ms=10)
        ax2.axhline(0.0, linestyle='--', color=self.colors['excluded'], linewidth=1.2)
        ylabel = r"$\frac{\mathrm{MSE}^\mathrm{A}}{\mathrm{MSE}^\mathrm{F}} - 1$"
        ax2.set_ylabel(ylabel, fontsize=15)
        ax2.set_xlabel("noise standard deviation", fontsize=15)
        ax2.set_ylim(-1.2, 0.3)  # set limits to show the ratio clearly
        for ax in (ax1, ax2):
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        if title is not None:
            fig.suptitle(title, fontsize=17, y=0.95)
        extra = f"_{extra_info}" if extra_info else ""
        self.h.save_fig(f"error_vs_noise_order{self.derivative_order}{extra}.pdf")
        self.h.save_fig(f"error_vs_noise_order{self.derivative_order}{extra}.png")
        plt.show()

    def plot_ecdf_errors(self, noise_std=0.01, trials=200, title=None, extra_info=None):
        """
        Plot empirical cumulative distribution functions (ECDFs) of squared errors.

        This method compares the distribution of squared errors from finite difference and
        adaptive derivative estimators by plotting their ECDFs under repeated noisy evaluations.

        Parameters
        ----------
        noise_std : float, optional
            Standard deviation of the Gaussian noise added to the function (default is 0.01).
        trials : int, optional
            Number of Monte Carlo trials to generate error samples (default is 200).
        title : str, optional
            Optional title for the plot.
        extra_info : str, optional
            Additional string to append to the saved filename.

        Notes
        -----
        - ECDF provides a full view of the error distribution, not just summary statistics.
        - The reference derivative value is used to compute true squared error per trial.
        - Useful for visualizing which method tends to produce smaller errors more often.
        """

        true_val = self.h.reference_derivative(self.central_value)
        rng = np.random.default_rng(123)

        fin_err, ad_err = [], []
        for _ in range(trials):
            seed = int(rng.integers(0, 2 ** 31 - 1))
            func = self.h.make_shared_noisy_func(noise_std, seed=seed)
            kit = DerivativeKit(func, self.central_value)
            f_fin = kit.finite.compute(derivative_order=self.derivative_order,
                                       stepsize=0.01 * (abs(self.central_value) or 1.0))
            f_ad = kit.adaptive.compute(derivative_order=self.derivative_order)
            fin_err.append((f_fin - true_val) ** 2)
            ad_err.append((f_ad - true_val) ** 2)

        def ecdf(a):
            x = np.sort(a)
            y = np.arange(1, len(a) + 1) / len(a)
            return x, y

        x1, y1 = ecdf(fin_err)
        x2, y2 = ecdf(ad_err)
        plt.figure(figsize=(7, 5))
        ms = 10
        plt.plot(x1, y1, "o-", label="finite", color=self.color("finite"), lw=self.lw, markersize=ms)
        plt.plot(x2, y2, "o-", label="adaptive", color=self.color("adaptive"), lw=self.lw, markersize=ms)

        plt.xlabel(r"$t$ (squared error)", fontsize=15)
        plt.ylabel(f"$\\widehat{{F}}_{{e^2}} (t)$", fontsize=15)
        if title is not None:
            plt.title(title, fontsize=17)
        plt.legend(frameon=True, fontsize=15, loc="best", framealpha=0.5)
        extra = f"_{extra_info}" if extra_info else ""
        self.h.save_fig(f"ecdf_errors_order{self.derivative_order}{extra}.png")
        self.h.save_fig(f"ecdf_errors_order{self.derivative_order}{extra}.pdf")
        plt.show()

    def plot_paired_error_differences(self, noise_std=0.01, trials=200, title=None, extra_info=None):
        """
        Plot paired squared error differences between adaptive and finite methods.

        This method runs repeated noisy derivative trials and computes the squared error
        for both adaptive and finite difference methods. It then plots the pairwise
        differences: Δ = error²_adaptive − error²_finite.

        Points are jittered horizontally to show density and grouped into:
        - Adaptive better (Δ < 0)
        - Finite better (Δ > 0)
        - Tie (Δ = 0)

        Parameters
        ----------
        noise_std : float, optional
            Standard deviation of the Gaussian noise added to the function (default is 0.01).
        trials : int, optional
            Number of paired Monte Carlo trials to run (default is 200).
        title : str, optional
            Optional title for the plot.
        extra_info : str, optional
            Additional string to append to the saved filename.

        Notes
        -----
        - A horizontal line at Δ = 0 indicates equal performance.
        - The estimated win rate (P(Δ < 0)) is shown in the legend.
        - This visualization provides intuitive insight into method-wise reliability.
        """

        true_val = self.h.reference_derivative(self.central_value)
        rng = np.random.default_rng(123)

        diffs = []
        for _ in range(trials):
            seed = int(rng.integers(0, 2 ** 31 - 1))
            func = self.h.make_shared_noisy_func(noise_std, seed=seed)
            kit = DerivativeKit(func, self.central_value)
            e_f = kit.finite.compute(derivative_order=self.derivative_order,
                                     stepsize=0.01 * (abs(self.central_value) or 1.0)) - true_val
            e_a = kit.adaptive.compute(derivative_order=self.derivative_order) - true_val
            diffs.append(e_a ** 2 - e_f ** 2)  # Δ (adaptive − finite)

        diffs = np.asarray(diffs, dtype=float)
        m_ad = diffs < 0  # adaptive wins (below 0)
        m_fin = diffs > 0  # finite wins (above 0)
        m_tie = diffs == 0  # ties (rare)

        # Win rate; count ties as half-wins (optional)
        n_ad = np.count_nonzero(m_ad)
        n_fin = np.count_nonzero(m_fin)
        n_tie = np.count_nonzero(m_tie)
        p = (n_ad + 0.5 * n_tie) / len(diffs)

        # jitter for display (reproducible)
        x = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(diffs))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.axhline(0, color=self.color("central"), lw=2, ls="-",
                   label="equal‑error $(\\Delta = 0)$", zorder=0)

        # losers first so winners draw on top
        markersize=100
        ax.scatter(x[m_fin], diffs[m_fin], s=markersize, alpha=0.9,
                   color=self.color("finite"),
                   label=f"finite better $(n = {n_fin})$", zorder=3)
        ax.scatter(x[m_ad], diffs[m_ad], s=markersize, alpha=0.9,
                   color=self.color("adaptive"),
                   label=f"adaptive better $(n = {n_ad})$", zorder=3)
        if np.any(m_tie):
            ax.scatter(x[m_tie], diffs[m_tie], s=markersize, alpha=0.9,
                       color=self.color("central"),
                       label=f"tie $(n = {n_tie})$", zorder=3)

        ax.scatter([], [], c="white", label=f"$\\widehat P(\\Delta<0)={p:.2f}$")
        ax.set_xlabel("paired trials (jittered $x$)", fontsize=17)
        ax.set_ylabel(f"$e^2_\\mathrm{{A}} - e^2_\\mathrm{{F}}$", fontsize=17)
        ax.set_xticks([])
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if title is not None:
            plt.title(title, fontsize=17)
        ax.legend(frameon=True, loc="lower left", fontsize=15)
        extra = f"_{extra_info}" if extra_info else ""
        self.h.save_fig(f"paired_differences_order{self.derivative_order}{extra}.pdf")
        self.h.save_fig(f"paired_differences_order{self.derivative_order}{extra}.png")
        plt.show()


# we also need to add an extra function to check different derivative orders
def plot_multi_order_error_vs_noise(
    function,
    central_value,
    snr_values=np.logspace(1, 5, 20),
    orders=(1, 2, 3),
    trials=50,
    fit_tolerance=0.05,
    clip_threshold=1e6,
    title=None,
    extra_info=None,
    plot_dir="plots"
):
    """
    Plot mean squared error (MSE) versus inverse signal-to-noise ratio (1/SNR)
    for multiple derivative orders and estimation methods.

    The function injects Gaussian noise into the target function, scaled inversely
    with SNR and proportionally to |f(x)|. It compares the performance of finite
    difference and adaptive polynomial fitting methods.

    Parameters
    ----------
    function : callable
        Target function to differentiate.
    central_value : float
        Point at which the derivative is computed.
    snr_values : array-like, optional
        List or array of SNR values (default is log-spaced from 10 to 1e5).
    orders : tuple of int, optional
        Derivative orders to evaluate (default: (1, 2, 3)).
    trials : int, optional
        Number of Monte Carlo trials per SNR and order (default: 50).
    fit_tolerance : float or dict, optional
        Residual tolerance used in adaptive fitting. Can be a float or a dict keyed by order.
    clip_threshold : float, optional
        Maximum allowed squared error; larger values are clipped to prevent skew (default: 1e6).
    title : str, optional
        Title for the plot.
    extra_info : str, optional
        Additional string to append to the saved filename.
    plot_dir : str, optional
        Directory in which to save generated plots (default is "plots").

    Notes
    -----
    - Uses a fixed random seed for reproducibility.
    - Derivatives are computed using both finite differences and adaptive polynomial fitting.
    - Noise is scaled as: `sigma = max(|f(x)|, 1e-3) / SNR` to avoid division by very small amplitudes.
    - Plots MSE vs. 1/SNR on log-log axes for interpretability.
    """
    def make_snr_scaled_noisy_func(f, snr, seed):
        rng = np.random.default_rng(seed)
        def noisy(x):
            fx = f(x)
            scale = max(abs(fx), 1e-3)  # floor to avoid tiny amplitudes
            sigma = scale / snr
            return fx + rng.normal(loc=0.0, scale=sigma)
        return noisy

    finite_results = {}
    adaptive_results = {}

    for order in orders:
        tol = fit_tolerance[str(order)] if isinstance(fit_tolerance, dict) else fit_tolerance

        plotter = PlotKit(
            function=function,
            central_value=central_value,
            derivative_order=order,
            fit_tolerance=tol,
            plot_dir=plot_dir,
        )
        true_val = plotter.h.reference_derivative(central_value)
        rng = np.random.default_rng(42)

        stencil_mse, adaptive_mse = [], []

        for snr in snr_values:
            st_errs, ad_errs = [], []
            for _ in range(trials):
                seed = int(rng.integers(0, 2**31 - 1))
                noisy_f = make_snr_scaled_noisy_func(function, snr, seed)
                kit = DerivativeKit(noisy_f, central_value)

                st_est = kit.finite.compute(derivative_order=order,
                                            stepsize=0.01 * (abs(central_value) or 1.0))
                ad_est = kit.adaptive.compute(derivative_order=order)

                if np.isfinite(st_est):
                    err = (st_est - true_val) ** 2
                    if err < clip_threshold:
                        st_errs.append(err)

                if np.isfinite(ad_est):
                    err = (ad_est - true_val) ** 2
                    if err < clip_threshold:
                        ad_errs.append(err)

            stencil_mse.append(np.mean(st_errs) if st_errs else np.nan)
            adaptive_mse.append(np.mean(ad_errs) if ad_errs else np.nan)

        finite_results[order] = stencil_mse
        adaptive_results[order] = adaptive_mse

    # --- Plot all orders in one figure ---
    plt.figure(figsize=(8, 6))
    inv_snr = 1 / np.array(snr_values)
    colors = GRADIENT_COLORS
    ms = 10

    # Plot finite and adaptive for each order
    for order in orders:
        label_deriv = f"$\\mathrm{{d}}^{{{order}}}f/\\mathrm{{d}}x^{{{order}}}$"

        plt.plot(inv_snr, finite_results[order],
                 linestyle="-", marker="o", ms=ms,
                 label=f"finite {label_deriv}", color=colors[f"finite_{order}"])

        plt.plot(inv_snr, adaptive_results[order],
                 linestyle="-", marker="o", ms=ms,
                 label=f"adaptive {label_deriv}", color=colors[f"adaptive_{order}"])

    plt.xlabel("1 / SNR", fontsize=15)
    plt.ylabel("mean squared error (MSE)", fontsize=15)
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(frameon=True, fontsize=15)
    if title is not None:
        plt.title(title, fontsize=17)
    plt.grid(False)
    plt.tight_layout()

    extra = f"_{extra_info}" if extra_info else ""
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"multi_order_error_vs_snr{extra}.pdf"), dpi=300)
    plt.savefig(os.path.join(plot_dir, f"multi_order_error_vs_snr{extra}.png"), dpi=300)
    plt.show()


def slow_function(x, sleep_time):
    """
    Simulates a slow, compute-intensive mathematical function.

    This function applies a non-linear transformation to the input `x` involving
    sine, exponential decay, and logarithmic growth, repeated 100 times to mimic
    computational load. Additionally, it pauses execution for a specified duration
    using `time.sleep`.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s) to be processed.
    sleep_time : float
        Time in seconds to pause execution before starting computation.

    Returns
    -------
    float or np.ndarray
        Transformed value(s) after applying the repeated operation.
    """

    time.sleep(sleep_time)
    for _ in range(100):
        x = np.sin(x**2) * np.exp(-x**2) + np.log(1 + x**2)
    return x


def make_slow_func(sleep_time):
    """
    Creates a lambda function that wraps `slow_function` with a fixed sleep time.

    Parameters
    ----------
    sleep_time : float
        Time in seconds that the wrapped function will sleep before computation.

    Returns
    -------
    function
        A lambda function that takes `x` as input and calls `slow_function(x, sleep_time)`.
    """

    return lambda x: slow_function(x, sleep_time)


def plot_timing_for_order(order, sleep_times, timings_adaptive, timings_finite):
    """
    Plot timing benchmarks comparing adaptive and finite difference methods
    for a given derivative order as a function of simulated function evaluation time.

    This function assumes access to global dictionaries `timings_adaptive` and
    `timings_finite`, which store timing results for various derivative orders,
    as well as `sleep_times`, a list of simulated function evaluation times.
    It fits a linear model to each method's timing data, computes the slowdown
    factor of the adaptive method relative to finite differences, and visualizes
    the timing comparison.

    Parameters
    ----------
    order : int
        The derivative order for which to generate the timing comparison plot.
    sleep_times : list of float
        List of simulated function evaluation times (in seconds) used for the x-axis.
    timings_adaptive : dict
        Dictionary containing timing results for the adaptive method, keyed by derivative order.
    timings_finite : dict
        Dictionary containing timing results for the finite difference method, keyed by derivative order.

    Returns
    -------
    None
        Displays a matplotlib plot and prints the slowdown factor in the legend.
    """
    x = np.array(sleep_times)
    y_adaptive = np.array(timings_adaptive[order])
    y_finite   = np.array(timings_finite[order])

    # Fit slopes and compute slowdown ratio
    a_ad, _ = np.polyfit(x, y_adaptive, 1)
    a_fin, _ = np.polyfit(x, y_finite, 1)
    slowdown = a_ad / a_fin
    note = f"adaptive ≈ {slowdown:.1f}× slower"

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(x, y_adaptive, 'o-', label="Adaptive Method",
             color=GRADIENT_COLORS[f"adaptive_{order}"],
             markersize=10, lw=DEFAULT_LINEWIDTH)

    plt.plot(x, y_finite, 'o-', label="Finite Difference",
             color=GRADIENT_COLORS[f"finite_{order}"],
             markersize=10, lw=DEFAULT_LINEWIDTH)

    plt.plot([], [], ' ', label=note)

    plt.xlabel("Simulated Function Evaluation Time [s]", fontsize=14)
    plt.ylabel("Total Derivative Evaluation Time [s]", fontsize=14)
    plt.title(f"Timing Benchmark (Derivative Order {order})", fontsize=16)
    plt.legend(fontsize=13, frameon=True)
    plt.xscale("log")
    plt.tight_layout()
    plt.show()


def benchmark_derivative_timing_vs_order(function,
                                         central_value,
                                         orders,
                                         stencil_points=5,
                                         stencil_stepsize=0.01):
    """
    Benchmark and plot the evaluation time of adaptive vs. finite difference
    derivative estimation methods across multiple derivative orders.

    This function measures and compares the runtime of the `DerivativeKit`'s
    adaptive and finite difference methods for computing derivatives of the
    provided function at a given point, over a range of derivative orders.
    It generates a plot showing how the evaluation time scales with derivative order
    and the relative slowdown of adaptive vs. finite difference.

    Parameters
    ----------
    function : callable
        The function to differentiate.
    central_value : float
        The point at which the derivative is evaluated.
    orders : list of int
        The derivative orders to benchmark.
    stencil_points: int, optional
        Number of stencil points to use for the finite difference method (default is 5).
    stencil_stepsize: float, optional
        Step size for the finite difference stencil (default is 0.01).

    Returns
    -------
    None
        Displays a matplotlib plot comparing timing for each method and their ratio.
    """
    adaptive_times = []
    finite_times = []

    for order in orders:
        kit = DerivativeKit(function, central_value)

        # Time adaptive method
        start = perf_counter()
        _ = kit.adaptive.compute(derivative_order=order)
        adaptive_times.append(perf_counter() - start)

        # Time finite difference method
        start = perf_counter()
        _ = kit.finite.compute(derivative_order=order, stepsize=stencil_stepsize, num_points=stencil_points)
        finite_times.append(perf_counter() - start)

    # Compute ratio
    adaptive_times = np.array(adaptive_times)
    finite_times = np.array(finite_times)
    ratio = adaptive_times / finite_times - 1.0

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1})

    # Top panel: absolute times
    ax1.plot(orders, adaptive_times, 'o-', label="adaptive",
             color=DEFAULT_COLORS["adaptive"], lw=DEFAULT_LINEWIDTH, markersize=10)
    ax1.plot(orders, finite_times, 'o-', label="finite",
             color=DEFAULT_COLORS["finite"], lw=DEFAULT_LINEWIDTH, markersize=10)
    ax1.set_ylabel("time [s]", fontsize=14)

    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_powerlimits((0, 0))
    ax1.yaxis.set_major_formatter(sci_formatter)

    ax1.legend(fontsize=12)
    ax1.tick_params(labelsize=12)

    # Bottom panel: slowdown ratio
    ax2.plot(orders, ratio, 'o-', color=DEFAULT_COLORS["central"], lw=DEFAULT_LINEWIDTH, markersize=10)
    ax2.axhline(0.0, color=DEFAULT_COLORS["excluded"], linestyle="--", lw=1)
    ax2.set_ylabel("relative slowdown", fontsize=14)
    ax2.set_xlabel("derivative order", fontsize=14)
    ax2.tick_params(labelsize=12)

    ax1.minorticks_off()
    ax2.minorticks_off()

    plt.tight_layout()
    plt.show()


def plot_adaptive_timing_sweeps(
    sleep_time=0.01,
    central_value=0.0005,
    deriv_orders=(1, 2, 3),
    fit_tolerance_fixed=0.01,
    min_used_points_fixed=11,
    min_pts_list=None,
    fit_tolerances=None,
    save_fig=True,
    output_filename=None,
    n_workers=1,
):
    """
    Benchmark and plot timing results for adaptive derivative estimation as a function of:
    (1) `min_used_points` with fixed `fit_tolerance`
    (2) `fit_tolerance` with fixed `min_used_points`

    Parameters
    ----------
    sleep_time : float
        Artificial delay in function evaluation.
    central_value : float
        Point at which to evaluate the derivative.
    deriv_orders : tuple of int
        Derivative orders to benchmark and plot.
    fit_tolerance_fixed : float
        The fixed fit tolerance used when sweeping `min_used_points`.
    min_used_points_fixed : int
        The fixed number of points used when sweeping `fit_tolerance`.
    min_pts_list : array-like, optional
        Values of `min_used_points` to sweep. Default: np.arange(5, 30, 2).
    fit_tolerances : array-like, optional
        Values of `fit_tolerance` to sweep. Default: np.logspace(-4, log10(0.2), 14).
    save_fig : bool
        If True, saves the figure to 'plots/adaptive_timing_sweeps.{png,pdf}'.
    output_filename : str or None
        If provided, also saves the benchmark results to a `.npz` file.
    n_workers: int, optional
            Number of worker to use in multiprocessing. Default is 1 (no multiprocessing).

    Returns
    -------
    None
        Displays the plot. Optionally saves results and/or figure.
    """
    # === Run benchmark
    if min_pts_list is None:
        min_pts_list = np.arange(5, 30, 2)
    if fit_tolerances is None:
        fit_tolerances = np.logspace(-4, np.log10(0.2), 14)

    timings_minpts = {order: [] for order in deriv_orders}
    timings_tol = {order: [] for order in deriv_orders}

    for order in deriv_orders:
        for min_pts in min_pts_list:
            f = make_slow_func(sleep_time)
            kit = DerivativeKit(f, central_value)
            kit.adaptive.fit_tolerance = fit_tolerance_fixed
            kit.adaptive.min_used_points = min_pts
            start = perf_counter()
            _ = kit.adaptive.compute(derivative_order=order, 
                                     n_workers=n_workers)
            timings_minpts[order].append(perf_counter() - start)

    for order in deriv_orders:
        for tol in fit_tolerances:
            f = make_slow_func(sleep_time)
            kit = DerivativeKit(f, central_value)
            kit.adaptive.fit_tolerance = tol
            kit.adaptive.min_used_points = min_used_points_fixed
            start = perf_counter()
            _ = kit.adaptive.compute(derivative_order=order, 
                                     n_workers=n_workers)
            
            timings_tol[order].append(perf_counter() - start)
    
            
    if output_filename:
        np.savez(
            output_filename,
            timings_minpts=timings_minpts,
            timings_tol=timings_tol,
            min_pts_list=min_pts_list,
            fit_tolerances=fit_tolerances
        )

    # === Plotting
    gradient_colors = GRADIENT_COLORS
    markers = {order: "o" for order in deriv_orders}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    title = 'Multiprocessing '
    if n_workers > 1:
        title += f'on, #workers={n_workers}'
    else:
        title += 'off'
    fig.suptitle(title, fontsize=20)
    marker_size=10
    font_size=20

    for order in deriv_orders:
        axes[0].plot(
            min_pts_list,
            timings_minpts[order],
            markers[order] + "-",
            label=f"order = {order}",
            color=gradient_colors[f"adaptive_{order}"],
            markersize=marker_size
        )
    axes[0].set_xlabel("min_used_points", fontsize=font_size)
    axes[0].set_ylabel("timing [s]", fontsize=font_size)
    axes[0].legend(title=f"tol = {fit_tolerance_fixed}", title_fontsize=font_size, fontsize=font_size)

    for order in deriv_orders:
        axes[1].plot(
            fit_tolerances,
            timings_tol[order],
            markers[order] + "-",
            label=f"order = {order}",
            color=gradient_colors[f"adaptive_{order}"],
            markersize=marker_size
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("fit_tolerance", fontsize=font_size)
    axes[1].legend(title=f"min pts = {min_used_points_fixed}", title_fontsize=font_size, fontsize=font_size)

    axes[0].minorticks_off()
    axes[1].minorticks_off()

    axes[0].tick_params(labelsize=16)
    axes[1].tick_params(labelsize=16)

    plt.tight_layout()
    if save_fig:
        fig.savefig("plots/adaptive_timing_sweeps.pdf", dpi=300)
        fig.savefig("plots/adaptive_timing_sweeps.png", dpi=300)
    plt.show()


def plot_function_with_residuals(function, central_value, derivative_order=1, dx=0.5,
                                 title="Function and Tangents + Residuals + Fractional Diff",
                                 show_ratio_panel=True, save_fig=True,
                                 function_name=None):
    """
    Visualize a function and its tangent approximations with residual diagnostics.

    This function plots the input function alongside its finite difference and adaptive
    tangent approximations at a given point. It includes diagnostics such as residuals,
    fractional residuals, and optionally the ratio of fractional residuals (adaptive / finite),
    helping to assess the accuracy and behavior of each method.

    Parameters
    ----------
    function : callable
        The target function to differentiate.
    central_value : float
        The point x₀ at which the derivative is computed and tangents are anchored.
    derivative_order : int, optional
        The order of the derivative to estimate (default is 1).
    dx : float, optional
        Half-width of the plotting interval around `central_value` (default is 0.5).
    title : str, optional
        Title to display at the top of the figure (default is descriptive).
    show_ratio_panel : bool, optional
        If True, adds a fourth panel showing the ratio of fractional residuals (default is True).
    save_fig : bool, optional
        If True, saves the resulting plot to 'plots/' directory as PDF and PNG (default is True).
    function_name : str, optional
        Name of the function used in saved filenames (default is None).

    Notes
    -----
    - Uses `DerivativeKit` to estimate derivatives via both adaptive fitting and finite differences.
    - Residuals are defined as: f(x) − T(x), where T(x) is the tangent.
    - Fractional residuals are normalized by the function value.
    - The ratio panel shows (adaptive / finite) − 1 to highlight relative performance.
    """
    x0 = central_value
    x = np.linspace(x0 - dx, x0 + dx, 400)
    y = function(x)
    f0 = function(x0)

    kit = DerivativeKit(function, central_value=x0)
    slope_adaptive = kit.adaptive.compute(derivative_order=derivative_order)
    slope_finite = kit.finite.compute(derivative_order=derivative_order)

    tangent_adapt = slope_adaptive * (x - x0) + f0
    tangent_finite = slope_finite * (x - x0) + f0

    resid_adapt = y - tangent_adapt
    resid_finite = y - tangent_finite

    frac_adapt = resid_adapt / y
    frac_finite = resid_finite / y

    # Avoid divide-by-zero
    ratio = np.divide(frac_adapt, frac_finite, out=np.full_like(frac_adapt, np.nan), where=frac_finite != 0)

    colors = DEFAULT_COLORS
    lw = 3
    fs = 15

    if show_ratio_panel:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7, 9), sharex=True,
                                                 gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 7), sharex=True,
                                            gridspec_kw={'height_ratios': [2, 1, 1]})

    # 1. Function and tangents
    ax1.plot(x, y, label='function', color=colors["excluded"], lw=lw)
    ax1.plot(x, tangent_adapt, lw=lw, label='adaptive tangent', color=colors["adaptive"])
    ax1.plot(x, tangent_finite, ':', lw=lw, label='finite tangent', color=colors["finite"])
    ax1.set_ylabel("$f(x)$", fontsize=fs)
    ax1.set_title(title, fontsize=fs)
    ax1.legend(fontsize=fs)

    # 2. Residuals
    ax2.axhline(0, ls='--', color=colors["central"], lw=lw)
    ax2.plot(x, resid_adapt, label='adaptive', color=colors["adaptive"], lw=lw)
    ax2.plot(x, resid_finite, label='finite', color=colors["finite"], lw=lw, ls=":")
    ax2.set_ylabel(r"$f(x) - T(x)$", fontsize=fs)
    ax2.legend(fontsize=fs-2)

    # 3. Fractional residuals
    ax3.axhline(0, ls='--', color=colors["central"], lw=lw)
    ax3.plot(x, frac_adapt, label='adaptive', color=colors["adaptive"], lw=lw)
    ax3.plot(x, frac_finite, label='finite', color=colors["finite"], lw=lw, ls=":")
    ax3.set_ylabel(r"$\frac{f(x) - T(x)}{f(x)}$", fontsize=fs)
    if not show_ratio_panel:
        ax3.set_xlabel("evaluation point $x$", fontsize=fs)
    ax3.legend(fontsize=fs-2)

    # 4. Ratio panel
    if show_ratio_panel:
        ax4.axhline(0, ls='--', color=colors["central"], lw=lw)
        ax4.plot(x, ratio-1, color="lightgray", lw=lw, label=r"adaptive / finite")
        ax4.set_ylabel(r"frac. ratio", fontsize=fs)
        ax4.set_xlabel("evaluation point $x$", fontsize=fs)
        ax4.legend(fontsize=fs-2)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.03)
    if function_name is not None:
       fname = f"function_with_residuals_{function_name}_order{derivative_order}"
    if save_fig:
        fig.savefig(f"plots/{fname}.pdf", dpi=300)
        fig.savefig(f"plots/{fname}.png", dpi=300)
    else:
        plt.tight_layout()
    plt.show()
