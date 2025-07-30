import os
import warnings

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np

from .kit import DerivativeKit


class DerivativePlotter:
    def __init__(self, function, x_center, derivative_order=1, plot_dir="plots"):
        self.function = function
        self.x_center = x_center
        self.plot_dir = plot_dir
        self.derivative_order = derivative_order
        os.makedirs(self.plot_dir, exist_ok=True)
        self.derivs = DerivativeKit(function, x_center)

    def _nd_derivative(self, x):
        return nd.Derivative(self.function)(x)

    def _save_fig(self, filename):
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/{filename}", dpi=300)

    def plot_histograms(self, stencil_vals, adaptive_vals, bins=20):
        plt.figure(figsize=(20, 10))
        lw, alpha = 2.5, 1
        plt.rcParams.update({
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'legend.fontsize': 20, 'axes.labelsize': 20,
            'xtick.labelsize': 15, 'ytick.labelsize': 15
        })

        nd_derivative = self._nd_derivative(self.x_center)

        for i, (vals, label, color) in enumerate([
            (stencil_vals, 'Stencil', 'hotpink'),
            (adaptive_vals, 'Adaptive Fit', 'yellowgreen')
        ]):
            plt.subplot(1, 2, i + 1)
            plt.hist(vals, bins=bins, color=color, alpha=alpha)
            plt.axvline(np.mean(vals), color='k', linestyle=':', lw=lw, label=f'Mean: {np.mean(vals):.2f}')
            plt.axvline(np.median(vals), color='gray', linestyle='--', lw=lw, label=f'Median: {np.median(vals):.2f}')
            plt.plot([], [], ' ', label=f"ND Derivative: {nd_derivative:.2f}")
            plt.plot([], [], ' ', label=f"Variance: {np.var(vals):.2f}")
            plt.title(f"{label} Derivative Distribution", fontsize=25)
            plt.xlabel("Derivative")
            plt.ylabel("Frequency")
            plt.legend(frameon=False, loc='upper left')

        self._save_fig("derivative_histograms.png")
        plt.show()

    def plot_overlaid_histograms(self, noise_std=0.01, trials=100, bins=20):
        """
        Runs noisy derivative trials for stencil, adaptive, and Numdifftools methods
        and plots an overlaid histogram comparing their distributions.
        """
        lw = 2.5
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({
            'xtick.direction': 'in', 'ytick.direction': 'in',
            'legend.fontsize': 15, 'axes.labelsize': 18,
            'xtick.labelsize': 14, 'ytick.labelsize': 14
        })

        # Run trials for each method
        stencil_vals = self.run_derivative_trials_with_noise(method="finite", noise_std=noise_std, trials=trials)
        adaptive_vals = self.run_derivative_trials_with_noise(method="adaptive", noise_std=noise_std, trials=trials)
        nd_vals = self.run_derivative_trials_with_noise(method="numdifftools", noise_std=noise_std, trials=trials)

        # Plot all three histograms
        plt.hist(stencil_vals, bins=bins, histtype='step', color='hotpink', label='Stencil', linewidth=lw)
        plt.hist(adaptive_vals, bins=bins, histtype='step', color='yellowgreen', label='Adaptive Fit', linewidth=lw)
        plt.hist(nd_vals, bins=bins, histtype='step', color='lightskyblue', label='Numdifftools', linewidth=lw)

        # Labels and styling
        plt.xlabel("Derivative")
        plt.ylabel("Frequency")
        plt.title("Overlayed Derivative Histograms (Noisy Trials)", fontsize=20)
        plt.legend(frameon=False, loc='upper right')

        self._save_fig("overlaid_histograms_with_trials.png")
        plt.show()

    def demonstrate_adaptive_fit(self, num_points=20, noise_std=0.01):
        """
        Visualizes the adaptive fit process using a noisy version of the function.

        - Evaluates f(x) + ε on a symmetric set of x values.
        - Runs the adaptive fitting routine with the noisy function.
        - Highlights which noisy points were used or excluded from the polynomial fit.

        Parameters
        ----------
        num_points : int
            Number of total x points sampled around x_center.
        noise_std : float
            Standard deviation of Gaussian noise added to f(x).
        """
        # 1. Generate symmetric x-values around the central point
        x_vals = np.linspace(self.x_center - 0.1, self.x_center + 0.1, num_points)

        # 2. Define a noisy version of the function for this plot
        def noisy_func(x):
            return self.function(x) + np.random.normal(0, noise_std)

        # 3. Temporarily override the function in the adaptive fitter
        original_func = self.derivs.adaptive.function
        self.derivs.adaptive.function = noisy_func
        _, diagnostics = self.derivs.adaptive.compute(
            diagnostics=True
        )
        self.derivs.adaptive.function = original_func  # Restore clean function

        # 4. Extract diagnostic info
        x_all = diagnostics["x_all"]
        y_all = diagnostics["y_all"].flatten()
        used_mask = diagnostics["used_mask"][0] if diagnostics["used_mask"] else np.zeros_like(x_all, dtype=bool)
        x_used = x_all[used_mask]
        y_used = y_all[used_mask]
        x_excluded = x_all[~used_mask]
        y_excluded = y_all[~used_mask]

        # 5. Plot results
        plt.figure(figsize=(7, 5))

        # Excluded points
        plt.scatter(x_excluded, y_excluded, color='black', label='Excluded from fit', s=50, zorder=2)

        # Used points (in fit)
        if len(x_used) >= 2:
            plt.scatter(x_used, y_used, color='hotpink', edgecolor='black', label='Used in fit', s=60, zorder=3)

            # Fit a line to the used noisy points
            slope, intercept, _ = self._adaptive_fit_with_outlier_removal(
                x_used, y_used, return_inliers=True
            )
            x_fit = np.linspace(min(x_all), max(x_all), 100)
            y_fit = slope * x_fit + intercept
            plt.plot(x_fit, y_fit, label='Fit (noisy data)', color='hotpink', lw=3, zorder=4)
        else:
            print(f"[WARNING] Only {len(x_used)} point(s) used — skipping fit.")

        plt.axvline(self.x_center, color='darkgray', linestyle='--', lw=2, label='Central Value')
        plt.title('Adaptive Fit on Noisy Function', fontsize=15)
        plt.xlabel('$x$')
        plt.ylabel('$f(x) + \\epsilon$')
        plt.legend(frameon=False)
        self._save_fig("adaptive_fit_with_noise.png")
        plt.show()

    def _adaptive_fit_with_outlier_removal(self, x_vals, y_vals, return_inliers=False):
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

    def plot_box_comparison(self, iterations=100, noise_std=0.01):
        stencil_vals, adaptive_vals = [], []
        for _ in range(iterations):
            noisy_func = lambda x: self.function(x) + np.random.randn() * noise_std
            kit = DerivativeKit(noisy_func, self.x_center)
            stencil_vals.append(kit.finite.compute())
            adaptive_vals.append(kit.adaptive.compute())

        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes[0].boxplot(stencil_vals, labels=['Stencil'], vert=False,
                        patch_artist=True, boxprops=dict(facecolor='hotpink'),
                        medianprops=dict(color='black'))
        axes[1].boxplot(adaptive_vals, labels=['Adaptive Fit'], vert=False,
                        patch_artist=True, boxprops=dict(facecolor='yellowgreen'),
                        medianprops=dict(color='black'))
        axes[1].set_xlabel('Derivative')
        self._save_fig("derivative_boxplot.png")
        plt.show()

    def plot_convergence_vs_h(self, h_values):
        nd_val = self._nd_derivative(self.x_center)
        stencil_vals, adaptive_vals = [], []

        for h in h_values:
            stencil_vals.append(self.derivs.finite.estimate(step_size=h))
            adaptive_vals.append(self.derivs.adaptive.estimate(step_size=h))

        plt.figure(figsize=(8, 5))
        plt.plot(h_values, stencil_vals, label="Stencil", marker='o')
        plt.plot(h_values, adaptive_vals, label="Adaptive Fit", marker='s')
        plt.axhline(nd_val, color='gray', linestyle='--', label='Numdifftools Ref')
        plt.xscale('log')
        plt.xlabel("Step size (h)")
        plt.ylabel("Derivative")
        plt.title("Convergence with Decreasing h")
        plt.legend()
        self._save_fig("convergence_vs_h.png")
        plt.show()

    def plot_error_vs_noise(self, noise_levels, trials=50):
        true_val = self._nd_derivative(self.x_center)
        stencil_mse, adaptive_mse = [], []

        for sigma in noise_levels:
            stencil_errors, adaptive_errors = [], []
            for _ in range(trials):
                noisy_f = lambda x: self.function(x) + np.random.normal(0, sigma)
                kit = DerivativeKit(noisy_f, self.x_center)
                stencil_errors.append((kit.finite.compute() - true_val)**2)
                adaptive_errors.append((kit.adaptive.compute() - true_val)**2)

            stencil_mse.append(np.mean(stencil_errors))
            adaptive_mse.append(np.mean(adaptive_errors))

        plt.figure(figsize=(7, 5))
        plt.plot(noise_levels, stencil_mse, label='Stencil', marker='o')
        plt.plot(noise_levels, adaptive_mse, label='Adaptive Fit', marker='s')
        plt.xlabel("Noise Standard Deviation")
        plt.ylabel("Mean Squared Error")
        plt.yscale("log")
        plt.title("Error vs Noise Level")
        plt.legend()
        self._save_fig("error_vs_noise.png")
        plt.show()

    def plot_relative_and_absolute_errors(self, x_min=None, x_max=None, num=100, **kwargs):
        if x_min is None: x_min = self.x_center - 0.5
        if x_max is None: x_max = self.x_center + 0.5
        x_vals = np.linspace(x_min, x_max, num)

        abs_errs_stencil, abs_errs_adaptive = [], []
        rel_errs_stencil, rel_errs_adaptive = [], []

        for x in x_vals:
            try:
                true = self._nd_derivative(x)
                stencil_val = self.derivs.finite.compute(**kwargs)
                adaptive_val = self.derivs.adaptive.compute(**kwargs)
                abs_errs_stencil.append(np.abs(stencil_val - true))
                abs_errs_adaptive.append(np.abs(adaptive_val - true))
                rel_errs_stencil.append(np.abs((stencil_val - true) / (true + 1e-12)))
                rel_errs_adaptive.append(np.abs((adaptive_val - true) / (true + 1e-12)))
            except Exception as e:
                warnings.warn(f"Skipping x={x:.4f} due to error: {e}")
                abs_errs_stencil.append(np.nan)
                abs_errs_adaptive.append(np.nan)
                rel_errs_stencil.append(np.nan)
                rel_errs_adaptive.append(np.nan)

        fig, axs = plt.subplots(1, 2, figsize=(7, 5), sharex=True)

        axs[0].plot(x_vals, abs_errs_stencil, label="Stencil", color='hotpink', marker='o', markersize=4)
        axs[0].plot(x_vals, abs_errs_adaptive, label="Adaptive Fit", color='yellowgreen', marker='s', markersize=4)
        axs[0].set_yscale("log")
        axs[0].set_ylabel("Absolute Error")
        axs[0].set_xlabel("x")
        axs[0].set_title("Absolute Error Across Domain")
        axs[0].legend(frameon=False)

        axs[1].plot(x_vals, rel_errs_stencil, label="Stencil", color='hotpink', marker='o', markersize=4)
        axs[1].plot(x_vals, rel_errs_adaptive, label="Adaptive Fit", color='yellowgreen', marker='s', markersize=4)
        axs[1].set_yscale("log")
        axs[1].set_ylabel("Relative Error")
        axs[1].set_xlabel("x")
        axs[1].set_title("Relative Error Across Domain")
        axs[1].legend(frameon=False)

        self._save_fig("error_comparison_across_domain.png")
        plt.show()

    def run_derivative_trials_with_noise(self, method="finite", order=1, noise_std=0.01, trials=100):
        results = []

        for i in range(trials):
            noisy_f = self.make_noisy_interpolated_function(
                self.function, self.x_center, noise_std=noise_std, seed=i
            )

            if method == "finite":
                result = DerivativeKit(noisy_f, self.x_center, derivative_order=order).finite.compute()
            elif method == "adaptive":
                result = DerivativeKit(noisy_f, self.x_center, derivative_order=order).adaptive.compute()
            elif method == "numdifftools":
                result = nd.Derivative(noisy_f, n=order)(self.x_center)

            results.append(result)

        return results

    def plot_bias_variance_tradeoff(self, noise_std=0.01, trials=100):
        true_val = self._nd_derivative(self.x_center)

        def collect(method):
            return self.run_derivative_trials_with_noise(method=method, noise_std=noise_std, trials=trials)

        stencil_vals = collect("finite")
        adaptive_vals = collect("adaptive")
        nd_vals = collect("numdifftools")

        def stats(arr):
            bias = np.mean(arr) - true_val
            var = np.var(arr)
            return bias ** 2, var, bias ** 2 + var

        labels = ['Bias²', 'Variance', 'MSE']
        stencil_stats = stats(stencil_vals)
        adaptive_stats = stats(adaptive_vals)
        nd_stats = stats(nd_vals)

        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(x - width, stencil_stats, width, label='Stencil', color='hotpink')
        ax.bar(x, adaptive_stats, width, label='Adaptive', color='yellowgreen')
        ax.bar(x + width, nd_stats, width, label='Numdifftools', color='lightskyblue')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Error")
        ax.set_title("Bias-Variance Tradeoff")
        ax.legend()
        self._save_fig("bias_variance_all_methods.png")
        plt.show()

    def make_noisy_interpolated_function(self, func, x_center, width=0.2, resolution=100, noise_std=0.01, seed=42):
        rng = np.random.default_rng(seed)
        x_grid = np.linspace(x_center - width, x_center + width, resolution)
        y_grid = np.array([func(x) + rng.normal(0, noise_std) for x in x_grid])

        def noisy_interp(x):
            return np.interp(x, x_grid, y_grid)

        return noisy_interp
