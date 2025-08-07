import os

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np

from .hybrid import HybridDerivativeCalculator


class DerivativePlotter:
    def __init__(self, function, x_center, plot_dir="plots"):
        self.function = function
        self.x_center = x_center
        self.plot_dir = plot_dir

        # Ensure the plot directory exists
        os.makedirs(self.plot_dir, exist_ok=True)
        self.hdc = HybridDerivativeCalculator(function, x_center)

    def plot_derivatives_hist(self,
                              method1_data,
                              method2_data,
                              method1_name='Stencil',
                              method2_name='STEM',
                              bins=20,
                              save_fig=False):
        """
        Plots histograms of derivative calculations for two methods side by side,
        and compares with numdifftools derivative calculation.

        Parameters:
            method1_data (list): The derivative data from the first method.
            method2_data (list): The derivative data from the second method.
            method1_name (str): The name of the first method.
            method2_name (str): The name of the second method.
            bins (int): Number of bins for the histogram.
            save_fig (bool): Whether to save the figure to a file. Default is False.

        Returns:
            None: Displays the histograms of the derivative distributions.
        """
        # Set the figure size
        plt.figure(figsize=(21, 10))
        lw = 2.5  # Line width
        alpha = 1  # Transparency

        # Set the global parameters for all plots
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['legend.fontsize'] = 20
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 15  # Set x-tick label font size
        plt.rcParams['ytick.labelsize'] = 15

        # Calculate derivative using numdifftools
        nd_derivative = nd.Derivative(self.function)(self.x_center)

        # Plot for method 1
        plt.subplot(1, 2, 1)
        plt.hist(method1_data, bins=bins, color='hotpink', alpha=alpha, edgecolor='black')
        plt.axvline(np.mean(method1_data), color='k', linestyle=':', lw=lw, label=f'Mean: {np.mean(method1_data):.2f}')
        plt.axvline(np.median(method1_data), color='gray', linestyle='--', lw=lw,
                    label=f'Median: {np.median(method1_data):.2f}')
        plt.plot([], [], ' ', label=f"ND Derivative: {nd_derivative:.2f}")  # Empty plot for legend numdifftools
        plt.plot([], [], ' ', label=f"Variance: {np.var(method1_data):.2f}")  # Empty plot for legend variance
        plt.title(f"{method1_name} Derivative Distribution", fontsize=25)
        plt.xlabel("Derivative")
        plt.ylabel("Frequency")
        plt.legend(loc='upper left')

        # Plot for method 2
        plt.subplot(1, 2, 2)
        plt.hist(method2_data, bins=bins, color='yellowgreen', alpha=alpha, edgecolor='black')
        plt.axvline(np.mean(method2_data), color='k', linestyle=':', lw=lw, label=f'Mean: {np.mean(method2_data):.2f}')
        plt.axvline(np.median(method2_data), color='gray', linestyle='--', lw=lw,
                    label=f'Median: {np.median(method2_data):.2f}')
        plt.plot([], [], ' ', label=f"ND Derivative: {nd_derivative:.2f}")  # Empty plot for legend numdifftools
        plt.plot([], [], ' ', label=f"Variance: {np.var(method2_data):.2f}")  # Empty plot for legend variance
        plt.title(f"{method2_name} Derivative Distribution", fontsize=25)
        plt.xlabel("Derivative")
        plt.ylabel("Frequency")
        plt.legend(loc='upper left')
        # Adjust layout to ensure everything fits without clipping
        plt.tight_layout()

        if save_fig:
            plt.savefig(f"{self.plot_dir}/derivation_comparison_hist.png", dpi=300)

        plt.show()

    def demonstrate_stem_method(self, num_points=20, noise_std=0.01, save_fig=False):
        """
        Demonstrates the stem method with fitting using noisy data points.
        It generates noisy data points, fits a line using the stem method,
        and plots the noisy data points and the fitted line.
        Please note that this is a simplified version of the stem_method() function.

        Parameters:
            num_points (int): Number of data points to generate and fit.
            noise_std (float): Standard deviation of the noise to add to data points.
            save_fig (bool): Whether to save the plot to a file. Default to False.

        Returns:
            plot: The plot of noisy data points and fitted line.
        """
        # Generate noisy data points
        x_values = np.linspace(self.x_center - 0.1, self.x_center + 0.1, num_points)
        y_values = [self.function(x) + np.random.randn(1)[0] * noise_std for x in x_values]

        # Fit a line using the stem method
        slope, intercept, max_spread = self.stem_method_with_fitting(x_values, y_values)

        # Set the global parameters for all plots
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        # Plot the noisy data points and fitted line
        plt.figure(figsize=(7, 5))

        plt.axvline(self.x_center, color='darkgray', linestyle='--', lw=2, label='Evaluation Point')
        fit = slope * x_values + intercept
        plt.scatter(x_values, y_values, label='Sampled Function Values', edgecolors='k', color='yellowgreen', s=100)
        plt.plot(x_values, fit, label='STEM Fitted Tangent', color='yellowgreen', lw=3)
        plt.scatter(x_values, y_values, edgecolors='k', color='yellowgreen', s=100)

        plt.title('STEM Tangent Estimation via Local Sampling', fontsize=15)
        plt.xlabel('$x$ (central value and stem values)', fontsize=12)
        plt.ylabel('$y$ (evaluated function values)', fontsize=12)
        plt.legend(fontsize=12, frameon=False)
        plt.xticks(fontsize=10)  # Adjust the font size for x-axis tick labels as needed
        plt.yticks(fontsize=10)
        # Adjust layout to ensure everything fits without clipping
        plt.tight_layout()

        if save_fig:
            # Save the plot to the specified directory
            plt.savefig(f"{self.plot_dir}/stem_method_demo.png", dpi=300)

        plt.show()

    def stem_method_with_fitting(self, x_values, y_values, min_samples=5):
        """
        Calculates the stem method with fitting using provided data points.
        It fits a line to the data points and removes the outlier points iteratively.
        The fitting is done using numpy.polyfit.
        Other fitting methods can be used instead.

        Parameters:
            x_values (array-like): x values of data points.
            y_values (array-like): y values of data points.
            min_samples (int): Minimum number of samples required to perform fitting.

        Returns:
            Returns:
                tuple: (slope, intercept, max_spread) — coefficients of the fitted line and
                 the maximum relative spread.

        """
        # Initial fitting
        slope, intercept = np.polyfit(x_values, y_values, 1)
        y_fitted = slope * x_values + intercept
        spread = np.abs((y_fitted - y_values) / y_values)  # Spread of the fitted line
        max_spread = np.max(spread)  # Maximum spread

        # Refinement with removal of outliers and re-fitting
        while max_spread >= 0.01 and len(x_values) >= min_samples:
            # Find the index of the point with maximum spread
            max_spread_index = np.argmax(spread)

            # Remove the outlier point
            x_values = np.delete(x_values, max_spread_index)
            y_values = np.delete(y_values, max_spread_index)

            # Fit the line again
            slope, intercept = np.polyfit(x_values, y_values, 1)
            y_fitted = slope * x_values + intercept
            spread = np.abs((y_fitted - y_values) / y_values)
            max_spread = np.max(spread)

        return float(slope), float(intercept), float(max_spread)

    def plot_derivatives_boxplot(self, iterations=100, noise_std=0.01, save_fig=False):
        """
        Plots a boxplot comparing the distributions of derivatives calculated.
        Args:
            iterations: Number of iterations to calculate derivatives with noise.
            noise_std: Amount of noise to add to the function evaluations.
            save_fig: Whether to save the figure to a file. Default is False.

        Returns:
            None: Displays a boxplot comparing the distributions of derivatives
                  calculated using the stem method and the five-point stencil method.
        """
        stem_derivatives = []
        stencil_derivatives = []

        for _ in range(iterations):
            def noisy_func(x, func=self.function):
                return func(x) + np.random.randn(1)[0] * noise_std

            hdc_noisy = HybridDerivativeCalculator(noisy_func, self.x_center)
            stem_derivatives.append(hdc_noisy.stem_method())
            stencil_derivatives.append(hdc_noisy.five_point_stencil_method())

        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=False)

        stem_color = 'yellowgreen'
        stencil_color = 'hotpink'

        # Stencil boxplot
        b1 = axes[0].boxplot(stencil_derivatives, vert=False, patch_artist=True,
                             boxprops=dict(facecolor=stencil_color),
                             medianprops=dict(color='black'))
        axes[0].set_yticks([])  # remove label ticks
        axes[0].legend([b1["boxes"][0]], ['Five-Point Stencil'], loc='upper right', frameon=True)

        # Stem boxplot
        b2 = axes[1].boxplot(stem_derivatives, vert=False, patch_artist=True,
                             boxprops=dict(facecolor=stem_color),
                             medianprops=dict(color='black'))
        axes[1].set_yticks([])
        axes[1].legend([b2["boxes"][0]], ['STEM Method'], loc='upper right', frameon=True)

        axes[1].set_xlabel('Distribution of Estimated Derivatives', fontsize=12)

        plt.tight_layout()

        if save_fig:
            plt.savefig(f"{self.plot_dir}/derivation_comparison_boxplot.png", dpi=300)
        plt.show()
