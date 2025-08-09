from copy import deepcopy

import numpy as np

from derivkit.kit import DerivativeKit

class ForecastKit:
    """
    Provides tools for facilitating experimental forecasts.

    Parameters
    ----------
    function : callable
        The scalar or vector-valued function to differentiate.
        It should accept a list or array of parameter values as input and return
        either a scalar or a NumPy array of observable values.
    central_values : array-like
        The point(s) at which the derivative is evaluated.
        A 1D array or list of parameter values matching the expected input of the function.
    covariance_matrix : array-like
        The covariance matrix of the observables.
        Should be a square matrix with shape (n_observables, n_observables), where 
        n_observables is the number of observables returned by the function.
    """

    def __init__(self, function, central_values, covariance_matrix):
        self.function = function
        self.central_values = np.atleast_1d(central_values)
        self.covariance_matrix = covariance_matrix
        self.n_parameters = len(self.central_values)
        self.n_observables = len(covariance_matrix)

    def get_derivatives(self, derivative_order):
        """
        Returns derivatives of the observables, according to the requested 
            order of the derivatives.

        Parameters
        ----------
        derivative_order : int
            The requested order, d, of the derivatives. 
                d = 1 returns first-order derivatives.
                d = 2 returns second-order derivatives.
                Currently only d = 1, 2 are supported.

        Returns
        -------
        np.ndarray
            An array of derivative values.
                d = 1 returns an array with shape (n_parameters, n_observables)  containing first-order derivatives.
                d = 2 returns an array with shape (n_parameters, n_parameters, n_observables) containing second-order derivatives.
        """
        if derivative_order not in [1, 2]:
            raise ValueError("Only first- and second-order derivatives are currently supported.")

        if derivative_order == 1:
            # Get the first-order derivatives
            first_order_derivatives = np.zeros((self.n_parameters, self.n_observables))
            for m in range(self.n_parameters):
                # 1 parameter to differentiate, and n_parameters-1 parameters to hold fixed
                central_values_x = deepcopy(self.central_values)
                function_to_diff = self._get_partial_function(self.function, m, central_values_x)
                kit = DerivativeKit(function_to_diff,
                                    self.central_values[m]
                                    )
                first_order_derivatives[m] = kit.adaptive.compute(derivative_order=1)
            return first_order_derivatives

        elif derivative_order == 2:
            # Get the second-order derivatives
            second_order_derivatives = np.zeros((self.n_parameters, self.n_parameters, self.n_observables))
            for m1 in range(self.n_parameters):
                for m2 in range(self.n_parameters):
                    if m1==m2:
                        # 1 parameter to differentiate twice, and n_parameters-1 parameters to hold fixed
                        central_values_x = deepcopy(self.central_values)
                        function_to_diff1 = self._get_partial_function(self.function, m1, central_values_x)
                        kit1 = DerivativeKit(function_to_diff1,
                                             self.central_values[m1]
                                             )
                        second_order_derivatives[m1][m2] = kit1.adaptive.compute(derivative_order=2)

                    else:
                        # 2 parameters to differentiate once, with other parameters held fixed
                        def function_to_diff2(y):
                            central_values_y = deepcopy(self.central_values)
                            central_values_y[m2] = y
                            function_to_diff1 = self._get_partial_function(self.function, m1, central_values_y)
                            kit1 = DerivativeKit(function_to_diff1,
                                                 self.central_values[m1]
                                                 )
                            return kit1.adaptive.compute(derivative_order=1)
                        kit2 = DerivativeKit(function_to_diff2,
                                             self.central_values[m2]
                                             )
                        second_order_derivatives[m1][m2] = kit2.adaptive.compute(derivative_order=1)

            return second_order_derivatives

        raise RuntimeError("Unreachable code reached in get_forecast_tensors.")

    def get_forecast_tensors(self, forecast_order=1):
        """
        Returns a set of tensors, according to the requested order of the forecast.

        Parameters
        ----------
        forecast_order : int
            The requested order, D, of the forecast. 
                D = 1 returns a Fisher matrix.
                D = 2 returns the 3-d and 4-d tensors required for the 
                    doublet-DALI approximation.
                D = 3 would be the triplet-DALI approximation.
                Currently only D = 1, 2 are supported.

        Returns
        -------
        np.ndarray
            A list of numpy arrays.
                D = 1 returns a square matrix of size n_parameters, where n_parameters is the number of
                    parameters included in the forecast.
                D = 2 returns one array of shapes (n_parameters, n_parameters, n_parameters) and one array of 
                shape (n_parameters, n_parameters, n_parameters, n_parameters), where n_parameters is the
                    number of parameters included in the forecast.
        """
        if forecast_order not in [1, 2]:
            raise ValueError("Only Fisher (order 1) and doublet-DALI (order 2) forecasts are currently supported.")

        # Invert the covariance matrix
        try:
            inverse_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        except np.linalg.LinAlgError:
            print("Standard inversion failed. Trying pseudoinverse.")
            try:
                inverse_covariance_matrix = np.linalg.pinv(self.covariance_matrix)
            except Exception as e:
                print(f"Pseudoinverse also failed: {e}")
                inverse_covariance_matrix = np.full((self.n_observables, self.n_observables),
                                                    np.nan)

        if forecast_order == 1:
            # Compute Fisher matrix
            dfp1 = self.get_derivatives(derivative_order=1)
            # F_ab = sum(i,j) df_i/dp_a * InvCov_ij * df_j/dp_b
            fisher_ab = np.einsum('ai,ij,bj->ab', dfp1, inverse_covariance_matrix, dfp1)
            return fisher_ab

        elif forecast_order == 2:
            # Compute doublet-DALI tensors
            dfp1 = self.get_derivatives(derivative_order=1)
            dfp2 = self.get_derivatives(derivative_order=2)
            # G_abc = sum(i,j) df_i/(dp_a dp_b )* InvCov_ij * df_j/dp_c
            g_abc = np.einsum('abi,ij,cj->abc', dfp2, inverse_covariance_matrix, dfp1)
            # H_abcd = sum(i,j) df_i/(dp_a dp_b) * InvCov_ij * df_j/(dp_c dp_d)
            h_abcd = np.einsum('abi,ij,cdj->abcd', dfp2, inverse_covariance_matrix, dfp2)
            return g_abc, h_abcd

        raise RuntimeError("Unreachable code reached in get_forecast_tensors.")

    def _get_partial_function(self, full_function, variable_index, fixed_values):
        """
        Returns a single-variable version of a multivariate function, where all
        parameters except one are held fixed.

        Parameters
        ----------
        full_function : callable
            A function that takes a list of n_parameters parameters and returns 
            a vector of n_observables observables.
        variable_index : int
            Index of the parameter to treat as the variable.
        fixed_values : list or np.ndarray
            The list of parameter values to use as fixed inputs for all parameters
            except the one being varied.

        Returns
        -------
        callable
            A function of a single variable, suitable for use in differentiation.
        """

        def partial_function(x):
            params = deepcopy(fixed_values)
            params[variable_index] = x
            return np.atleast_1d(full_function(params))

        return partial_function
