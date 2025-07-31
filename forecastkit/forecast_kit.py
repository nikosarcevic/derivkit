import numpy as np
from copy import deepcopy

from derivkit.kit import DerivativeKit

class ForecastKit:
    """
    Provides tools for facilitating experimental forecasts.

    Parameters
    ----------
    function : callable
        The scalar or vector-valued function to differentiate.
        For N observables with M model parameters, this should be a list of
            N functions, with each one a function of 1 list of M parameters.
    central_value : float
        The point at which the derivative is evaluated.
        For M model parameters, this should be 1 list of M values,
            provided in the same order as the requested input of the functions
            to be differentiated.
    covariance_matrix : float
        The covariance matrix of the observables.
        For N observables, this should be an NxN matrix, with the order of the 
            matrix entries matching the order of the functions to be differentiated.
    """

    def __init__(self, function, central_value, covariance_matrix):
        self.function = function
        self.central_value = central_value
        self.covariance_matrix = covariance_matrix
        self.M = len(central_value)
        self.N = len(covariance_matrix)

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
                d = 1 returns a MxN array of first-order derivatives.
                d = 2 returns a MxMxN array of second-order derivatives.
        """
        if derivative_order not in [1, 2]:
            raise ValueError("Only first- and second-order derivatives are currently supported.")

        if derivative_order == 1:
            # Get the first-order derivatives
            first_order_derivatives = np.zeros((self.M,self.N))
            for m in range(self.M):
                # 1 parameter to differentiate, and M-1 parameters to hold fixed
                functionToDiff = self._reduceFunctionTo1Var(self.function, m, self.central_value)
                # Currently done with adaptive
                kit = DerivativeKit(functionToDiff, \
                                    self.central_value[m], \
                                    derivative_order=1)
                first_order_derivatives[m] = kit.adaptive.compute()
            return first_order_derivatives

        elif derivative_order == 2:
            # Get the second-order derivatives
            second_order_derivatives = np.zeros((self.M,self.M,self.N))
            for m1 in range(self.M):
                for m2 in range(self.M):
                    if m1==m2:
                        # 1 parameter to differentiate twice, and M-1 parameters to hold fixed
                        functionToDiff1 = self._reduceFunctionTo1Var(self.function, m1, self.central_value)
                        kit1 = DerivativeKit(functionToDiff1, \
                                             self.central_value[m1], \
                                             derivative_order = 2)
                        # Currently done with adaptive
                        second_order_derivatives[m1][m2] = kit1.adaptive.compute()
                    else:
                        # 2 parameters to differentiate once, with other parameters held fixed
                        def functionToDiff2(y):
                            central_values_y = deepcopy(self.central_value)
                            central_values_y[m2] = y
                            functionToDiff1 = self._reduceFunctionTo1Var(self.function, m1, central_values_y)
                            kit1 = DerivativeKit(functionToDiff1, \
                                                 self.central_value[m1], \
                                                 derivative_order = 1)
                            # Currently done with adaptive
                            return kit1.adaptive.compute()
                        kit2 = DerivativeKit(functionToDiff2, \
                                             self.central_value[m2], \
                                             derivative_order = 1)
                        # Currently done with adaptive
                        second_order_derivatives[m1][m2] = kit2.adaptive.compute()
            return second_order_derivatives

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
                D = 1 returns an MxM matrix, where M is the number of
                    parameters included in the forecast.
                D = 2 returns MxMxM and MxMxMxM arrays, where M is the
                    number of parameters included in the forecast.
        """
        if forecast_order not in [1, 2]:
            raise ValueError("Only Fisher (order 1) and doublet-DALI (order 2) forecasts are currently supported.")

        # Invert the covariance matrix
        try:
            inverse_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        except:
            print("Covariance inversion failed!")
            inverse_covariance_matrix = np.empty((self.N,self.N))
            inverse_covariance_matrix.fill(np.nan)

        if forecast_order == 1:
            # Compute Fisher matrix
            dfp1 = self.get_derivatives(derivative_order = 1)
            # F_ab = sum(i,j) df_i/dp_a * InvCov_ij * df_j/dp_b
            F_ab = np.einsum('ai,ij,bj->ab', dfp1, inverse_covariance_matrix, dfp1)
            return F_ab

        elif forecast_order == 2:
            # Compute doublet-DALI tensors
            dfp1 = self.get_derivatives(derivative_order = 1)
            dfp2 = self.get_derivatives(derivative_order = 2)
            # G_abc = sum(i,j) df_i/(dp_a dp_b )* InvCov_ij * df_j/dp_c
            G_abc = np.einsum('abi,ij,cj->abc', dfp2, inverse_covariance_matrix, dfp1)
            # H_abcd = sum(i,j) df_i/(dp_a dp_b) * InvCov_ij * df_j/(dp_c dp_d)
            H_abcd = np.einsum('abi,ij,cdj->abcd', dfp2, inverse_covariance_matrix, dfp2)
            return G_abc, H_abcd

    def _reduceFunctionTo1Var(self, someFunction, indexToVary, fixedParamValues):
        """
        Reduces a multi-variable function to a single-variable function,
            which can then be passed to the DerivativeKit.

        Parameters
        ----------
        someFunction : callable
            The function to be reduced. The function should take one argument,
                a single list of M variables.
        indexToVary : int
            The index of the parameter to be kept as the independent variable.
        fixedParamValues : list
            A list of floats, fiducial parameter values for the parameters 
                which will be kept fixed.

        Returns
        -------
        function : callable
            A function which takes a single float variable as its input.
        """
        # Reduces the multi-variable function to a single-variable function
        def function1Var(x):
            fixedParamValues[indexToVary] = x
            return someFunction(fixedParamValues)
        return function1Var
