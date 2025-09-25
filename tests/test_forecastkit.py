"""Tests for ForecastKit.

Note: long numeric arrays are kept inline for readability.
"""

import numpy as np
import pytest

from derivkit.forecast_kit import ForecastKit


def test_pseudoinverse_path_no_nan():
    """If inversion fails, pseudoinverse path should still return finite numbers."""
    # Singular covariance -> forces pinv path
    forecaster = ForecastKit(
        lambda x: x,
        np.array([1.0]),
        np.array([[0.0]]),
    )
    fisher = forecaster.fisher()
    assert np.isfinite(fisher).all()
    tensor_g, tensor_h = forecaster.dali()
    assert np.isfinite(tensor_g).all()
    assert np.isfinite(tensor_h).all()


@pytest.mark.parametrize(
    (
        "model, "
        "fiducials, "
        "covariance_matrix, "
        "expected_fisher, "
        "expected_dali_g, "
        "expected_dali_h"
    ),
    [
        pytest.param(
            lambda x: 0.4 * x**2,
            np.array([2.11]),
            np.array([[2.75]]),
            np.array([[1.03612509]]),
            np.array([[[0.49105455]]]),
            np.array([[[[0.23272727]]]]),
        ),
        pytest.param(
            lambda x: 0.4 * x**2,
            np.array([1.1, 0.4]),
            np.array(
                [
                    [1.0, 2.75],
                    [3.2, 0.1],
                ]
            ),
            np.array(
                [
                    [-0.00890115, 0.08901149],
                    [0.10357701, -0.01177011],
                ]
            ),
            np.array(
                [
                    [
                        [-8.09195402e-03, 8.09195402e-02],
                        [-1.46382975e-16, -2.08644092e-16],
                    ],
                    [
                        [-1.42668169e-16, -3.75546474e-16],
                        [2.58942529e-01, -2.94252874e-02],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [
                            [-7.35632184e-03, -1.11448615e-16],
                            [-1.06393661e-16, 2.02298851e-01],
                        ],
                        [
                            [-1.33075432e-16, 7.15511183e-31],
                            [1.01887953e-30, -5.21610229e-16],
                        ],
                    ],
                    [
                        [
                            [-1.29698336e-16, 9.78598871e-31],
                            [1.29608820e-30, -9.38866185e-16],
                        ],
                        [
                            [2.35402299e-01, -6.14828927e-16],
                            [-1.10097326e-15, -7.35632184e-02],
                        ],
                    ],
                ]
            ),
        ),
        pytest.param(
            lambda x: np.exp(-0.5 * x**2),
            np.array([2.11]),
            np.array([[2.75]]),
            np.array([[0.01890366]]),
            np.array([[[-0.03087509]]]),
            np.array([[[[0.05042786]]]]),
        ),
        pytest.param(
            lambda x: np.exp(-0.5 * x**2),
            np.array([1.1, 0.4]),
            np.array(
                [
                    [1.0, 2.75],
                    [3.2, 0.1],
                ]
            ),
            np.array(
                [
                    [-0.00414466, 0.07008167],
                    [0.08154958, -0.01566948],
                ]
            ),
            np.array(
                [
                    [
                        [7.89639690e-04, -1.33519460e-02],
                        [6.87814470e-15, -1.84147323e-15],
                    ],
                    [
                        [6.59290723e-17, -1.11478872e-15],
                        [1.71255860e-01, -3.29062482e-02],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [
                            [-1.50441995e-04, -1.12697772e-15],
                            [-1.25607936e-17, -2.80393711e-02],
                        ],
                        [
                            [-1.31042274e-15, -2.06221337e-28],
                            [-1.09410605e-28, -3.86713302e-15],
                        ],
                    ],
                    [
                        [
                            [-1.25607936e-17, -9.40943024e-29],
                            [-1.04873334e-30, -2.34108006e-15],
                        ],
                        [
                            [-3.26276318e-02, -4.04783117e-15],
                            [-2.72416588e-15, -6.91038222e-02],
                        ],
                    ],
                ]
            ),
        ),
    ],
)
def test_forecast(
    model,
    fiducials,
    covariance_matrix,
    expected_fisher,
    expected_dali_g,
    expected_dali_h,
):
    """Compare forecast tensors to reference values.

    Args:
        model: Callable mapping x -> observable.
        fiducials: Fixed values at which the tensors are computed.
        covariance_matrix: Covariance used to compute forecast tensors.
        expected_fisher: Fisher matrix (2nd-order derivatives).
        expected_dali_g: First 3rd-order forecast tensor.
        expected_dali_h: Second 3rd-order forecast tensor.
    """
    forecaster = ForecastKit(model, fiducials, covariance_matrix)

    # Small magnitudes: use atol=0 (relative tolerance only).
    fisher_matrix = forecaster.fisher()
    assert np.allclose(fisher_matrix, expected_fisher, atol=0)

    dali_g, dali_h = forecaster.dali()
    assert np.allclose(dali_g, expected_dali_g, atol=0)
    assert np.allclose(dali_h, expected_dali_h, atol=0)
