"""Test functions for the ForecastKit class."""
import numpy
import pytest
import scipy

from derivkit.forecast_kit import ForecastKit


def test_derivative_order():
    """Tests if derivatives of high order raise a ValueError

    High order means higher than 2. The function tests an order of 3
    and a random number between 4 and 30 inclusive.
    """
    forecaster = ForecastKit(lambda x: x,
        numpy.array([1]),
        numpy.array([1])
    )
    with pytest.raises(ValueError) as exception:
        forecaster.get_derivatives(derivative_order=3)
    assert exception.type is ValueError
    with pytest.raises(ValueError) as exception:
        forecaster.get_derivatives(
            derivative_order=numpy.random.randint(low=4, high=30)
        )
    assert exception.type is ValueError


def test_forecast_order():
    """Tests if high forecast orders raise a ValueError.

    The forecast is an expansions of the likelihood function so really this
    tests if high order expansions raise a ValueError.

    High order means higher than 2. The function tests an order of 3
    and a random number between 4 and 30 inclusive.
    """
    forecaster = ForecastKit(lambda x: x,
        numpy.array([1]),
        numpy.array([1]),
    )
    with pytest.raises(ValueError) as exception:
        forecaster.get_forecast_tensors(forecast_order=3)
    assert exception.type is ValueError
    with pytest.raises(ValueError) as exception:
        forecaster.get_forecast_tensors(
            forecast_order=numpy.random.randint(low=4, high=30)
        )
    assert exception.type is ValueError


def test_failed_pseudoinverse():
    """Tests if failed pseudoinverse generates NaNs."""
    forecaster = ForecastKit(lambda x: x,
        numpy.array([1]),
        numpy.array([1]),
    )
    for i in range(1, 3):
        result = forecaster.get_forecast_tensors(forecast_order=i)
        assert map(numpy.isnan, result)

@pytest.mark.parametrize(
    (
        "model, "
        "fiducials, "
        "covariance_matrix, "
        "expected_fisher, "
        "expected_DALI_G, "
        "expected_DALI_H"
    ),
    [
        pytest.param( 
            lambda x: 0.4 * x**2,
            numpy.array([2.11]),
            numpy.array([[2.75]]),
            numpy.array([[1.03612509]]),
            numpy.array([[[0.49105455]]]),
            numpy.array([[[[0.23272727]]]])
        ),
        pytest.param( 
            lambda x: 0.4 * x**2,
            numpy.array([1.1, 0.4]),
            numpy.array([
                [1.0, 2.75],
                [3.2, 0.1]
            ]),
            numpy.array([
                [-0.00890115,  0.08901149],
                [ 0.10357701, -0.01177011]
            ]),
            numpy.array([
                [
                    [-8.09195402e-03,  8.09195402e-02],
                    [-1.46382975e-16, -2.08644092e-16]
                ],
                [
                    [-1.42668169e-16, -3.75546474e-16],
                    [ 2.58942529e-01, -2.94252874e-02]
                ]
            ]),
            numpy.array([
                [
                    [
                        [-7.35632184e-03, -1.11448615e-16],
                        [-1.06393661e-16,  2.02298851e-01]
                    ],
                    [
                        [-1.33075432e-16,  7.15511183e-31],
                        [ 1.01887953e-30, -5.21610229e-16]
                    ]
                ],
                [
                    [
                        [-1.29698336e-16,  9.78598871e-31],
                        [ 1.29608820e-30, -9.38866185e-16]
                    ],
                    [
                        [ 2.35402299e-01, -6.14828927e-16],
                        [-1.10097326e-15, -7.35632184e-02]
                    ]
                ]
            ])
        ),
        pytest.param( 
            lambda x: numpy.exp(-0.5 * x**2),
            numpy.array([2.11]),
            numpy.array([[2.75]]),
            numpy.array([[0.01890366]]),
            numpy.array([[[-0.03087509]]]),
            numpy.array([[[[0.05042786]]]])
        ),
        pytest.param( 
            lambda x: numpy.exp(-0.5 * x**2),
            numpy.array([1.1, 0.4]),
            numpy.array([
                [1.0, 2.75],
                [3.2, 0.1]
            ]),
            numpy.array([
                [-0.00414466, 0.07008167],
                [0.08154958, -0.01566948]
            ]),
            numpy.array([
                [
                    [ 7.89639690e-04, -1.33519460e-02],
                    [ 6.87814470e-15, -1.84147323e-15]
                ],
                [
                    [ 6.59290723e-17, -1.11478872e-15],
                    [ 1.71255860e-01, -3.29062482e-02]
                ]
            ]),
            numpy.array([
                [
                    [
                        [-1.50441995e-04, -1.12697772e-15],
                        [-1.25607936e-17, -2.80393711e-02]
                    ],
                    [
                        [-1.31042274e-15, -2.06221337e-28],
                        [-1.09410605e-28, -3.86713302e-15]
                    ]
                ],
                [
                    [
                        [-1.25607936e-17, -9.40943024e-29],
                        [-1.04873334e-30, -2.34108006e-15]
                    ],
                    [
                        [-3.26276318e-02, -4.04783117e-15],
                        [-2.72416588e-15, -6.91038222e-02]
                    ]
                ]
            ])
        ),
   ]
)
def test_forecast(
    model,
    fiducials,
    covariance_matrix,
    expected_fisher,
    expected_DALI_G,
    expected_DALI_H
):
    """Test the output of get_forecast_tensors to reference values.

    Args:
        model (callable): A function that parametrises a quantity expressed
            in terms of fiducial values
        fiducials (numpy.array): The fixed values at which the tensors are
            computed
        covariance_matrix (numpy.array): The covariance matrix used in the
            computation of the forecast tensors
        expected_fisher (numpy.array): The second order (derivative) forecast
            tensor (the Fisher matrix)
        expected_DALI_G (numpy.array): The first third order forecast tensor,
            third order in derivatives
        expected_DALI_H (numpy.array): The second third order forecast tensor,
            fourth order in derivatives
    """
    observables = model
    fiducial_values = fiducials
    covmat = covariance_matrix

    forecaster = ForecastKit(observables, fiducial_values, covmat)
    number_fiducial_values = len(fiducial_values)

    # It is possible for the computed (and expected) values of the tensors
    # to be much smaller than 0. The default value of the parameter atol of
    # numpy.isclose is then not appropriate: see the numpy documentation at
    # https://numpy.org/doc/stable/reference/generated/numpy.isclose.html.
    # The value has been set to 0 instead, so the tolerance is quantified
    # by only the relative difference.
    fisher_matrix = forecaster.get_forecast_tensors(forecast_order = 1)
    assert numpy.allclose(fisher_matrix, expected_fisher, atol=0)

    DALI_G, DALI_H = forecaster.get_forecast_tensors(forecast_order = 2)
    assert numpy.allclose(DALI_G, expected_DALI_G, atol=0)
    assert numpy.allclose(DALI_H, expected_DALI_H, atol=0)
