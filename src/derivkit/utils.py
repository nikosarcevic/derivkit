import numpy as np


def log_debug_message(message, debug=False, log_file=None, log_to_file=None):
    """
    Logs a debug message to stdout and optionally to a file.

    Args:
        message (str): The debug message to print/log.
        debug (bool): Whether to print the message.
        log_file (str or None): Path to the log file.
        log_to_file (bool or None): Whether to write the message to the file.
    """
    if not debug:
        return

    print(message)

    if log_to_file and log_file:
        try:
            with open(log_file, "a") as f:
                f.write(message + "\n")
        except (IOError, OSError) as e:
            print(f"[log_debug_message] Failed to write to log file: {e}")


def is_finite_and_differentiable(function, x, delta=1e-5, tol=1e-2):
    """
    Checks if a function is finite and numerically differentiable at a point.

    Args:
        function (callable): The function to test.
        x (float or np.ndarray): The input value(s).
        delta (float): Step size for finite difference.
        tol (float): Tolerance for differentiability check.

    Returns:
        bool: True if function is finite and differentiable at x, False otherwise.
    """
    try:
        f0 = np.asarray(function(x))
        f_minus = np.asarray(function(x - delta))
        f_plus = np.asarray(function(x + delta))

        # Check for finiteness
        if not np.isfinite(f0).all() or not np.isfinite(f_minus).all() or not np.isfinite(f_plus).all():
            return False

        # Numerical derivative comparison (left and right)
        left = (f0 - f_minus) / delta
        right = (f_plus - f0) / delta

        return np.all(np.abs(left - right) < tol)

    except (ValueError, TypeError, ArithmeticError):
        return False


def normalize_derivative(derivative, reference):
    """
    Computes the relative error between estimated and reference derivative.

    Args:
        derivative (float or np.ndarray): Estimated derivative.
        reference (float or np.ndarray): True/reference derivative.

    Returns:
        float or np.ndarray: Normalized relative error.
    """
    return (derivative - reference) / (np.abs(reference) + 1e-12)


def central_difference_error_estimate(step_size, order=1):
    """
    Provides a rough truncation error estimate for central differences.

    Args:
        step_size (float): Finite difference step size.
        order (int): Order of the derivative (1 to 4).

    Returns:
        float: Estimated error magnitude.
    """
    if order == 1:
        return step_size**2 / 6
    elif order == 2:
        return step_size**2 / 12
    elif order == 3:
        return step_size**2 / 8
    elif order == 4:
        return step_size**2 / 6
    else:
        raise ValueError("Only derivative orders 1–4 are supported.")


def is_symmetric_grid(x_vals):
    """
    Checks if evaluation grid is symmetric around 0.

    Args:
        x_vals (np.ndarray): Evaluation points (1D).

    Returns:
        bool: True if grid is symmetric, False otherwise.
    """
    x_vals = np.sort(np.asarray(x_vals))
    n = len(x_vals)
    if n % 2 == 0:
        return False
    mid = n // 2
    return np.allclose(x_vals[:mid], -x_vals[:mid:-1])


def generate_test_function(name="sin"):
    """
    Returns a known test function and its first/second derivatives.

    Args:
        name (str): One of 'sin', 'exp', 'polynomial', 'gaussian'.

    Returns:
        tuple: (f(x), df/dx, d2f/dx2)
    """
    if name == "sin":
        return (
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x: -np.sin(x)
        )
    elif name == "exp":
        return (
            lambda x: np.exp(x),
            lambda x: np.exp(x),
            lambda x: np.exp(x)
        )
    elif name == "polynomial":
        return (
            lambda x: x**3 + 2*x**2 - x + 5,
            lambda x: 3*x**2 + 4*x - 1,
            lambda x: 6*x + 4
        )
    elif name == "gaussian":
        return (
            lambda x: np.exp(-x**2),
            lambda x: -2*x*np.exp(-x**2),
            lambda x: (4*x**2 - 2)*np.exp(-x**2)
        )
    else:
        raise ValueError("Supported names: 'sin', 'exp', 'polynomial', 'gaussian'")
