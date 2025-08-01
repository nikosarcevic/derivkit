import numpy as np


# Define functions to calculate derivatives of

# First we define just basic function with known analytic derivatives
# These functions are used to demo the derivative estimation methods
test_functions = {
    "x_squared": {
        "func": lambda x: x**2,
        "label": r"$f(x) = x^2$",
        "reference": {
            1: lambda x: 2 * x,
            2: lambda x: 2,
            3: lambda x: 0
        }
    },
    "cubic_poly": {
        "func": lambda x: x**3 - 2 * x + 1,
        "label": r"$f(x) = x^3 - 2x + 1$",
        "reference": {
            1: lambda x: 3 * x**2 - 2,
            2: lambda x: 6 * x,
            3: lambda x: 6
        }
    },
    "sin_x": {
        "func": lambda x: np.sin(x),
        "label": r"$f(x) = \sin(x)$",
        "reference": {
            1: lambda x: np.cos(x),
            2: lambda x: -np.sin(x),
            3: lambda x: -np.cos(x)
        }
    },
    "cos_x": {
        "func": lambda x: np.cos(x),
        "label": r"$f(x) = \cos(x)$",
        "reference": {
            1: lambda x: -np.sin(x),
            2: lambda x: -np.cos(x),
            3: lambda x: np.sin(x)
        }
    },
    "exp_x": {
        "func": lambda x: np.exp(x),
        "label": r"$f(x) = e^x$",
        "reference": {
            1: lambda x: np.exp(x),
            2: lambda x: np.exp(x),
            3: lambda x: np.exp(x)
        }
    },
    "exp_neg_x_squared": {
        "func": lambda x: np.exp(-x**2),
        "label": r"$f(x) = e^{-x^2}$",
        "reference": {
            1: lambda x: -2 * x * np.exp(-x**2),
            2: lambda x: (4 * x**2 - 2) * np.exp(-x**2),
            3: lambda x: (-8 * x**3 + 12 * x) * np.exp(-x**2)
        }
    },
    "tanh_x": {
        "func": lambda x: np.tanh(x),
        "label": r"$f(x) = \tanh(x)$",
        "reference": {
            1: lambda x: 1 - np.tanh(x)**2,
            2: lambda x: -2 * np.tanh(x) * (1 - np.tanh(x)**2),
            3: lambda x: 2 * (1 - np.tanh(x)**2) * (3 * np.tanh(x)**2 - 1)
        }
    },
    "log1p_x": {
        "func": lambda x: np.log1p(x),
        "label": r"$f(x) = \log(1 + x)$",
        "reference": {
            1: lambda x: 1 / (1 + x),
            2: lambda x: -1 / (1 + x)**2,
            3: lambda x: 2 / (1 + x)**3
        }
    },
}

# Then we define a dictionary with standard functions
# which will probably behave well under polynomial fitting
# This we use in the notebook calle `03_standard_fn_demo.ipynb`
standard_functions = {
    'linear': {
        'func': lambda x: 2 * x + 3,
        'label': r"$f(x) = 2x + 3$"
    },
    'quadratic': {
        'func': lambda x: x ** 2 - 4 * x + 2,
        'label': r"$f(x) = x^2 - 4x + 2$"
    },
    'trigonometric': {
        'func': lambda x: np.sin(x),
        'label': r"$f(x) = \sin(x)$"
    },
    'cubic': {
        'func': lambda x: x**3 - 3 * x**2 + 2 * x + 1,
        'label': r"$f(x) = x^3 - 3x^2 + 2x + 1$"
    },
    'gaussian': {
        'func': lambda x: np.exp(-x**2),
        'label': r"$f(x) = e^{-x^2}$"
    },
}

# Define functions that have singularities or discontinuities
# These functions are expected to have issues with polynomial fitting
# and are used to test the robustness of the derivative estimation methods
# They are often used to test the limits of numerical methods
# and to see how well they can handle functions that are not smooth
# or have points where they are not defined
# This we use in the notebook called `04_blowup_fn_demo.ipynb`
blowup_functions = {
    '1_over_x': {
        'func': lambda x: 1 / x,
        'label': r"$f(x) = \frac{1}{x}$"
    },
    'log_x': {
        'func': lambda x: np.log(x),
        'label': r"$f(x) = \log(x)$"
    },
    'abs_x': {
        'func': lambda x: np.abs(x),
        'label': r"$f(x) = |x|$"
    },
    'step': {
        'func': lambda x: np.where(x < 0, -1, 1),
        'label': r"$f(x) = \mathrm{sign}(x)$"
    },
    'root_abs': {
    'func': lambda x: np.sqrt(np.abs(x)),
    'label': r"$f(x) = \sqrt{|x|}$"
    },

}

# here we define the fit tolerances for different polynomial orders
# these are used to determine how well the polynomial fit should approximate the function
fit_tolerance = {
    "1": 0.05,
    "2": 0.1,
    "3": 0.2,
    "4": 0.3,
}

# and here we define a litle function to convert numbers to their ordinal representation
# i use this to label the order of derivatives in plots and tables
def get_ordinal(n):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")
