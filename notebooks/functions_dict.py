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

test_functions_residuals = [
    {"name": "x_squared", "label": r"$x^2$", "func": lambda x: x**2},
    {"name": "x_cubed", "label": r"$x^3$", "func": lambda x: x**3},
    {"name": "sin", "label": r"$\sin(x)$", "func": lambda x: np.sin(x)},
    {"name": "cos", "label": r"$\cos(x)$", "func": lambda x: np.cos(x)},
    {"name": "tan", "label": r"$\tan(x)$", "func": lambda x: np.tan(x)},
    {"name": "exp", "label": r"$\exp(x)$", "func": lambda x: np.exp(x)},
    {"name": "log_shifted", "label": r"$\ln(x + 2)$", "func": lambda x: np.log(x + 2)},
    {"name": "sqrt_shifted", "label": r"$\sqrt{x + 1}$", "func": lambda x: np.sqrt(x + 1)},
    {"name": "rational", "label": r"$\frac{1}{1 + x^2}$", "func": lambda x: 1 / (1 + x**2)},
    {"name": "x_sin", "label": r"$x\,\sin(x)$", "func": lambda x: x * np.sin(x)},
    {"name": "x2_cos", "label": r"$x^2 \cos(x)$", "func": lambda x: x**2 * np.cos(x)},
    {"name": "x_exp_negx2", "label": r"$x\,e^{-x^2}$", "func": lambda x: x * np.exp(-x**2)},
    {"name": "gauss_sin_5x", "label": r"$e^{-x^2} \sin(5x)$", "func": lambda x: np.exp(-x**2) * np.sin(5 * x)},
    {"name": "logsin_combo", "label": r"$\ln(1 + x^2)\sin(x)$", "func": lambda x: np.log(1 + x**2) * np.sin(x)},
    {"name": "sin_cos", "label": r"$\sin(x)\cos(x)$", "func": lambda x: np.sin(x) * np.cos(x)},
    {"name": "abs", "label": r"$|x|$", "func": lambda x: np.abs(x)},
    {"name": "piecewise_quad_sqrt", "label": r"$\begin{cases} x^2, & x<0 \\ \sqrt{x+1}, & x \geq 0 \end{cases}$", "func": lambda x: x**2 if x < 0 else np.sqrt(x + 1)},
    {"name": "piecewise_sin_cos", "label": r"$\begin{cases} \sin(x), & x<1 \\ \cos(x), & x \geq 1 \end{cases}$", "func": lambda x: np.sin(x) if x < 1 else np.cos(x)},
    {"name": "relu", "label": r"$\mathrm{ReLU}(x)$", "func": lambda x: max(0, x)},
    {"name": "x_absx", "label": r"$x\,|x|$", "func": lambda x: x * abs(x)},
    {"name": "heaviside", "label": r"$\Theta(x)$", "func": lambda x: 0 if x < 0 else 1},
    {"name": "sin_x2", "label": r"$\sin(x^2)$", "func": lambda x: np.sin(x**2)},
    {"name": "exp_sinx", "label": r"$\exp(\sin(x))$", "func": lambda x: np.exp(np.sin(x))},
    {"name": "log_cos2_plus1", "label": r"$\ln(\cos^2(x) + 1)$", "func": lambda x: np.log(np.cos(x)**2 + 1)},
    {"name": "poly_exp_mix", "label": r"$x^3 \cos(x^2) + e^{-x^2}$", "func": lambda x: x**3 * np.cos(x**2) + np.exp(-x**2)},
    {"name": "log_abs_sin", "label": r"$\ln(|\sin(x)| + 1)$", "func": lambda x: np.log(np.abs(np.sin(x)) + 1)},
    {"name": "gauss_cos10x", "label": r"$e^{-x^2} \cos(10x)$", "func": lambda x: np.exp(-x**2) * np.cos(10 * x)},
    {"name": "atan_exp", "label": r"$\arctan(e^x)$", "func": lambda x: np.arctan(np.exp(x))},
    {"name": "log_abs_sin_eps", "label": r"$\ln(|\sin(x)| + 10^{-3})$", "func": lambda x: np.log(np.abs(np.sin(x)) + 1e-3)},
    {"name": "sqrt_trig_identity", "label": r"$\sqrt{\sin^2(x) + \cos^2(x)}$", "func": lambda x: np.sqrt(np.sin(x)**2 + np.cos(x)**2)},
    {"name": "sigmoid", "label": r"$\sigma(x) = \frac{1}{1 + e^{-x}}$", "func": lambda x: 1 / (1 + np.exp(-x))},
    {"name": "softplus", "label": r"$\ln(1 + e^x)$", "func": lambda x: np.log(1 + np.exp(x))},
    {"name": "x_tanh", "label": r"$x \tanh(x)$", "func": lambda x: x * np.tanh(x)},
    {"name": "gaussian_bump", "label": r"$\exp\left(-\frac{1}{1 - x^2}\right)$", "func": lambda x: np.exp(-1 / (1 - x**2)) if abs(x) < 1 else 0},
    {"name": "x_pow_x_safe", "label": r"$x^x$", "func": lambda x: x**x if x > 0 else 0},
    {"name": "sin_exp", "label": r"$\sin(e^x)$", "func": lambda x: np.sin(np.exp(x))},
    {"name": "cos_sin_x2", "label": r"$\cos(\sin(x^2))$", "func": lambda x: np.cos(np.sin(x**2))},
    {"name": "sinc", "label": r"$\mathrm{sinc}(x)$", "func": lambda x: np.sinc(x / np.pi)},
]


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
