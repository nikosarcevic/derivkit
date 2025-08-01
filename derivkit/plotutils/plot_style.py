import matplotlib as mpl


DEFAULT_LINEWIDTH = 2.0

GRADIENT_COLORS = {
    "finite_1": "#3b9ab2",
    "finite_2": "#76b8c9",
    "finite_3": "#b1d7e0",
    "adaptive_1": "#e1af00",
    "adaptive_2": "#eac74d",
    "adaptive_3": "#f3df99",
}

DEFAULT_COLORS = {
    "finite": "#3b9ab2",
    "finite_lite": "#77b7c5",
    "adaptive": "#e1af00",
    "adaptive_lite": "#eac74d",
    "central": "#f65e4d",
    "excluded": "#f21901"
}

MID_GRAY   = "#4a4a4a"
DARK_GRAY  = "#555555"
LIGHT_GRAY = "#E0E0E0"

def apply_plot_style():
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.serif": ["Computer Modern Sans serif"],
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "mathtext.rm": "sans",
        "mathtext.default": "regular",
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage{sfmath}",

        # Color styling
        "text.color": MID_GRAY,
        "axes.labelcolor": MID_GRAY,
        "axes.titlecolor": MID_GRAY,
        "axes.edgecolor": MID_GRAY,
        "xtick.color": DARK_GRAY,
        "ytick.color": DARK_GRAY,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.edgecolor": MID_GRAY,
        "legend.facecolor": "white",
        "legend.framealpha": 0.9,
        "legend.labelcolor": MID_GRAY,
        "grid.color": LIGHT_GRAY,
        "grid.alpha": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    })
