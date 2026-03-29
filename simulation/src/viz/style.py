"""Shared color palette and style helpers for all viz modules."""

import matplotlib.pyplot as plt

# Consistent palette across all plots
DEMAND_COLOR = "#3b82f6"       # blue
SUPPLY_COLOR = "#22c55e"       # green
LOCAL_ALLOC_COLOR = "#f59e0b"  # amber/orange
GRID_IMPORT_COLOR = "#ef4444"  # red
GRID_EXPORT_COLOR = "#8b5cf6"  # purple
COST_COLOR = "#0ea5e9"         # sky blue
NEUTRAL_COLOR = "#6b7280"      # gray

# Prosumer palette (for per-prosumer lines/bars)
PROSUMER_CMAP = "tab20"


def apply_style() -> None:
    """Apply a clean default style for all figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",

        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.edgecolor": "black",

        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,

        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 100,
    })
