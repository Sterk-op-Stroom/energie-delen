"""Pricing & cost distribution visualizations.

All functions take a PricingResult and return a matplotlib Figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..core_types import PricingResult
from .style import (
    COST_COLOR,
    LOCAL_ALLOC_COLOR,
    GRID_IMPORT_COLOR,
    NEUTRAL_COLOR,
    PROSUMER_CMAP,
    apply_style,
)


def plot_community_cost(result: PricingResult) -> Figure:
    """Community total local cost over time."""
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.fill_between(result.timestamp, result.total_local_cost_eur, alpha=0.3, color=COST_COLOR)
    ax.plot(result.timestamp, result.total_local_cost_eur, color=COST_COLOR, linewidth=0.8)

    total = float(result.total_local_cost_eur.sum())
    ax.set_ylabel("EUR per interval")
    ax.set_title(f"Community Local Energy Cost — Total: {total:,.2f} EUR")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_prosumer_cost_bars(result: PricingResult) -> Figure:
    """Horizontal bar chart of total EUR per prosumer, sorted descending."""
    apply_style()

    ids = sorted(
        result.prosumer_ids,
        key=lambda m: result.total_local_cost_eur_by_prosumer[m],
        reverse=True,
    )
    costs = [result.total_local_cost_eur_by_prosumer[m] for m in ids]

    fig, ax = plt.subplots(figsize=(8, max(3, len(ids) * 0.35)))
    bars = ax.barh(range(len(ids)), costs, color=COST_COLOR, alpha=0.7)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("EUR (total)")
    ax.set_title("Per-Prosumer Local Energy Cost")

    # Value labels
    for bar, cost in zip(bars, costs):
        ax.text(
            bar.get_width() + max(costs) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{cost:.2f}", va="center", fontsize=8,
        )

    fig.tight_layout()
    return fig


def plot_cost_vs_kwh(result: PricingResult) -> Figure:
    """Scatter: total kWh received (x) vs. total EUR paid (y) per prosumer."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    kwh_totals = [float(result.local_kwh_priced[m].sum()) for m in result.prosumer_ids]
    eur_totals = [result.total_local_cost_eur_by_prosumer[m] for m in result.prosumer_ids]

    ax.scatter(kwh_totals, eur_totals, color=COST_COLOR, s=50, alpha=0.7, edgecolors="white", linewidth=0.5)

    # Reference line: expected linear relationship
    if max(kwh_totals) > 0:
        x_line = np.array([0, max(kwh_totals) * 1.05])
        ax.plot(x_line, x_line * result.fixed_price_eur_per_kwh, color=NEUTRAL_COLOR, linestyle="--",
                linewidth=0.8, label=f"@ {result.fixed_price_eur_per_kwh:.4f} EUR/kWh")

    for m, x, y in zip(result.prosumer_ids, kwh_totals, eur_totals):
        ax.annotate(m, (x, y), fontsize=7, textcoords="offset points", xytext=(5, 3))

    ax.set_xlabel("kWh received (local)")
    ax.set_ylabel("EUR paid (local)")
    ax.set_title("Cost vs. Energy Received")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_cumulative_cost(result: PricingResult) -> Figure:
    """Cumulative EUR over time per prosumer."""
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    cmap = plt.get_cmap(PROSUMER_CMAP)
    n = len(result.prosumer_ids)
    for i, m in enumerate(result.prosumer_ids):
        cumulative = np.cumsum(result.local_cost_eur[m])
        ax.plot(result.timestamp, cumulative, linewidth=1, color=cmap(i / max(n - 1, 1)), label=m)

    ax.set_ylabel("Cumulative EUR")
    ax.set_title("Cumulative Local Energy Cost per Prosumer")
    if n <= 15:
        ax.legend(loc="upper left", fontsize=7, ncol=max(1, n // 5))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_savings_estimate(
    result: PricingResult,
    grid_tariff_eur_per_kwh: float = 0.25,
) -> Figure:
    """Per-prosumer savings: hypothetical grid cost minus actual local cost.

    Args:
        result: PricingResult from the pipeline.
        grid_tariff_eur_per_kwh: Assumed grid electricity price for comparison.
    """
    apply_style()

    ids = result.prosumer_ids
    local_costs = [result.total_local_cost_eur_by_prosumer[m] for m in ids]
    grid_costs = [float(result.local_kwh_priced[m].sum()) * grid_tariff_eur_per_kwh for m in ids]
    savings = [g - l for g, l in zip(grid_costs, local_costs)]

    # Sort by savings descending
    order = sorted(range(len(ids)), key=lambda i: savings[i], reverse=True)
    ids_sorted = [ids[i] for i in order]
    local_sorted = [local_costs[i] for i in order]
    grid_sorted = [grid_costs[i] for i in order]
    savings_sorted = [savings[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.8), 5))
    x = range(len(ids_sorted))
    width = 0.35

    ax.bar([i - width / 2 for i in x], grid_sorted, width, color=GRID_IMPORT_COLOR, alpha=0.6, label=f"Grid cost @ {grid_tariff_eur_per_kwh:.2f} EUR/kWh")
    ax.bar([i + width / 2 for i in x], local_sorted, width, color=LOCAL_ALLOC_COLOR, alpha=0.6, label=f"Local cost @ {result.fixed_price_eur_per_kwh:.4f} EUR/kWh")

    # Savings annotation
    for i, s in enumerate(savings_sorted):
        ax.annotate(
            f"{s:+.2f}",
            (i, max(grid_sorted[i], local_sorted[i])),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
            color="green" if s > 0 else "red",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(ids_sorted, fontsize=8, rotation=45 if len(ids) > 10 else 0, ha="right" if len(ids) > 10 else "center")
    ax.set_ylabel("EUR (total)")
    ax.set_title("Savings Estimate: Local vs. Grid Pricing")
    ax.legend()
    fig.tight_layout()
    return fig
