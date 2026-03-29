"""Supply & demand pattern visualizations.

All functions take an AggregatedStep and return a matplotlib Figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..core_types import AggregatedStep
from .style import (
    DEMAND_COLOR,
    SUPPLY_COLOR,
    NEUTRAL_COLOR,
    apply_style,
)


def plot_supply_vs_demand(step: AggregatedStep) -> Figure:
    """Time series of supply_total and demand_total with shaded gap."""
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    ts = step.timestamp
    ax.plot(ts, step.demand_total, color=DEMAND_COLOR, linewidth=0.8, label="Demand")
    ax.plot(ts, step.supply_total, color=SUPPLY_COLOR, linewidth=0.8, label="Supply")
    ax.fill_between(
        ts,
        step.supply_total,
        step.demand_total,
        where=step.supply_total >= step.demand_total,
        alpha=0.15,
        color=SUPPLY_COLOR,
        label="Surplus",
    )
    ax.fill_between(
        ts,
        step.supply_total,
        step.demand_total,
        where=step.supply_total < step.demand_total,
        alpha=0.15,
        color=DEMAND_COLOR,
        label="Deficit",
    )

    ax.set_ylabel("kWh per interval")
    ax.set_title("Community Supply vs. Demand")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_supply_demand_ratio(step: AggregatedStep) -> Figure:
    """Supply / demand ratio over time. Ratio > 1 = surplus."""
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 4))

    ratio = np.where(
        step.demand_total > 0,
        step.supply_total / step.demand_total,
        np.nan,
    )
    ax.plot(step.timestamp, ratio, color=NEUTRAL_COLOR, linewidth=0.7)
    ax.axhline(1.0, color=SUPPLY_COLOR, linestyle="--", linewidth=0.8, label="Balanced (1.0)")
    ax.fill_between(
        step.timestamp, ratio, 1.0,
        where=ratio >= 1.0, alpha=0.15, color=SUPPLY_COLOR,
    )
    ax.fill_between(
        step.timestamp, ratio, 1.0,
        where=ratio < 1.0, alpha=0.15, color=DEMAND_COLOR,
    )

    ax.set_ylabel("Supply / Demand")
    ax.set_title("Self-Sufficiency Ratio Over Time")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_daily_profile(step: AggregatedStep) -> Figure:
    """Average-day profile: hour-of-day vs. mean kWh for supply and demand."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    df = pd.DataFrame({
        "demand": step.demand_total,
        "supply": step.supply_total,
        "hour": step.timestamp.hour + step.timestamp.minute / 60,
    })
    grouped = df.groupby("hour").mean()

    ax.plot(grouped.index, grouped["demand"], color=DEMAND_COLOR, linewidth=1.5, label="Demand (avg)")
    ax.plot(grouped.index, grouped["supply"], color=SUPPLY_COLOR, linewidth=1.5, label="Supply (avg)")
    ax.fill_between(grouped.index, grouped["demand"], alpha=0.1, color=DEMAND_COLOR)
    ax.fill_between(grouped.index, grouped["supply"], alpha=0.1, color=SUPPLY_COLOR)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("kWh (average)")
    ax.set_title("Average Daily Profile")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_active_participants(step: AggregatedStep) -> Figure:
    """Stacked area of n_demanders and n_suppliers over time."""
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.fill_between(step.timestamp, step.n_demanders, alpha=0.5, color=DEMAND_COLOR, label="Demanders")
    ax.fill_between(step.timestamp, step.n_suppliers, alpha=0.5, color=SUPPLY_COLOR, label="Suppliers")

    ax.set_ylabel("Number of meters")
    ax.set_title("Active Participants Over Time")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_weekly_heatmap(step: AggregatedStep) -> Figure:
    """Day-of-week x hour heatmap of net balance (supply - demand)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    net = step.supply_total - step.demand_total
    df = pd.DataFrame({
        "net": net,
        "dow": step.timestamp.dayofweek,
        "hour": step.timestamp.hour,
    })
    pivot = df.pivot_table(values="net", index="dow", columns="hour", aggfunc="mean")

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
    )
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day")
    ax.set_title("Weekly Net Balance Heatmap (green = surplus, red = deficit)")
    fig.colorbar(im, ax=ax, label="Net kWh (supply − demand)")
    fig.tight_layout()
    return fig
