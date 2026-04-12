"""Pure functions: pipeline result objects → DataFrames suitable for hvplot.

No Panel imports. All functions are independently testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Type-only imports — avoids pulling in simulation code at import time in tests
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core_types import AggregatedStep, AllocationResult, PricingResult


def make_supply_demand_df(step: AggregatedStep) -> pd.DataFrame:
    """Community-level supply and demand per timestep.

    Returns columns: timestamp, demand_kWh, supply_kWh, net_kWh
    net_kWh = supply - demand (positive = surplus, negative = deficit)
    """
    return pd.DataFrame(
        {
            "timestamp": step.timestamp,
            "demand_kWh": step.demand_total.astype(float),
            "supply_kWh": step.supply_total.astype(float),
            "net_kWh": (step.supply_total - step.demand_total).astype(float),
        }
    )


def make_allocation_df(allocation: AllocationResult) -> pd.DataFrame:
    """Community-level allocation and grid flows per timestep.

    Returns columns: timestamp, local_allocation_kWh, grid_import_kWh, grid_export_kWh
    """
    total_allocated = sum(
        allocation.allocations[m] for m in allocation.prosumer_ids
    ).astype(float)
    return pd.DataFrame(
        {
            "timestamp": allocation.timestamp,
            "local_allocation_kWh": total_allocated,
            "grid_import_kWh": allocation.grid_import.astype(float),
            "grid_export_kWh": allocation.grid_export.astype(float),
        }
    )


def make_efficiency_df(
    step: AggregatedStep, allocation: AllocationResult
) -> pd.DataFrame:
    """Self-sufficiency and self-consumption rates per timestep.

    Returns columns: timestamp, self_sufficiency_pct, self_consumption_pct
    Both in [0, 1]. Timesteps with zero demand/supply → 0 for the respective rate.
    """
    total_allocated = sum(
        allocation.allocations[m] for m in allocation.prosumer_ids
    ).astype(float)
    demand = step.demand_total.astype(float)
    supply = step.supply_total.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ss = np.where(demand > 0, total_allocated / demand, 0.0)
        sc = np.where(supply > 0, total_allocated / supply, 0.0)
    return pd.DataFrame({
        "timestamp": step.timestamp,
        "self_sufficiency_pct": ss,
        "self_consumption_pct": sc,
    })


def make_community_cost_df(pricing: PricingResult) -> pd.DataFrame:
    """Community-level cost per timestep.

    Returns columns: timestamp, cost_eur, cumulative_cost_eur
    """
    cost = pricing.total_local_cost_eur.astype(float)
    return pd.DataFrame(
        {
            "timestamp": pricing.timestamp,
            "cost_eur": cost,
            "cumulative_cost_eur": cost.cumsum(),
        }
    )


def make_prosumer_timeseries_df(
    allocation: AllocationResult, pricing: PricingResult
) -> pd.DataFrame:
    """Long-form per-prosumer time series.

    Returns columns: timestamp, meter_id, allocated_kWh, cost_eur
    One row per (timestep × prosumer).
    """
    rows = []
    for pid in allocation.prosumer_ids:
        alloc = allocation.allocations[pid].astype(float)
        cost = pricing.local_cost_eur[pid].astype(float)
        n = len(alloc)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": allocation.timestamp,
                    "meter_id": [pid] * n,
                    "allocated_kWh": alloc,
                    "cost_eur": cost,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def available_profiles(step: AggregatedStep) -> list[str]:
    """Return profile granularities that have at least 2 complete periods in the data.

    Checks: daily (≥2 days), weekly (≥2 weeks), yearly (≥2 years).
    """
    ts = pd.DatetimeIndex(step.timestamp)
    profiles = []
    if ts.normalize().nunique() >= 2:
        profiles.append("Daily")
    if ts.to_period("W").nunique() >= 2:
        profiles.append("Weekly")
    if ts.to_period("Y").nunique() >= 2:
        profiles.append("Yearly")
    return profiles


def make_avg_profile_df(step: AggregatedStep, profile: str) -> pd.DataFrame:
    """Average profile DataFrame for the given granularity.

    Daily   → columns: hour_of_day (0.0–23.75), demand_kWh, supply_kWh
    Weekly  → columns: week_position (0.0–~6.99, day + time/24), demand_kWh, supply_kWh
              15-min resolution; mean kWh per slot across all matching weeks.
    Yearly  → columns: day_of_year (1–365), demand_kWh, supply_kWh
              Input resampled to daily before averaging.
    """
    ts = pd.DatetimeIndex(step.timestamp)
    base = pd.DataFrame({
        "timestamp": ts,
        "demand_kWh": step.demand_total.astype(float),
        "supply_kWh": step.supply_total.astype(float),
    })

    if profile == "Daily":
        base["hour_of_day"] = ts.hour + ts.minute / 60
        return (
            base.groupby("hour_of_day")[["demand_kWh", "supply_kWh"]]
            .mean()
            .reset_index()
        )

    if profile == "Weekly":
        # Group by (day_of_week, 15-min slot) — mean kWh per slot across all matching weeks
        base["day_of_week"] = ts.dayofweek
        base["slot"] = ts.hour + ts.minute / 60          # 0.0, 0.25, …, 23.75
        base["week_position"] = base["day_of_week"] + base["slot"] / 24  # 0.0 … ~6.9948
        return (
            base.groupby("week_position")[["demand_kWh", "supply_kWh"]]
            .mean()
            .reset_index()
        )

    if profile == "Yearly":
        indexed = base.set_index("timestamp")[["demand_kWh", "supply_kWh"]]
        daily = indexed.resample("1D").sum().reset_index()
        daily["year"] = daily["timestamp"].dt.year
        daily["day_of_year"] = daily["timestamp"].dt.dayofyear
        return (
            daily.groupby("day_of_year")[["demand_kWh", "supply_kWh"]]
            .mean()
            .reset_index()
        )

    raise ValueError(f"Unknown profile: {profile!r}")


def resample_df(
    df: pd.DataFrame,
    freq: str,
    timestamp_col: str = "timestamp",
    agg: str = "sum",
) -> pd.DataFrame:
    """Resample a DataFrame to a lower frequency.

    Args:
        df: DataFrame with a timestamp column.
        freq: Pandas offset string, e.g. '1h', '1D', '1W'.
        timestamp_col: Name of the timestamp column.
        agg: Aggregation method — 'sum' or 'mean'.

    Returns:
        Resampled DataFrame with the same columns. The timestamp column
        contains period-start timestamps.
    """
    numeric_cols = [c for c in df.columns if c != timestamp_col]
    indexed = df.set_index(timestamp_col)
    resampler = indexed[numeric_cols].resample(freq)
    if agg == "sum":
        resampled = resampler.sum()
    elif agg == "mean":
        resampled = resampler.mean()
    else:
        raise ValueError(f"Unsupported agg={agg!r}. Use 'sum' or 'mean'.")
    return resampled.reset_index().rename(columns={"index": timestamp_col})
