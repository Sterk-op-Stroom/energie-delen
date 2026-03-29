"""Metadata and reporting types for the energy sharing simulation.

These types describe data *about* the simulation (coverage, inspection results)
rather than data that flows *through* the pipeline. For pipeline contract types
(MeterTimeSeries, AggregatedStep, AllocationResult, PricingResult), see core_types.py.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .core_types import MeterTimeSeries


@dataclass
class CoverageReport:
    """Report describing coverage of meters against a SimulationConfig.

    Attributes:
        missing_prosumers: list of prosumer meter_ids missing any timestamps in the requested period
        missing_production_assets: list of asset meter_ids missing any timestamps in the requested period
        per_meter_full_span: dict mapping meter_id -> pd.Timedelta of the longest contiguous fully-covered span
        overall_longest_full_span: pd.Timedelta, maximum across all meters
        per_meter_missing_count: dict mapping meter_id -> count of missing timesteps in period
        per_meter_missing_fraction: dict mapping meter_id -> fraction [0,1] of missing timesteps
        per_meter_first_missing: dict mapping meter_id -> first timestamp with missing value (or None)
        per_meter_last_missing: dict mapping meter_id -> last timestamp with missing value (or None)
    """

    missing_prosumers: list[str] = field(default_factory=list)
    missing_production_assets: list[str] = field(default_factory=list)
    per_meter_full_span: dict[str, pd.Timedelta] = field(default_factory=dict)
    overall_longest_full_span: pd.Timedelta = pd.Timedelta(0)
    per_meter_missing_count: dict[str, int] = field(default_factory=dict)
    per_meter_missing_fraction: dict[str, float] = field(default_factory=dict)
    per_meter_first_missing: dict[str, Optional[pd.Timestamp]] = field(default_factory=dict)
    per_meter_last_missing: dict[str, Optional[pd.Timestamp]] = field(default_factory=dict)


@dataclass
class MeterInfo:
    """Per-meter summary from inspect_dataset."""

    meter_id: str
    role: str  # "prosumer" or "asset"
    start: pd.Timestamp
    end: pd.Timestamp
    n_points: int
    nan_fraction: float
    freq: str | None


@dataclass
class InspectResult:
    """Result of inspect_dataset(), usable from scripts and notebooks.

    Attributes:
        meters: List of MeterInfo summaries.
        global_start: Earliest timestamp across all meters.
        global_end: Latest timestamp across all meters.
        common_start: Latest start across all meters (overlap begin).
        common_end: Earliest end across all meters (overlap end).
        has_overlap: Whether common_start <= common_end.
        overlap_days: Duration of common overlap in days (0 if no overlap).
        frequencies: Dict mapping meter_id -> inferred freq string.
        freq_consistent: Whether all meters share the same frequency.
        suggested_start: Suggested --start for cli run (date string).
        suggested_end: Suggested --end for cli run (date string).
        suggested_freq: Suggested --freq for cli run.
        raw_meters: The raw MeterTimeSeries objects (for coverage plots).
    """

    meters: list[MeterInfo]
    global_start: pd.Timestamp
    global_end: pd.Timestamp
    common_start: pd.Timestamp
    common_end: pd.Timestamp
    has_overlap: bool
    overlap_days: float
    frequencies: dict[str, str | None]
    freq_consistent: bool
    suggested_start: str | None
    suggested_end: str | None
    suggested_freq: str
    raw_meters: list[MeterTimeSeries] = field(repr=False)
