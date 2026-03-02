"""Core data types for the energy sharing simulation."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MeterTimeSeries:
    """Time series data for a single meter (prosumer or production asset).

    Uses numpy arrays for memory efficiency and vectorized operations.
    At scale (1000+ meters × 35k timesteps), this saves ~3-4× RAM vs Python lists.

    Attributes:
        meter_id: Unique identifier for the meter.
        timestamp: Datetime index of the time series (tz-aware UTC).
        value: Contiguous numpy array of float32 meter values.
               NaN values represent missing data (to be handled by fill policies).
        unit: Unit string (currently "kWh" only).

    Notes:
        - NaN is allowed and treated as "missing data" (not an error).
        - Downstream modules (fill policies, alignment) handle NaN as missing.
        - If you want to reject NaN, apply a fill policy or alignment with fill_method.
    """

    meter_id: str
    timestamp: pd.DatetimeIndex
    value: np.ndarray  # shape: (n_timesteps,), dtype: float32, NaN = missing
    unit: str

    def __post_init__(self):
        """Validate data consistency."""
        if self.unit.lower() not in {"kwh"}:
            raise ValueError(f"Unsupported unit: {self.unit}. Only 'kWh' is supported now.")

        # Ensure value is a numpy array of float32
        if not isinstance(self.value, np.ndarray):
            self.value = np.asarray(self.value, dtype=np.float32)
        elif self.value.dtype != np.float32:
            self.value = self.value.astype(np.float32)

        length = len(self.timestamp)
        if len(self.value) != length:
            raise ValueError(
                f"Length mismatch: {len(self.value)} values "
                f"but {length} timestamps"
            )

        # NaN is now allowed and treated as missing data.
        # No validation of NaN here — let downstream modules handle it.


@dataclass
class LoadedDataset:
    """Container for loaded and validated time series data.

    Attributes:
        prosumers: List of MeterTimeSeries objects (prosumer data).
        production_assets: List of MeterTimeSeries objects (production/generation data).
        timestamp_index: Common timezone-aware UTC DatetimeIndex across all data.
        timezone: Timezone string (currently always "UTC").
        metadata: Optional metadata dict (dataset name, source, processing notes, etc.).
    """

    prosumers: list[MeterTimeSeries] = field(default_factory=list)
    production_assets: list[MeterTimeSeries] = field(default_factory=list)
    timestamp_index: Optional[pd.DatetimeIndex] = None
    timezone: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def get_prosumer_ids(self) -> list[str]:
        """Return sorted list of all prosumer meter IDs."""
        return sorted([p.meter_id for p in self.prosumers])

    def get_asset_ids(self) -> list[str]:
        """Return sorted list of all production asset meter IDs."""
        return sorted([a.meter_id for a in self.production_assets])

    def __post_init__(self):
        """Validate dataset consistency."""
        if not self.prosumers and not self.production_assets:
            raise ValueError("Dataset must contain at least prosumers or production assets")


@dataclass
class SimulationConfig:
    """Configuration for a simulation run.

    Attributes:
        start: Simulation start timestamp (inclusive). Accepts str or pd.Timestamp.
        end: Simulation end timestamp (inclusive). Accepts str or pd.Timestamp.
        freq: Pandas frequency string (default '15min').
        allow_partial: If False, DatasetLoader will raise an error when any meter is
            missing data inside the requested period. If True, loader will log a
            coverage report but continue.
        align_to_clock: If True, reindex all series to the canonical simulation clock
            and apply fill_method to missing values. Only used when allow_partial=True.
        fill_method: How to fill missing values after alignment. Options: 'zero' (fill with 0),
            'ffill' (forward-fill), None (no filling, leave NaN). Default 'zero'.
    """

    start: pd.Timestamp | str
    end: pd.Timestamp | str
    freq: str = "15min"
    allow_partial: bool = False
    align_to_clock: bool = False
    fill_method: Optional[str] = "zero"  # 'zero', 'ffill', or None

    def to_index(self, tz: str = "UTC") -> pd.DatetimeIndex:
        """Return a timezone-aware DatetimeIndex for the configured period.

        The index will be created with closed='right' semantics consistent with
        P4 interval-end timestamps if callers require that convention. The
        caller is responsible for interpreting interval semantics.
        """
        start_ts = pd.to_datetime(self.start)
        end_ts = pd.to_datetime(self.end)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize(tz)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize(tz)
        return pd.date_range(start=start_ts, end=end_ts, freq=self.freq)


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

