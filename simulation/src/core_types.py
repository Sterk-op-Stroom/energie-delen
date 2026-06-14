"""Core data types for the energy sharing simulation."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .utils import ensure_array


@dataclass
class AggregatedStep:
    """Per-timestep aggregated supply and demand totals.

    Output of the Aggregator module; input to the Allocation module.

    Sign convention (aligned with loader):
        demand_total: sum of positive meter values per timestep (consumption, kWh)
        supply_total: sum of |negative meter values| per timestep (production available, kWh)
    Both arrays are non-negative float32.

    Attributes:
        timestamp: Timezone-aware UTC DatetimeIndex.
        demand_total: Total consumption per timestep (kWh), non-negative.
        supply_total: Total available production per timestep (kWh), non-negative.
        n_demanders: Number of meters with net demand (value > 0) per timestep.
        n_suppliers: Number of meters with net supply (value < 0) per timestep.
        unit: Unit of values (currently "kWh").
        freq: Frequency string (e.g. "15min"), or None if unknown.
        metadata: Optional metadata dict (prosumer/asset IDs, nan policy, etc.).
    """

    timestamp: pd.DatetimeIndex
    demand_total: np.ndarray  # float32, non-negative
    supply_total: np.ndarray  # float32, non-negative
    n_demanders: np.ndarray   # int32
    n_suppliers: np.ndarray   # int32
    unit: str = "kWh"
    freq: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.timestamp)
        for arr_name in ("demand_total", "supply_total", "n_demanders", "n_suppliers"):
            arr = getattr(self, arr_name)
            if len(arr) != n:
                raise ValueError(
                    f"Length mismatch: {arr_name} has {len(arr)} elements but timestamp has {n}"
                )
        self.demand_total = ensure_array(self.demand_total, np.float32)
        self.supply_total = ensure_array(self.supply_total, np.float32)
        self.n_demanders = ensure_array(self.n_demanders, np.int32)
        self.n_suppliers = ensure_array(self.n_suppliers, np.int32)

    def summary(self) -> str:
        return (
            f"AggregatedStep: {len(self.timestamp)} steps | "
            f"demand={float(self.demand_total.sum()):.1f} {self.unit} | "
            f"supply={float(self.supply_total.sum()):.1f} {self.unit}"
        )


@dataclass
class EnergyFlow:
    """A directional per-prosumer energy stream, ready for pricing.

    The common input to all PricingModel implementations. Builder functions
    in src/flows.py construct EnergyFlow objects from upstream pipeline results.

    Attributes:
        timestamp: Timezone-aware UTC DatetimeIndex.
        prosumer_ids: Ordered list of meter IDs.
        kwh: Per-prosumer non-negative kWh arrays, dict[meter_id → float32 (n_timesteps,)].
        direction: "demand" — energy consumed (cost to prosumer);
                   "supply" — energy produced (revenue for prosumer).
        flow_type: Stream identity. One of:
            "local_shared"          — allocated local energy (consumers)
            "grid_import"           — residual demand drawn from the grid
            "grid_export"           — residual supply exported to the grid
            "counterfactual_import" — full demand without any local sharing
            "counterfactual_export" — full supply without any local sharing
        freq: Pandas frequency string (e.g. "15min"), or None.
        metadata: Optional dict for strategy params, source info, etc.
    """

    VALID_DIRECTIONS = ("demand", "supply")
    VALID_FLOW_TYPES = (
        "local_shared",
        "grid_import",
        "grid_export",
        "counterfactual_import",
        "counterfactual_export",
    )

    timestamp: pd.DatetimeIndex
    prosumer_ids: list[str]
    kwh: dict[str, np.ndarray]
    direction: str
    flow_type: str
    freq: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.timestamp)
        if self.direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid direction={self.direction!r}. "
                f"Must be one of: {', '.join(self.VALID_DIRECTIONS)}"
            )
        if self.flow_type not in self.VALID_FLOW_TYPES:
            raise ValueError(
                f"Invalid flow_type={self.flow_type!r}. "
                f"Must be one of: {', '.join(self.VALID_FLOW_TYPES)}"
            )
        if not self.prosumer_ids:
            raise ValueError("EnergyFlow must have at least one prosumer_id.")
        missing = [m for m in self.prosumer_ids if m not in self.kwh]
        if missing:
            raise ValueError(
                f"{len(missing)} prosumer_id(s) have no entry in kwh: {missing[:5]}"
            )
        coerced: dict[str, np.ndarray] = {}
        for meter_id, arr in self.kwh.items():
            arr = ensure_array(arr, np.float32)
            if len(arr) != n:
                raise ValueError(
                    f"kwh[{meter_id!r}] has length {len(arr)}, expected {n}"
                )
            neg_mask = arr < 0
            if np.any(neg_mask):
                raise ValueError(
                    f"kwh[{meter_id!r}] contains negative values. EnergyFlow.kwh must be non-negative."
                )
            coerced[meter_id] = arr
        self.kwh = coerced


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
        - If you want to reject NaN, use missing_data='error' in SimulationConfig.
    """

    meter_id: str
    timestamp: pd.DatetimeIndex
    value: np.ndarray  # shape: (n_timesteps,), dtype: float32, NaN = missing
    unit: str

    def __post_init__(self):
        """Validate data consistency."""
        if self.unit.lower() not in {"kwh"}:
            raise ValueError(f"Unsupported unit: {self.unit}. Only 'kWh' is supported now.")

        self.value = ensure_array(self.value, np.float32)

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

    def summary(self) -> str:
        n = len(self.timestamp_index) if self.timestamp_index is not None else 0
        period = (
            f"{self.timestamp_index[0].date()} → {self.timestamp_index[-1].date()}"
            if n > 0 else "no index"
        )
        return (
            f"LoadedDataset: {len(self.prosumers)} prosumers | "
            f"{len(self.production_assets)} assets | "
            f"{n} timesteps ({period})"
        )


@dataclass
class SimulationConfig:
    """Configuration for a simulation run.

    Attributes:
        start: Simulation start timestamp (inclusive). Accepts str or pd.Timestamp.
        end: Simulation end timestamp (inclusive). Accepts str or pd.Timestamp.
        freq: Pandas frequency string (default '15min').
        missing_data: How to handle missing data during loading (loader stage).
            Controls both the gate (error vs. continue) and the fill strategy
            when aligning meters to the simulation clock. Default 'fill_zero'.
        nan_policy: How the aggregator handles NaN values when computing
            per-timestep totals (aggregation stage). Default 'treat_as_zero'.
        tz: Timezone for naive start/end strings (default 'UTC').
            Dutch meter data is typically CET/CEST — pass 'Europe/Amsterdam' if needed.

    missing_data options (loader stage — structural gaps):
        'error'        — Raise ValueError if any meter has missing data in the
                         simulation period. Use this when you need guaranteed
                         complete data (e.g. billing, regulatory reporting).
        'fill_zero'    — Align all meters to the simulation clock and fill gaps
                         with 0.0. Best default for most simulations: missing
                         meters are treated as having zero consumption/production.
        'fill_forward' — Align and forward-fill gaps (last known value carries
                         forward). Useful when meter data has small gaps and the
                         underlying signal is relatively stable.
        'keep_nan'     — Align to the simulation clock but leave gaps as NaN.
                         Downstream modules handle NaN explicitly via nan_policy.
                         Use when you want full control over how missing data
                         propagates through the pipeline.

    nan_policy options (aggregation stage — NaN in values):
        'treat_as_zero' — NaN meter values contribute 0 to demand/supply totals.
                          The meter is effectively absent for that timestep.
        'propagate'      — Any NaN meter at a timestep makes that timestep's
                          total NaN. Useful for detecting incomplete data in
                          aggregated results.

    Date format:
        Dates are parsed with dayfirst=True (Dutch convention: DD-MM-YYYY).
        "02-03-2025" is interpreted as 2 March 2025, not February 3rd.
        ISO format ("2025-03-02") and pd.Timestamp objects also work.

    Examples:
        # Strict: fail if any data is missing
        SimulationConfig(start="01-01-2025", end="07-01-2025", missing_data="error")

        # Default: fill gaps with zero, NaN treated as zero in aggregation
        SimulationConfig(start="01-01-2025", end="07-01-2025")

        # Keep NaN through loading, let aggregator propagate them
        SimulationConfig(
            start="01-01-2025", end="07-01-2025",
            missing_data="keep_nan", nan_policy="propagate",
        )
    """

    VALID_MISSING_DATA = ("error", "fill_zero", "fill_forward", "keep_nan")
    VALID_NAN_POLICY = ("treat_as_zero", "propagate")

    start: pd.Timestamp | str
    end: pd.Timestamp | str
    freq: str = "15min"
    missing_data: str = "fill_zero"
    nan_policy: str = "treat_as_zero"
    tz: str = "UTC"

    def __post_init__(self):
        if self.missing_data not in self.VALID_MISSING_DATA:
            raise ValueError(
                f"Invalid missing_data={self.missing_data!r}. "
                f"Must be one of: {', '.join(self.VALID_MISSING_DATA)}"
            )
        if self.nan_policy not in self.VALID_NAN_POLICY:
            raise ValueError(
                f"Invalid nan_policy={self.nan_policy!r}. "
                f"Must be one of: {', '.join(self.VALID_NAN_POLICY)}"
            )

    def to_index(self) -> pd.DatetimeIndex:
        """Return a timezone-aware DatetimeIndex for the configured period.

        Naive start/end timestamps are localized to self.tz (default "UTC").
        Timezone-aware timestamps are used as-is. Dutch meter data is typically
        in CET/CEST — pass tz="Europe/Amsterdam" to SimulationConfig if needed.

        Date strings are parsed with dayfirst=True (Dutch DD-MM-YYYY convention).
        """
        start_ts = pd.to_datetime(self.start, dayfirst=True)
        end_ts = pd.to_datetime(self.end, dayfirst=True)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize(self.tz)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize(self.tz)
        return pd.date_range(start=start_ts, end=end_ts, freq=self.freq)

    def summary(self) -> str:
        return (
            f"SimulationConfig: {self.start} → {self.end} | "
            f"freq={self.freq} | missing_data={self.missing_data} | nan_policy={self.nan_policy}"
        )


@dataclass
class AllocationResult:
    """Per-timestep allocation of local supply to each prosumer.

    Output of the Allocation module; direct input contract for the Pricing module.

    Sign convention: all arrays are non-negative kWh values.

    Constraints (enforced by allocation strategies, not by __post_init__):
        - allocations[meter_id][t] <= demand[meter_id][t]   (no over-allocation)
        - sum(allocations[:,t]) <= supply_total[t]          (no excess)
        - grid_import[t] >= 0, grid_export[t] >= 0

    NaN convention: NaN demand is treated as 0 (prosumer absent).
    NaN supply is treated as 0 (no local generation that timestep).

    Attributes:
        timestamp: Timezone-aware UTC DatetimeIndex.
        prosumer_ids: Ordered list of prosumer meter IDs.
        allocations: Per-prosumer allocated kWh —
            dict mapping meter_id → float32 array of shape (n_timesteps,).
        grid_import: Community-level unmet demand per timestep (kWh), non-negative.
            Equal to: demand_total − sum(allocations).
        grid_export: Community-level unallocated local supply per timestep (kWh), non-negative.
            Equal to: supply_total − sum(allocations).
        residual_demand: Per-prosumer unmet demand after allocation (kWh), non-negative —
            dict mapping meter_id → float32 array of shape (n_timesteps,).
            Populated by run_allocation(); empty dict if not computed.
        strategy: Name of the allocation strategy (e.g. "equal_allocation").
        unit: Unit of values (currently "kWh").
        freq: Frequency string (e.g. "15min"), or None if unknown.
        metadata: Optional metadata dict (strategy params, counts, etc.).
    """

    timestamp: pd.DatetimeIndex
    prosumer_ids: list[str]
    allocations: dict[str, np.ndarray]   # meter_id → float32 (n_timesteps,)
    grid_import: np.ndarray              # float32 (n_timesteps,), non-negative
    grid_export: np.ndarray              # float32 (n_timesteps,), non-negative
    strategy: str
    unit: str = "kWh"
    freq: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    residual_demand: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.timestamp)
        coerced: dict[str, np.ndarray] = {}
        for meter_id, arr in self.allocations.items():
            arr = ensure_array(arr, np.float32)
            if len(arr) != n:
                raise ValueError(
                    f"Allocation array for {meter_id!r} has length {len(arr)}, expected {n}"
                )
            coerced[meter_id] = arr
        self.allocations = coerced
        for arr_name in ("grid_import", "grid_export"):
            arr = ensure_array(getattr(self, arr_name), np.float32)
            if len(arr) != n:
                raise ValueError(f"{arr_name} has length {len(arr)}, expected {n}")
            setattr(self, arr_name, arr)
        if self.residual_demand:
            coerced_rd: dict[str, np.ndarray] = {}
            for meter_id, arr in self.residual_demand.items():
                arr = ensure_array(arr, np.float32)
                if len(arr) != n:
                    raise ValueError(
                        f"residual_demand[{meter_id!r}] has length {len(arr)}, expected {n}"
                    )
                coerced_rd[meter_id] = arr
            self.residual_demand = coerced_rd

    def summary(self) -> str:
        allocated = sum(a.sum() for a in self.allocations.values())
        return (
            f"AllocationResult: strategy={self.strategy} | "
            f"{len(self.prosumer_ids)} prosumers | "
            f"allocated={float(allocated):.1f} {self.unit} | "
            f"grid_import={float(self.grid_import.sum()):.1f} | "
            f"grid_export={float(self.grid_export.sum()):.1f}"
        )


@dataclass
class PricingResult:
    """Per-timestep energy charges for each prosumer.

    Output of the Pricing module. Can cover any energy stream (local allocated,
    grid import, grid export, counterfactual, etc.) as indicated by flow_type.

    Sign convention: all arrays are non-negative EUR values.

    Pricing invariants (enforced by pricing strategies):
        - cost_eur[meter_id][t] == kwh_priced[meter_id][t] * fixed_price
        - total_cost_eur[t] == sum(cost_eur[:,t])
        - Sign of costs follows sign of fixed_price (negative price = subsidy/rebate).

    NaN convention: NaN kWh values are treated as 0 (prosumer absent that timestep).

    Attributes:
        timestamp: Timezone-aware UTC DatetimeIndex.
        prosumer_ids: Ordered list of prosumer meter IDs.
        cost_eur: Per-prosumer energy charges —
            dict mapping meter_id → float32 array of shape (n_timesteps,).
        kwh_priced: Per-prosumer kWh that were priced —
            dict mapping meter_id → float32 array of shape (n_timesteps,).
        total_cost_eur: Community total energy charges per timestep (EUR).
        total_cost_eur_by_prosumer: Total charges per prosumer over all timesteps —
            dict mapping meter_id → float.
        fixed_price_eur_per_kwh: The fixed price applied (EUR/kWh).
        flow_type: The energy stream that was priced (e.g. "local_shared", "grid_import").
        strategy: Name of the pricing strategy (e.g. "fixed_price").
        unit: Unit of monetary values (currently "EUR").
        freq: Frequency string (e.g. "15min"), or None if unknown.
        metadata: Optional metadata dict.
    """

    timestamp: pd.DatetimeIndex
    prosumer_ids: list[str]
    cost_eur: dict[str, np.ndarray]              # meter_id → float32 (n_timesteps,)
    kwh_priced: dict[str, np.ndarray]            # meter_id → float32 (n_timesteps,)
    total_cost_eur: np.ndarray                   # float32 (n_timesteps,), non-negative
    total_cost_eur_by_prosumer: dict[str, float]
    fixed_price_eur_per_kwh: float
    flow_type: str
    strategy: str
    unit: str = "EUR"
    freq: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.timestamp)
        for field_name in ("cost_eur", "kwh_priced"):
            raw = getattr(self, field_name)
            coerced: dict[str, np.ndarray] = {}
            for meter_id, arr in raw.items():
                arr = ensure_array(arr, np.float32)
                if len(arr) != n:
                    raise ValueError(
                        f"{field_name}[{meter_id!r}] has length {len(arr)}, expected {n}"
                    )
                coerced[meter_id] = arr
            setattr(self, field_name, coerced)
        self.total_cost_eur = ensure_array(self.total_cost_eur, np.float32)
        if len(self.total_cost_eur) != n:
            raise ValueError(
                f"total_cost_eur has length {len(self.total_cost_eur)}, expected {n}"
            )

    def summary(self) -> str:
        total_eur = float(self.total_cost_eur.sum())
        total_kwh = sum(a.sum() for a in self.kwh_priced.values())
        return (
            f"PricingResult: strategy={self.strategy} | flow={self.flow_type} | "
            f"price={self.fixed_price_eur_per_kwh:.4f} EUR/kWh | "
            f"total={total_eur:.2f} EUR | "
            f"priced={float(total_kwh):.1f} kWh"
        )

