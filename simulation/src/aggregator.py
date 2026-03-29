"""Aggregator module for the energy sharing simulation.

Takes a LoadedDataset and computes per-timestep totals for the allocation module:
  - Total demand  (sum of positive values across all meters)
  - Total supply  (sum of |negative values| across all meters)
  - Count of demanding and supplying meters per timestep

Sign convention (inherited from loader):
  positive value = consumption (demand)
  negative value = production (supply available for sharing)
"""

import logging
from enum import Enum
import numpy as np
import pandas as pd

from .core_types import AggregatedStep, LoadedDataset
from .utils import infer_freq

logger = logging.getLogger(__name__)


class NanPolicy(str, Enum):
    """Policy for handling NaN values during aggregation."""

    TREAT_AS_ZERO = "treat_as_zero"  # NaN → 0 (meter contributes nothing to totals)
    PROPAGATE = "propagate"          # NaN propagates through sums (any NaN → NaN in result)


class Aggregator:
    """Aggregates a LoadedDataset into per-timestep supply/demand totals.

    Processes all prosumers and production assets together. Each timestep produces:
      - demand_total: sum of positive meter values (consumption)
      - supply_total: sum of |negative meter values| (production for sharing)
      - n_demanders: count of meters with net demand (value > 0)
      - n_suppliers: count of meters with net supply (value < 0)

    NaN meters do not contribute to demand/supply totals (under 'treat_as_zero')
    and are never counted as demanders or suppliers.
    """

    def __init__(self, nan_policy: NanPolicy | str = NanPolicy.TREAT_AS_ZERO):
        """Initialize aggregator.

        Args:
            nan_policy: How to handle NaN values in meter series.
                Accepts NanPolicy enum or string ('treat_as_zero', 'propagate').
                'treat_as_zero': NaN counts as 0 (absent from totals).
                'propagate': any NaN at a timestep makes that timestep's total NaN.
        """
        if isinstance(nan_policy, str):
            nan_policy = NanPolicy(nan_policy)
        self.nan_policy = nan_policy

    def aggregate(self, dataset: LoadedDataset) -> AggregatedStep:
        """Aggregate a LoadedDataset into per-timestep supply/demand totals.

        All prosumers and production assets are aggregated together. The dataset
        should be aligned (same timestamps across all meters) before calling this
        method. Load with a SimulationConfig to ensure alignment.

        Args:
            dataset: Loaded and validated dataset.

        Returns:
            AggregatedStep with per-timestep totals and counts.

        Raises:
            ValueError: If dataset has no meters, or series have mismatched lengths.
        """
        all_series = dataset.prosumers + dataset.production_assets

        if not all_series:
            raise ValueError("Dataset contains no meters (prosumers or production assets)")

        # Canonical timestamp index
        timestamp: pd.DatetimeIndex
        if dataset.timestamp_index is not None:
            timestamp = dataset.timestamp_index
        else:
            timestamp = all_series[0].timestamp

        n_timesteps = len(timestamp)

        # Validate all series match the expected length
        mismatched = [s.meter_id for s in all_series if len(s.value) != n_timesteps]
        if mismatched:
            raise ValueError(
                f"Series length mismatch: {len(mismatched)} meter(s) have lengths "
                f"!= {n_timesteps}. Mismatched IDs: {mismatched[:5]}. "
                "Load with a SimulationConfig to align all series first."
            )

        logger.info(
            "Aggregating %d meters (%d prosumers, %d assets) over %d timesteps",
            len(all_series),
            len(dataset.prosumers),
            len(dataset.production_assets),
            n_timesteps,
        )

        # Stack into 2D matrix: (n_meters, n_timesteps)
        matrix = np.stack([s.value for s in all_series], axis=0)  # float32

        # Count demanders/suppliers on raw data (NaN > 0 → False, so NaN excluded)
        n_demanders = np.sum(matrix > 0, axis=0).astype(np.int32)
        n_suppliers = np.sum(matrix < 0, axis=0).astype(np.int32)

        # Apply NaN policy for totals
        if self.nan_policy == NanPolicy.TREAT_AS_ZERO:
            matrix = np.where(np.isnan(matrix), 0.0, matrix)
        # PROPAGATE: numpy sum naturally propagates NaN

        # Demand: sum of positive parts per timestep
        demand_total = np.sum(np.clip(matrix, 0, None), axis=0).astype(np.float32)

        # Supply: sum of absolute values of negative parts per timestep
        supply_total = np.sum(np.clip(-matrix, 0, None), axis=0).astype(np.float32)

        freq = infer_freq(timestamp)

        metadata = {
            "n_prosumers": len(dataset.prosumers),
            "n_assets": len(dataset.production_assets),
            "nan_policy": self.nan_policy.value,
            "prosumer_ids": dataset.get_prosumer_ids(),
            "asset_ids": dataset.get_asset_ids(),
        }

        logger.info(
            "Aggregation complete: avg demand/step=%.3f kWh, avg supply/step=%.3f kWh",
            float(np.nanmean(demand_total)),
            float(np.nanmean(supply_total)),
        )

        return AggregatedStep(
            timestamp=timestamp,
            demand_total=demand_total,
            supply_total=supply_total,
            n_demanders=n_demanders,
            n_suppliers=n_suppliers,
            unit="kWh",
            freq=freq,
            metadata=metadata,
        )


