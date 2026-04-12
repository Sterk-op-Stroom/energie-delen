"""Allocation module for the energy sharing simulation.

Distributes available local supply (kWh) among prosumers per timestep.

This module is intentionally pricing-free: it works in kWh only and has
no knowledge of tariffs or monetary values.

Sign convention (inherited from loader/aggregator):
    demand_matrix values are non-negative  (consumption)
    supply_total values are non-negative   (generation available for sharing)

NaN convention:
    NaN demand is treated as 0 — the prosumer does not participate that timestep.
    NaN supply is treated as 0 — no local generation available that timestep.
"""

import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from .core_types import AggregatedStep, AllocationResult, LoadedDataset
from .utils import infer_freq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class AllocationModel(ABC):
    """Abstract base class for allocation strategies.

    All strategies operate on plain numpy arrays. Project-specific types
    (LoadedDataset, AggregatedStep) are handled by run_allocation(), keeping
    the strategy interface independent of loader/aggregator internals.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier (e.g. 'equal_allocation')."""

    @abstractmethod
    def allocate(
        self,
        timestamp: pd.DatetimeIndex,
        prosumer_ids: list[str],
        demand_matrix: np.ndarray,
        supply_total: np.ndarray,
    ) -> AllocationResult:
        """Allocate available local supply among prosumers.

        Args:
            timestamp: UTC DatetimeIndex of length n_timesteps.
            prosumer_ids: Prosumer meter IDs, length n_prosumers.
            demand_matrix: Per-prosumer demand, shape (n_prosumers, n_timesteps).
                Non-negative float. NaN treated as 0.
            supply_total: Available local supply per timestep, shape (n_timesteps,).
                Non-negative float. NaN treated as 0.

        Returns:
            AllocationResult with per-prosumer allocations and grid import/export.

        Raises:
            ValueError: On dimension mismatch or empty inputs.
        """


# ---------------------------------------------------------------------------
# Shared input validation
# ---------------------------------------------------------------------------


def _validate_and_sanitize(
    timestamp: pd.DatetimeIndex,
    prosumer_ids: list[str],
    demand_matrix: np.ndarray,
    supply_total: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate dimensions and return sanitized (demand, supply) float32 arrays.

    NaN → 0, negative values → 0.

    Returns:
        (demand_matrix, supply_total) as float32 numpy arrays, NaN-free and clipped to ≥0.

    Raises:
        ValueError: On any dimension mismatch.
    """
    n_timesteps = len(timestamp)

    if demand_matrix.ndim != 2:
        raise ValueError(f"demand_matrix must be 2D, got shape {demand_matrix.shape}")

    n_p, n_t = demand_matrix.shape

    if n_t != n_timesteps:
        raise ValueError(
            f"demand_matrix has {n_t} timesteps but timestamp has {n_timesteps}"
        )
    if n_p != len(prosumer_ids):
        raise ValueError(
            f"demand_matrix has {n_p} rows but prosumer_ids has {len(prosumer_ids)} entries"
        )
    if len(supply_total) != n_timesteps:
        raise ValueError(
            f"supply_total has length {len(supply_total)}, expected {n_timesteps}"
        )

    demand = np.where(np.isnan(demand_matrix), 0.0, demand_matrix).astype(np.float32)
    supply = np.where(np.isnan(supply_total), 0.0, supply_total).astype(np.float32)
    demand = np.clip(demand, 0.0, None)
    supply = np.clip(supply, 0.0, None)
    return demand, supply


# ---------------------------------------------------------------------------
# Convenience function: extract inputs from project types
# ---------------------------------------------------------------------------


def run_allocation(
    dataset: LoadedDataset,
    step: AggregatedStep,
    model: AllocationModel,
) -> AllocationResult:
    """Extract allocation inputs from project types and run the allocation model.

    The allocation model itself only sees plain numpy arrays, so it stays
    decoupled from loader/aggregator internals. This function is the bridge.

    Args:
        dataset: Aligned LoadedDataset (loaded with a SimulationConfig).
        step: AggregatedStep produced by the aggregator.
        model: Allocation strategy to apply.

    Returns:
        AllocationResult.

    Raises:
        ValueError: If dataset has no prosumers or series lengths don't match step.
    """
    if not dataset.prosumers:
        raise ValueError("LoadedDataset contains no prosumers.")

    n_timesteps = len(step.timestamp)

    # Include both prosumers and production assets as potential demanders.
    # Production assets may have positive (consumption) values and should
    # compete equally for local supply under EqualAllocation.
    all_meters = dataset.prosumers + dataset.production_assets

    mismatched = [m.meter_id for m in all_meters if len(m.value) != n_timesteps]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} meter(s) have lengths != {n_timesteps}. "
            f"IDs: {mismatched[:5]}. Load with a SimulationConfig to align series."
        )

    prosumer_ids = [m.meter_id for m in all_meters]
    # Clip to [0, ∞): negative values are local generation already captured
    # in supply_total; we only want the consumption side here.
    demand_matrix = np.stack(
        [np.clip(m.value, 0.0, None) for m in all_meters], axis=0
    )  # (n_meters, n_timesteps)

    return model.allocate(
        timestamp=step.timestamp,
        prosumer_ids=prosumer_ids,
        demand_matrix=demand_matrix,
        supply_total=step.supply_total,
    )


# ---------------------------------------------------------------------------
# Strategy: Equal Allocation
# ---------------------------------------------------------------------------


class EqualAllocation(AllocationModel):
    """Equal absolute allocation with iterative redistribution under caps.

    Rule:
        At each timestep, available local supply is split equally among all
        active demanders (demand > 0). If any prosumer's equal share would
        exceed their actual demand, they are capped at their demand and the
        surplus is redistributed equally among the remaining active demanders.
        This continues until all supply is allocated or all demand is met.

    Properties:
        - Every active demander receives the same uncapped share as a starting point.
        - No prosumer receives more than their demand.
        - All available supply is allocated if total demand >= supply.
        - Prosumers with zero demand at a timestep are excluded from that timestep.

    NaN handling:
        NaN demand → treated as 0 (prosumer excluded from allocation).
        NaN supply → treated as 0 (no local energy available).
    """

    @property
    def name(self) -> str:
        return "equal_allocation"

    def allocate(
        self,
        timestamp: pd.DatetimeIndex,
        prosumer_ids: list[str],
        demand_matrix: np.ndarray,
        supply_total: np.ndarray,
    ) -> AllocationResult:
        demand, supply = _validate_and_sanitize(
            timestamp, prosumer_ids, demand_matrix, supply_total
        )
        n_prosumers = len(prosumer_ids)
        n_timesteps = len(timestamp)

        logger.info(
            "EqualAllocation: %d prosumers × %d timesteps", n_prosumers, n_timesteps
        )

        allocation_matrix = np.zeros((n_prosumers, n_timesteps), dtype=np.float32)
        remaining_supply = supply.copy()
        remaining_demand = demand.copy()

        # Relative tolerance to guard against float32 precision accumulation across
        # redistribution rounds. float32 epsilon is ~1.2e-7, so a threshold of 1e-9
        # falls below the noise floor for supply values near 1.0 kWh.
        # Scaling by the original supply magnitude keeps the threshold meaningful
        # regardless of the order of magnitude of the input.
        supply_tol = np.finfo(np.float32).eps * np.where(
            supply > 0, supply, np.finfo(np.float32).eps
        )

        # At most n_prosumers rounds: in the worst case one prosumer hits their
        # demand cap per round, so we converge in at most n_prosumers iterations.
        for _ in range(n_prosumers + 1):
            active = remaining_demand > 0                   # (n_prosumers, n_timesteps)
            n_active = active.sum(axis=0).astype(np.float32)  # (n_timesteps,)

            has_supply = remaining_supply > supply_tol
            has_demanders = n_active > 0

            if not (has_supply & has_demanders).any():
                break

            # Equal share per timestep; 0 where no supply or no active demanders
            per_person = np.where(
                has_supply & has_demanders,
                remaining_supply / np.where(n_active > 0, n_active, 1.0),
                0.0,
            )  # (n_timesteps,)

            # Each active prosumer gets min(their remaining demand, equal share)
            round_alloc = np.where(
                active,
                np.minimum(remaining_demand, per_person[np.newaxis, :]),
                0.0,
            )  # (n_prosumers, n_timesteps)

            allocation_matrix += round_alloc
            remaining_supply = np.clip(
                remaining_supply - round_alloc.sum(axis=0), 0.0, None
            )
            remaining_demand = np.clip(remaining_demand - round_alloc, 0.0, None)

        demand_total = demand.sum(axis=0)
        total_allocated = allocation_matrix.sum(axis=0)
        grid_import = np.clip(demand_total - total_allocated, 0.0, None).astype(np.float32)
        grid_export = np.clip(supply - total_allocated, 0.0, None).astype(np.float32)

        allocations = {
            meter_id: allocation_matrix[i]
            for i, meter_id in enumerate(prosumer_ids)
        }

        logger.info(
            "Allocation complete: avg allocated/step=%.3f kWh, "
            "avg grid_import=%.3f kWh, avg grid_export=%.3f kWh",
            float(total_allocated.mean()),
            float(grid_import.mean()),
            float(grid_export.mean()),
        )

        freq = infer_freq(timestamp)

        return AllocationResult(
            timestamp=timestamp,
            prosumer_ids=prosumer_ids,
            allocations=allocations,
            grid_import=grid_import,
            grid_export=grid_export,
            strategy=self.name,
            unit="kWh",
            freq=freq,
            metadata={
                "n_prosumers": n_prosumers,
                "prosumer_ids": prosumer_ids,
            },
        )


