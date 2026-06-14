"""Flow builder functions — extract EnergyFlow objects from pipeline results.

Each function takes existing pipeline types and returns a ready-to-price
EnergyFlow. The three principal stream pairs are:

  local_sharing      — energy allocated locally among prosumers (demand)
  residual           — the demand/supply remainder that flows via the grid
  counterfactual     — demand/supply as if no local sharing occurred
"""

import numpy as np

from .core_types import AggregatedStep, AllocationResult, EnergyFlow, LoadedDataset


def local_sharing_flow(allocation: AllocationResult) -> EnergyFlow:
    """Demand-side flow of locally allocated energy per prosumer."""
    return EnergyFlow(
        timestamp=allocation.timestamp,
        prosumer_ids=allocation.prosumer_ids,
        kwh={m: allocation.allocations[m].copy() for m in allocation.prosumer_ids},
        direction="demand",
        flow_type="local_shared",
        freq=allocation.freq,
        metadata={"allocation_strategy": allocation.strategy},
    )


def residual_import_flow(allocation: AllocationResult) -> EnergyFlow:
    """Per-prosumer unmet demand drawn from the grid after local sharing.

    Requires AllocationResult built by run_allocation() (residual_demand populated).
    """
    if not allocation.residual_demand:
        raise ValueError(
            "AllocationResult.residual_demand is empty. "
            "Use run_allocation() to build AllocationResult."
        )
    missing = [m for m in allocation.prosumer_ids if m not in allocation.residual_demand]
    if missing:
        raise ValueError(
            f"{len(missing)} prosumer_id(s) in prosumer_ids have no entry in "
            f"residual_demand: {missing[:5]}. Use run_allocation() to build AllocationResult."
        )
    return EnergyFlow(
        timestamp=allocation.timestamp,
        prosumer_ids=allocation.prosumer_ids,
        kwh={m: allocation.residual_demand[m].copy() for m in allocation.prosumer_ids},
        direction="demand",
        flow_type="grid_import",
        freq=allocation.freq,
        metadata={"allocation_strategy": allocation.strategy},
    )


def residual_export_flow(
    allocation: AllocationResult,
    dataset: LoadedDataset,
    step: AggregatedStep,
) -> EnergyFlow:
    """Per-producer residual supply exported to the grid after local sharing.

    Community grid_export is distributed pro-rata by each meter's original
    production contribution.
    """
    all_meters = {m.meter_id: m for m in dataset.prosumers + dataset.production_assets}
    supply_total = step.supply_total.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        export_frac = np.where(
            supply_total > 0,
            allocation.grid_export.astype(np.float64) / supply_total,
            0.0,
        )

    kwh: dict[str, np.ndarray] = {}
    for meter_id in allocation.prosumer_ids:
        meter = all_meters.get(meter_id)
        production = (
            np.clip(-meter.value, 0.0, None).astype(np.float64)
            if meter is not None
            else np.zeros(len(allocation.timestamp), dtype=np.float64)
        )
        kwh[meter_id] = (production * export_frac).astype(np.float32)

    return EnergyFlow(
        timestamp=allocation.timestamp,
        prosumer_ids=allocation.prosumer_ids,
        kwh=kwh,
        direction="supply",
        flow_type="grid_export",
        freq=allocation.freq,
        metadata={"allocation_strategy": allocation.strategy},
    )


def counterfactual_import_flow(
    allocation: AllocationResult,
    dataset: LoadedDataset,
) -> EnergyFlow:
    """Per-prosumer demand as if no local sharing took place.

    Each meter's counterfactual import = their full positive (consumption) values.
    """
    all_meters = {m.meter_id: m for m in dataset.prosumers + dataset.production_assets}
    kwh: dict[str, np.ndarray] = {}
    for meter_id in allocation.prosumer_ids:
        meter = all_meters.get(meter_id)
        kwh[meter_id] = (
            np.clip(meter.value, 0.0, None).astype(np.float32)
            if meter is not None
            else np.zeros(len(allocation.timestamp), dtype=np.float32)
        )
    return EnergyFlow(
        timestamp=allocation.timestamp,
        prosumer_ids=allocation.prosumer_ids,
        kwh=kwh,
        direction="demand",
        flow_type="counterfactual_import",
        freq=allocation.freq,
    )


def counterfactual_export_flow(
    allocation: AllocationResult,
    dataset: LoadedDataset,
) -> EnergyFlow:
    """Per-producer supply as if no local sharing took place.

    Each meter's counterfactual export = their full production (absolute negative values).
    """
    all_meters = {m.meter_id: m for m in dataset.prosumers + dataset.production_assets}
    kwh: dict[str, np.ndarray] = {}
    for meter_id in allocation.prosumer_ids:
        meter = all_meters.get(meter_id)
        kwh[meter_id] = (
            np.clip(-meter.value, 0.0, None).astype(np.float32)
            if meter is not None
            else np.zeros(len(allocation.timestamp), dtype=np.float32)
        )
    return EnergyFlow(
        timestamp=allocation.timestamp,
        prosumer_ids=allocation.prosumer_ids,
        kwh=kwh,
        direction="supply",
        flow_type="counterfactual_export",
        freq=allocation.freq,
    )
