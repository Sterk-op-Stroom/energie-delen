"""Pricing module for the energy sharing simulation.

Converts already-allocated local energy (kWh) into charges (EUR) per prosumer
per timestep. This module is strictly downstream of allocation: it must not
recompute allocation, access raw prosumer demand, or access raw production data.

Phase 1 scope: local allocated energy only.
Explicit non-goals: grid import pricing, grid export compensation, dynamic
pricing, market-indexed pricing, taxes, VAT, network tariffs, member
differentiation, full invoice logic.

NaN convention:
    NaN allocations are treated as 0 — the prosumer did not receive local
    energy that timestep and incurs no local charge.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .core_types import AllocationResult, PricingResult
from .utils import infer_freq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class PricingModel(ABC):
    """Abstract base class for pricing strategies.

    All strategies operate on an AllocationResult and return a PricingResult.
    The interface is intentionally minimal so alternative pricing rules
    (market-linked, cost-plus, differentiated) can be added later by
    subclassing without touching simulation orchestration code.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier (e.g. 'fixed_price')."""

    @abstractmethod
    def price(self, allocation: AllocationResult) -> PricingResult:
        """Convert an AllocationResult into a PricingResult.

        Args:
            allocation: Output of the allocation module for the period.

        Returns:
            PricingResult with per-prosumer local energy charges.

        Raises:
            ValueError: On invalid configuration or malformed allocation fields.
        """


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_pricing(allocation: AllocationResult, model: PricingModel) -> PricingResult:
    """Apply a pricing model to an AllocationResult.

    This thin wrapper exists for symmetry with run_allocation() and to give
    orchestration code a single consistent call pattern.

    Args:
        allocation: AllocationResult from the allocation module.
        model: Pricing strategy to apply.

    Returns:
        PricingResult.
    """
    return model.price(allocation)


# ---------------------------------------------------------------------------
# Strategy: Fixed Price
# ---------------------------------------------------------------------------


class FixedPricePricing(PricingModel):
    """Fixed local price per kWh of allocated local energy.

    Rule:
        For every timestep, each prosumer pays:
            local_cost_eur[meter_id][t] = allocations[meter_id][t] * fixed_price

    The price is uniform across all prosumers and all timesteps.

    Phase 1 scope:
        - Prices local allocated energy only.
        - Does not price grid import or grid export.
        - No standing charges, VAT, taxes, or network fees.

    Args:
        fixed_price_eur_per_kwh: Price per kWh of locally allocated energy, in EUR.
            May be negative (e.g. to model a subsidy or rebate).

    """

    def __init__(self, fixed_price_eur_per_kwh: float) -> None:
        self._price = fixed_price_eur_per_kwh

    @property
    def name(self) -> str:
        return "fixed_price"

    def price(self, allocation: AllocationResult) -> PricingResult:
        _validate_allocation(allocation)

        n_timesteps = len(allocation.timestamp)
        prosumer_ids = allocation.prosumer_ids

        logger.info(
            "FixedPricePricing: %d prosumers x %d timesteps @ %.4f EUR/kWh",
            len(prosumer_ids),
            n_timesteps,
            self._price,
        )

        local_kwh_priced: dict[str, np.ndarray] = {}
        local_cost_eur: dict[str, np.ndarray] = {}
        total_local_cost_eur_by_prosumer: dict[str, float] = {}

        for meter_id in prosumer_ids:
            kwh = allocation.allocations[meter_id].copy()
            # NaN → 0: absent prosumer incurs no charge
            kwh = np.where(np.isnan(kwh), 0.0, kwh).astype(np.float32)
            cost = (kwh * self._price).astype(np.float32)
            local_kwh_priced[meter_id] = kwh
            local_cost_eur[meter_id] = cost
            total_local_cost_eur_by_prosumer[meter_id] = float(cost.sum())

        total_local_cost_eur = np.zeros(n_timesteps, dtype=np.float32)
        for meter_id in prosumer_ids:
            total_local_cost_eur += local_cost_eur[meter_id]

        freq = infer_freq(allocation.timestamp)

        logger.info(
            "Pricing complete: total community local charge=%.2f EUR, "
            "avg per timestep=%.4f EUR",
            float(total_local_cost_eur.sum()),
            float(total_local_cost_eur.mean()),
        )

        return PricingResult(
            timestamp=allocation.timestamp,
            prosumer_ids=prosumer_ids,
            local_cost_eur=local_cost_eur,
            local_kwh_priced=local_kwh_priced,
            total_local_cost_eur=total_local_cost_eur,
            total_local_cost_eur_by_prosumer=total_local_cost_eur_by_prosumer,
            fixed_price_eur_per_kwh=self._price,
            strategy=self.name,
            unit="EUR",
            freq=freq,
            metadata={
                "n_prosumers": len(prosumer_ids),
                "prosumer_ids": prosumer_ids,
                "allocation_strategy": allocation.strategy,
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_allocation(allocation: AllocationResult) -> None:
    """Raise ValueError if the allocation result is malformed."""
    n = len(allocation.timestamp)
    if not allocation.prosumer_ids:
        raise ValueError("AllocationResult contains no prosumer_ids.")
    missing = [m for m in allocation.prosumer_ids if m not in allocation.allocations]
    if missing:
        raise ValueError(
            f"{len(missing)} prosumer_id(s) in prosumer_ids have no entry in "
            f"allocations: {missing[:5]}"
        )
    for meter_id in allocation.prosumer_ids:
        arr = allocation.allocations[meter_id]
        if len(arr) != n:
            raise ValueError(
                f"allocations[{meter_id!r}] has length {len(arr)}, "
                f"expected {n} (timestamp length)"
            )
        neg_mask = arr < 0
        if np.any(neg_mask):
            raise ValueError(
                f"allocations[{meter_id!r}] contains negative values "
                f"(first at index {int(np.argmax(neg_mask))}: {arr[np.argmax(neg_mask)]:.6f}). "
                "AllocationResult must be non-negative."
            )
    for arr_name in ("grid_import", "grid_export"):
        arr = getattr(allocation, arr_name)
        neg_mask = arr < 0
        if np.any(neg_mask):
            raise ValueError(
                f"AllocationResult.{arr_name} contains negative values "
                f"(first at index {int(np.argmax(neg_mask))}: {arr[np.argmax(neg_mask)]:.6f}). "
                "Must be non-negative."
            )


