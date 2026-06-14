"""Pricing module for the energy sharing simulation.

Converts an EnergyFlow (kWh) into charges (EUR) per prosumer per timestep.
This module is strictly downstream of allocation: it must not recompute
allocation, access raw prosumer demand, or access raw production data.

Phase 1 scope: any energy stream represented as an EnergyFlow.
Explicit non-goals: dynamic pricing, market-indexed pricing, taxes, VAT,
network tariffs, member differentiation, full invoice logic.

NaN convention:
    NaN kWh values are treated as 0 — the prosumer did not receive/produce
    energy that timestep and incurs no charge.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from .core_types import EnergyFlow, PricingResult
from .utils import infer_freq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class PricingModel(ABC):
    """Abstract base class for pricing strategies.

    All strategies operate on an EnergyFlow and return a PricingResult.
    The interface is intentionally minimal so alternative pricing rules
    (market-linked, cost-plus, differentiated) can be added later by
    subclassing without touching simulation orchestration code.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier (e.g. 'fixed_price')."""

    @abstractmethod
    def price(self, flow: EnergyFlow) -> PricingResult:
        """Convert an EnergyFlow into a PricingResult.

        Args:
            flow: EnergyFlow describing the energy stream to price.

        Returns:
            PricingResult with per-prosumer energy charges.

        Raises:
            ValueError: On invalid configuration or malformed flow fields.
        """


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_pricing(flow: EnergyFlow, model: PricingModel) -> PricingResult:
    """Apply a pricing model to an EnergyFlow.

    This thin wrapper exists for symmetry with run_allocation() and to give
    orchestration code a single consistent call pattern.

    Args:
        flow: EnergyFlow from the flows module.
        model: Pricing strategy to apply.

    Returns:
        PricingResult.
    """
    return model.price(flow)


# ---------------------------------------------------------------------------
# Strategy: Fixed Price
# ---------------------------------------------------------------------------


class FixedPricePricing(PricingModel):
    """Fixed price per kWh of energy in the flow.

    Rule:
        For every timestep, each prosumer pays:
            cost_eur[meter_id][t] = flow.kwh[meter_id][t] * fixed_price

    The price is uniform across all prosumers and all timesteps.

    Phase 1 scope:
        - Prices any energy stream represented as an EnergyFlow.
        - No standing charges, VAT, taxes, or network fees.

    Args:
        fixed_price_eur_per_kwh: Price per kWh of energy, in EUR.
            May be negative (e.g. to model a subsidy or rebate).

    """

    def __init__(self, fixed_price_eur_per_kwh: float) -> None:
        self._price = fixed_price_eur_per_kwh

    @property
    def name(self) -> str:
        return "fixed_price"

    def price(self, flow: EnergyFlow) -> PricingResult:
        n_timesteps = len(flow.timestamp)
        prosumer_ids = flow.prosumer_ids

        logger.info(
            "FixedPricePricing: %d prosumers × %d timesteps @ %.4f EUR/kWh (flow=%s)",
            len(prosumer_ids), n_timesteps, self._price, flow.flow_type,
        )

        kwh_priced: dict[str, np.ndarray] = {}
        cost_eur: dict[str, np.ndarray] = {}
        total_cost_eur_by_prosumer: dict[str, float] = {}

        for meter_id in prosumer_ids:
            kwh = flow.kwh[meter_id].copy()
            # NaN → 0: absent prosumer incurs no charge
            kwh = np.where(np.isnan(kwh), 0.0, kwh).astype(np.float32)
            cost = (kwh * self._price).astype(np.float32)
            kwh_priced[meter_id] = kwh
            cost_eur[meter_id] = cost
            total_cost_eur_by_prosumer[meter_id] = float(cost.sum())

        total_cost_eur = np.zeros(n_timesteps, dtype=np.float32)
        for meter_id in prosumer_ids:
            total_cost_eur += cost_eur[meter_id]

        freq = flow.freq or infer_freq(flow.timestamp)

        logger.info(
            "Pricing complete: total=%s %.2f EUR, avg/timestep=%.4f EUR",
            flow.flow_type,
            float(total_cost_eur.sum()),
            float(total_cost_eur.mean()),
        )

        return PricingResult(
            timestamp=flow.timestamp,
            prosumer_ids=prosumer_ids,
            cost_eur=cost_eur,
            kwh_priced=kwh_priced,
            total_cost_eur=total_cost_eur,
            total_cost_eur_by_prosumer=total_cost_eur_by_prosumer,
            fixed_price_eur_per_kwh=self._price,
            flow_type=flow.flow_type,
            strategy=self.name,
            unit="EUR",
            freq=freq,
            metadata={
                "n_prosumers": len(prosumer_ids),
                "prosumer_ids": prosumer_ids,
                "direction": flow.direction,
                **flow.metadata,
            },
        )
