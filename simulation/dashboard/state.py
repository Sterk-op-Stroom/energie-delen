"""Central application state shared across all dashboard pages."""

from __future__ import annotations

import param


class AppState(param.Parameterized):
    """Single source of truth for the dashboard.

    Pages read and write to this object. The param library makes all
    attributes observable — UI components can react to changes via
    @pn.depends / .param.watch without polling.

    Ownership:
        DataInputPage  → prosumer_path, production_path, inspect_result
        SimulationPage → start_date, end_date, freq, missing_data,
                         nan_policy, price_eur_per_kwh, pipeline, run_status
        ResultsPage    → reads pipeline (read-only)
    """

    # --- Data paths (set by DataInputPage) ---
    prosumer_path = param.Parameter(default=None, doc="Path to prosumer Parquet file or folder")
    production_path = param.Parameter(default=None, doc="Path to production Parquet file or folder")
    inspect_result = param.Parameter(default=None, doc="InspectResult from cli.inspect_dataset()")
    inspect_status = param.String(default="idle", doc="idle | loading | done | error: ...")

    # --- Simulation config (set by SimulationPage widgets) ---
    start_date = param.String(default="", doc="Simulation start in DD-MM-YYYY format")
    end_date = param.String(default="", doc="Simulation end in DD-MM-YYYY format")
    freq = param.String(default="15min", doc="Pandas frequency string, e.g. '15min', '30S'")
    freq_infer = param.Boolean(default=False, doc="Infer frequency from data instead of manual entry")
    missing_data = param.Selector(
        default="fill_zero",
        objects=["fill_zero", "fill_forward", "keep_nan", "error"],
    )
    nan_policy = param.Selector(
        default="treat_as_zero",
        objects=["treat_as_zero", "propagate"],
    )
    price_eur_per_kwh = param.Number(default=0.075, doc="Fixed local price in EUR/kWh")

    # --- Pipeline output (set after a successful run) ---
    pipeline = param.Parameter(default=None, doc="PipelineResult | None")
    run_status = param.String(default="idle", doc="idle | running | done | error: ...")

    # --- Navigation ---
    active_page = param.Selector(
        default="data_input",
        objects=["data_input", "simulation", "results"],
    )
