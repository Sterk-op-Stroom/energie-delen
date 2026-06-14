"""Central application state shared across all dashboard pages."""

from __future__ import annotations

import param


class AppState(param.Parameterized):
    """Single source of truth for the dashboard.

    Pages read and write to this object. The param library makes all
    attributes observable — UI components can react to changes via
    @pn.depends / .param.watch without polling.

    Ownership:
        DataInputPage  → prosumer_files, production_files,
                         selected_prosumer_indices, selected_production_indices,
                         prosumer_path, production_path, inspect_result,
                         file_sets_version
        SimulationPage → start_date, end_date, freq, missing_data,
                         nan_policy, pricing_model, price_eur_per_kwh,
                         market_pricing_model, market_price_import_eur_per_kwh,
                         market_price_export_eur_per_kwh,
                         counterfactual_pricing_model,
                         counterfactual_price_import_eur_per_kwh,
                         counterfactual_price_export_eur_per_kwh,
                         pipeline, run_status
        ResultsPage    → reads pipeline (read-only)
    """

    # --- Uploaded file stacks (set by DataInputPage) ---
    # Each entry is a (Path, str) tuple: (temp_path, original_filename).
    # The two lists are independent; files are selected individually per role.
    prosumer_files = param.List(default=[], doc="Uploaded prosumer files as (Path, str) tuples")
    production_files = param.List(default=[], doc="Uploaded production files as (Path, str) tuples")

    # Indices into prosumer_files / production_files that are currently checked.
    # All checked files are merged and used together when running inspect or simulation.
    selected_prosumer_indices = param.List(default=[], doc="Checked indices into prosumer_files")
    selected_production_indices = param.List(default=[], doc="Checked indices into production_files")

    file_sets_version = param.Integer(default=0, doc="Incremented on every file-list change to trigger reactive re-renders")

    # --- Active data paths (set by DataInputPage._on_inspect, read by SimulationPage) ---
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
    pricing_model = param.Selector(
        default="fixed_price",
        objects=["fixed_price"],
        doc="Pricing strategy for locally shared energy",
    )
    price_eur_per_kwh = param.Number(default=0.075, doc="Fixed local price in EUR/kWh")
    market_pricing_model = param.Selector(
        default="none",
        objects=["none", "fixed_price"],
        doc="Pricing strategy for market (grid) flows",
    )
    market_price_import_eur_per_kwh = param.Number(default=0.25, doc="Market import price in EUR/kWh")
    market_price_export_eur_per_kwh = param.Number(default=0.09, doc="Feed-in tariff in EUR/kWh")
    counterfactual_pricing_model = param.Selector(
        default="none",
        objects=["none", "fixed_price"],
        doc="Pricing strategy for counterfactual (no local sharing) flows",
    )
    counterfactual_price_import_eur_per_kwh = param.Number(default=0.25, doc="Counterfactual import price in EUR/kWh")
    counterfactual_price_export_eur_per_kwh = param.Number(default=0.09, doc="Counterfactual feed-in tariff in EUR/kWh")

    # --- Pipeline output (set after a successful run) ---
    pipeline = param.Parameter(default=None, doc="PipelineResult | None")
    run_status = param.String(default="idle", doc="idle | running | done | error: ...")

    # --- Navigation ---
    active_page = param.Selector(
        default="data_input",
        objects=["data_input", "simulation", "results"],
    )
