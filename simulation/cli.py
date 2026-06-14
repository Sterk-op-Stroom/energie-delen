"""CLI entrypoint for the energy sharing simulation."""

import argparse
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.aggregator import Aggregator
from src.allocation import EqualAllocation, run_allocation
from src.core_types import (
    AggregatedStep,
    AllocationResult,
    LoadedDataset,
    MeterTimeSeries,
    PricingResult,
    SimulationConfig,
)
from src.flows import (
    counterfactual_export_flow,
    counterfactual_import_flow,
    local_sharing_flow,
    residual_export_flow,
    residual_import_flow,
)
from src.loader import DatasetLoader, MeterLoader
from src.pricing import FixedPricePricing, run_pricing
from src.report_types import InspectResult, MeterInfo
from src.sample_data import SampleDataGenerator
from src.utils import infer_freq


class PipelineResult(NamedTuple):
    dataset: LoadedDataset
    step: AggregatedStep
    allocation: AllocationResult
    pricing: PricingResult
    config: SimulationConfig
    pricing_market_import: PricingResult | None = None
    pricing_market_export: PricingResult | None = None
    pricing_counterfactual_import: PricingResult | None = None
    pricing_counterfactual_export: PricingResult | None = None

    def __str__(self) -> str:
        lines = [
            self.config.summary(),
            self.dataset.summary(),
            self.step.summary(),
            self.allocation.summary(),
            self.pricing.summary(),
        ]
        if self.pricing_market_import is not None:
            lines.append(self.pricing_market_import.summary())
        if self.pricing_market_export is not None:
            lines.append(self.pricing_market_export.summary())
        if self.pricing_counterfactual_import is not None:
            lines.append(self.pricing_counterfactual_import.summary())
        if self.pricing_counterfactual_export is not None:
            lines.append(self.pricing_counterfactual_export.summary())
        return "\n".join(lines)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_pipeline(
    start: str,
    end: str,
    prosumer_path: Path | None = None,
    production_path: Path | None = None,
    freq: str = "15min",
    missing_data: str = "fill_zero",
    nan_policy: str = "treat_as_zero",
    price_eur_per_kwh: float = 0.075,
    market_price_import_eur_per_kwh: float | None = None,
    market_price_export_eur_per_kwh: float | None = None,
    counterfactual_price_import_eur_per_kwh: float | None = None,
    counterfactual_price_export_eur_per_kwh: float | None = None,
) -> PipelineResult:
    """Run the full simulation pipeline: load → aggregate → allocate → price."""
    if not prosumer_path and not production_path:
        raise ValueError("Provide at least prosumer_path or production_path")

    # --- Load ---
    print("\n[1/4] Loading data...")
    loader = DatasetLoader()
    simulation_config = SimulationConfig(
        start=start,
        end=end,
        freq=freq,
        missing_data=missing_data,
        nan_policy=nan_policy,
    )
    dataset, _ = loader.load(
        prosumer_data_path=prosumer_path,
        production_data_path=production_path,
        simulation_config=simulation_config,
    )
    n_prosumers = len(dataset.prosumers)
    n_assets = len(dataset.production_assets)
    n_timesteps = len(dataset.timestamp_index) if dataset.timestamp_index is not None else 0
    print(f"  {n_prosumers} prosumers, {n_assets} assets, {n_timesteps} timesteps")

    # --- Aggregate ---
    print("\n[2/4] Aggregating supply and demand...")
    aggregator = Aggregator(nan_policy=simulation_config.nan_policy)
    step = aggregator.aggregate(dataset)

    demand_sum = float(step.demand_total.sum())
    supply_sum = float(step.supply_total.sum())
    ratio = supply_sum / demand_sum if demand_sum > 0 else np.inf
    print(f"  Total demand:  {demand_sum:.2f} kWh")
    print(f"  Total supply:  {supply_sum:.2f} kWh")
    print(f"  Supply/demand: {ratio:.2f}")

    # --- Allocate ---
    print("\n[3/4] Allocating local supply (strategy: equal)...")
    allocation = run_allocation(dataset, step, EqualAllocation())

    total_alloc = sum(allocation.allocations[m] for m in allocation.prosumer_ids)
    local_fraction = float(total_alloc.sum()) / supply_sum if supply_sum > 0 else 0.0
    print(f"  Locally allocated: {total_alloc.sum():.2f} kWh ({local_fraction:.1%} of supply)")
    print(f"  Grid import:       {allocation.grid_import.sum():.2f} kWh")
    print(f"  Grid export:       {allocation.grid_export.sum():.2f} kWh")

    # --- Price local sharing ---
    print(f"\n[4/4] Pricing at {price_eur_per_kwh:.4f} EUR/kWh...")
    flow = local_sharing_flow(allocation)
    result = run_pricing(flow, FixedPricePricing(price_eur_per_kwh))

    community_total = float(result.total_cost_eur.sum())
    total_kwh = sum(result.kwh_priced[m].sum() for m in result.prosumer_ids)
    print(f"  Community local charge: {community_total:.2f} EUR")
    print(f"  Total kWh priced:      {total_kwh:.2f} kWh")

    # --- Per-prosumer summary ---
    print("\nPer-prosumer summary:")
    for pid in result.prosumer_ids[:5]:
        kwh = result.kwh_priced[pid].sum()
        eur = result.total_cost_eur_by_prosumer[pid]
        print(f"  {pid}: {kwh:.2f} kWh -> {eur:.2f} EUR")
    if len(result.prosumer_ids) > 5:
        print(f"  ... and {len(result.prosumer_ids) - 5} more")

    # --- Market pricing (residual grid flows) ---
    pricing_market_import = None
    pricing_market_export = None

    if market_price_import_eur_per_kwh is not None or market_price_export_eur_per_kwh is not None:
        print("\n[4b] Pricing market (grid) flows...")

    if market_price_import_eur_per_kwh is not None:
        import_flow = residual_import_flow(allocation)
        pricing_market_import = run_pricing(
            import_flow, FixedPricePricing(market_price_import_eur_per_kwh)
        )
        print(f"  Grid import charge:    {float(pricing_market_import.total_cost_eur.sum()):.2f} EUR")

    if market_price_export_eur_per_kwh is not None:
        export_flow = residual_export_flow(allocation, dataset, step)
        pricing_market_export = run_pricing(
            export_flow, FixedPricePricing(market_price_export_eur_per_kwh)
        )
        print(f"  Grid export revenue:   {float(pricing_market_export.total_cost_eur.sum()):.2f} EUR")

    # --- Counterfactual pricing (as-if no local sharing) ---
    pricing_counterfactual_import = None
    pricing_counterfactual_export = None

    if counterfactual_price_import_eur_per_kwh is not None or counterfactual_price_export_eur_per_kwh is not None:
        print("\n[4c] Pricing counterfactual (normale energieleverancier)...")

    if counterfactual_price_import_eur_per_kwh is not None:
        cf_import_flow = counterfactual_import_flow(allocation, dataset)
        pricing_counterfactual_import = run_pricing(
            cf_import_flow, FixedPricePricing(counterfactual_price_import_eur_per_kwh)
        )
        print(f"  Counterfactual import: {float(pricing_counterfactual_import.total_cost_eur.sum()):.2f} EUR")

    if counterfactual_price_export_eur_per_kwh is not None:
        cf_export_flow = counterfactual_export_flow(allocation, dataset)
        pricing_counterfactual_export = run_pricing(
            cf_export_flow, FixedPricePricing(counterfactual_price_export_eur_per_kwh)
        )
        print(f"  Counterfactual export: {float(pricing_counterfactual_export.total_cost_eur.sum()):.2f} EUR")

    print("\nDone.")

    return PipelineResult(
        dataset, step, allocation, result, simulation_config,
        pricing_market_import, pricing_market_export,
        pricing_counterfactual_import, pricing_counterfactual_export,
    )


def _find_longest_complete_period(
    raw_meters: list[MeterTimeSeries],
    freq: str,
    common_start: pd.Timestamp,
    common_end: pd.Timestamp,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the day-aligned start/end of the longest contiguous period where
    all meters have non-NaN data at every timestep.

    Uses a shared DatetimeIndex and vectorised numpy AND — O(n_meters × n_steps),
    no Python-level loops over timestamps.
    """
    if not raw_meters:
        return None, None

    idx = pd.date_range(start=common_start, end=common_end, freq=freq)
    if len(idx) == 0:
        return None, None

    # Build one boolean valid-mask per meter on the shared index, then AND them.
    all_valid = np.ones(len(idx), dtype=bool)
    for m in raw_meters:
        s = pd.Series(m.value.astype(float), index=m.timestamp)
        reindexed = s.reindex(idx)
        all_valid &= ~np.isnan(reindexed.values)

    if not all_valid.any():
        return None, None

    # Find the longest contiguous True run with numpy diff.
    padded = np.empty(len(all_valid) + 2, dtype=bool)
    padded[0] = False
    padded[1:-1] = all_valid
    padded[-1] = False
    diffs = np.diff(padded.view(np.int8))
    run_starts = np.where(diffs == 1)[0]
    run_ends = np.where(diffs == -1)[0]  # exclusive
    lengths = run_ends - run_starts
    best = int(np.argmax(lengths))

    best_start = idx[run_starts[best]]
    best_end = idx[run_ends[best] - 1]

    # Align to day boundaries, keeping only fully-covered days.
    step = pd.Timedelta(freq)
    # day_start: ceil to midnight so partial first days are excluded.
    day_start = best_start.normalize()
    if best_start != day_start:
        day_start += pd.Timedelta(days=1)
    # day_end: only include a day if its last expected timestep is covered.
    day_end = best_end.normalize()
    last_ts_of_day = day_end + pd.Timedelta(days=1) - step
    if best_end < last_ts_of_day:
        day_end -= pd.Timedelta(days=1)

    if day_start > day_end:
        return None, None
    return day_start, day_end


def inspect_dataset(
    prosumer_path: Path | None = None,
    production_path: Path | None = None,
) -> InspectResult:
    """Inspect raw meter data and return structured metadata.

    Loads data without requiring start/end/freq parameters, so users can
    determine the correct values from the result.

    Example (script / notebook)::

        from cli import inspect_dataset
        from pathlib import Path

        info = inspect_dataset(
            prosumer_path=Path("data/prosumers.parquet"),
            production_path=Path("data/production.parquet"),
        )
        print(info.suggested_start, info.suggested_end, info.suggested_freq)
        for m in info.meters:
            print(f"{m.meter_id}: {m.n_points} points, {m.nan_fraction:.1%} NaN")
    """
    if not prosumer_path and not production_path:
        raise ValueError("Provide at least prosumer_path or production_path")

    loader = MeterLoader()
    raw_meters: list[MeterTimeSeries] = []
    meter_infos: list[MeterInfo] = []
    frequencies: dict[str, str | None] = {}

    for path, role in [(prosumer_path, "prosumer"), (production_path, "asset")]:
        if path is None:
            continue
        meters, _meta = loader.load(path, unit="kWh")
        for m in meters:
            n = len(m.value)
            nan_frac = float(np.isnan(m.value).sum() / n) if n > 0 else 0.0
            freq = infer_freq(m.timestamp)
            frequencies[m.meter_id] = freq

            meter_infos.append(MeterInfo(
                meter_id=m.meter_id,
                role=role,
                start=m.timestamp.min(),
                end=m.timestamp.max(),
                n_points=n,
                nan_fraction=nan_frac,
                freq=freq,
            ))
            # Tag role for downstream coverage plots
            m._role = role  # type: ignore[attr-defined]
            raw_meters.append(m)

    if not raw_meters:
        raise ValueError("No meters found in the provided paths")

    all_starts = [m.start for m in meter_infos]
    all_ends = [m.end for m in meter_infos]
    global_start = min(all_starts)
    global_end = max(all_ends)
    common_start = max(all_starts)
    common_end = min(all_ends)
    has_overlap = common_start <= common_end
    overlap_days = (common_end - common_start).total_seconds() / 86400 if has_overlap else 0.0

    unique_freqs = set(f for f in frequencies.values() if f is not None)
    freq_consistent = len(unique_freqs) <= 1
    suggested_freq = next(iter(unique_freqs)) if unique_freqs else "15min"

    if has_overlap:
        complete_start, complete_end = _find_longest_complete_period(
            raw_meters, suggested_freq, common_start, common_end
        )
        if complete_start is not None and complete_end is not None:
            suggested_start = complete_start.strftime("%d-%m-%Y")
            suggested_end = complete_end.strftime("%d-%m-%Y")
        else:
            suggested_start = common_start.normalize().strftime("%d-%m-%Y")
            suggested_end = common_end.normalize().strftime("%d-%m-%Y")
    else:
        suggested_start = None
        suggested_end = None

    return InspectResult(
        meters=sorted(meter_infos, key=lambda m: m.meter_id),
        frequencies=frequencies,
        freq_consistent=freq_consistent,
        suggested_start=suggested_start,
        suggested_end=suggested_end,
        suggested_freq=suggested_freq,
        raw_meters=raw_meters,
    )


def cmd_run(args) -> None:
    """Run the full simulation pipeline."""
    prosumer_path = Path(args.prosumers) if args.prosumers else None
    production_path = Path(args.production) if args.production else None

    if not prosumer_path and not production_path:
        print("Error: Provide at least --prosumers or --production")
        return

    if not args.start or not args.end:
        print("Error: --start and --end are required")
        return

    try:
        run_pipeline(
            prosumer_path=prosumer_path,
            production_path=production_path,
            start=args.start,
            end=args.end,
            freq=args.freq,
            missing_data=args.missing_data,
            nan_policy=args.nan_policy,
            price_eur_per_kwh=args.price,
        )
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


def cmd_inspect(args) -> None:
    """Inspect dataset to help choose simulation parameters."""
    prosumer_path = Path(args.prosumers) if args.prosumers else None
    production_path = Path(args.production) if args.production else None

    try:
        info = inspect_dataset(prosumer_path, production_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # --- Print summary ---
    n_prosumers = sum(1 for m in info.meters if m.role == "prosumer")
    n_assets = sum(1 for m in info.meters if m.role == "asset")
    if n_prosumers:
        print(f"\nProsumers: {n_prosumers} meter(s)")
    if n_assets:
        print(f"Production assets: {n_assets} meter(s)")

    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    global_start = min(m.start for m in info.meters)
    global_end = max(m.end for m in info.meters)
    print(f"  Total meters:     {len(info.meters)}")
    print(f"  Global range:     {global_start}  to  {global_end}")
    if info.suggested_start is None:
        print("  WARNING: No complete overlap — meters have disjoint or fully-NaN time ranges!")
    else:
        sug_start = pd.to_datetime(info.suggested_start, dayfirst=True)
        sug_end = pd.to_datetime(info.suggested_end, dayfirst=True)
        print(f"  Complete period:  {info.suggested_start}  to  {info.suggested_end}  ({(sug_end - sug_start).days} days)")

    if info.freq_consistent:
        print(f"  Frequency:        {info.suggested_freq} (consistent)")
    else:
        print(f"  Frequencies:      {dict(sorted(info.frequencies.items()))} (INCONSISTENT)")

    # --- Per-meter table ---
    print(f"\n{'='*60}")
    print("PER-METER DETAILS")
    print(f"{'='*60}")
    print(f"  {'meter_id':<15} {'role':<10} {'start':<25} {'end':<25} {'points':>7} {'NaN%':>6} {'freq':<8}")
    print(f"  {'-'*15} {'-'*10} {'-'*25} {'-'*25} {'-'*7} {'-'*6} {'-'*8}")

    for m in info.meters:
        print(
            f"  {m.meter_id:<15} {m.role:<10} "
            f"{str(m.start):<25} {str(m.end):<25} "
            f"{m.n_points:>7} {m.nan_fraction * 100:>5.1f}% {m.freq or 'unknown':<8}"
        )

    # --- Suggested parameters ---
    print(f"\n{'='*60}")
    print("SUGGESTED PARAMETERS")
    print(f"{'='*60}")
    if info.suggested_start:
        print(f"  --start {info.suggested_start}")
        print(f"  --end   {info.suggested_end}")
        print(f"  --freq  {info.suggested_freq}")
        print(f"\n  Full command:")
        parts = ["  uv run python -m cli run"]
        if prosumer_path:
            parts.append(f"--prosumers {prosumer_path}")
        if production_path:
            parts.append(f"--production {production_path}")
        parts.append(f"--start {info.suggested_start}")
        parts.append(f"--end {info.suggested_end}")
        parts.append(f"--freq {info.suggested_freq}")
        print(" \\\n    ".join(parts))
    else:
        print("  Cannot suggest parameters — no common time overlap across meters.")



def cmd_generate(args) -> None:
    """Generate sample data for testing."""
    print("Generating sample dataset...")
    print(f"  Prosumers: {args.prosumers}, Assets: {args.assets}, Days: {args.days}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    prosumer_file, production_file = SampleDataGenerator.generate_sample_dataset(
        output_dir=output_dir,
        num_prosumers=args.prosumers,
        num_assets=args.assets,
        num_days=args.days,
    )

    print(f"  Prosumers:  {prosumer_file}")
    print(f"  Production: {production_file}")
    print("Done.")


def cmd_demo(args) -> None:
    """Run a complete demo: generate sample data, then run the full pipeline."""
    print("Running full demo...\n")

    demo_dir = Path(args.demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample data...")
    prosumer_file, production_file = SampleDataGenerator.generate_sample_dataset(
        output_dir=demo_dir,
        num_prosumers=5,
        num_assets=2,
        num_days=7,
    )
    print(f"  {prosumer_file.name}, {production_file.name}")

    # Infer date range from generated data
    df = pd.read_parquet(prosumer_file)
    start = str(df["timestamp"].min().date())
    end = str(df["timestamp"].max().date())

    run_pipeline(
        prosumer_path=prosumer_file,
        production_path=production_file,
        start=start,
        end=end,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Energy sharing simulation - modular P2P energy allocation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect dataset to choose parameters
  python -m cli inspect --prosumers data/prosumers.parquet --production data/production.parquet

  # Run full pipeline (dates in DD-MM-YYYY format)
  python -m cli run --prosumers data/prosumers.parquet --production data/production.parquet --start 01-01-2025 --end 07-01-2025

  # Generate sample data
  python -m cli generate --prosumers 10 --assets 3 --days 30

  # Run complete demo (generate + run)
  python -m cli demo --demo-dir /tmp/demo
""",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Inspect command
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect dataset: time ranges, frequencies, coverage, suggested parameters"
    )
    inspect_parser.add_argument("--prosumers", type=str, help="Path to prosumer data file/folder")
    inspect_parser.add_argument("--production", type=str, help="Path to production data file/folder")
    inspect_parser.set_defaults(func=cmd_inspect)

    # Run command
    run_parser = subparsers.add_parser(
        "run", help="Run full pipeline: load -> aggregate -> allocate -> price"
    )
    run_parser.add_argument("--prosumers", type=str, help="Path to prosumer data file/folder")
    run_parser.add_argument("--production", type=str, help="Path to production data file/folder")
    run_parser.add_argument("--start", type=str, help="Simulation start date (DD-MM-YYYY, e.g., '01-01-2025')")
    run_parser.add_argument("--end", type=str, help="Simulation end date (DD-MM-YYYY, e.g., '07-01-2025')")
    run_parser.add_argument("--freq", type=str, default="15min", help="Frequency (default: 15min)")
    run_parser.add_argument(
        "--missing-data",
        choices=["error", "fill_zero", "fill_forward", "keep_nan"],
        default="fill_zero",
        help="How to handle missing data (default: fill_zero)",
    )
    run_parser.add_argument(
        "--nan-policy",
        choices=["treat_as_zero", "propagate"],
        default="treat_as_zero",
        help="NaN handling in aggregator (default: treat_as_zero)",
    )
    run_parser.add_argument(
        "--price",
        type=float,
        default=0.075,
        help="Fixed local price in EUR/kWh (default: 0.075)",
    )
    run_parser.set_defaults(func=cmd_run)

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate sample data for testing")
    gen_parser.add_argument("--prosumers", type=int, default=5, help="Number of prosumers")
    gen_parser.add_argument("--assets", type=int, default=2, help="Number of production assets")
    gen_parser.add_argument("--days", type=int, default=7, help="Number of days of data")
    gen_parser.add_argument("--output", type=str, default="data", help="Output directory")
    gen_parser.set_defaults(func=cmd_generate)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Generate sample data and run full pipeline")
    demo_parser.add_argument("--demo-dir", type=str, default="demo_data", help="Demo output directory")
    demo_parser.set_defaults(func=cmd_demo)

    args = parser.parse_args()

    setup_logging(args.verbose)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
