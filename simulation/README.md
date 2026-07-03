# Energie Delen — Simulatie

A simulation framework for peer-to-peer energy sharing in Dutch energy cooperatives. It takes meter data from households and local renewable production, and models what happens when members share that energy with each other before drawing from the grid.

## What it does

Energy sharing communities pool the output of local solar panels (or other renewable assets) and distribute it among members who need it at that moment. The simulation runs through historical meter data, timestep by timestep, and answers:

- How much local energy was available, and how much was consumed?
- Who got what share of that local energy, and when?
- What did it cost — compared to buying everything from the grid?
- How self-sufficient was the community overall?

It doesn't set policy — it shows you the consequences of the policies you choose.

## Core concepts

**Prosumers** are the households or members of the cooperative. They consume energy (and may also produce it via their own panels). The simulation tracks their demand from meter data.

**Production assets** are shared renewable sources — typically a communal solar installation or wind turbine — whose output belongs to the community as a whole. Their meter data is tracked separately from prosumers.

**Allocation** is the rule for distributing local production among prosumers at each moment. The current rule splits available supply equally among active consumers, capping each member at their actual demand and redistributing any unclaimed surplus to others.

**Local pricing** converts the allocated kWh into costs. A fixed rate (€/kWh) is applied uniformly. This is where you set the local energy price — typically lower than the market tariff.

**Market pricing** (optional) adds grid costs on top: what members pay for energy they couldn't get locally, and what the cooperative earns from exporting surplus back to the grid.

## How it works

Each timestep (typically every 15 minutes):

1. **Load** — Read prosumer demand and production asset output from your data files.
2. **Aggregate** — Sum total community demand and total available local supply.
3. **Allocate** — Distribute local supply among prosumers using the chosen rule.
4. **Price** — Apply the pricing model to calculate costs per prosumer.

Energy that can't be covered locally is imported from the grid (residual demand). Local supply that exceeds total demand is exported to the grid (surplus). The simulation tracks five flows per prosumer per timestep: local energy shared, grid import, grid export, and counterfactual import/export (what would have happened without local sharing).

## How to use it

### Option 1: Dashboard (recommended)

The dashboard is the main interface. No coding required.

First make sure you have a local copy of the repository by [downloading](https://github.com/Sterk-op-Stroom/energie-delen/archive/refs/heads/main.zip) the project and unpacking the zip. Then inside the simulation folder:

**Windows:** double-click `launch_dashboard.bat`  
**macOS / Linux:** run `./launch_dashboard.sh`

The dashboard has three pages:

**1 · Data Input** — Load your prosumer and production data files (Parquet format). The tool inspects what's there — meter IDs, time ranges, data gaps, coverage — and pre-fills suggested simulation parameters. You can also generate sample data to try things out first without any real data.

**2 · Simulatie** — Set the date range, time resolution (e.g. 15min), local price, and optionally market import/export prices. Choose how to handle missing data. Click Run. A log shows progress, and a summary of key figures appears when it's done.

**3 · Resultaten** — Interactive charts for energy flows, self-sufficiency, costs, and per-prosumer breakdowns. Adjust the time resolution (raw, hourly, daily) with the aggregation selector. Download the prosumer summary as CSV.

See [`docs/DASHBOARD.md`](docs/DASHBOARD.md) for a full page-by-page walkthrough.

### Option 2: Command line

```bash
# Try it with generated sample data
uv run python -m cli demo --demo-dir /tmp/demo

# Inspect your data before running
uv run python -m cli inspect \
  --prosumers data/prosumers.parquet \
  --production data/production.parquet

# Run a simulation
uv run python -m cli run \
  --prosumers data/prosumers.parquet \
  --production data/production.parquet \
  --start 01-01-2025 --end 31-01-2025 \
  --price 0.075
```

For Python integration, see the API reference (coming soon).

## What you need

Your data must be in Parquet format with three columns: `timestamp` (UTC), `meter_id` (string), `value` (kWh).

- Prosumer files: positive values = consumption
- Production asset files: positive values = output (the simulation negates these internally)

Files can be a single Parquet file or a folder of Parquet files — the loader discovers them automatically.

See [`docs/data_formats.md`](docs/data_formats.md) for the full specification, including how to handle multiple meters per file, missing data, and pseudonymization.

## Privacy

Pseudonymize meter IDs before loading real data. See [`docs/PRIVACY.md`](docs/PRIVACY.md) for guidance on GDPR compliance and the Dutch regulatory context.

## Installation

The dashboar launchers take care of everything you need. If you want to use it through the command line you will have to ensure yourself that requirements are installed/

Requires Python 3.13+ and [uv](https://astral.sh/uv).

```bash
cd simulation
uv sync --all-groups
```

## License

- Code (`src/`): AGPL-3.0-or-later
- Documentation: CC-BY-4.0
