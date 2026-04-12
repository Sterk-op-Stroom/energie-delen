# Getting Started

## I want to use my own data for the simulation

1. **Prepare your Parquet files:** see `docs/data_formats.md` for the required schema and sign convention.
2. **Pseudonymize prosumer IDs:** see `docs/PRIVACY.md` before loading real data.
3. **Run the simulation:** using the dashboard (see `docs/DASHBOARD.md`), the Command Line Interface (see .../cli.py) or the python API (see .../cli.py)

## I want to use the dashboard

The dashboard is the easiest way to run simulations.

**Windows:** double-click `launch_dashboard.bat`
**macOS / Linux:** run `./launch_dashboard.sh`

The browser opens automatically. Follow the three-step workflow:
1. **Data Input** — load your Parquet files (or generate sample data) and inspect coverage
2. **Simulation** — configure start/end date, frequency, pricing, and click Run
3. **Results** — explore interactive charts and download CSV exports

See `docs/DASHBOARD.md` for a full walkthrough.
