"""Sample data generation for testing and demonstrations."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class SampleDataGenerator:
    """Generate synthetic time series data for testing."""

    @staticmethod
    def generate_prosumer_data(
        num_prosumers: int = 5,
        num_days: int = 7,
        start_date: str = "2025-01-01",
        seed: int = 42,
        output_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Generate synthetic prosumer demand data.

        Args:
            num_prosumers: Number of prosumers to generate.
            num_days: Number of days of data.
            start_date: Start date for the time series.
            seed: Random seed for reproducibility.
            output_file: If provided, save to Parquet file.

        Returns:
            DataFrame with columns: timestamp, meter_id, value
        """
        np.random.seed(seed)

        # Create 15-minute timestamp index
        timestamps = pd.date_range(
            start_date, periods=num_days * 96, freq="15min", tz="UTC"
        )

        data = []
        for prosumer_id in range(1, num_prosumers + 1):
            # Base demand pattern (higher during day, lower at night)
            hour_of_day = timestamps.hour + timestamps.minute / 60
            base_pattern = 0.5 + 0.4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            base_pattern = np.maximum(base_pattern, 0.1)  # Minimum demand

            # Random daily variation and noise
            daily_variation = np.random.gamma(2, 1, len(timestamps))
            noise = np.random.normal(0, 0.05, len(timestamps))

            demand = (base_pattern * daily_variation + noise) * 2.5  # Scale to 0-3.5 kW

            for timestamp, demand_val in zip(timestamps, demand):
                data.append({
                    "timestamp": timestamp,
                    "meter_id": f"p{prosumer_id:03d}",
                    "value": round(demand_val, 3),
                })

        df = pd.DataFrame(data)

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_file, index=False)
            print(f"Generated {len(df)} prosumer records -> {output_file}")

        return df

    @staticmethod
    def generate_production_data(
        num_assets: int = 2,
        num_days: int = 7,
        start_date: str = "2025-01-01",
        seed: int = 42,
        output_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Generate synthetic production asset data (e.g., PV systems).

        Args:
            num_assets: Number of production assets.
            num_days: Number of days of data.
            start_date: Start date for the time series.
            seed: Random seed for reproducibility.
            output_file: If provided, save to Parquet file.

        Returns:
            DataFrame with columns: timestamp, meter_id, value
        """
        np.random.seed(seed)

        # Create 15-minute timestamp index
        timestamps = pd.date_range(
            start_date, periods=num_days * 96, freq="15min", tz="UTC"
        )

        data = []
        asset_types = ["PV", "wind"]

        for asset_idx in range(1, num_assets + 1):
            asset_type = asset_types[(asset_idx - 1) % len(asset_types)]

            if asset_type == "PV":
                # PV production follows solar irradiance pattern
                hour_of_day = timestamps.hour + timestamps.minute / 60
                solar_pattern = np.maximum(
                    np.sin(np.pi * (hour_of_day - 6) / 12), 0
                )

                # Capacity: 5 kW per PV system
                capacity = 5.0

                # Cloud cover variation
                cloud_factor = np.random.gamma(2, 0.5, len(timestamps))
                cloud_factor = np.minimum(cloud_factor, 1.0)

                production = solar_pattern * capacity * cloud_factor

            else:  # Wind
                # Wind production is more random
                capacity = 10.0
                production = np.random.weibull(2, len(timestamps)) * capacity * 0.4
                production = np.minimum(production, capacity)

            for timestamp, prod_val in zip(timestamps, production):
                data.append({
                    "timestamp": timestamp,
                    "meter_id": f"a{asset_idx:02d}",
                    "value": round(prod_val, 3),
                })

        df = pd.DataFrame(data)

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_file, index=False)
            print(f"Generated {len(df)} production records -> {output_file}")

        return df

    @staticmethod
    def generate_sample_dataset(
        output_dir: Path,
        num_prosumers: int = 5,
        num_assets: int = 2,
        num_days: int = 7,
    ) -> tuple[Path, Path]:
        """Generate complete sample dataset (prosumers + assets).

        Args:
            output_dir: Directory to save generated files.
            num_prosumers: Number of prosumers.
            num_assets: Number of production assets.
            num_days: Number of days of data.

        Returns:
            Tuple of (prosumer_file_path, production_file_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prosumer_file = output_dir / "prosumers.parquet"
        production_file = output_dir / "production.parquet"

        SampleDataGenerator.generate_prosumer_data(
            num_prosumers=num_prosumers,
            num_days=num_days,
            output_file=prosumer_file,
        )

        SampleDataGenerator.generate_production_data(
            num_assets=num_assets,
            num_days=num_days,
            output_file=production_file,
        )

        return prosumer_file, production_file

