"""Data loader module for prosumer and production asset time series.

This module handles:
1. Loading Parquet files from single files or directories
2. Schema validation (timestamps, columns, data types)
3. Timezone handling
4. Missing value handling policies
5. Value sign convention: positive values represent consumption, negative values represent production
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .core_types import LoadedDataset, MeterTimeSeries, SimulationConfig
from .report_types import CoverageReport

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates Parquet file schemas for prosumers and production assets."""

    # Both prosumers and production assets use the same tidy schema
    TIDY_REQUIRED_COLUMNS = {"timestamp", "value", "meter_id"}

    @classmethod
    def validate_tidy_schema(cls, df: pd.DataFrame, file_path: Optional[str] = None) -> None:
        """Validate tidy meter-style data schema (timestamp, meter_id, value).

        Args:
            df: DataFrame loaded from Parquet.
            file_path: Path to the file (for error messages).

        Raises:
            ValueError: If schema is invalid.
        """
        missing_cols = cls.TIDY_REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            msg = f"Tidy meter data missing columns: {missing_cols}"
            if file_path:
                msg += f" (file: {file_path})"
            raise ValueError(msg)

        # Check timestamp column
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            raise ValueError(
                f"Column 'timestamp' must be datetime type, got {df['timestamp'].dtype}"
            )

        # Check value column is numeric
        if not pd.api.types.is_numeric_dtype(df["value"]):
            raise ValueError(
                f"Column 'value' must be numeric, got {df['value'].dtype}"
            )


class TimestampValidator:
    """Validates timestamp consistency across time series."""

    @classmethod
    def validate_timestamps(
        cls,
        timestamps: pd.DatetimeIndex,
        series_id: Optional[str] = None,
        freq: Optional[str] = None,
    ) -> Tuple[pd.DatetimeIndex, Optional[str]]:
        """Validate timestamps: check for duplicates, ensure timezone, infer frequency.

        This is a raw validation step — it warns on gaps but never errors on them.
        Missing-data policy decisions are made later by DatasetLoader using
        SimulationConfig.missing_data.

        Args:
            timestamps: The timestamp index to validate.
            series_id: Identifier of the series (for logging).
            freq: Expected frequency (e.g., "15min"). If None, will infer from data.

        Returns:
            Tuple of (validated_timestamps, inferred_or_provided_freq).

        Raises:
            ValueError: If timestamps are empty, contain duplicates, or frequency
                cannot be inferred.
        """
        if len(timestamps) == 0:
            raise ValueError("Timestamp index is empty")

        # Check for duplicates
        if timestamps.duplicated().any():
            dup_count = timestamps.duplicated().sum()
            msg = f"Found {dup_count} duplicate timestamps"
            if series_id:
                msg += f" in {series_id}"
            raise ValueError(msg)

        # Check for timezone
        if timestamps.tz is None:
            logger.warning("Timestamps have no timezone info; assuming UTC")
            timestamps = timestamps.tz_localize("UTC")

        # Infer frequency if not provided
        inferred_freq = freq
        if freq is None:
            inferred_freq = cls._infer_frequency(timestamps, series_id)
            logger.info(f"Inferred frequency {inferred_freq} from timestamps")

        # Check for gaps (warn only — policy decisions happen in DatasetLoader)
        gaps = cls._find_gaps(timestamps, freq=inferred_freq)
        if gaps:
            gap_msg = f"Found {len(gaps)} gaps in timestamps"
            if series_id:
                gap_msg += f" in {series_id}"
            logger.warning(gap_msg)

        return timestamps, inferred_freq

    @classmethod
    def _infer_frequency(cls, timestamps: pd.DatetimeIndex, series_id: Optional[str] = None) -> str:
        """Infer frequency from timestamp deltas with safeguards.

        Args:
            timestamps: The timestamp index.
            series_id: Identifier of the series (for logging).

        Returns:
            Inferred frequency string (e.g., "15min", "h", "D").

        Raises:
            ValueError: If frequency cannot be inferred reliably.
        """
        if len(timestamps) < 2:
            raise ValueError(f"Cannot infer frequency from single timestamp (series_id={series_id})")

        sorted_ts = timestamps.sort_values()
        deltas = sorted_ts.to_series().diff()[1:]  # Skip first NaT

        # Check if all deltas are the same (constant frequency)
        unique_deltas = deltas.unique()
        if len(unique_deltas) == 1:
            delta = unique_deltas[0]
        else:
            # Multiple different deltas: infer from most common
            delta_counts = deltas.value_counts()
            most_common_delta = delta_counts.index[0]
            common_count = delta_counts.iloc[0]
            total_count = len(deltas)
            coverage = common_count / total_count

            if coverage < 0.9:  # Less than 90% regular spacing
                msg = f"Frequency inconsistent: only {coverage*100:.1f}% of intervals match most common delta"
                if series_id:
                    msg += f" (series_id={series_id})"
                raise ValueError(msg)

            logger.warning(f"Frequency inferred from {coverage*100:.1f}% of intervals (has gaps)")
            delta = most_common_delta

        # Convert Timedelta to frequency string
        # Common intervals: 15min, 30min, 1H, 1D, etc.
        total_seconds = int(delta.total_seconds())

        if total_seconds % 86400 == 0:  # Daily or longer
            days = total_seconds // 86400
            return f"{days}D"
        elif total_seconds % 3600 == 0:  # Hourly
            hours = total_seconds // 3600
            return f"{hours}h"
        elif total_seconds % 60 == 0:  # Minutes
            minutes = total_seconds // 60
            return f"{minutes}min"
        else:
            return f"{total_seconds}s"

    @classmethod
    def _find_gaps(cls, timestamps: pd.DatetimeIndex, freq: str) -> list[tuple]:
        """Find gaps in timestamp sequence.

        Args:
            timestamps: The timestamp index.
            freq: Expected frequency (e.g., "15min").

        Returns:
            List of (gap_start, gap_end, gap_size_minutes) tuples.
        """
        if len(timestamps) < 2:
            return []

        gaps = []
        sorted_ts = timestamps.sort_values()
        diffs = sorted_ts.to_series().diff()

        # Parse frequency string to Timedelta
        # Create a dummy DatetimeIndex and check the inferred frequency
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
            expected_delta = pd.Timedelta(offset)
        except Exception:
            # Fallback: try to parse directly
            expected_delta = pd.Timedelta(freq)

        gap_mask = diffs > expected_delta

        for idx in diffs[gap_mask].index:
            gap_start = sorted_ts[sorted_ts.get_loc(idx) - 1]
            gap_end = idx
            gap_minutes = int((idx - gap_start).total_seconds() / 60)
            gaps.append((gap_start, gap_end, gap_minutes))

        return gaps

class MeterLoader:
    """Loader for time series data for meter id's.

    Loads raw meter data from Parquet files. Validates schema and timestamps
    but does not enforce missing-data policies — that is handled by
    DatasetLoader using SimulationConfig.missing_data.
    """

    def __init__(self):
        self.validator = SchemaValidator()
        self.ts_validator = TimestampValidator()

    def load(self, path: Path, unit: str, pattern: str = "*.parquet", freq: Optional[str] = None) -> Tuple[list[MeterTimeSeries], dict]:
        """Load from a file or folder, auto-detecting which.

        If path is a file: loads that single Parquet file.
        If path is a folder: loads all matching Parquet files from the folder.

        Args:
            path: Path to file or folder.
            unit: Unit of the values (e.g., "kWh").
            pattern: Glob pattern for folder matching (default: *.parquet).
                     Ignored if path is a file.
            freq: Expected frequency (e.g., "15min"). If None, will infer from data.

        Returns:
            Tuple of (list of MeterTimeSeries, metadata dict with inferred frequencies).

        Raises:
            ValueError: If path doesn't exist or is invalid.
        """
        path = Path(path)

        if path.is_file():
            logger.info(f"Path is file: {path}")
            return self._load_file(path, unit=unit, freq=freq)
        elif path.is_dir():
            logger.info(f"Path is folder: {path}")
            return self._load_folder(path, unit=unit, pattern=pattern, freq=freq)
        else:
            raise ValueError(f"Path does not exist: {path}")

    def _load_file(self, file_path: Path, unit: str, freq: Optional[str] = None) -> Tuple[list[MeterTimeSeries], dict]:
        """Load from a single Parquet file.

        Expected format: tidy structure with columns {timestamp, meter_id, value}.
        Multiple meter ids can be in the same file (grouped by meter_id).


        Returns raw series with native timestamps (one meter may have different
        timestamp coverage than another). No canonical reindexing here.

        Args:
            file_path: Path to Parquet file.
            unit: Unit of the values (e.g., "kWh").
            freq: Expected frequency. If None, will infer from data.

        Returns:
            Tuple of (list of MeterTimeSeries, metadata dict).
        """
        logger.info(f"Loading meters from {file_path}")

        df = pd.read_parquet(file_path)
        self.validator.validate_tidy_schema(df, str(file_path))

        meters = []
        metadata = {"inferred_frequencies": {}}
        inferred_freq = freq

        for meter_id, group in df.groupby("meter_id"):
            # Sort by timestamp
            group = group.sort_values("timestamp").reset_index(drop=True)

            # Validate timestamps and infer freq if needed
            timestamps = pd.DatetimeIndex(group["timestamp"])
            timestamps, inferred_freq_for_meter = self.ts_validator.validate_timestamps(
                timestamps, series_id=f"meter {meter_id}", freq=freq or inferred_freq
            )

            # Store inferred frequency for this meter
            metadata["inferred_frequencies"][str(meter_id)] = inferred_freq_for_meter

            # Update global inferred_freq from first meter if not provided
            if inferred_freq is None and freq is None:
                inferred_freq = inferred_freq_for_meter

            # No reindexing here — return each meter on its native timestamps
            meter = MeterTimeSeries(
                meter_id=str(meter_id),
                timestamp=timestamps,
                value=group["value"].to_numpy(dtype="float32"),
                unit=unit,
            )
            meters.append(meter)
            logger.debug(f"Loaded meter {meter_id}: {len(timestamps)} timesteps at freq {inferred_freq_for_meter}")

        return meters, metadata

    def _load_folder(self, folder_path: Path, unit: str, pattern: str = "*.parquet", freq: Optional[str] = None) -> Tuple[list[MeterTimeSeries], dict]:
        """Load from all Parquet files in a folder.

        Each meter retains its native timestamps. No alignment across files.
        Frequency consistency is still validated (all meters must have same freq).

        Args:
            folder_path: Path to folder containing Parquet files.
            unit: Unit of the values (e.g., "kWh").
            pattern: Glob pattern for file matching (default: *.parquet).
            freq: Expected frequency. If None, will infer from data.

        Returns:
            Tuple of (list of MeterTimeSeries, metadata dict).
        """
        folder_path = Path(folder_path)
        files = sorted(folder_path.glob(pattern))

        if not files:
            raise ValueError(f"No files matching {pattern} found in {folder_path}")

        logger.info(f"Found {len(files)} Parquet files in {folder_path}")

        all_meters = []
        all_metadata = {"inferred_frequencies": {}, "files_processed": []}
        inferred_freq = freq

        for file_path in files:
            meters, file_metadata = self._load_file(file_path, unit=unit, freq=freq or inferred_freq)
            all_meters.extend(meters)
            all_metadata["inferred_frequencies"].update(file_metadata.get("inferred_frequencies", {}))
            all_metadata["files_processed"].append(str(file_path))
            # Update global inferred_freq from first file if not provided
            if inferred_freq is None and freq is None and meters:
                inferred_freq = file_metadata["inferred_frequencies"][meters[0].meter_id]

        # Check for duplicate meter IDs across files
        meter_ids = [m.meter_id for m in all_meters]
        if len(meter_ids) != len(set(meter_ids)):
            duplicates = [mid for mid in set(meter_ids) if meter_ids.count(mid) > 1]
            logger.warning(f"Duplicate meter IDs across files: {duplicates}")


        return all_meters, all_metadata



class DatasetLoader:
    """High-level loader orchestrating prosumer and production asset loading.

    Missing-data behavior is controlled by SimulationConfig.missing_data,
    not by a separate policy on the loader. See SimulationConfig for options.
    """

    def __init__(self):
        self.prosumer_loader = MeterLoader()
        self.asset_loader = MeterLoader()

    def load(
        self,
        prosumer_data_path: Optional[Path] = None,
        production_data_path: Optional[Path] = None,
        simulation_config: Optional["SimulationConfig"] = None,
    ) -> Tuple[LoadedDataset, Optional[CoverageReport]]:
        """Load a complete dataset.

        MeterLoader automatically detects if paths are files or folders.
        Frequency can be provided in SimulationConfig, or will be inferred from data.

        Args:
            prosumer_data_path: Path to prosumer data (file or folder).
            production_data_path: Path to production data (file or folder).
            simulation_config: Optional SimulationConfig with freq and other settings.

        Returns:
            Tuple of (LoadedDataset, CoverageReport or None).
            CoverageReport is None if no simulation_config provided.

        Raises:
            ValueError: If no data paths provided, or if frequency validation fails.
        """
        if not prosumer_data_path and not production_data_path:
            raise ValueError("At least one of prosumer_data_path or production_data_path must be provided")

        # Extract freq from simulation_config if provided
        freq = simulation_config.freq if simulation_config else None

        prosumers = []
        production_assets = []
        all_metadata = {"inferred_frequencies": {}, "data_sources": {}}

        if prosumer_data_path:
            prosumers, prosumer_metadata = self.prosumer_loader.load(prosumer_data_path, unit="kWh", freq=freq)
            all_metadata["inferred_frequencies"].update(prosumer_metadata.get("inferred_frequencies", {}))
            all_metadata["data_sources"]["prosumers"] = prosumer_metadata

        if production_data_path:
            production_assets, asset_metadata = self.asset_loader.load(production_data_path, unit="kWh", freq=freq)
            # Negate production values to make them negative (uniform convention: positive=consumption, negative=production)
            for asset in production_assets:
                asset.value = -asset.value
            all_metadata["inferred_frequencies"].update(asset_metadata.get("inferred_frequencies", {}))
            all_metadata["data_sources"]["production_assets"] = asset_metadata

        # Validate frequency consistency across all meters
        self._validate_frequency_consistency(all_metadata["inferred_frequencies"], freq)

        # If simulation_config provided, compute coverage and apply missing_data policy
        coverage_report = None
        if simulation_config is not None:
            expected_index = simulation_config.to_index()

            # --- Coverage analysis ---
            missing_prosumers = []
            missing_production_assets = []
            per_meter_full_span = {}
            per_meter_missing_count = {}
            per_meter_missing_fraction = {}
            per_meter_first_missing = {}
            per_meter_last_missing = {}

            def check_coverage_per_series(series_list, missing_list):
                """Check coverage: timestamp exists AND value is not NaN."""
                longest_span = pd.Timedelta(0)
                for s in series_list:
                    ts = pd.DatetimeIndex(s.timestamp)
                    values = pd.Series(s.value, index=ts).reindex(expected_index)
                    present = values.notna().to_numpy()

                    missing_count = (~present).sum()
                    missing_fraction = missing_count / len(present) if len(present) > 0 else 0.0

                    missing_indices = np.where(~present)[0]
                    first_missing = expected_index[missing_indices[0]] if len(missing_indices) > 0 else None
                    last_missing = expected_index[missing_indices[-1]] if len(missing_indices) > 0 else None

                    per_meter_missing_count[s.meter_id] = int(missing_count)
                    per_meter_missing_fraction[s.meter_id] = float(missing_fraction)
                    per_meter_first_missing[s.meter_id] = first_missing
                    per_meter_last_missing[s.meter_id] = last_missing

                    if not present.all():
                        missing_list.append(s.meter_id)

                    if present.any():
                        arr = present.astype(int)
                        padded = np.concatenate([[0], arr, [0]])
                        diff = np.diff(padded)
                        starts = np.where(diff == 1)[0]
                        ends = np.where(diff == -1)[0]
                        counts = (ends - starts)
                        max_len = (counts * pd.Timedelta(simulation_config.freq)).max() if len(counts) > 0 else pd.Timedelta(0)
                    else:
                        max_len = pd.Timedelta(0)

                    per_meter_full_span[s.meter_id] = max_len
                    if max_len > longest_span:
                        longest_span = max_len
                return longest_span

            prosumer_span = check_coverage_per_series(prosumers, missing_prosumers)
            asset_span = check_coverage_per_series(production_assets, missing_production_assets)
            overall_longest = max(prosumer_span, asset_span)

            coverage_report = CoverageReport(
                missing_prosumers=missing_prosumers,
                missing_production_assets=missing_production_assets,
                per_meter_full_span=per_meter_full_span,
                overall_longest_full_span=overall_longest,
                per_meter_missing_count=per_meter_missing_count,
                per_meter_missing_fraction=per_meter_missing_fraction,
                per_meter_first_missing=per_meter_first_missing,
                per_meter_last_missing=per_meter_last_missing,
            )

            # --- Gate: error mode raises on any missing data ---
            has_missing = missing_prosumers or missing_production_assets
            if has_missing and simulation_config.missing_data == "error":
                logger.warning(
                    "Missing data detected for the requested simulation period. "
                    f"Missing prosumers: {missing_prosumers}; Missing assets: {missing_production_assets}"
                )
                logger.info(
                    f"Longest full-coverage contiguous subperiod: {overall_longest}"
                )
                raise ValueError(
                    "Simulation period contains missing data for one or more meters. "
                    "Set missing_data='fill_zero' (or 'fill_forward'/'keep_nan') to continue."
                )

            # --- Align all series to the simulation clock and fill ---
            missing_data = simulation_config.missing_data

            def align_and_fill(series_list):
                """Reindex series to canonical clock and apply fill strategy."""
                aligned = []
                for s in series_list:
                    df = pd.DataFrame({"value": s.value}, index=s.timestamp)
                    df_reindexed = df.reindex(expected_index)

                    if missing_data == "fill_zero":
                        df_reindexed["value"] = df_reindexed["value"].fillna(0.0)
                    elif missing_data == "fill_forward":
                        df_reindexed["value"] = df_reindexed["value"].ffill()
                    # "keep_nan" and "error" (no missing data): leave as-is

                    aligned.append(MeterTimeSeries(
                        meter_id=s.meter_id,
                        timestamp=expected_index,
                        value=df_reindexed["value"].to_numpy(dtype="float32"),
                        unit=s.unit,
                    ))
                return aligned

            prosumers = align_and_fill(prosumers)
            production_assets = align_and_fill(production_assets)

        dataset = LoadedDataset(
            prosumers=prosumers,
            production_assets=production_assets,
            timestamp_index=simulation_config.to_index() if simulation_config else None,
            timezone=simulation_config.tz if simulation_config else None,
            metadata=all_metadata,
        )

        return dataset, coverage_report

    @staticmethod
    def _validate_frequency_consistency(inferred_frequencies: dict, provided_freq: Optional[str]) -> None:
        """Validate that all meters have consistent frequency.

        Args:
            inferred_frequencies: dict mapping meter_id -> inferred frequency string
            provided_freq: frequency from SimulationConfig (if any)

        Raises:
            ValueError: If frequencies are inconsistent or don't match provided freq.
        """
        if not inferred_frequencies:
            return  # No meters loaded, nothing to validate

        unique_freqs = set(inferred_frequencies.values())

        if len(unique_freqs) > 1:
            # Multiple different frequencies across meters
            freq_breakdown = {}
            for meter_id, freq in inferred_frequencies.items():
                if freq not in freq_breakdown:
                    freq_breakdown[freq] = []
                freq_breakdown[freq].append(meter_id)

            msg = (
                f"Frequency mismatch across meters. Cannot create single simulation clock.\n"
                f"Found {len(unique_freqs)} different frequencies:\n"
            )
            for freq, meter_list in sorted(freq_breakdown.items()):
                msg += f"  {freq}: {len(meter_list)} meters (e.g., {meter_list[:3]})\n"
            msg += (
                "\nEnsure all data files have consistent frequency or provide an explicit freq in SimulationConfig."
            )
            raise ValueError(msg)

        inferred_freq = unique_freqs.pop()

        if provided_freq is not None:
            # User provided a frequency; verify it matches
            if inferred_freq != provided_freq:
                msg = (
                    f"Frequency mismatch: SimulationConfig.freq='{provided_freq}' "
                    f"but data infers '{inferred_freq}'.\n"
                    f"Check your data or update SimulationConfig.freq to match."
                )
                raise ValueError(msg)
            logger.info(f"Frequency validation passed: all meters match provided freq '{provided_freq}'")
        else:
            # Frequency was inferred; log the result
            logger.info(f"Frequency validation passed: all meters have consistent inferred freq '{inferred_freq}'")
