import logging
from pathlib import Path
import re
from typing import Iterable, Sequence

import xarray as xr

logger = logging.getLogger(__name__)


def preprocess_era5_variable(
    var_letter: str,
    output_name: str,
    lat_range: Sequence[float],
    lon_range: Sequence[float],
    input_location: str | Path = "./raw_data",
    output_location: str | Path = "./outputs",
) -> xr.Dataset | None:
    """
    Read ERA5 GRIB files for a single variable, subset geographically, stitch
    fragmented pressure-level files per year, merge across years, and save as NetCDF.
    """
    input_dir = Path(input_location)
    output_dir = Path(output_location)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(f"era5_{var_letter}_*.grib"))
    if not files:
        logger.warning(f"No files found matching pattern: {input_dir / f'era5_{var_letter}_*.grib'}")
        return None

    lat_max, lat_min = max(lat_range), min(lat_range)
    lon_min, lon_max = min(lon_range), max(lon_range)

    # 1. Group files by year to handle overlapping pressure levels cleanly
    files_by_year: dict[str, list[Path]] = {}
    for file_path in files:
        match = re.search(r"(20\d{2})", file_path.name)
        year = match.group(1) if match else "all"
        files_by_year.setdefault(year, []).append(file_path)

    yearly_datasets: list[xr.Dataset] = []

    # 2. Process and merge fragmented pressure levels for each specific year
    for year, year_files in files_by_year.items():
        ds_list: list[xr.Dataset] = []
        for file_path in year_files:
            logger.debug(f"Reading {file_path}...")
            with xr.open_dataset(file_path, engine="cfgrib") as ds:
                ds_subset = ds.sel(
                    latitude=slice(lat_max, lat_min),
                    longitude=slice(lon_min, lon_max),
                ).load()
            ds_list.append(ds_subset)

        if len(ds_list) > 1:
            logger.debug(f"Concatenating {len(ds_list)} pressure fragments for year {year}...")
            if "isobaricInhPa" in ds_list[0].coords:
                ds_year = xr.concat(ds_list, dim="isobaricInhPa", combine_attrs="drop_conflicts")
                ds_year = ds_year.sortby("isobaricInhPa")
            else:
                ds_year = xr.concat(ds_list, dim="time", combine_attrs="drop_conflicts")
        else:
            ds_year = ds_list[0]
            if "isobaricInhPa" in ds_year.coords:
                ds_year = ds_year.sortby("isobaricInhPa")

        yearly_datasets.append(ds_year)

    # 3. Combine assembled years across the time dimension
    logger.info(f"Merging {len(yearly_datasets)} yearly datasets for {var_letter}...")
    if len(yearly_datasets) > 1:
        merged = xr.concat(yearly_datasets, dim="time", combine_attrs="drop_conflicts").sortby("time")
    else:
        merged = yearly_datasets[0]
        if "time" in merged.coords:
            merged = merged.sortby("time")

    output_path = output_dir / output_name
    logger.info(f"Saving merged dataset to {output_path}...")
    merged.to_netcdf(output_path)

    return merged


def preprocess_era5_variables(
    variables: Iterable[str],
    lat_range: Sequence[float],
    lon_range: Sequence[float],
    input_location: str | Path = "./raw_data",
    output_location: str | Path = "./outputs",
    skip_existing: bool = True,
) -> dict[str, Path]:
    """
    Preprocess multiple ERA5 variables into `era5_<var>.nc` files inside `output_location`.
    Returns a mapping of variable name to output NetCDF path for processed/existing files.
    """
    output_dir = Path(output_location)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed: dict[str, Path] = {}

    for var in variables:
        out_name = f"era5_{var}.nc"
        out_path = output_dir / out_name
        if skip_existing and out_path.exists():
            logger.info(f"{out_name} already exists. Skipping preprocessing.")
            processed[var] = out_path
            continue

        logger.info(f"Initiating preprocessing for variable: {var}")
        result = preprocess_era5_variable(
            var_letter=var,
            output_name=out_name,
            lat_range=lat_range,
            lon_range=lon_range,
            input_location=input_location,
            output_location=output_dir,
        )
        if result is not None:
            processed[var] = out_path

    return processed


def load_era5_datasets(
    data_dir: str | Path,
    variables: Iterable[str],
) -> dict[str, xr.Dataset]:
    """
    Load preprocessed `era5_<var>.nc` datasets from `data_dir`.
    Raises FileNotFoundError if any requested file is missing.
    """
    base_dir = Path(data_dir)
    datasets: dict[str, xr.Dataset] = {}
    missing: list[Path] = []

    for var in variables:
        nc_path = base_dir / f"era5_{var}.nc"
        if not nc_path.exists():
            missing.append(nc_path)
            continue
        datasets[var] = xr.open_dataset(nc_path)

    if missing:
        for ds in datasets.values():
            ds.close()
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing preprocessed NetCDF file(s): {missing_str}")

    return datasets
