import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Sequence

import xarray as xr

from src.identifier import (
    IDENTIFICATION_METHODS,
    classify_front_type,
    identify_fronts_by_method,
)
from src.plot import (
    plot_classified_fronts,
    plot_front_samples,
    plot_method_comparison,
)
from src.utils import load_era5_datasets, preprocess_era5_variables

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the ERA5 front identification pipeline."""

    method: str = "f_diagnostic"
    threshold: float | None = None
    pressure_level: int = 850
    lat_range: tuple[float, float] = (35.25, 45.0)
    lon_range: tuple[float, float] = (25.0, 34.75)
    input_dir: Path = Path("./raw_data")
    output_dir: Path = Path("./outputs")
    time_range: tuple[str, str] | None = ("2025-10-05", "2025-10-10")
    max_plots: int = 100
    preprocess: bool = False
    classify: bool = False
    compare: bool = False
    force: bool = False
    log_level: str = "INFO"


def configure_logging(log_level: str = "INFO", log_file: str | Path = "pipeline.log") -> None:
    """Configure root logging with file and console handlers."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> PipelineConfig:
    """Parse CLI arguments into a PipelineConfig."""
    method_choices = [*sorted(IDENTIFICATION_METHODS), "all"]
    parser = argparse.ArgumentParser(
        description="Identify and plot atmospheric fronts from ERA5 pressure-level data."
    )
    parser.add_argument(
        "--method",
        choices=method_choices,
        default="f_diagnostic",
        help="Front identification method to run (default: f_diagnostic).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override for the selected identification method.",
    )
    parser.add_argument(
        "--pressure-level",
        type=int,
        default=850,
        help="Target pressure level in hPa (default: 850).",
    )
    parser.add_argument(
        "--lat-range",
        type=float,
        nargs=2,
        default=(35.25, 45.0),
        metavar=("LAT_MIN", "LAT_MAX"),
        help="Latitude bounds for GRIB preprocessing (default: 35.25 45.0).",
    )
    parser.add_argument(
        "--lon-range",
        type=float,
        nargs=2,
        default=(25.0, 34.75),
        metavar=("LON_MIN", "LON_MAX"),
        help="Longitude bounds for GRIB preprocessing (default: 25.0 34.75).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./raw_data"),
        help="Directory containing raw ERA5 GRIB files (default: ./raw_data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Directory for preprocessed NetCDF files, masks, and plots (default: ./outputs).",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        default=("2025-10-05", "2025-10-10"),
        metavar=("START", "END"),
        help="Inclusive date/time slice for plotting (pass --no-time-range to plot from start).",
    )
    parser.add_argument(
        "--no-time-range",
        action="store_true",
        help="Disable time-range filtering and plot the first `max-plots` timesteps.",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=100,
        help="Maximum number of plots to generate (default: 100).",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run GRIB-to-NetCDF preprocessing for variables q, t, u, v before identification.",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Also classify detected fronts into cold/warm/stationary via temperature advection.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate a multi-method comparison plot when running `--method all`.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute masks even if the output NetCDF file already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO).",
    )

    args = parser.parse_args(argv)
    return PipelineConfig(
        method=args.method,
        threshold=args.threshold,
        pressure_level=args.pressure_level,
        lat_range=(args.lat_range[0], args.lat_range[1]),
        lon_range=(args.lon_range[0], args.lon_range[1]),
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        time_range=None if args.no_time_range else (args.time_range[0], args.time_range[1]),
        max_plots=args.max_plots,
        preprocess=args.preprocess,
        classify=args.classify,
        compare=args.compare,
        force=args.force,
        log_level=args.log_level,
    )


def _mask_filename(method: str, pressure_level: int, single_default: bool) -> str:
    """Keep legacy `front_mask_850hPa.nc` for default f_diagnostic runs, or include method name."""
    if single_default and method == "f_diagnostic":
        return f"front_mask_{pressure_level}hPa.nc"
    return f"front_mask_{method}_{pressure_level}hPa.nc"


def run_pipeline(config: PipelineConfig) -> None:
    """Execute preprocessing, front identification, optional classification, and plotting."""
    logger.info("Starting ERA5 pipeline.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = config.output_dir / "plots"

    # 1. Optional GRIB -> NetCDF preprocessing
    if config.preprocess:
        preprocess_era5_variables(
            variables=("q", "t", "u", "v"),
            lat_range=config.lat_range,
            lon_range=config.lon_range,
            input_location=config.input_dir,
            output_location=config.output_dir,
            skip_existing=not config.force,
        )

    methods_to_run = (
        list(IDENTIFICATION_METHODS.keys()) if config.method == "all" else [config.method]
    )
    single_default = config.method == "f_diagnostic"

    # Determine required NetCDF variables for the requested method(s)
    needed_vars: set[str] = {"t"}
    for m in methods_to_run:
        needed_vars.update(IDENTIFICATION_METHODS[m]["required_vars"])
        needed_vars.update(IDENTIFICATION_METHODS[m].get("optional_vars", ()))
    if config.classify:
        needed_vars.update({"t", "u", "v"})

    # Load only available optional vars alongside required vars
    required_vars: set[str] = {"t"}
    for m in methods_to_run:
        required_vars.update(IDENTIFICATION_METHODS[m]["required_vars"])
    if config.classify:
        required_vars.update({"u", "v"})

    datasets: dict[str, object] = {}
    t_file = config.output_dir / "era5_t.nc"

    try:
        try:
            datasets = load_era5_datasets(config.output_dir, sorted(required_vars))
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return

        # Load optional variables (e.g., 'q' for TFP) if present
        for opt_var in sorted(needed_vars - required_vars):
            opt_path = config.output_dir / f"era5_{opt_var}.nc"
            if opt_path.exists():
                datasets.update(load_era5_datasets(config.output_dir, [opt_var]))

        generated_masks = {}

        # 2. Front identification
        for method in methods_to_run:
            mask_path = config.output_dir / _mask_filename(
                method, config.pressure_level, single_default=single_default
            )
            if mask_path.exists() and not config.force:
                logger.info(f"Front mask already exists at {mask_path}. Skipping identification.")
                with xr.open_dataset(mask_path) as existing_ds:
                    front_mask_da = existing_ds["front_mask"].load()
            else:
                logger.info(f"Calculating front masks with method '{method}'...")
                front_mask_da = identify_fronts_by_method(
                    method=method,
                    datasets=datasets,  # type: ignore[arg-type]
                    threshold=config.threshold,
                    pressure_level=config.pressure_level,
                )
                front_mask_da.to_netcdf(mask_path)
                logger.info(f"Successfully saved front mask to {mask_path}")

            generated_masks[method] = front_mask_da

            # 3. Optional front-type classification
            if config.classify:
                type_path = config.output_dir / f"front_type_{method}_{config.pressure_level}hPa.nc"
                front_type_da = classify_front_type(
                    front_mask_da,
                    datasets["t"],  # type: ignore[arg-type]
                    datasets["u"],  # type: ignore[arg-type]
                    datasets["v"],  # type: ignore[arg-type]
                    pressure_level=config.pressure_level,
                )
                front_type_da.to_netcdf(type_path)
                logger.info(f"Saved front type classification to {type_path}")
                plot_classified_fronts(
                    front_type_path=type_path,
                    t_path=t_file,
                    output_dir=plots_dir / method if len(methods_to_run) > 1 else plots_dir,
                    time_range=config.time_range,
                    max_plots=config.max_plots,
                    pressure_level=config.pressure_level,
                )

            # 4. Standard binary mask plotting
            method_plot_dir = plots_dir / method if len(methods_to_run) > 1 else plots_dir
            plot_front_samples(
                mask_path=mask_path,
                t_path=t_file,
                output_dir=method_plot_dir,
                time_range=config.time_range,
                max_plots=config.max_plots,
                pressure_level=config.pressure_level,
            )

        if config.compare and len(generated_masks) > 1:
            plot_method_comparison(
                masks=generated_masks,
                t_da=datasets["t"]["t"],  # type: ignore[index]
                output_dir=plots_dir,
                time_index=0,
                pressure_level=config.pressure_level,
            )
    finally:
        for ds in datasets.values():
            if hasattr(ds, "close"):
                ds.close()

    logger.info("Pipeline finished.")


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    configure_logging(log_level=config.log_level)
    run_pipeline(config)


if __name__ == "__main__":
    main()
