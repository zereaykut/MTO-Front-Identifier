import logging
from pathlib import Path
from typing import Mapping, Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr

logger = logging.getLogger(__name__)


def _format_time_str(t_step: object) -> str:
    """Format a numpy/xarray timestamp into YYYY-MM-DD_HH for filenames and titles."""
    return str(t_step)[:13].replace("T", "_")


def _add_map_features(ax: plt.Axes, gridlines: bool = False) -> None:
    """Add coastline, borders, and optional lat/lon gridlines to a Cartopy axis."""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False


def _select_temperature(ds_t: xr.Dataset, pressure_level: int | float) -> xr.DataArray:
    """Extract temperature DataArray at the target pressure level."""
    t_da = ds_t["t"]
    if "isobaricInhPa" in t_da.coords:
        t_da = t_da.sel(isobaricInhPa=pressure_level)
    return t_da


def plot_front_samples(
    mask_path: str | Path,
    t_path: str | Path,
    output_dir: str | Path = "./outputs/plots",
    time_range: Sequence[str] | None = None,
    max_plots: int = 10,
    pressure_level: int | float = 850,
    title_prefix: str | None = None,
    gridlines: bool = False,
) -> list[Path]:
    """
    Plot binary front masks overlaid on 850 hPa temperature fields.
    Returns a list of generated image paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading data for plotting...")

    try:
        ds_mask = xr.open_dataset(mask_path)
        ds_t = xr.open_dataset(t_path)
    except FileNotFoundError as e:
        logger.error(f"Could not load files for plotting: {e}")
        return []

    generated_files: list[Path] = []
    try:
        front_mask = ds_mask["front_mask"]
        t_level = _select_temperature(ds_t, pressure_level)

        if time_range:
            logger.info(f"Filtering data for time range: {time_range[0]} to {time_range[1]}")
            front_mask = front_mask.sel(time=slice(time_range[0], time_range[1]))
            t_level = t_level.sel(time=slice(time_range[0], time_range[1]))

        times_to_plot = front_mask.time.values[:max_plots]
        if len(times_to_plot) == 0:
            logger.warning("No time steps found in the specified range. Skipping plotting.")
            return []

        logger.info(f"Generating {len(times_to_plot)} plot(s)...")
        heading = title_prefix or f"Frontal Zones and {pressure_level} hPa Temperature"

        for t_step in times_to_plot:
            time_str = _format_time_str(t_step)
            logger.debug(f"Plotting time step: {time_str}")

            fig = plt.figure(figsize=(10, 8))
            try:
                ax = plt.axes(projection=ccrs.PlateCarree())
                _add_map_features(ax, gridlines=gridlines)

                temp_data = t_level.sel(time=t_step) - 273.15
                c = ax.contourf(
                    temp_data.longitude,
                    temp_data.latitude,
                    temp_data,
                    levels=20,
                    cmap="coolwarm",
                    transform=ccrs.PlateCarree(),
                )
                plt.colorbar(
                    c,
                    ax=ax,
                    orientation="horizontal",
                    pad=0.05,
                    label=f"Temperature (°C) at {pressure_level} hPa",
                )

                mask_data = front_mask.sel(time=t_step)
                if float(mask_data.max()) >= 0.5 > float(mask_data.min()):
                    ax.contour(
                        mask_data.longitude,
                        mask_data.latitude,
                        mask_data,
                        levels=[0.5],
                        colors="black",
                        linewidths=2.5,
                        transform=ccrs.PlateCarree(),
                    )

                plt.title(f"{heading}\n{time_str}")
                out_file = out_dir / f"front_map_{time_str}.png"
                plt.savefig(out_file, dpi=150, bbox_inches="tight")
                generated_files.append(out_file)
            finally:
                plt.close(fig)
    finally:
        ds_mask.close()
        ds_t.close()

    logger.info(f'Plotting complete. Check the "{out_dir}" directory.')
    return generated_files


def plot_classified_fronts(
    front_type_path: str | Path,
    t_path: str | Path,
    output_dir: str | Path = "./outputs/plots",
    time_range: Sequence[str] | None = None,
    max_plots: int = 10,
    pressure_level: int | float = 850,
    gridlines: bool = True,
) -> list[Path]:
    """
    Plot classified front types (1=Cold, 2=Warm, 3=Stationary) over temperature.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds_type = xr.open_dataset(front_type_path)
        ds_t = xr.open_dataset(t_path)
    except FileNotFoundError as e:
        logger.error(f"Could not load files for classified plotting: {e}")
        return []

    generated_files: list[Path] = []
    try:
        front_type = ds_type["front_type"]
        t_level = _select_temperature(ds_t, pressure_level)

        if time_range:
            front_type = front_type.sel(time=slice(time_range[0], time_range[1]))
            t_level = t_level.sel(time=slice(time_range[0], time_range[1]))

        times_to_plot = front_type.time.values[:max_plots]
        if len(times_to_plot) == 0:
            logger.warning("No time steps found for classified front plotting.")
            return []

        type_styles = [
            (1, "#0044cc", "Cold Front"),
            (2, "#cc2200", "Warm Front"),
            (3, "#7b2cbf", "Stationary Front"),
        ]

        for t_step in times_to_plot:
            time_str = _format_time_str(t_step)
            fig = plt.figure(figsize=(10, 8))
            try:
                ax = plt.axes(projection=ccrs.PlateCarree())
                _add_map_features(ax, gridlines=gridlines)

                temp_data = t_level.sel(time=t_step) - 273.15
                c = ax.contourf(
                    temp_data.longitude,
                    temp_data.latitude,
                    temp_data,
                    levels=20,
                    cmap="coolwarm",
                    alpha=0.85,
                    transform=ccrs.PlateCarree(),
                )
                plt.colorbar(
                    c,
                    ax=ax,
                    orientation="horizontal",
                    pad=0.06,
                    label=f"Temperature (°C) at {pressure_level} hPa",
                )

                type_slice = front_type.sel(time=t_step)
                legend_handles: list[Line2D] = []
                for code, color, label in type_styles:
                    binary = (type_slice == code).astype(int)
                    if int(binary.max()) == 1 and int(binary.min()) == 0:
                        ax.contour(
                            binary.longitude,
                            binary.latitude,
                            binary,
                            levels=[0.5],
                            colors=color,
                            linewidths=2.4,
                            transform=ccrs.PlateCarree(),
                        )
                        legend_handles.append(Line2D([0], [0], color=color, lw=2.4, label=label))

                if legend_handles:
                    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

                plt.title(f"Classified Frontal Zones and {pressure_level} hPa Temperature\n{time_str}")
                out_file = out_dir / f"front_classified_{time_str}.png"
                plt.savefig(out_file, dpi=150, bbox_inches="tight")
                generated_files.append(out_file)
            finally:
                plt.close(fig)
    finally:
        ds_type.close()
        ds_t.close()

    return generated_files


def plot_method_comparison(
    masks: Mapping[str, xr.DataArray],
    t_da: xr.DataArray,
    output_dir: str | Path = "./outputs/plots",
    time_index: int = 0,
    pressure_level: int | float = 850,
) -> Path | None:
    """
    Generate a side-by-side comparison figure across multiple front identification methods
    for a single timestep.
    """
    if not masks:
        logger.warning("No masks provided for method comparison plot.")
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_level = t_da.sel(isobaricInhPa=pressure_level) if "isobaricInhPa" in t_da.coords else t_da
    t_step = t_level.time.values[time_index]
    time_str = _format_time_str(t_step)
    temp_data = t_level.sel(time=t_step) - 273.15

    n_methods = len(masks)
    cols = min(2, n_methods)
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(7 * cols, 6 * rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    out_file = out_dir / f"front_comparison_{time_str}.png"
    try:
        flat_axes = axes.ravel()
        last_cf = None
        for idx, (method_name, mask_da) in enumerate(masks.items()):
            ax = flat_axes[idx]
            _add_map_features(ax, gridlines=False)
            last_cf = ax.contourf(
                temp_data.longitude,
                temp_data.latitude,
                temp_data,
                levels=20,
                cmap="coolwarm",
                transform=ccrs.PlateCarree(),
            )
            mask_slice = mask_da.sel(time=t_step)
            if float(mask_slice.max()) >= 0.5 > float(mask_slice.min()):
                ax.contour(
                    mask_slice.longitude,
                    mask_slice.latitude,
                    mask_slice,
                    levels=[0.5],
                    colors="black",
                    linewidths=2.0,
                    transform=ccrs.PlateCarree(),
                )
            ax.set_title(method_name)

        for idx in range(n_methods, len(flat_axes)):
            flat_axes[idx].set_visible(False)

        if last_cf is not None:
            fig.colorbar(
                last_cf,
                ax=flat_axes[:n_methods].tolist(),
                orientation="horizontal",
                fraction=0.04,
                pad=0.06,
                label=f"Temperature (°C) at {pressure_level} hPa",
            )

        fig.suptitle(f"Front Identification Method Comparison ({time_str})", fontsize=14)
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    return out_file
