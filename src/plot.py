import os
import logging
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

logger = logging.getLogger(__name__)

def plot_front_samples(mask_path, t_path, output_dir="./outputs/plots", time_range=None, max_plots=10):
    """
    Plots the front mask over the 850 hPa temperature field.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Loading data for plotting...")
    
    try:
        ds_mask = xr.open_dataset(mask_path)
        ds_t = xr.open_dataset(t_path)
    except FileNotFoundError as e:
        logger.error(f"Could not load files for plotting: {e}")
        return

    # Extract the mask and the 850 hPa temperature for the background
    front_mask = ds_mask.front_mask
    t_850 = ds_t.t.sel(isobaricInhPa=850)
    
    # Filter by time range if provided
    if time_range:
        logger.info(f"Filtering data for time range: {time_range[0]} to {time_range[1]}")
        front_mask = front_mask.sel(time=slice(time_range[0], time_range[1]))
        t_850 = t_850.sel(time=slice(time_range[0], time_range[1]))
        
    # Limit the number of plots to avoid generating too many files at once
    times_to_plot = front_mask.time.values[:max_plots]
    
    if len(times_to_plot) == 0:
        logger.warning("No time steps found in the specified range. Skipping plotting.")
        return

    logger.info(f"Generating {len(times_to_plot)} plot(s)...")

    for t_step in times_to_plot:
        # Format the timestamp for the filename (e.g., 2023-01-01_12)
        time_str = str(t_step)[:13].replace("T", "_") 
        logger.debug(f"Plotting time step: {time_str}")
        
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
        
        # 1. Plot background temperature (converted to Celsius)
        temp_data = t_850.sel(time=t_step) - 273.15 
        c = ax.contourf(temp_data.longitude, temp_data.latitude, temp_data, 
                        levels=20, cmap='coolwarm', transform=ccrs.PlateCarree())
        plt.colorbar(c, ax=ax, orientation='horizontal', pad=0.05, label='Temperature (°C) at 850 hPa')
        
        # 2. Overlay Front Mask
        mask_data = front_mask.sel(time=t_step)
        # We draw a contour exactly at the threshold of our binary mask
        ax.contour(mask_data.longitude, mask_data.latitude, mask_data, 
                   levels=[0.5], colors='black', linewidths=2.5, transform=ccrs.PlateCarree())
        
        plt.title(f"Frontal Zones and 850 hPa Temperature\n{time_str}")
        
        # Save figure
        out_file = os.path.join(output_dir, f"front_map_{time_str}.png")
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    logger.info(f"Plotting complete. Check the '{output_dir}' directory.")