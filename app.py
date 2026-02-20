import logging
import xarray as xr
import os
from src.utils import preprocess_era5_variable
from src.identifier import identify_fronts
from src.plot import plot_front_samples

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting ERA5 pipeline.")

    # ---------------------------------------------------------
    # CONFIGURATION: Set the time range for plot analysis

    # Target area
    lat_range = [35, 45] 
    lon_range = [25, 45]
    input_location = "/home/spidy/Projects/Data/mto/"          
    output_location = "./outputs" 

    # Format: ("YYYY-MM-DD", "YYYY-MM-DD") or None to plot the start of the dataset
    plot_time_range = ("2025-10-01", "2025-10-10") 

    vars_to_process = ["q", "t", "u", "v"]
    # ---------------------------------------------------------
    
    # 1. Preprocess Variables
    for var in vars_to_process:
        # Check if file already exists to save time during reruns
        if not os.path.exists(f"{output_location}/era5_{var}.nc"):
            logger.info(f"Initiating preprocessing for variable: {var}")
            preprocess_era5_variable(var, f"era5_{var}.nc", lat_range, lon_range, input_location, output_location)
        else:
            logger.info(f"era5_{var}.nc already exists. Skipping preprocessing.")

    # 2. Run Front Identification
    mask_output_path = os.path.join(output_location, "front_mask_850hPa.nc")
    
    if not os.path.exists(mask_output_path):
        logger.info("Loading preprocessed NetCDF files for front identification...")
        try:
            ds_t = xr.open_dataset(os.path.join(output_location, "era5_t.nc"))
            ds_q = xr.open_dataset(os.path.join(output_location, "era5_q.nc"))
            
            logger.info("Calculating front masks...")
            front_mask_da = identify_fronts(ds_t, ds_q, ds_sp=None)
            
            front_mask_da.to_netcdf(mask_output_path)
            logger.info(f"Successfully saved front mask to {mask_output_path}")

        except Exception as e:
            logger.error(f"An error occurred during front identification: {e}", exc_info=True)
    else:
        logger.info("Front mask already exists. Skipping identification.")

    # 3. Plotting
    logger.info("Starting plotting module...")
    try:
        t_file = os.path.join(output_location, "era5_t.nc")
        plot_front_samples(
            mask_path=mask_output_path, 
            t_path=t_file, 
            output_dir=os.path.join(output_location, "plots"), 
            time_range=plot_time_range,
            max_plots=5 # Change this to generate more/fewer frames
        )
    except Exception as e:
        logger.error(f"An error occurred during plotting: {e}", exc_info=True)

    logger.info("Pipeline finished.")

if __name__ == "__main__":
    main()