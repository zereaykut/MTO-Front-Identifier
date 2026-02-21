import xarray as xr
import glob
import os
import logging

logger = logging.getLogger(__name__)

def preprocess_era5_variable(var_letter, output_name, lat_range, lon_range, input_location="./raw_data", output_location="./outputs"):
    os.makedirs(output_location, exist_ok=True)

    search_pattern = f"{input_location}/era5_{var_letter}_*.grib"
    files = glob.glob(search_pattern)
    
    if not files:
        logger.warning(f"No files found matching pattern: {search_pattern}")
        return None

    datasets = []
    for f in sorted(files):
        logger.debug(f"Reading {f}...")
        ds = xr.open_dataset(f, engine="cfgrib")
        
        ds_subset = ds.sel(latitude=slice(max(lat_range), min(lat_range)), 
                           longitude=slice(min(lon_range), max(lon_range)))
        datasets.append(ds_subset)

    logger.info(f"Merging {len(datasets)} datasets for {var_letter}...")
    
    merged = xr.combine_by_coords(datasets, combine_attrs='drop_conflicts') # combine_attrs='drop_conflicts' to ignore the mismatched history metadata
    
    output_path = os.path.join(output_location, output_name)
    logger.info(f"Saving merged dataset to {output_path}...")
    merged.to_netcdf(output_path)
    
    return merged