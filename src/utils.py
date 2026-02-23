import xarray as xr
import glob
import os
import logging
import re

logger = logging.getLogger(__name__)

def preprocess_era5_variable(var_letter, output_name, lat_range, lon_range, input_location="./raw_data", output_location="./outputs"):
    os.makedirs(output_location, exist_ok=True)

    search_pattern = f"{input_location}/era5_{var_letter}_*.grib"
    files = glob.glob(search_pattern)
    
    if not files:
        logger.warning(f"No files found matching pattern: {search_pattern}")
        return None

    # 1. Group files by year to handle overlapping pressure levels cleanly
    files_by_year = {}
    for f in sorted(files):
        # Extract the 4-digit year from the filename
        match = re.search(r"(20\d{2})", f)
        year = match.group(1) if match else "all"
        files_by_year.setdefault(year, []).append(f)

    yearly_datasets = []
    
    # 2. Process and merge fragmented pressure levels for each specific year
    for year, year_files in files_by_year.items():
        ds_list = []
        for f in year_files:
            logger.debug(f"Reading {f}...")
            ds = xr.open_dataset(f, engine="cfgrib")
            
            # Subset geographically
            ds_subset = ds.sel(latitude=slice(max(lat_range), min(lat_range)), 
                               longitude=slice(min(lon_range), max(lon_range)))
            ds_list.append(ds_subset)
        
        # If there are fragmented pressure levels for this year, stitch them together
        if len(ds_list) > 1:
            logger.debug(f"Concatenating {len(ds_list)} pressure fragments for year {year}...")
            # Check if this is 3D data (has pressure levels) or 2D surface data
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

    # 3. Finally, combine the fully assembled years across the time dimension
    logger.info(f"Merging {len(yearly_datasets)} yearly datasets for {var_letter}...")
    
    if len(yearly_datasets) > 1:
        merged = xr.concat(yearly_datasets, dim="time", combine_attrs="drop_conflicts").sortby("time")
    else:
        merged = yearly_datasets[0]
        if "time" in merged.coords:
            merged = merged.sortby("time")
    
    output_path = os.path.join(output_location, output_name)
    logger.info(f"Saving merged dataset to {output_path}...")
    merged.to_netcdf(output_path)
    
    return merged