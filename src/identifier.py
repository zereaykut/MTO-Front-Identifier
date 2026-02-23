import metpy.calc as mpcalc
from metpy.units import units
import xarray as xr
import numpy as np
import logging

logger = logging.getLogger(__name__)

def identify_fronts(ds_t, ds_q, ds_sp=None):
    logger.info("Starting front identification at 850 hPa.")
    
    # 1. Calculate Potential Temperature
    logger.debug("Extracting Temperature and Specific Humidity at 850 hPa...")
    t_850 = ds_t.t.sel(isobaricInhPa=850) * units.kelvin
    q_850 = ds_q.q.sel(isobaricInhPa=850) * units("kg/kg")
    p_850 = 850 * units.hPa
    
    # 2. Calculate Dewpoint then Equivalent Potential Temperature
    logger.info("Calculating Equivalent Potential Temperature (theta_e)...")
    theta_e = mpcalc.equivalent_potential_temperature(
        p_850, 
        t_850, 
        mpcalc.dewpoint_from_specific_humidity(p_850, t_850, q_850)
    )

    # Ensure dimensional order is exactly as expected before we iterate
    theta_e = theta_e.transpose("time", "latitude", "longitude")

    # 3. Calculate Gradients
    logger.info("Calculating atmospheric gradients frame-by-frame...")
    dx, dy = mpcalc.lat_lon_grid_deltas(theta_e.longitude, theta_e.latitude)
    
    front_masks = []
    
    # Process each time step individually to ensure the 2D slices
    for i in range(theta_e.sizes["time"]):
        theta_e_slice = theta_e.isel(time=i)
        
        grad = mpcalc.gradient(theta_e_slice, deltas=(dy, dx))
        grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
        
        # 4. Masking
        front_mask_slice = grad_mag > (4e-5 * units("K/m")) 
        
        # FIX: Safely extract the array data regardless of how the boolean mask is wrapped
        if hasattr(front_mask_slice, "magnitude"):
            mask_array = front_mask_slice.magnitude
        elif hasattr(front_mask_slice, "values"):
            mask_array = front_mask_slice.values
        else:
            mask_array = front_mask_slice
            
        front_masks.append(np.asarray(mask_array, dtype=int))
    
    # Re-stack into a single multidimensional array matching the original metadata
    logger.debug("Formatting output mask...")
    front_mask_da = xr.DataArray(
        np.array(front_masks),  # Stacked shape will be (time, lat, lon)
        coords=theta_e.coords, 
        dims=theta_e.dims, 
        name="front_mask"
    )
    
    logger.info("Front identification complete.")
    return front_mask_da

def identify_fronts_f_diagnostic(ds_t, ds_u, ds_v, threshold=1.0):
    """
    Method 2: F-Diagnostic Front Identification.
    Combines both thermal (horizontal temperature gradient) and dynamic (isobaric relative vorticity) components.
    """
    logger.info("Starting F-Diagnostic front identification at 850 hPa.")
    
    # 1. Extract data at 850 hPa (We keep them as DataArrays here to maintain dimensional alignment)
    logger.debug("Extracting Temperature and Wind components (U, V) at 850 hPa...")
    t_850 = ds_t.t.sel(isobaricInhPa=850)
    u_850 = ds_u.u.sel(isobaricInhPa=850)
    v_850 = ds_v.v.sel(isobaricInhPa=850)
    
    # Ensure dimensional order is exactly as expected before iterating
    t_850 = t_850.transpose("time", "latitude", "longitude")
    u_850 = u_850.transpose("time", "latitude", "longitude")
    v_850 = v_850.transpose("time", "latitude", "longitude")

    # Calculate physical distance grids (dx, dy)
    dx, dy = mpcalc.lat_lon_grid_deltas(t_850.longitude, t_850.latitude)
    
    # 2. Calculate the Coriolis parameter (f) across the 2D Latitude grid
    # Bypassing xarray coordinates here to avoid metadata conflicts
    lon_2d, lat_2d = np.meshgrid(t_850.longitude.values, t_850.latitude.values)
    f = mpcalc.coriolis_parameter(lat_2d * units.degrees)
    
    # 3. Define the Characteristic Temperature Gradient (|∇T|0 ≈ 0.45 K/100km)
    grad_t_0 = 0.45 * units.kelvin / (100 * units.km)
    
    front_masks = []
    
    logger.info("Calculating F-parameters frame-by-frame...")
    for i in range(t_850.sizes["time"]):
        
        # Extract raw numpy values and apply pint units directly.
        t_slice = t_850.isel(time=i).values * units.kelvin
        u_slice = u_850.isel(time=i).values * units('m/s')
        v_slice = v_850.isel(time=i).values * units('m/s')
        
        # Calculate the magnitude of the horizontal temperature gradient (|∇Tp|)
        grad_t = mpcalc.gradient(t_slice, deltas=(dy, dx))
        grad_t_mag = np.sqrt(grad_t[0]**2 + grad_t[1]**2)
        
        # Calculate Isobaric Relative Vorticity (ζp) using wind components
        vort = mpcalc.vorticity(u_slice, v_slice, dx=dx, dy=dy)
        
        # Calculate the F-Parameter: (ζp / f) * (|∇Tp| / |∇T|0)
        term1 = (vort / f).to_base_units()
        term2 = (grad_t_mag / grad_t_0).to_base_units()
        
        f_param = term1 * term2
        
        # 4. Masking
        f_mask_slice = f_param > threshold
        
        # FIX: Safely extract the raw boolean array regardless of what type it returns
        if hasattr(f_mask_slice, "magnitude"):
            mask_array = f_mask_slice.magnitude
        elif hasattr(f_mask_slice, "values"):
            mask_array = f_mask_slice.values
        else:
            mask_array = f_mask_slice
            
        front_masks.append(np.asarray(mask_array, dtype=int))

    logger.debug("Formatting output mask...")
    front_mask_da = xr.DataArray(
        np.array(front_masks),
        coords=t_850.coords,
        dims=t_850.dims,
        name="front_mask"
    )
    
    logger.info("F-Diagnostic front identification complete.")
    return front_mask_da