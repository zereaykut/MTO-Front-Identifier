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
    q_850 = ds_q.q.sel(isobaricInhPa=850) * units('kg/kg')
    p_850 = 850 * units.hPa
    
    # 2. Calculate Dewpoint then Equivalent Potential Temperature
    logger.info("Calculating Equivalent Potential Temperature (theta_e)...")
    theta_e = mpcalc.equivalent_potential_temperature(
        p_850, 
        t_850, 
        mpcalc.dewpoint_from_specific_humidity(p_850, t_850, q_850)
    )

    # 3. Calculate Gradients
    logger.info("Calculating atmospheric gradients...")
    dx, dy = mpcalc.lat_lon_grid_deltas(theta_e.longitude, theta_e.latitude)
    grad = mpcalc.gradient(theta_e, deltas=(dy, dx))
    grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
    
    # 4. Masking
    logger.info("Applying frontal threshold mask (> 4K / 100km)...")
    front_mask = grad_mag > (4e-5 * units('K/m')) 
    
    # Convert MetPy pint array back to a standard xarray DataArray for NetCDF saving
    logger.debug("Formatting output mask...")
    front_mask_da = xr.DataArray(
        front_mask.magnitude.astype(int),  # 1 for front, 0 for no front
        coords=theta_e.coords, 
        dims=theta_e.dims, 
        name="front_mask"
    )
    
    logger.info("Front identification complete.")
    return front_mask_da