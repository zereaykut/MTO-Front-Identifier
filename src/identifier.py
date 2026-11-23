import logging
from typing import Any, Callable

import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _extract_level(
    ds: xr.Dataset,
    var_name: str,
    pressure_level: int | float = 850,
) -> xr.DataArray:
    """Extract a variable at the target pressure level in (time, latitude, longitude) order."""
    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in dataset. Available: {list(ds.data_vars)}")

    da = ds[var_name]
    if "isobaricInhPa" in da.coords:
        da = da.sel(isobaricInhPa=pressure_level)

    expected_dims = [dim for dim in ("time", "latitude", "longitude") if dim in da.dims]
    if len(expected_dims) != 3:
        raise ValueError(
            f"Variable '{var_name}' must have ('time', 'latitude', 'longitude') dimensions; got {da.dims}"
        )

    return da.transpose("time", "latitude", "longitude")


def _grid_deltas(da: xr.DataArray) -> tuple[Any, Any]:
    """Compute physical grid spacings (dx, dy) with explicit degree units."""
    return mpcalc.lat_lon_grid_deltas(
        da.longitude.values * units.degrees,
        da.latitude.values * units.degrees,
    )


def _to_binary_mask(mask_slice: Any) -> np.ndarray:
    """Convert a Pint Quantity, xarray DataArray, or numpy boolean mask into an integer array."""
    if hasattr(mask_slice, "magnitude"):
        mask_array = mask_slice.magnitude
    elif hasattr(mask_slice, "values"):
        mask_array = mask_slice.values
    else:
        mask_array = mask_slice

    return np.asarray(np.nan_to_num(mask_array, nan=False), dtype=int)


def _build_mask_dataarray(
    masks: list[np.ndarray],
    template_da: xr.DataArray,
    name: str = "front_mask",
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Stack per-timestep 2D masks into a DataArray preserving template coordinates and dimensions."""
    return xr.DataArray(
        np.array(masks),
        coords=template_da.coords,
        dims=template_da.dims,
        name=name,
        attrs=attrs or {},
    )


def identify_fronts(
    ds_t: xr.Dataset,
    ds_q: xr.Dataset,
    ds_sp: xr.Dataset | None = None,
    threshold: float = 4e-5,
    pressure_level: int | float = 850,
) -> xr.DataArray:
    """
    Method 1: Equivalent Potential Temperature Gradient (|∇θ_e|).

    Identifies frontal zones where the horizontal gradient magnitude of equivalent
    potential temperature at the given pressure level exceeds `threshold` (default 4e-5 K/m).
    """
    del ds_sp  # Kept in signature for backward compatibility.
    logger.info(f"Starting theta_e gradient front identification at {pressure_level} hPa.")

    logger.debug(f"Extracting Temperature and Specific Humidity at {pressure_level} hPa...")
    t_level = _extract_level(ds_t, "t", pressure_level) * units.kelvin
    q_level = _extract_level(ds_q, "q", pressure_level) * units("kg/kg")
    p_level = pressure_level * units.hPa

    logger.info("Calculating Equivalent Potential Temperature (theta_e)...")
    dewpoint = mpcalc.dewpoint_from_specific_humidity(p_level, q_level)
    theta_e = mpcalc.equivalent_potential_temperature(p_level, t_level, dewpoint)
    theta_e = theta_e.transpose("time", "latitude", "longitude")

    logger.info("Calculating atmospheric gradients frame-by-frame...")
    dx, dy = _grid_deltas(theta_e)
    threshold_q = threshold * units("K/m")

    front_masks: list[np.ndarray] = []
    for i in range(theta_e.sizes["time"]):
        theta_e_slice = theta_e.isel(time=i)
        grad = mpcalc.gradient(theta_e_slice, deltas=(dy, dx))
        grad_mag = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
        front_mask_slice = grad_mag > threshold_q
        front_masks.append(_to_binary_mask(front_mask_slice))

    logger.debug("Formatting output mask...")
    front_mask_da = _build_mask_dataarray(
        front_masks,
        theta_e,
        name="front_mask",
        attrs={
            "method": "theta_e_gradient",
            "pressure_level_hPa": pressure_level,
            "threshold": threshold,
            "threshold_units": "K/m",
        },
    )

    logger.info("Front identification complete.")
    return front_mask_da


def identify_fronts_f_diagnostic(
    ds_t: xr.Dataset,
    ds_u: xr.Dataset,
    ds_v: xr.Dataset,
    threshold: float = 1.0,
    pressure_level: int | float = 850,
) -> xr.DataArray:
    """
    Method 2: F-Diagnostic Front Identification.

    Combines thermal (horizontal temperature gradient |∇T_p|) and dynamic
    (isobaric relative vorticity ζ_p) components:
        F = (ζ_p / f) * (|∇T_p| / |∇T|_0), where |∇T|_0 = 0.45 K / 100 km.
    """
    logger.info(f"Starting F-Diagnostic front identification at {pressure_level} hPa.")

    logger.debug(f"Extracting Temperature and Wind components (U, V) at {pressure_level} hPa...")
    t_level = _extract_level(ds_t, "t", pressure_level)
    u_level = _extract_level(ds_u, "u", pressure_level)
    v_level = _extract_level(ds_v, "v", pressure_level)

    dx, dy = _grid_deltas(t_level)

    _, lat_2d = np.meshgrid(t_level.longitude.values, t_level.latitude.values)
    f = mpcalc.coriolis_parameter(lat_2d * units.degrees)
    # Guard against exact-equator grid points where f == 0
    f_safe = np.where(np.abs(f.magnitude) < 1e-12, np.nan, f.magnitude) * f.units

    grad_t_0 = 0.45 * units.kelvin / (100 * units.km)

    front_masks: list[np.ndarray] = []
    logger.info("Calculating F-parameters frame-by-frame...")
    for i in range(t_level.sizes["time"]):
        t_slice = t_level.isel(time=i).values * units.kelvin
        u_slice = u_level.isel(time=i).values * units("m/s")
        v_slice = v_level.isel(time=i).values * units("m/s")

        grad_t = mpcalc.gradient(t_slice, deltas=(dy, dx))
        grad_t_mag = np.sqrt(grad_t[0] ** 2 + grad_t[1] ** 2)
        vort = mpcalc.vorticity(u_slice, v_slice, dx=dx, dy=dy)

        term1 = (vort / f_safe).to_base_units()
        term2 = (grad_t_mag / grad_t_0).to_base_units()
        f_param = term1 * term2

        with np.errstate(invalid="ignore"):
            f_mask_slice = f_param > threshold

        front_masks.append(_to_binary_mask(f_mask_slice))

    logger.debug("Formatting output mask...")
    front_mask_da = _build_mask_dataarray(
        front_masks,
        t_level,
        name="front_mask",
        attrs={
            "method": "f_diagnostic",
            "pressure_level_hPa": pressure_level,
            "threshold": threshold,
            "threshold_units": "dimensionless",
        },
    )

    logger.info("F-Diagnostic front identification complete.")
    return front_mask_da


def identify_fronts_tfp(
    ds_t: xr.Dataset,
    ds_q: xr.Dataset | None = None,
    tfp_threshold: float = 0.0,
    grad_threshold: float = 2e-5,
    pressure_level: int | float = 850,
) -> xr.DataArray:
    """
    Method 3: Thermal Front Parameter (TFP; Renard & Clarke, 1965).

    Locates the leading (warm-side) edge of baroclinic frontal zones via:
        TFP = -∇|∇τ| · (∇τ / |∇τ|)
    where τ is equivalent potential temperature (θ_e) if `ds_q` is provided,
    or potential temperature (θ) otherwise. Masks regions where TFP > `tfp_threshold`
    (K/m²) and |∇τ| > `grad_threshold` (K/m).
    """
    thermal_var = "theta_e" if ds_q is not None else "theta"
    logger.info(
        f"Starting TFP front identification using {thermal_var} at {pressure_level} hPa."
    )

    t_level = _extract_level(ds_t, "t", pressure_level)
    p_level = pressure_level * units.hPa
    t_qty = t_level.values * units.kelvin

    if ds_q is not None:
        q_level = _extract_level(ds_q, "q", pressure_level)
        q_qty = q_level.values * units("kg/kg")
        dewpoint = mpcalc.dewpoint_from_specific_humidity(p_level, q_qty)
        tau_all = mpcalc.equivalent_potential_temperature(p_level, t_qty, dewpoint)
    else:
        tau_all = mpcalc.potential_temperature(p_level, t_qty)

    dx, dy = _grid_deltas(t_level)
    tfp_thresh_q = tfp_threshold * units("K/m**2")
    grad_thresh_q = grad_threshold * units("K/m")

    front_masks: list[np.ndarray] = []
    for i in range(t_level.sizes["time"]):
        tau_slice = tau_all[i]
        grad_tau = mpcalc.gradient(tau_slice, deltas=(dy, dx))
        grad_mag = np.sqrt(grad_tau[0] ** 2 + grad_tau[1] ** 2)
        grad_of_mag = mpcalc.gradient(grad_mag, deltas=(dy, dx))

        # Avoid division by zero in uniform-temperature regions
        safe_mag = np.where(grad_mag.magnitude > 0, grad_mag.magnitude, np.nan) * grad_mag.units
        tfp = -(
            grad_of_mag[0] * (grad_tau[0] / safe_mag)
            + grad_of_mag[1] * (grad_tau[1] / safe_mag)
        )

        with np.errstate(invalid="ignore"):
            mask_slice = (tfp > tfp_thresh_q) & (grad_mag > grad_thresh_q)

        front_masks.append(_to_binary_mask(mask_slice))

    front_mask_da = _build_mask_dataarray(
        front_masks,
        t_level,
        name="front_mask",
        attrs={
            "method": f"tfp_{thermal_var}",
            "pressure_level_hPa": pressure_level,
            "tfp_threshold": tfp_threshold,
            "tfp_threshold_units": "K/m^2",
            "grad_threshold": grad_threshold,
            "grad_threshold_units": "K/m",
        },
    )

    logger.info("TFP front identification complete.")
    return front_mask_da


def identify_fronts_frontogenesis(
    ds_t: xr.Dataset,
    ds_u: xr.Dataset,
    ds_v: xr.Dataset,
    threshold: float = 0.5,
    pressure_level: int | float = 850,
) -> xr.DataArray:
    """
    Method 4: 2D Horizontal Kinematic Frontogenesis (Petterssen, 1936).

    Computes horizontal frontogenesis of potential temperature (θ) at the target
    pressure level using wind components (u, v). Threshold is expressed in
    synoptic units of K / (100 km · 3 h) (default 0.5).
    """
    logger.info(f"Starting 2D Frontogenesis identification at {pressure_level} hPa.")

    t_level = _extract_level(ds_t, "t", pressure_level)
    u_level = _extract_level(ds_u, "u", pressure_level)
    v_level = _extract_level(ds_v, "v", pressure_level)

    dx, dy = _grid_deltas(t_level)
    p_level = pressure_level * units.hPa
    threshold_q = threshold * (units.kelvin / (100 * units.km * 3 * units.hour))

    front_masks: list[np.ndarray] = []
    for i in range(t_level.sizes["time"]):
        t_slice = t_level.isel(time=i).values * units.kelvin
        u_slice = u_level.isel(time=i).values * units("m/s")
        v_slice = v_level.isel(time=i).values * units("m/s")

        theta_slice = mpcalc.potential_temperature(p_level, t_slice)
        fronto = mpcalc.frontogenesis(theta_slice, u_slice, v_slice, dx=dx, dy=dy)

        with np.errstate(invalid="ignore"):
            mask_slice = fronto > threshold_q

        front_masks.append(_to_binary_mask(mask_slice))

    front_mask_da = _build_mask_dataarray(
        front_masks,
        t_level,
        name="front_mask",
        attrs={
            "method": "frontogenesis",
            "pressure_level_hPa": pressure_level,
            "threshold": threshold,
            "threshold_units": "K / (100 km * 3 h)",
        },
    )

    logger.info("Frontogenesis identification complete.")
    return front_mask_da


def classify_front_type(
    front_mask: xr.DataArray | xr.Dataset,
    ds_t: xr.Dataset,
    ds_u: xr.Dataset,
    ds_v: xr.Dataset,
    advection_threshold: float = 1e-5,
    pressure_level: int | float = 850,
) -> xr.DataArray:
    """
    Classify frontal zones into cold, warm, or stationary fronts using horizontal
    temperature advection (-V · ∇T).

    Output codes:
        0: No front
        1: Cold front (cold air advection < -advection_threshold K/s)
        2: Warm front (warm air advection > +advection_threshold K/s)
        3: Stationary / neutral front (|advection| <= advection_threshold K/s)
    """
    logger.info(f"Classifying frontal zones via temperature advection at {pressure_level} hPa.")

    if isinstance(front_mask, xr.Dataset):
        mask_da = front_mask["front_mask"]
    else:
        mask_da = front_mask

    mask_da = mask_da.transpose("time", "latitude", "longitude")
    t_level = _extract_level(ds_t, "t", pressure_level)
    u_level = _extract_level(ds_u, "u", pressure_level)
    v_level = _extract_level(ds_v, "v", pressure_level)

    dx, dy = _grid_deltas(t_level)
    adv_thresh_q = advection_threshold * units("K/s")

    classified_slices: list[np.ndarray] = []
    for i in range(t_level.sizes["time"]):
        mask_slice = np.asarray(mask_da.isel(time=i).values, dtype=bool)
        t_slice = t_level.isel(time=i).values * units.kelvin
        u_slice = u_level.isel(time=i).values * units("m/s")
        v_slice = v_level.isel(time=i).values * units("m/s")

        advection = mpcalc.advection(t_slice, u=u_slice, v=v_slice, dx=dx, dy=dy)

        classified = np.zeros(mask_slice.shape, dtype=int)
        with np.errstate(invalid="ignore"):
            cold = mask_slice & _to_binary_mask(advection < -adv_thresh_q).astype(bool)
            warm = mask_slice & _to_binary_mask(advection > adv_thresh_q).astype(bool)
            stationary = mask_slice & ~cold & ~warm

        classified[cold] = 1
        classified[warm] = 2
        classified[stationary] = 3
        classified_slices.append(classified)

    return _build_mask_dataarray(
        classified_slices,
        t_level,
        name="front_type",
        attrs={
            "description": "Frontal type classification via horizontal temperature advection",
            "legend": "0=no_front, 1=cold_front, 2=warm_front, 3=stationary_front",
            "pressure_level_hPa": pressure_level,
            "advection_threshold_K_s": advection_threshold,
        },
    )


IDENTIFICATION_METHODS: dict[str, dict[str, Any]] = {
    "theta_e": {
        "func": identify_fronts,
        "required_vars": ("t", "q"),
        "default_threshold": 4e-5,
        "description": "Equivalent potential temperature gradient magnitude |∇θ_e|",
    },
    "f_diagnostic": {
        "func": identify_fronts_f_diagnostic,
        "required_vars": ("t", "u", "v"),
        "default_threshold": 1.0,
        "description": "Hewson/Parfitt F-diagnostic (vorticity × temperature gradient)",
    },
    "tfp": {
        "func": identify_fronts_tfp,
        "required_vars": ("t",),
        "optional_vars": ("q",),
        "default_threshold": 2e-5,
        "description": "Renard & Clarke Thermal Front Parameter (TFP)",
    },
    "frontogenesis": {
        "func": identify_fronts_frontogenesis,
        "required_vars": ("t", "u", "v"),
        "default_threshold": 0.5,
        "description": "Petterssen 2D horizontal kinematic frontogenesis",
    },
}


def identify_fronts_by_method(
    method: str,
    datasets: dict[str, xr.Dataset],
    threshold: float | None = None,
    pressure_level: int | float = 850,
) -> xr.DataArray:
    """
    Dispatch front identification by method name.

    Supported methods: 'theta_e', 'f_diagnostic', 'tfp', 'frontogenesis'.
    """
    if method not in IDENTIFICATION_METHODS:
        valid = ", ".join(sorted(IDENTIFICATION_METHODS))
        raise ValueError(f"Unknown identification method '{method}'. Choose from: {valid}")

    spec = IDENTIFICATION_METHODS[method]
    missing = [var for var in spec["required_vars"] if var not in datasets]
    if missing:
        raise ValueError(f"Method '{method}' requires datasets for variables: {missing}")

    effective_threshold = spec["default_threshold"] if threshold is None else threshold
    fn: Callable[..., xr.DataArray] = spec["func"]

    if method == "theta_e":
        return fn(
            datasets["t"],
            datasets["q"],
            threshold=effective_threshold,
            pressure_level=pressure_level,
        )
    if method == "f_diagnostic":
        return fn(
            datasets["t"],
            datasets["u"],
            datasets["v"],
            threshold=effective_threshold,
            pressure_level=pressure_level,
        )
    if method == "tfp":
        return fn(
            datasets["t"],
            ds_q=datasets.get("q"),
            grad_threshold=effective_threshold,
            pressure_level=pressure_level,
        )
    if method == "frontogenesis":
        return fn(
            datasets["t"],
            datasets["u"],
            datasets["v"],
            threshold=effective_threshold,
            pressure_level=pressure_level,
        )

    raise AssertionError(f"Unhandled method: {method}")
