import numpy as np
import pytest
import xarray as xr

from app import main as app_main
from src.identifier import (
    IDENTIFICATION_METHODS,
    classify_front_type,
    identify_fronts,
    identify_fronts_by_method,
    identify_fronts_f_diagnostic,
    identify_fronts_frontogenesis,
    identify_fronts_tfp,
)
from src.plot import plot_classified_fronts, plot_front_samples, plot_method_comparison
from src.utils import load_era5_datasets


def _make_synthetic_frontal_datasets(
    n_times: int = 2,
    pressure_levels: tuple[int, ...] = (850, 700),
) -> dict[str, xr.Dataset]:
    """Create synthetic ERA5-like datasets with a well-defined baroclinic frontal zone."""
    lats = np.linspace(45.0, 35.25, 40)
    lons = np.linspace(25.0, 34.75, 40)
    times = np.array(["2025-10-05T00:00:00", "2025-10-05T01:00:00"][:n_times], dtype="datetime64[ns]")

    lon_2d, lat_2d = np.meshgrid(lons, lats)
    # Sharp tilted frontal band across the center of the domain
    front_coord = (lat_2d - 40.0) - 0.35 * (lon_2d - 30.0)
    frontal_step = np.tanh(front_coord / 0.45)

    # Temperature: colder in NW (~274 K), warmer in SE (~288 K)
    t_base = 281.0 - 7.0 * frontal_step
    # Specific humidity: drier in cold air, moister in warm air
    q_base = 0.0065 - 0.0030 * frontal_step
    # Cyclonic wind shear + cross-front convergence for positive vorticity and frontogenesis
    u_base = -8.0 * frontal_step + 4.0
    v_base = -6.0 * frontal_step + 2.0

    def _build_var(name: str, field_2d: np.ndarray, units_str: str) -> xr.Dataset:
        data = np.zeros((n_times, len(pressure_levels), len(lats), len(lons)), dtype=np.float64)
        for ti in range(n_times):
            for pi in range(len(pressure_levels)):
                data[ti, pi] = field_2d - 0.1 * ti - 2.0 * pi
        return xr.Dataset(
            {
                name: xr.DataArray(
                    data,
                    dims=("time", "isobaricInhPa", "latitude", "longitude"),
                    coords={
                        "time": times,
                        "isobaricInhPa": list(pressure_levels),
                        "latitude": lats,
                        "longitude": lons,
                    },
                    attrs={"units": units_str},
                )
            }
        )

    return {
        "t": _build_var("t", t_base, "K"),
        "q": _build_var("q", q_base, "kg kg**-1"),
        "u": _build_var("u", u_base, "m s**-1"),
        "v": _build_var("v", v_base, "m s**-1"),
    }


def test_identify_fronts_theta_e_preserves_calculation_and_shape():
    ds = _make_synthetic_frontal_datasets()
    mask = identify_fronts(ds["t"], ds["q"])

    assert mask.name == "front_mask"
    assert mask.dims == ("time", "latitude", "longitude")
    assert mask.shape == (2, 40, 40)
    assert set(np.unique(mask.values)).issubset({0, 1})
    assert int(mask.sum()) > 0


def test_identify_fronts_f_diagnostic_preserves_calculation_and_shape():
    ds = _make_synthetic_frontal_datasets()
    mask = identify_fronts_f_diagnostic(ds["t"], ds["u"], ds["v"], threshold=1.0)

    assert mask.name == "front_mask"
    assert mask.dims == ("time", "latitude", "longitude")
    assert mask.shape == (2, 40, 40)
    assert set(np.unique(mask.values)).issubset({0, 1})
    assert int(mask.sum()) > 0


def test_identify_fronts_tfp_with_and_without_moisture():
    ds = _make_synthetic_frontal_datasets()
    mask_dry = identify_fronts_tfp(ds["t"])
    mask_moist = identify_fronts_tfp(ds["t"], ds_q=ds["q"])

    for mask in (mask_dry, mask_moist):
        assert mask.name == "front_mask"
        assert mask.dims == ("time", "latitude", "longitude")
        assert set(np.unique(mask.values)).issubset({0, 1})
        assert int(mask.sum()) > 0


def test_identify_fronts_frontogenesis():
    ds = _make_synthetic_frontal_datasets()
    mask = identify_fronts_frontogenesis(ds["t"], ds["u"], ds["v"], threshold=0.5)

    assert mask.name == "front_mask"
    assert mask.dims == ("time", "latitude", "longitude")
    assert set(np.unique(mask.values)).issubset({0, 1})
    assert int(mask.sum()) > 0


def test_classify_front_type_detects_cold_or_warm_advection():
    ds = _make_synthetic_frontal_datasets()
    mask = identify_fronts_f_diagnostic(ds["t"], ds["u"], ds["v"], threshold=1.0)
    classified = classify_front_type(mask, ds["t"], ds["u"], ds["v"])

    assert classified.name == "front_type"
    assert classified.dims == ("time", "latitude", "longitude")
    assert set(np.unique(classified.values)).issubset({0, 1, 2, 3})
    # Non-frontal pixels remain 0; frontal pixels become 1, 2, or 3
    assert np.all((classified.values > 0) == (mask.values == 1))


def test_uniform_field_produces_empty_mask_without_errors():
    ds = _make_synthetic_frontal_datasets(n_times=1)
    for var, const_val in (("t", 280.0), ("q", 0.005), ("u", 3.0), ("v", -2.0)):
        ds[var][var].values[:] = const_val

    for method in IDENTIFICATION_METHODS:
        mask = identify_fronts_by_method(method, ds)
        assert int(mask.sum()) == 0


def test_transposed_input_dims_and_dispatcher_validation():
    ds = _make_synthetic_frontal_datasets(n_times=1)
    ds_t_perm = ds["t"].transpose("latitude", "longitude", "time", "isobaricInhPa")
    ds_u_perm = ds["u"].transpose("longitude", "time", "latitude", "isobaricInhPa")
    mask = identify_fronts_f_diagnostic(ds_t_perm, ds_u_perm, ds["v"])
    assert mask.dims == ("time", "latitude", "longitude")

    with pytest.raises(ValueError, match="Unknown identification method"):
        identify_fronts_by_method("nonexistent", ds)

    with pytest.raises(ValueError, match="requires datasets"):
        identify_fronts_by_method("theta_e", {"t": ds["t"]})


def test_plotting_and_cli_pipeline_end_to_end(tmp_path):
    ds = _make_synthetic_frontal_datasets(n_times=2)
    for var, dset in ds.items():
        dset.to_netcdf(tmp_path / f"era5_{var}.nc")

    loaded = load_era5_datasets(tmp_path, ["t", "q", "u", "v"])
    try:
        assert set(loaded.keys()) == {"t", "q", "u", "v"}
    finally:
        for d in loaded.values():
            d.close()

    # Run full CLI across all methods with classification and comparison
    app_main(
        [
            "--method",
            "all",
            "--output-dir",
            str(tmp_path),
            "--no-time-range",
            "--max-plots",
            "1",
            "--classify",
            "--compare",
        ]
    )

    plots_dir = tmp_path / "plots"
    assert (tmp_path / "front_mask_f_diagnostic_850hPa.nc").exists()
    assert (tmp_path / "front_mask_theta_e_850hPa.nc").exists()
    assert (tmp_path / "front_mask_tfp_850hPa.nc").exists()
    assert (tmp_path / "front_mask_frontogenesis_850hPa.nc").exists()
    assert (plots_dir / "front_comparison_2025-10-05_00.png").exists()
    assert (plots_dir / "f_diagnostic" / "front_map_2025-10-05_00.png").exists()
    assert (plots_dir / "f_diagnostic" / "front_classified_2025-10-05_00.png").exists()


def test_direct_plot_helpers(tmp_path):
    ds = _make_synthetic_frontal_datasets(n_times=1)
    t_path = tmp_path / "era5_t.nc"
    ds["t"].to_netcdf(t_path)

    mask = identify_fronts_f_diagnostic(ds["t"], ds["u"], ds["v"])
    mask_path = tmp_path / "mask.nc"
    mask.to_netcdf(mask_path)

    ftype = classify_front_type(mask, ds["t"], ds["u"], ds["v"])
    ftype_path = tmp_path / "ftype.nc"
    ftype.to_netcdf(ftype_path)

    out_plots = plot_front_samples(mask_path, t_path, output_dir=tmp_path / "p1", max_plots=1)
    assert len(out_plots) == 1 and out_plots[0].exists()

    out_cls = plot_classified_fronts(ftype_path, t_path, output_dir=tmp_path / "p2", max_plots=1)
    assert len(out_cls) == 1 and out_cls[0].exists()

    comp = plot_method_comparison({"f_diagnostic": mask}, ds["t"]["t"], output_dir=tmp_path / "p3")
    assert comp is not None and comp.exists()


def test_exact_equivalence_with_original_formulas():
    """Verify preserved methods match original inline calculations element-for-element."""
    import metpy.calc as mpcalc
    from metpy.units import units

    ds = _make_synthetic_frontal_datasets(n_times=1)
    for var in ds.values():
        var.latitude.attrs["units"] = "degrees_north"
        var.longitude.attrs["units"] = "degrees_east"

    # Reference Method 1 (theta_e)
    t_850 = ds["t"].t.sel(isobaricInhPa=850) * units.kelvin
    q_850 = ds["q"].q.sel(isobaricInhPa=850) * units("kg/kg")
    p_850 = 850 * units.hPa
    theta_e = mpcalc.equivalent_potential_temperature(
        p_850, t_850, mpcalc.dewpoint_from_specific_humidity(p_850, q_850)
    ).transpose("time", "latitude", "longitude")
    dx, dy = mpcalc.lat_lon_grid_deltas(theta_e.longitude, theta_e.latitude)
    grad = mpcalc.gradient(theta_e.isel(time=0), deltas=(dy, dx))
    ref_theta_e = np.asarray(np.sqrt(grad[0] ** 2 + grad[1] ** 2) > (4e-5 * units("K/m")), dtype=int)

    got_theta_e = identify_fronts(ds["t"], ds["q"]).isel(time=0).values
    assert np.array_equal(got_theta_e, ref_theta_e)

    # Reference Method 2 (F-diagnostic)
    t_raw = ds["t"].t.sel(isobaricInhPa=850).isel(time=0).values * units.kelvin
    u_raw = ds["u"].u.sel(isobaricInhPa=850).isel(time=0).values * units("m/s")
    v_raw = ds["v"].v.sel(isobaricInhPa=850).isel(time=0).values * units("m/s")
    _, lat_2d = np.meshgrid(ds["t"].longitude.values, ds["t"].latitude.values)
    f = mpcalc.coriolis_parameter(lat_2d * units.degrees)
    grad_t_0 = 0.45 * units.kelvin / (100 * units.km)
    gt = mpcalc.gradient(t_raw, deltas=(dy, dx))
    gt_mag = np.sqrt(gt[0] ** 2 + gt[1] ** 2)
    vort = mpcalc.vorticity(u_raw, v_raw, dx=dx, dy=dy)
    f_param = (vort / f).to_base_units() * (gt_mag / grad_t_0).to_base_units()
    ref_f_diag = np.asarray(f_param > 1.0, dtype=int)

    got_f_diag = identify_fronts_f_diagnostic(ds["t"], ds["u"], ds["v"], threshold=1.0).isel(time=0).values
    assert np.array_equal(got_f_diag, ref_f_diag)
