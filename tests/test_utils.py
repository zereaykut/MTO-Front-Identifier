from unittest.mock import patch

import numpy as np
import xarray as xr

from src.utils import preprocess_era5_variable, preprocess_era5_variables


def test_preprocess_era5_variable_stitches_levels_and_years(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()

    # Create dummy GRIB placeholder paths spanning two years and two pressure fragments
    grib_files = [
        raw_dir / "era5_t_2024_p1.grib",
        raw_dir / "era5_t_2024_p2.grib",
        raw_dir / "era5_t_2025_p1.grib",
        raw_dir / "era5_t_2025_p2.grib",
    ]
    for path in grib_files:
        path.write_text("placeholder")

    lats = np.array([46.0, 44.0, 40.0, 35.0, 33.0])
    lons = np.array([24.0, 26.0, 30.0, 34.0, 36.0])

    def _fake_open_dataset(file_path, engine=None):
        assert engine == "cfgrib"
        name = file_path.name
        year = "2024" if "2024" in name else "2025"
        level = 850 if "p1" in name else 700
        times = np.array([f"{year}-01-01T00:00:00"], dtype="datetime64[ns]")
        data = np.full((1, 1, len(lats), len(lons)), float(level), dtype=np.float64)
        return xr.Dataset(
            {
                "t": xr.DataArray(
                    data,
                    dims=("time", "isobaricInhPa", "latitude", "longitude"),
                    coords={
                        "time": times,
                        "isobaricInhPa": [level],
                        "latitude": lats,
                        "longitude": lons,
                    },
                )
            }
        )

    with patch("src.utils.xr.open_dataset", side_effect=_fake_open_dataset):
        merged = preprocess_era5_variable(
            var_letter="t",
            output_name="era5_t.nc",
            lat_range=[35.25, 45.0],
            lon_range=[25.0, 34.75],
            input_location=raw_dir,
            output_location=out_dir,
        )

    assert merged is not None
    assert (out_dir / "era5_t.nc").exists()
    assert list(merged.isobaricInhPa.values) == [700, 850]
    assert merged.sizes["time"] == 2
    assert merged.sizes["latitude"] == 2  # 44.0, 40.0 within [35.25, 45.0]
    assert merged.sizes["longitude"] == 3  # 26.0, 30.0, 34.0 within [25.0, 34.75]


def test_preprocess_era5_variables_skips_existing_and_handles_missing(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    out_dir.mkdir()

    existing = out_dir / "era5_t.nc"
    xr.Dataset({"t": ("x", [1.0])}).to_netcdf(existing)

    processed = preprocess_era5_variables(
        variables=("t", "q"),
        lat_range=(35.0, 45.0),
        lon_range=(25.0, 35.0),
        input_location=raw_dir,
        output_location=out_dir,
        skip_existing=True,
    )

    assert processed == {"t": existing}
