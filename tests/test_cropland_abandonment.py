from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


MODULE_PATH = Path(__file__).resolve().parents[1] / "function" / "cropland_abandonment.py"
SPEC = importlib.util.spec_from_file_location("cropland_abandonment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reclassify_esa_cci_simplified9_and_binary_contract():
    landcover = xr.DataArray(
        [[10, 160, 110, 121, 150, 180, 190, 201, 210, np.nan]],
        coords={"lat": [1], "lon": list(range(10))},
        dims=("lat", "lon"),
        name="lccs_class",
    )

    simplified = MODULE.reclassify_esa_cci(landcover, mode="simplified9")
    binary = MODULE.reclassify_esa_cci(landcover, mode="binary_crop")

    np.testing.assert_array_equal(simplified.values, np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=np.uint8))
    np.testing.assert_array_equal(binary.values, np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8))
    assert simplified.encoding["_FillValue"] is None
    assert binary.encoding["_FillValue"] is None


def test_aggregate_categorical_mode_keeps_source_classes_and_prefers_center_on_tie():
    values = np.array([[10, 20, 10], [20, 20, 10], [30, 30, 40]], dtype=np.int16)
    landcover = xr.DataArray(values, coords={"lat": [2, 1, 0], "lon": [100, 101, 102]}, dims=("lat", "lon"))
    result = MODULE.aggregate_categorical_mode(landcover, lat_factor=3, lon_factor=3)
    assert result.shape == (1, 1)
    assert int(result.item()) == 20
    assert int(result.item()) in set(values.reshape(-1))
    assert float(result["lat"].item()) == pytest.approx(1.0)
    assert float(result["lon"].item()) == pytest.approx(101.0)


def test_categorical_mode_block_matches_window_reference_with_nan_and_dask():
    da = pytest.importorskip("dask.array")
    rng = np.random.default_rng(20260819)
    values = rng.integers(0, 10, size=(2, 6, 9)).astype(np.float32)
    values[0, :3, :3] = np.nan
    data = xr.DataArray(
        da.from_array(values, chunks=(1, 4, 5)),
        coords={"time": [2000, 2001], "lat": np.arange(6)[::-1], "lon": np.arange(9)},
        dims=("time", "lat", "lon"),
    )
    observed = MODULE.aggregate_categorical_mode(data, lat_factor=3, lon_factor=3).compute()
    expected = np.full((2, 2, 3), np.nan, dtype=np.float32)
    for time_index in range(2):
        for row in range(2):
            for col in range(3):
                window = values[time_index, row * 3 : (row + 1) * 3, col * 3 : (col + 1) * 3]
                valid = window[np.isfinite(window)]
                if valid.size == 0:
                    continue
                classes, counts = np.unique(valid, return_counts=True)
                winners = classes[counts == counts.max()]
                center = window[1, 1]
                expected[time_index, row, col] = center if np.isfinite(center) and center in winners else winners.min()
    np.testing.assert_allclose(observed.values, expected, equal_nan=True)


def test_categorical_mode_histogram_matches_unique_path_for_uint8():
    rng = np.random.default_rng(20260819)
    values = rng.integers(0, 10, size=(2, 12, 18), dtype=np.uint8)
    values[0, 0, 0] = values[0, 0, 1]
    landcover = xr.DataArray(
        values,
        coords={"time": [2000, 2001], "lat": np.arange(12)[::-1], "lon": np.arange(18)},
        dims=("time", "lat", "lon"),
    )
    observed = MODULE.aggregate_categorical_mode(landcover, lat_factor=3, lon_factor=3)
    expected = np.empty((2, 4, 6), dtype=np.uint8)
    for time_index in range(2):
        for row in range(4):
            for col in range(6):
                window = values[time_index, row * 3 : (row + 1) * 3, col * 3 : (col + 1) * 3]
                classes, counts = np.unique(window, return_counts=True)
                winners = classes[counts == counts.max()]
                center = window[1, 1]
                expected[time_index, row, col] = center if center in winners else winners.min()
    np.testing.assert_array_equal(observed.values, expected)


def test_temporal_majority_filter_uses_partial_windows_at_edges():
    years = pd.date_range("1992-01-01", periods=5, freq="YS")
    series = xr.DataArray([[10], [20], [20], [10], [10]], coords={"time": years, "lat": [40]}, dims=("time", "lat"))
    result = MODULE.temporal_majority_filter(series, window=5)
    np.testing.assert_array_equal(result[:, 0].values, np.array([20, 20, 10, 10, 10]))


def test_temporal_majority_filter_block_matches_series_reference_with_ties_and_nan():
    values = np.array(
        [
            [[1, 1], [np.nan, 2]],
            [[2, 1], [np.nan, 2]],
            [[1, 2], [3, 2]],
            [[2, 2], [3, np.nan]],
            [[2, 1], [3, 1]],
            [[1, 1], [np.nan, 1]],
            [[1, 2], [2, 1]],
        ],
        dtype=np.float32,
    )
    data = xr.DataArray(
        values,
        coords={"time": pd.date_range("2000-01-01", periods=7, freq="YS"), "lat": [1, 0], "lon": [10, 11]},
        dims=("time", "lat", "lon"),
    )

    expected = np.full_like(values, np.nan)
    for lat_index in range(values.shape[1]):
        for lon_index in range(values.shape[2]):
            series = values[:, lat_index, lon_index]
            for time_index in range(values.shape[0]):
                start = max(0, time_index - 2)
                stop = min(values.shape[0], time_index + 3)
                valid = series[start:stop][np.isfinite(series[start:stop])]
                if valid.size == 0:
                    continue
                classes, counts = np.unique(valid, return_counts=True)
                winners = classes[counts == counts.max()]
                center = series[time_index]
                expected[time_index, lat_index, lon_index] = center if np.isfinite(center) and center in winners else winners.min()

    result = MODULE.temporal_majority_filter(data, window=5)
    np.testing.assert_allclose(result.values, expected, equal_nan=True)


def test_detect_abandonment_selects_latest_valid_event_and_cutoff_requires_elapsed_duration():
    years = pd.date_range("1992-01-01", periods=15, freq="YS")
    values = np.array([10, 10, 10, 10, 10, 50, 50, 50, 50, 10, 10, 50, 50, 50, 50], dtype=np.int16).reshape(15, 1, 1)
    landcover = xr.DataArray(values, coords={"time": years, "lat": [40], "lon": [100]}, dims=("time", "lat", "lon"), attrs={"crs": "EPSG:4326"})
    result = MODULE.detect_abandonment(
        landcover,
        baseline_years=(1992, 1994),
        min_abandonment_years=4,
        required_through_year=2002,
        analysis_end_year=2006,
        recultivation_years=1,
    )
    assert int(result["abandonment_year"].item()) == 2003
    assert int(result["abandonment_duration"].item()) == 4
    assert int(result["qualifies_at_cutoff"].item()) == 0
    assert int(result["current_abandonment"].item()) == 1


def test_detect_abandonment_rejects_excluded_transition_within_first_five_years_and_truncates_after():
    years = pd.date_range("1992-01-01", periods=12, freq="YS")
    values = np.array(
        [[10, 10], [10, 10], [10, 10], [10, 10], [10, 10], [50, 50], [50, 50], [190, 50], [50, 50], [50, 190], [50, 50], [50, 50]],
        dtype=np.int16,
    ).reshape(12, 1, 2)
    landcover = xr.DataArray(values, coords={"time": years, "lat": [40], "lon": [100, 101]}, dims=("time", "lat", "lon"))
    result = MODULE.detect_abandonment(
        landcover,
        baseline_years=(1992, 1994),
        min_abandonment_years=3,
        required_through_year=1999,
        analysis_end_year=2003,
    )
    assert np.isnan(result["abandonment_year"].isel(lat=0, lon=0).item())
    assert int(result["abandonment_year"].isel(lat=0, lon=1).item()) == 1997
    assert int(result["abandonment_end_year"].isel(lat=0, lon=1).item()) == 2000


def test_detect_abandonment_xie_window_allows_configured_anomalies():
    years = pd.date_range("1992-01-01", periods=12, freq="YS")
    values = np.array([10, 10, 50, 10, 10, 50, 50, 10, 50, 50, 50, 50], dtype=np.int16).reshape(12, 1, 1)
    landcover = xr.DataArray(values, coords={"time": years, "lat": [40], "lon": [100]}, dims=("time", "lat", "lon"))
    result = MODULE.detect_abandonment(
        landcover,
        detector="xie_window",
        pre_window=5,
        post_window=5,
        max_noncrop_pre_years=1,
        max_crop_post_years=1,
        min_abandonment_years=3,
        required_through_year=2000,
        analysis_end_year=2003,
    )
    assert int(result["abandonment_year"].item()) == 1997


def test_detect_abandonment_handles_simplified_schema_and_missing_years():
    years = pd.date_range("1992-01-01", periods=6, freq="YS")
    simplified = xr.DataArray(np.array([1, 1, 1, 0, 2, 2], dtype=np.uint8).reshape(6, 1, 1), coords={"time": years, "lat": [40], "lon": [100]}, dims=("time", "lat", "lon"))
    result = MODULE.detect_abandonment(
        simplified,
        schema="simplified9",
        baseline_years=(1992, 1994),
        min_abandonment_years=2,
        required_through_year=1996,
        analysis_end_year=1997,
    )
    assert np.isnan(result["abandonment_year"].item())


def test_detect_abandonment_handles_raw_esa_zero_as_nodata():
    years = pd.date_range("1992-01-01", periods=6, freq="YS")
    raw = xr.DataArray(
        np.array([10, 10, 10, 0, 50, 50], dtype=np.int16).reshape(6, 1, 1),
        coords={"time": years, "lat": [40], "lon": [100]},
        dims=("time", "lat", "lon"),
    )
    result = MODULE.detect_abandonment(
        raw,
        schema="raw_esa",
        baseline_years=(1992, 1994),
        min_abandonment_years=2,
        required_through_year=1996,
        analysis_end_year=1997,
    )
    assert np.isnan(result["abandonment_year"].item())


def test_reclassify_lcmap_lcpri_maps_known_values_and_fails_unknown():
    lcmap = xr.DataArray([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dims=("y", "x"))
    mapped = MODULE.reclassify_lcmap_lcpri(lcmap)
    np.testing.assert_array_equal(mapped.values, np.array([[0, 7, 1, 4, 2, 9, 6, 8, 5]], dtype=np.uint8))
    with pytest.raises(ValueError, match="unknown values"):
        MODULE.reclassify_lcmap_lcpri(xr.DataArray([[9]], dims=("y", "x")))


def test_reclassify_lcmap_lcpri_attrs_are_netcdf_serializable(tmp_path):
    source = xr.DataArray(
        np.arange(9, dtype=np.uint8).reshape(1, 3, 3),
        dims=("time", "y", "x"),
        coords={"time": [2020], "y": [60.0, 30.0, 0.0], "x": [0.0, 30.0, 60.0]},
    )
    result = MODULE.reclassify_lcmap_lcpri(source, output_name="lccs_class")
    destination = tmp_path / "lcmap_reclass.nc"
    result.to_dataset().to_netcdf(destination)
    with xr.open_dataset(destination) as observed:
        assert observed.lccs_class.attrs["source_mapping"].startswith("{")


def test_cell_area_m2_reuses_unique_lon_width_calls(monkeypatch):
    class FakeGeod:
        def __init__(self):
            self.calls = []

        def polygon_area_perimeter(self, lons, lats):
            self.calls.append((tuple(lons), tuple(lats)))
            width = abs(lons[1] - lons[0])
            height = abs(lats[2] - lats[1])
            return width * height, 0.0

    fake = FakeGeod()
    monkeypatch.setattr(MODULE, "WGS84_GEOD", fake)
    area = MODULE.cell_area_m2(np.array([1.5, 0.5]), np.array([100.5, 101.5, 102.5]))
    assert area.shape == (2, 3)
    assert len(fake.calls) == 2


def test_cell_area_m2_supports_projected_native_crs_and_yx_dims():
    area = MODULE.cell_area_m2(
        np.array([1000.0, 900.0]),
        np.array([500.0, 600.0, 700.0]),
        spatial_dims=("y", "x"),
        crs="EPSG:5070",
    )
    assert area.dims == ("y", "x")
    np.testing.assert_array_equal(area.values, np.full((2, 3), 10000.0))


def test_write_validate_summarize_publish_and_reference_compare(tmp_path: Path):
    dataset = xr.Dataset(
        data_vars={
            "abandonment_year": (("lat", "lon"), np.array([[1997.0, np.nan]], dtype=np.float32)),
            "abandonment_duration": (("lat", "lon"), np.array([[5, 0]], dtype=np.int16)),
            "recultivation": (("lat", "lon"), np.array([[0, 0]], dtype=np.uint8)),
            "current_abandonment": (("lat", "lon"), np.array([[1, 0]], dtype=np.uint8)),
            "abandonment_end_year": (("lat", "lon"), np.array([[2001.0, np.nan]], dtype=np.float32)),
            "qualifies_at_cutoff": (("lat", "lon"), np.array([[1, 0]], dtype=np.uint8)),
        },
        coords={"lat": [40.0], "lon": [100.0, 101.0]},
        attrs={"crs": "EPSG:4326", "class_codes": np.array([1, 6, 7], dtype=np.int16)},
    )
    for name in ("recultivation", "current_abandonment", "qualifies_at_cutoff"):
        dataset[name].attrs.update({"flag_values": [0, 1], "flag_meanings": "false true"})
        dataset[name].encoding["_FillValue"] = None
    area = xr.DataArray(np.array([[10_000.0, 20_000.0]]), coords=dataset["current_abandonment"].coords, dims=dataset["current_abandonment"].dims)
    state_codes = xr.DataArray(np.array([[6, 0]]), coords=dataset["current_abandonment"].coords, dims=dataset["current_abandonment"].dims)

    chunk_paths = MODULE.write_abandonment_chunks(dataset, tmp_path / "chunks", chunk_slices=[{"lon": slice(0, 2)}], chunk_indices=[(0, 1)])
    reopened = xr.open_dataset(chunk_paths[0])
    checks = MODULE.validate_roundtrip(dataset, reopened)
    invalid_encoding = reopened.copy(deep=True)
    invalid_encoding["current_abandonment"].encoding["_FillValue"] = np.uint8(255)
    invalid_checks = MODULE.validate_roundtrip(dataset, invalid_encoding)
    summaries = MODULE.summarize_key_numbers(dataset, area, state_codes=state_codes)
    comparison = MODULE.compare_reference_abandonment(
        dataset,
        xr.DataArray(np.array([[1997.0, np.nan]], dtype=np.float32), coords=dataset["abandonment_year"].coords, dims=dataset["abandonment_year"].dims),
        area,
        state_codes=state_codes,
    )
    comparison_interval = MODULE.compare_reference_abandonment(
        dataset,
        xr.DataArray(np.array([[1997.0, 2001.0]], dtype=np.float32), coords=dataset["abandonment_year"].coords, dims=dataset["abandonment_year"].dims),
        area,
        comparison_years=(1997, 1999),
        state_codes=state_codes,
    )
    written = MODULE.publish_validation_results(
        run_id="unit-test",
        repo_result_root=tmp_path / "publish",
        manifest={"run_id": "unit-test", "hash": "abc"},
        summaries=summaries,
        report_text="# Validation\n",
    )

    assert chunk_paths[0].name == "chunk_000_001.nc"
    assert checks["passed"] is True
    assert invalid_checks["encoding_fillvalue_equal"]["current_abandonment"] is False
    assert invalid_checks["passed"] is False
    assert summaries["overall"]["pixels"].tolist() == [1, 1, 1, 0]
    assert summaries["by_state"]["state_code"].tolist() == [6]
    assert comparison["overall"]["pixels"].tolist() == [1, 1, 0, 0]
    assert comparison["by_state"]["overlap_area_ha"].tolist() == [1.0]
    assert comparison_interval["overall"]["comparison_start_year"].tolist() == [1997, 1997, 1997, 1997]
    assert comparison_interval["overall"]["comparison_end_year"].tolist() == [1999, 1999, 1999, 1999]
    assert comparison_interval["by_year"]["abandonment_year"].tolist() == [1997]
    assert written["manifest"].exists()
    assert not list((tmp_path / "publish").rglob("*.tmp*"))


def test_write_abandonment_chunks_supports_resume_chunk_size_and_manifest(tmp_path: Path):
    dataset = xr.Dataset(
        {
            "abandonment_year": (("y", "x"), np.array([[1997.0, np.nan], [1998.0, 1999.0]], dtype=np.float32)),
            "abandonment_duration": (("y", "x"), np.array([[5.0, np.nan], [4.0, 3.0]], dtype=np.float32)),
            "recultivation": (("y", "x"), np.array([[0, 0], [0, 0]], dtype=np.uint8)),
            "current_abandonment": (("y", "x"), np.array([[1, 0], [1, 1]], dtype=np.uint8)),
            "abandonment_end_year": (("y", "x"), np.array([[2001.0, np.nan], [2001.0, 2001.0]], dtype=np.float32)),
            "qualifies_at_cutoff": (("y", "x"), np.array([[1, 0], [1, 1]], dtype=np.uint8)),
        },
        coords={"y": [40.0, 39.0], "x": [100.0, 101.0]},
    )
    first = MODULE.write_abandonment_chunks(
        dataset,
        tmp_path / "xie_chunks",
        spatial_dims=("y", "x"),
        chunk_size=1,
        resume=False,
        manifest={"source": "unit"},
    )
    second = MODULE.write_abandonment_chunks(
        dataset,
        tmp_path / "xie_chunks",
        spatial_dims=("y", "x"),
        chunk_size=1,
        resume=True,
        manifest={"source": "unit"},
    )
    manifest = json.loads((tmp_path / "xie_chunks" / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(first) == 4
    assert len(second) == 4
    assert manifest["spatial_dims"] == ["y", "x"]
    assert manifest["stats"]["skipped"] == 4


def test_compare_reference_abandonment_rejects_invalid_year_interval():
    dataset = xr.Dataset(
        data_vars={"abandonment_year": (("lat", "lon"), np.array([[1997.0]], dtype=np.float32))},
        coords={"lat": [40.0], "lon": [100.0]},
    )
    area = xr.DataArray(np.array([[10_000.0]]), coords=dataset["abandonment_year"].coords, dims=dataset["abandonment_year"].dims)
    reference = xr.DataArray(np.array([[1997.0]], dtype=np.float32), coords=dataset["abandonment_year"].coords, dims=dataset["abandonment_year"].dims)
    with pytest.raises(ValueError, match="start <="):
        MODULE.compare_reference_abandonment(dataset, reference, area, comparison_years=(2000, 1999))


def test_xie_reference_to_event_dataset_and_chunk_interface(tmp_path: Path):
    reference = xr.DataArray(
        np.array([[1997.0, np.nan], [1998.0, 1999.0]], dtype=np.float32),
        coords={"y": [40.0, 39.0], "x": [100.0, 101.0]},
        dims=("y", "x"),
        attrs={"source_crs": "EPSG:4326"},
    )
    events = MODULE.xie_reference_to_event_dataset(
        reference,
        current_year=2020,
        cutoff_year=2020,
        min_abandonment_years=5,
        spatial_dims=("y", "x"),
    )
    assert events["current_abandonment"].dtype == np.uint8
    assert events["current_abandonment"].encoding.get("_FillValue") is None
    assert int(events["qualifies_at_cutoff"].isel(y=0, x=0).item()) == 1
    written = MODULE.write_xie_event_chunks(
        reference,
        tmp_path / "event_chunks",
        current_year=2020,
        cutoff_year=2020,
        min_abandonment_years=5,
        spatial_dims=("y", "x"),
        chunk_size=1,
    )
    assert len(written["chunk_paths"]) == 4
    manifest = json.loads((tmp_path / "event_chunks" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_contract"] == "xie_event_geotiff"


def test_legacy_pipeline_supports_yx_spatial_dims(tmp_path: Path):
    years = pd.date_range("1992-01-01", periods=8, freq="YS")
    landcover = xr.DataArray(
        np.array(
            [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
                [[0, 0], [7, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            dtype=np.uint8,
        ),
        coords={"time": years, "y": [40.0, 39.0], "x": [100.0, 101.0]},
        dims=("time", "y", "x"),
        name="lccs_class",
        attrs={"crs": "EPSG:5070"},
    )
    mask = MODULE.build_legacy_candidate_mask(landcover)
    detected = MODULE.detect_legacy_equivalent_chunk(landcover, mask, spatial_dims=("y", "x"))
    land_path = tmp_path / "land.nc"
    mask_path = tmp_path / "mask.nc"
    landcover.to_dataset(name="lccs_class").to_netcdf(land_path)
    mask.to_dataset(name="final_mask").to_netcdf(mask_path)
    manifest = MODULE.write_legacy_equivalent_chunks(
        land_path,
        mask_path,
        tmp_path / "legacy_chunks",
        chunk_size=1,
        spatial_dims=("y", "x"),
        time_dim="time",
        landcover_var="lccs_class",
        mask_var="final_mask",
    )
    report = MODULE.validate_legacy_chunk_set(
        land_path,
        mask_path,
        tmp_path / "legacy_chunks",
        chunk_size=1,
        spatial_dims=("y", "x"),
        time_dim="time",
        landcover_var="lccs_class",
        mask_var="final_mask",
    )
    merged = MODULE.merge_landcover_into_legacy_chunks(
        land_path,
        tmp_path / "legacy_chunks",
        tmp_path / "legacy_merged",
        spatial_dims=("y", "x"),
        time_dim="time",
        landcover_var="lccs_class",
    )
    assert detected.sizes["y"] == 2
    assert manifest["spatial_dims"] == ["y", "x"]
    assert report["passed"] is True
    assert merged["passed"] is True


def test_fetch_reference_raster_archive_chain(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tif_bytes = b"fake-tif-binary"
    archive_member_name = "nested/reference.tif"
    archive_source = tmp_path / "source.tar.gz"
    with tarfile.open(archive_source, "w:gz") as archive:
        temp_tif = source_dir / "reference.tif"
        temp_tif.write_bytes(tif_bytes)
        archive.add(temp_tif, arcname=archive_member_name)

    archive_sha = hashlib.sha256(archive_source.read_bytes()).hexdigest()
    raster_sha = hashlib.sha256(tif_bytes).hexdigest()
    output = MODULE.fetch_reference_raster(
        archive_source=archive_source,
        archive_path=tmp_path / "copied.tar.gz",
        archive_member_name=archive_member_name,
        extracted_raster_path=tmp_path / "reference.tif",
        expected_archive_sha256=archive_sha,
        expected_archive_size=archive_source.stat().st_size,
        expected_raster_sha256=raster_sha,
        expected_raster_size=len(tif_bytes),
    )
    assert output.exists()
    assert output.read_bytes() == tif_bytes


def test_validate_reference_raster_uses_stats_tags_without_full_read(monkeypatch, tmp_path: Path):
    class FakeDataset:
        width = 214419
        height = 93324
        count = 1
        dtypes = ("int16",)
        crs = type("Crs", (), {"to_string": lambda self: "EPSG:4326"})()
        transform = (1, 0, 0, 0, -1, 0)
        nodata = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def tags(self, band):
            return {"STATISTICS_MINIMUM": "1991", "STATISTICS_MAXIMUM": "2014"}

        def read(self, *args, **kwargs):
            raise AssertionError("full raster read should not occur")

    rasterio = pytest.importorskip("rasterio")
    path = tmp_path / "fake.tif"
    path.write_bytes(b"abc")
    monkeypatch.setattr(rasterio, "open", lambda _: FakeDataset())
    metadata = MODULE.validate_reference_raster(
        path,
        expected_crs="EPSG:4326",
        expected_count=1,
        expected_dtype="int16",
        expected_year_min=1991,
        expected_year_max=2014,
        max_pixels_for_scan=1,
    )
    assert metadata["year_min"] == 1991
    assert metadata["year_max"] == 2014


def test_open_reference_abandonment_year_is_lazy_and_avoids_rasterio_read(monkeypatch, tmp_path: Path):
    rioxarray = pytest.importorskip("rioxarray")
    da = pytest.importorskip("dask.array")

    data = da.from_array(np.array([[[1997.0, 0.0], [1998.0, 1999.0]]], dtype=np.float32), chunks=(1, 2, 2))
    opened = xr.DataArray(data, dims=("band", "y", "x"), coords={"band": [1], "y": [40.0, 39.0], "x": [100.0, 101.0]})
    opened = opened.rio.write_crs("EPSG:4326")

    monkeypatch.setattr(rioxarray, "open_rasterio", lambda *args, **kwargs: opened)
    result = MODULE.open_reference_abandonment_year(tmp_path / "lazy.tif", nodata=0)
    assert hasattr(result.data, "compute")
    assert np.isnan(result.compute().isel(y=0, x=1).item())


@pytest.mark.parametrize(
    ("method", "expected_resampling"),
    [("nearest", "nearest"), ("mode", "mode")],
)
def test_align_reference_abandonment_uses_bounded_reproject(
    monkeypatch, tmp_path: Path, method: str, expected_resampling: str
):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.enums import Resampling

    calls = {}

    class FakeDataset:
        transform = (1, 0, 0, 0, -1, 0)
        crs = "EPSG:4326"
        nodata = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_open(path):
        calls["opened"] = str(path)
        return FakeDataset()

    def fake_band(src, index):
        calls["band"] = index
        return ("band", index)

    def fake_reproject(**kwargs):
        calls["destination_shape"] = kwargs["destination"].shape
        calls["resampling"] = kwargs["resampling"]
        kwargs["destination"][:] = 2001.0

    monkeypatch.setattr(rasterio, "open", fake_open)
    monkeypatch.setattr(rasterio, "band", fake_band)
    monkeypatch.setattr("rasterio.warp.reproject", fake_reproject)

    ref = xr.DataArray(np.zeros((1, 1), dtype=np.float32), dims=("y", "x"), attrs={"source_path": str(tmp_path / "src.tif")})
    target = xr.DataArray(np.zeros((2, 2), dtype=np.float32), coords={"lat": [40.0, 39.0], "lon": [100.0, 101.0]}, dims=("lat", "lon"))
    aligned = MODULE.align_reference_abandonment(ref, target, method=method)
    assert aligned.shape == (2, 2)
    assert float(aligned.isel(lat=0, lon=0).item()) == 2001.0
    assert calls["destination_shape"] == (2, 2)
    assert calls["resampling"] == getattr(Resampling, expected_resampling)


def test_reference_local_file_end_to_end_small(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    rioxarray = pytest.importorskip("rioxarray")
    from rasterio.transform import from_origin

    source = tmp_path / "source.tif"
    transform = from_origin(99.5, 40.5, 1.0, 1.0)
    data = np.array([[1997, 0], [1998, 1999]], dtype=np.uint16)
    with rasterio.open(source, "w", driver="GTiff", width=2, height=2, count=1, dtype="uint16", crs="EPSG:4326", transform=transform, nodata=0) as dst:
        dst.write(data, 1)
        dst.update_tags(1, STATISTICS_MINIMUM="1997", STATISTICS_MAXIMUM="1999")

    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    fetched = MODULE.fetch_reference_raster(source=source, target_path=tmp_path / "copied.tif", expected_sha256=sha256, expected_size=source.stat().st_size)
    metadata = MODULE.validate_reference_raster(fetched, expected_sha256=sha256, expected_size=source.stat().st_size, expected_nodata=0, expected_year_min=1997, expected_year_max=1999)
    opened = MODULE.open_reference_abandonment_year(fetched, nodata=0)

    assert fetched.exists()
    assert metadata["year_min"] == 1997
    assert np.isnan(opened.compute().isel(y=0, x=1).item())


def test_dask_compatibility_when_available():
    da = pytest.importorskip("dask.array")
    years = pd.date_range("1992-01-01", periods=6, freq="YS")
    data = da.from_array(np.array([10, 10, 10, 50, 50, 50], dtype=np.int16).reshape(6, 1, 1), chunks=(3, 1, 1))
    landcover = xr.DataArray(data, coords={"time": years, "lat": [40], "lon": [100]}, dims=("time", "lat", "lon"))
    filtered = MODULE.temporal_majority_filter(landcover, window=3)
    detected = MODULE.detect_abandonment(landcover, baseline_years=(1992, 1994), min_abandonment_years=2, required_through_year=1996, analysis_end_year=1997)
    summary = MODULE.summarize_key_numbers(detected, MODULE.cell_area_m2(detected["lat"], detected["lon"]))
    roundtrip = MODULE.validate_roundtrip(detected, detected.compute())
    assert hasattr(filtered.data, "compute")
    assert hasattr(detected["abandonment_year"].data, "compute")
    assert summary["overall"]["pixels"].tolist() == [1, 1, 1, 0]
    assert roundtrip["passed"] is True


def test_mode_reclass_monitor_and_legacy_chunk_pipeline(tmp_path: Path):
    years = pd.date_range("1992-01-01", periods=10, freq="YS")
    values = np.array(
        [
            [[1, 1], [0, 1]],
            [[1, 1], [0, 1]],
            [[2, 2], [0, 7]],
            [[2, 2], [0, 2]],
            [[2, 2], [0, 2]],
            [[2, 2], [0, 2]],
            [[2, 2], [0, 2]],
            [[2, 1], [0, 2]],
            [[2, 1], [0, 2]],
            [[2, 2], [0, 2]],
        ],
        dtype=np.uint8,
    )
    landcover = xr.Dataset(
        {"lccs_class": (("time", "lat", "lon"), values)},
        coords={"time": years, "lat": [1.0, 0.0], "lon": [10.0, 11.0]},
        attrs={"spatial_aggregation": "3x3_categorical_mode"},
    )
    source_path = tmp_path / "reclass_lccs_1km.nc"
    landcover.to_netcdf(source_path)
    readiness = MODULE.monitor_mode_reclass_file(
        source_path,
        poll_seconds=0,
        stable_seconds=0,
        timeout_seconds=1,
        validation_kwargs={
            "expected_years": 10,
            "expected_lat": 2,
            "expected_lon": 2,
            "expected_resolution": 1.0,
        },
    )
    assert readiness["passed"] is True

    mask = MODULE.build_legacy_candidate_mask(landcover.lccs_class)
    np.testing.assert_array_equal(mask.values, np.array([[1, 1], [0, 0]], dtype=np.uint8))
    mask_path = tmp_path / "mask.nc"
    mask.to_dataset().to_netcdf(mask_path)
    chunk_dir = tmp_path / "abandon_mode_2"
    manifest = MODULE.write_legacy_equivalent_chunks(
        source_path,
        mask_path,
        chunk_dir,
        chunk_size=1,
    )
    assert manifest["stats"] == {"total": 2, "generated": 2, "skipped": 0, "failed": 0}
    detection_manifest_path = chunk_dir / "run_manifest.json"
    detection_manifest_mtime = detection_manifest_path.stat().st_mtime_ns
    resumed_detection = MODULE.write_legacy_equivalent_chunks(
        source_path,
        mask_path,
        chunk_dir,
        chunk_size=1,
        resume=True,
    )
    assert resumed_detection["stats"] == manifest["stats"]
    assert detection_manifest_path.stat().st_mtime_ns == detection_manifest_mtime
    with xr.open_dataset(chunk_dir / "chunk_0_0.nc") as detected:
        assert float(detected.abandonment_year.item()) == 1994.0
        assert float(detected.abandonment_duration.item()) == 8.0
        assert int(detected.current_abandonment.item()) == 1
    with xr.open_dataset(chunk_dir / "chunk_0_1.nc") as detected:
        assert float(detected.abandonment_year.item()) == 1994.0
        assert float(detected.abandonment_duration.item()) == 5.0
        assert int(detected.recultivation.item()) == 1
        assert int(detected.current_abandonment.item()) == 0

    report = MODULE.validate_legacy_chunk_set(
        source_path,
        mask_path,
        chunk_dir,
        chunk_size=1,
        sample_pixels=10,
    )
    assert report["passed"] is True
    merged_dir = tmp_path / "merged_chunk_mode_2"
    merged = MODULE.merge_landcover_into_legacy_chunks(
        source_path,
        chunk_dir,
        merged_dir,
        chunk_size=1,
        compression_level=1,
        resume=True,
    )
    assert merged["passed"] is True
    resumed_merge = MODULE.merge_landcover_into_legacy_chunks(
        source_path,
        chunk_dir,
        merged_dir,
        chunk_size=1,
        compression_level=1,
        resume=True,
    )
    assert resumed_merge["generated"] == 0
    assert resumed_merge["skipped"] == 2
    with xr.open_dataset(merged_dir / "chunk_0_0.nc") as dataset:
        assert "landcover" in dataset
        assert dataset.landcover.sizes["time"] == 10
        assert dataset.landcover.encoding["complevel"] == 1


def test_legacy_equivalent_series_golden_cases():
    years = np.arange(1992, 2002)
    assert MODULE.legacy_equivalent_detect_series([1, 1, 2, 2, 2, 2, 2, 2, 2, 2], years) == (1994.0, 8.0, 0, 1)
    assert MODULE.legacy_equivalent_detect_series([1, 1, 2, 2, 2, 2, 2, 1, 1, 2], years) == (1994.0, 5.0, 1, 0)
    result = MODULE.legacy_equivalent_detect_series([1, 2, 2, 2, 2, 2, 2, 2, 2, 2], years)
    assert np.isnan(result[0]) and np.isnan(result[1]) and result[2:] == (0, 0)


def test_legacy_numba_kernel_matches_python_reference_when_available():
    if MODULE._legacy_detect_many_numba is None:
        pytest.skip("numba unavailable")
    rng = np.random.default_rng(20260818)
    series = rng.choice(np.array([0, 1, 2, 7], dtype=np.float32), size=(512, 31))
    years = np.arange(1992, 2023, dtype=np.int16)
    expected = MODULE._legacy_detect_many_python(series, years)
    observed = MODULE._legacy_detect_many_numba(np.ascontiguousarray(series), years)
    for left, right in zip(expected, observed):
        np.testing.assert_allclose(left, right, equal_nan=True)


def _cutoff_safe_series_result(
    values,
    *,
    years=None,
    window_year=5,
    init_cropland=2,
    current_end_year=2020,
    extend_validation=True,
):
    values = np.asarray(values, dtype=np.float32)
    years = np.asarray(years if years is not None else np.arange(1992, 1992 + values.size))
    landcover = xr.DataArray(
        values.reshape(values.size, 1, 1),
        coords={"time": years, "lat": [40.0], "lon": [-120.0]},
        dims=("time", "lat", "lon"),
        name="lccs_class",
    )
    return MODULE._detect_cutoff_safe_chunk(
        landcover,
        window_year=window_year,
        start_year=1992,
        current_end_year=current_end_year,
        init_cropland=init_cropland,
        extend_validation=extend_validation,
    )


def test_run_abandonment_detection_has_canonical_signature_and_defaults():
    signature = inspect.signature(MODULE.run_abandonment_detection)
    assert list(signature.parameters) == [
        "target_nc",
        "window_year",
        "start_year",
        "current_end_year",
        "init_cropland",
        "extend_validation",
    ]
    assert signature.parameters["target_nc"].default == Path(r"D:\xarray\reclass_lccs_1km.nc")
    assert signature.parameters["window_year"].default == 5
    assert signature.parameters["start_year"].default == 1992
    assert signature.parameters["current_end_year"].default == 2020
    assert signature.parameters["init_cropland"].default == 2
    assert signature.parameters["extend_validation"].default is True
    assert "run_abandonment_detection" in MODULE.__all__


def test_cutoff_safe_detector_intentionally_rejects_2018_min5_completed_only_by_extension():
    years = np.arange(1992, 2023)
    values = np.ones(years.size, dtype=np.float32)
    values[years >= 2018] = 2

    legacy = MODULE.legacy_equivalent_detect_series(values, years)
    observed = _cutoff_safe_series_result(values, years=years, window_year=5)

    assert legacy == (2018.0, 5.0, 0, 1)
    assert np.isnan(observed.abandonment_year.item())
    assert np.isnan(observed.abandonment_duration.item())
    assert int(observed.qualifies_at_cutoff.item()) == 0
    assert int(observed.persistence_validated.item()) == 0


@pytest.mark.parametrize("bad_extension", ([1, 2], [7, 2], [np.nan, 2], [0, 2], [10, 2]))
def test_cutoff_safe_extension_only_changes_persistence_inclusion(bad_extension):
    years = np.arange(1992, 2023)
    good_values = np.ones(years.size, dtype=np.float32)
    good_values[years >= 2016] = 2
    bad_values = good_values.copy()
    bad_values[-2:] = bad_extension

    good = _cutoff_safe_series_result(good_values, years=years)
    bad = _cutoff_safe_series_result(bad_values, years=years)

    for field in (
        "abandonment_year",
        "abandonment_duration",
        "abandonment_end_year",
        "qualifies_at_cutoff",
        "current_abandonment",
    ):
        assert bad[field].item() == good[field].item()
    assert float(good.abandonment_year.item()) == 2016.0
    assert float(good.abandonment_duration.item()) == 5.0
    assert int(good.persistence_validated.item()) == 1
    assert int(bad.persistence_validated.item()) == 0


def test_cutoff_safe_missing_extension_fails_closed_and_false_never_requires_it():
    full_years = np.arange(1992, 2023)
    full_values = np.ones(full_years.size, dtype=np.float32)
    full_values[full_years >= 2016] = 2
    good = _cutoff_safe_series_result(full_values, years=full_years)

    through_2021_years = np.arange(1992, 2022)
    through_2021_values = full_values[: through_2021_years.size]
    missing = _cutoff_safe_series_result(through_2021_values, years=through_2021_years)
    cutoff_only = _cutoff_safe_series_result(
        through_2021_values[:29],
        years=np.arange(1992, 2021),
        extend_validation=False,
    )

    for field in ("abandonment_year", "abandonment_duration", "qualifies_at_cutoff", "current_abandonment"):
        assert missing[field].item() == good[field].item()
    assert int(missing.persistence_validated.item()) == 0
    assert int(cutoff_only.persistence_validated.item()) == 1
    assert cutoff_only.attrs["extension_years"] == []


def test_cutoff_safe_window_and_init_cropland_control_both_eligibility_and_exit_prefix():
    years = np.arange(1992, 2023)
    exit_2017 = np.ones(years.size, dtype=np.float32)
    exit_2017[years >= 2017] = 2
    min4 = _cutoff_safe_series_result(exit_2017, years=years, window_year=4)
    min6 = _cutoff_safe_series_result(exit_2017, years=years, window_year=6)
    assert float(min4.abandonment_year.item()) == 2017.0
    assert float(min4.abandonment_duration.item()) == 4.0
    assert int(min4.persistence_validated.item()) == 1
    assert np.isnan(min6.abandonment_year.item())

    initial_failure = exit_2017.copy()
    initial_failure[1] = 2
    initial = _cutoff_safe_series_result(initial_failure, years=years, window_year=4)
    assert int(initial.eligible_cropland.item()) == 0
    assert np.isnan(initial.abandonment_year.item())

    prefix_failure = np.full(years.size, 2, dtype=np.float32)
    prefix_failure[:2] = 1
    prefix_failure[years == 2015] = 1
    prefix = _cutoff_safe_series_result(prefix_failure, years=years)
    assert int(prefix.eligible_cropland.item()) == 1
    assert np.isnan(prefix.abandonment_year.item())


def test_cutoff_safe_selects_latest_valid_exit_and_rejects_recultivated_event():
    years = np.arange(1992, 2023)
    multiple = np.full(years.size, 2, dtype=np.float32)
    multiple[:2] = 1
    multiple[(years == 2000) | (years == 2001)] = 1
    latest = _cutoff_safe_series_result(multiple, years=years)
    assert float(latest.abandonment_year.item()) == 2002.0
    assert float(latest.abandonment_duration.item()) == 19.0

    recultivated = np.full(years.size, 2, dtype=np.float32)
    recultivated[:2] = 1
    recultivated[(years == 2016) | (years == 2017)] = 1
    rejected = _cutoff_safe_series_result(recultivated, years=years)
    assert np.isnan(rejected.abandonment_year.item())
    assert int(rejected.current_abandonment.item()) == 0
    assert int(rejected.persistence_validated.item()) == 0


def test_cutoff_safe_rejects_invalid_parameters_and_core_year_axis():
    values = np.ones(31, dtype=np.float32)
    values[24:] = 2
    for field, value in (("window_year", 0), ("window_year", True), ("init_cropland", 0), ("init_cropland", True)):
        kwargs = {"window_year": 5, "init_cropland": 2}
        kwargs[field] = value
        with pytest.raises(ValueError):
            _cutoff_safe_series_result(values, **kwargs)
    with pytest.raises(ValueError, match="core analysis year"):
        _cutoff_safe_series_result(values[:-1], years=np.delete(np.arange(1992, 2023), 10))


def test_run_abandonment_detection_writes_and_resumes_fingerprinted_local_chunks(
    tmp_path: Path, monkeypatch
):
    years = np.arange(1992, 2023)
    values = np.ones((years.size, 2, 2), dtype=np.float32)
    values[years >= 2016, :, :] = 2
    values[1, 1, 1] = 2
    source = tmp_path / "reclass_lccs_1km.nc"
    xr.Dataset(
        {"lccs_class": (("time", "lat", "lon"), values)},
        coords={"time": years, "lat": [40.0, 39.0], "lon": [-121.0, -120.0]},
        attrs={"crs": "EPSG:4326"},
    ).to_netcdf(source)
    monkeypatch.setattr(MODULE, "_LOCAL_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "_LOCAL_PREDICTION_ENABLED", False)

    receipt = MODULE.run_abandonment_detection(target_nc=source)
    assert receipt["status"] == "detector_chunks_complete_prediction_pending"
    assert receipt["accepted"] is False
    assert receipt["feature"] == "1km_mode_min5"
    assert receipt["expected_chunk_count"] == receipt["verified_chunk_count"] == 1
    assert receipt["event_counts"] == {
        "eligible_cropland": 3,
        "qualifies_at_cutoff": 3,
        "current_abandonment": 3,
        "persistence_validated": 3,
    }
    detector_path = tmp_path / "abandon_2_1km_mode_min5" / "chunk_0_0.nc"
    merged_path = tmp_path / "merged_chunk_2_1km_mode_min5" / "chunk_0_0.nc"
    assert detector_path.is_file() and merged_path.is_file()
    with xr.open_dataset(detector_path) as detector:
        assert set(detector.data_vars) == set(MODULE.CUTOFF_SAFE_OUTPUT_VARS)
        assert detector.attrs["current_end_year"] == 2020
        assert detector.attrs["minimum_abandonment_years"] == 5
        assert detector.attrs["extension_effect"] == "persistence_validated_only"
    with xr.open_dataset(merged_path) as merged:
        assert "landcover" in merged
        assert merged.sizes["time"] == 31

    manifest_path = Path(receipt["abandonment_manifest_path"])
    manifest_mtime = manifest_path.stat().st_mtime_ns
    resumed = MODULE.run_abandonment_detection(target_nc=source)
    assert resumed["run_fingerprint"] == receipt["run_fingerprint"]
    assert manifest_path.stat().st_mtime_ns == manifest_mtime

    from function import embedding_pipeline

    delegated = {}

    def finalize(detector_receipt, **kwargs):
        delegated["receipt"] = detector_receipt
        delegated["kwargs"] = kwargs
        return {**detector_receipt, "status": "accepted", "accepted": True}

    monkeypatch.setattr(embedding_pipeline, "finalize_abandonment_prediction", finalize)
    monkeypatch.setattr(MODULE, "_LOCAL_PREDICTION_ENABLED", True)
    finalized = MODULE.run_abandonment_detection(target_nc=source)
    assert finalized["accepted"] is True
    assert delegated["receipt"]["parameters"]["current_end_year"] == 2020
    assert delegated["kwargs"]["feature_root"] == MODULE._LOCAL_FEATURE_ROOT
    monkeypatch.setattr(MODULE, "_LOCAL_PREDICTION_ENABLED", False)

    with xr.open_dataset(detector_path) as existing:
        tampered = existing.load()
    tampered["qualifies_at_cutoff"].values[0, 0] = 0
    tampered.to_netcdf(detector_path, mode="w")
    with pytest.raises(ValueError, match="data mismatch for qualifies_at_cutoff"):
        MODULE.run_abandonment_detection(target_nc=source)
