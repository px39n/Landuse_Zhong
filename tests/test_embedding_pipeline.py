from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box


MODULE_PATH = Path(__file__).resolve().parents[1] / "function" / "embedding_pipeline.py"
SPEC = importlib.util.spec_from_file_location("embedding_pipeline", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _build_small_inputs(tmp_path: Path):
    lat = np.array([31.0, 30.0, 29.0, 28.0])
    lon = np.array([-100.0, -99.0, -98.0, -97.0])
    times = pd.to_datetime(["2010-01-01", "2018-01-01", "2020-01-01"])
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    landcover = np.stack(
        [np.full((4, 4), 3, dtype=np.uint8), np.full((4, 4), 4, dtype=np.uint8), np.full((4, 4), 5, dtype=np.uint8)]
    )
    merged = xr.Dataset(
        {
            "abandonment_year": (("lat", "lon"), np.full((4, 4), 2010.0, dtype=np.float32)),
            "abandonment_duration": (("lat", "lon"), np.full((4, 4), 20.0, dtype=np.float32)),
            "recultivation": (("lat", "lon"), np.zeros((4, 4), dtype=np.uint8)),
            "current_abandonment": (("lat", "lon"), np.ones((4, 4), dtype=np.uint8)),
            "landcover": (("time", "lat", "lon"), landcover),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    merged.to_netcdf(merged_dir / "chunk_0_0.nc")

    feature_root = tmp_path / "features"
    feature_root.mkdir()
    for variable in MODULE.FEATURE_2D_VARS:
        xr.Dataset({variable: (("lat", "lon"), np.full((4, 4), 2.0))}, coords={"lat": lat, "lon": lon}).to_netcdf(
            feature_root / f"{variable}.nc"
        )
    for variable in MODULE.FEATURE_3D_VARS:
        for year in (2018, 2020):
            xr.Dataset(
                {variable: (("time", "lat", "lon"), np.full((1, 4, 4), float(year)))},
                coords={"time": pd.to_datetime([f"{year}-01-01"]), "lat": lat, "lon": lon},
            ).to_netcdf(feature_root / f"{variable}_{year}.nc")

    pv = pd.DataFrame(
        {
            "unique_id": [1, 2],
            "p_area": [10.0, 20.0],
            "capacity_m": [1.0, 2.0],
            "country": ["USA", "USA"],
            "year": [2018, 2020],
            "longitude": [-100.0, -99.0],
            "latitude": [31.0, 30.0],
        }
    )
    pv_path = tmp_path / "pv.csv"
    pv.to_csv(pv_path, index=False)
    return merged_dir, feature_root, pv_path


def test_build_training_embedding_preserves_schema_and_rederives_landcover_at_abandonment(tmp_path: Path):
    merged_dir, feature_root, pv_path = _build_small_inputs(tmp_path)
    result = MODULE.build_training_embedding(
        str(merged_dir / "*.nc"),
        feature_root,
        pv_path,
        chunk_size=500,
    )
    assert list(result.columns) == MODULE.EXPECTED_COLUMNS
    assert len(result) == 2
    assert result["landcover"].tolist() == [4.0, 5.0]
    assert result["landcover_at_abandonment"].tolist() == [3.0, 3.0]
    assert result["abandonment_year"].tolist() == [2010.0, 2010.0]
    assert result["GDPpc"].tolist() == [2018.0, 2020.0]


def test_reference_embedding_defines_exact_standard_grid_row_domain(tmp_path: Path):
    merged_dir, feature_root, pv_path = _build_small_inputs(tmp_path)
    domain_path = tmp_path / "previous.csv"
    pd.DataFrame(
        {
            "time": ["2018-01-01"],
            "lat": [31.000003],
            "lon": [-100.000006],
            "unique_id": [777],
            "p_area": [77.0],
            "capacity_m": [7.0],
            "country": ["USA"],
            "year": [2018],
        }
    ).to_csv(domain_path, index=False)
    result = MODULE.build_training_embedding(
        str(merged_dir / "*.nc"),
        feature_root,
        pv_path,
        row_domain_csv=domain_path,
        chunk_size=500,
    )
    assert len(result) == 1
    assert result.iloc[0]["unique_id"] == 777
    assert result.iloc[0]["p_area"] == 77.0


def test_pv_dedup_keeps_last_input_record_not_highest_unique_id(tmp_path: Path):
    path = tmp_path / "pv.csv"
    pd.DataFrame(
        {
            "unique_id": [99, 1],
            "p_area": [99.0, 1.0],
            "capacity_m": [9.0, 1.0],
            "country": ["USA", "USA"],
            "year": [2020, 2020],
            "longitude": [-100.0, -100.0],
            "latitude": [30.0, 30.0],
        }
    ).to_csv(path, index=False)
    observed = MODULE.load_aligned_pv_sites(path)
    assert len(observed) == 1
    assert observed.iloc[0]["unique_id"] == 1


def test_embedding_validation_outputs_and_transactional_promotion(tmp_path: Path):
    merged_dir, feature_root, pv_path = _build_small_inputs(tmp_path)
    candidate = MODULE.build_training_embedding(str(merged_dir / "*.nc"), feature_root, pv_path)
    candidate["lat"] = [30.995833333333334, 29.995833333333334]
    candidate["lon"] = [-100.00416666666666, -99.00416666666666]
    previous = candidate.copy()
    previous["lat"] += 3e-6
    previous["lon"] -= 6e-6
    previous.loc[0, "landcover"] = 9
    previous.loc[0, "abandonment_year"] = 2000
    validation = MODULE.validate_embedding_candidate(candidate, previous)
    assert validation["report"]["passed"] is True
    assert validation["report"]["allowed_changed_columns"]["landcover"] == 1
    assert validation["report"]["common_rows"] == len(candidate)
    assert len(validation["conus"]) == len(candidate)

    next_path = tmp_path / "training_embedding.next.csv"
    conus_path = tmp_path / "training_embedding_conus.csv"
    report_path = tmp_path / "embedding_validation.json"
    MODULE.write_embedding_outputs(
        candidate,
        previous,
        candidate_path=next_path,
        conus_path=conus_path,
        report_path=report_path,
    )
    current_path = tmp_path / "training_embedding.csv"
    backup_path = tmp_path / "training_embedding_us_oldversion_v2.csv"
    previous.to_csv(current_path, index=False)
    old_hash = MODULE.sha256_file(current_path)
    promoted = MODULE.promote_training_embedding(next_path, current_path, backup_path, expected_current_sha256=old_hash)
    assert backup_path.exists() and current_path.exists()
    assert promoted["backup_sha256"] == old_hash
    assert list(pd.read_csv(current_path).columns) == MODULE.EXPECTED_COLUMNS

    another_candidate = tmp_path / "another.next.csv"
    candidate.to_csv(another_candidate, index=False)
    with pytest.raises(FileExistsError):
        MODULE.promote_training_embedding(another_candidate, current_path, backup_path)


def test_project_abandonment_to_each_sample_year_uses_inclusive_end_year():
    frame = pd.DataFrame(
        {
            "time": ["2018-01-01", "2020-01-01", "2020-01-01"],
            "abandonment_year": [2015.0, 2015.0, 2021.0],
            "abandonment_duration": [4.0, 6.0, 5.0],
            "recultivation": [1.0, 1.0, 0.0],
            "current_abandonment": [0.0, 0.0, 1.0],
            "landcover_at_abandonment": [2.0, 2.0, 3.0],
        }
    )
    observed = MODULE.project_abandonment_to_sample_year(frame)
    assert observed.loc[0, "current_abandonment"] == 1
    assert observed.loc[1, "current_abandonment"] == 1
    assert observed.loc[0, "recultivation"] == 0
    assert observed.loc[1, "recultivation"] == 0
    assert observed.loc[2, MODULE.ABANDONMENT_VARS].isna().all()
    assert np.isnan(observed.loc[2, "landcover_at_abandonment"])


def test_authoritative_landcover_keeps_points_outside_sparse_candidate_chunks(tmp_path: Path):
    merged_dir, feature_root, pv_path = _build_small_inputs(tmp_path)
    chunk_path = merged_dir / "chunk_0_0.nc"
    with xr.open_dataset(chunk_path) as opened:
        full_landcover = opened[["landcover"]].rename({"landcover": "lccs_class"}).load()
        sparse_chunk = opened.isel(lat=slice(0, 2), lon=slice(0, 2)).load()
    sparse_chunk.to_netcdf(chunk_path, mode="w")
    landcover_path = tmp_path / "reclass_lccs_1km.nc"
    full_landcover.to_netcdf(landcover_path)
    pv = pd.read_csv(pv_path)
    pv.loc[1, ["longitude", "latitude"]] = [-97.0, 28.0]
    pv.to_csv(pv_path, index=False)

    result = MODULE.build_training_embedding(
        str(merged_dir / "*.nc"),
        feature_root,
        pv_path,
        landcover_path=landcover_path,
        chunk_size=2,
    )
    assert len(result) == 2
    assert result["landcover"].tolist() == [4.0, 5.0]
    assert result.loc[1, MODULE.ABANDONMENT_VARS].isna().all()


def test_publish_0819_candidate_does_not_modify_legacy_embedding(tmp_path: Path):
    legacy = tmp_path / "training_embedding.csv"
    candidate = tmp_path / "aligned_for_training0819.next.csv"
    final = tmp_path / "aligned_for_training0819.csv"
    legacy.write_text("legacy", encoding="utf-8")
    candidate.write_text("new", encoding="utf-8")
    legacy_hash = MODULE.sha256_file(legacy)
    report = MODULE.publish_embedding_candidate(candidate, final)
    assert report["reused"] is False
    assert final.read_text(encoding="utf-8") == "new"
    assert MODULE.sha256_file(legacy) == legacy_hash
    assert not candidate.exists()

    conflicting = tmp_path / "conflicting.next.csv"
    conflicting.write_text("different", encoding="utf-8")
    with pytest.raises(FileExistsError):
        MODULE.publish_embedding_candidate(conflicting, final)
    assert conflicting.exists()


def test_clip_prediction_to_conus_uses_within_and_excludes_non_conus_states(tmp_path: Path):
    states = gpd.GeoDataFrame(
        {"STATEFP": ["06", "02"], "STUSPS": ["CA", "AK"], "NAME": ["California", "Alaska"]},
        geometry=[box(-101.0, 29.0, -98.0, 32.0), box(-151.0, 60.0, -149.0, 62.0)],
        crs="EPSG:4326",
    )
    state_path = tmp_path / "states.gpkg"
    states.to_file(state_path, driver="GPKG")
    frame = pd.DataFrame(
        {
            "lat": [30.0, 29.0, 61.0],
            "lon": [-100.0, -101.0, -150.0],
        }
    )

    clipped, lookup = MODULE._clip_prediction_to_conus(frame, state_path)

    assert len(clipped) == 1
    assert clipped.iloc[0]["state_fips"] == "06"
    assert clipped.iloc[0]["state_code"] == "CA"
    assert set(lookup["state_fips"]) == {"06"}


def test_prediction_embedding_returns_row_aligned_state_membership_frame(tmp_path: Path):
    merged, features, states = _build_small_inputs(tmp_path)
    state_gdf = gpd.GeoDataFrame(
        {"STATEFP": ["06"], "STUSPS": ["CA"], "NAME": ["California"]},
        geometry=[box(-101.0, 27.0, -96.0, 32.0)],
        crs="EPSG:4326",
    )
    state_path = tmp_path / "states.gpkg"
    state_gdf.to_file(state_path, driver="GPKG")

    candidate, state_rows = MODULE.build_prediction_embedding(str(merged / "*.nc"), features, state_path)

    assert list(state_rows.columns) == MODULE.STATE_MEMBERSHIP_COLUMNS
    assert len(state_rows) == len(candidate)
    assert set(zip(state_rows["lat"], state_rows["lon"])) == set(zip(candidate["lat"], candidate["lon"]))
    assert state_rows["state_fips"].eq("06").all()
    summary = MODULE.state_membership_summary(state_rows)
    shuffled = MODULE.state_membership_summary(state_rows.sample(frac=1.0, random_state=8))
    assert summary == shuffled
    assert summary["row_count"] == len(candidate)
    california, subset = MODULE.reconstruct_state_subset(
        candidate,
        state_rows,
        state_fips="06",
        expected_membership_sha256=summary["membership_sha256"],
    )
    assert len(california) == len(candidate)
    assert subset["parent_membership_sha256"] == summary["membership_sha256"]
    with pytest.raises(ValueError, match="fingerprint"):
        MODULE.reconstruct_state_subset(
            candidate,
            state_rows,
            expected_membership_sha256="0" * 64,
        )


def test_prediction_aoi_contract_binds_bounds_components_and_join(tmp_path: Path):
    states = gpd.GeoDataFrame(
        {"STATEFP": ["06"], "STUSPS": ["CA"], "NAME": ["California"]},
        geometry=[box(-125.0, 25.0, -65.0, 49.0)],
        crs="EPSG:4326",
    )
    state_path = tmp_path / "states.gpkg"
    states.to_file(state_path, driver="GPKG")
    first = MODULE.prediction_aoi_contract(state_path)
    second = MODULE.prediction_aoi_contract(state_path, bounds=MODULE.DEFAULT_CONUS_BOUNDS)
    assert first["aoi_sha256"] == second["aoi_sha256"]
    assert first["bounds"] == MODULE.DEFAULT_CONUS_BOUNDS
    assert first["bounds_inclusive"] is True
    assert first["excluded_state_fips"] == list(MODULE.EXCLUDED_CONUS_STATE_FIPS)
    assert first["join"] == {"how": "inner", "predicate": "within", "point": "pixel_center"}
    assert first["state_components"][0]["sha256"] == MODULE.sha256_file(state_path)


def test_prediction_embedding_accepts_schema_valid_zero_event_result(tmp_path: Path):
    merged, features, _ = _build_small_inputs(tmp_path)
    chunk_path = merged / "chunk_0_0.nc"
    with xr.open_dataset(chunk_path) as opened:
        no_events = opened.load()
    no_events["abandonment_year"][:] = np.nan
    no_events["abandonment_duration"][:] = np.nan
    no_events.to_netcdf(chunk_path, mode="w")
    states = gpd.GeoDataFrame(
        {"STATEFP": ["06"], "STUSPS": ["CA"], "NAME": ["California"]},
        geometry=[box(-101.0, 27.0, -96.0, 32.0)],
        crs="EPSG:4326",
    )
    state_path = tmp_path / "states.gpkg"
    states.to_file(state_path, driver="GPKG")
    candidate, membership = MODULE.build_prediction_embedding(
        str(merged / "chunk_*.nc"),
        features,
        state_path,
    )
    assert candidate.empty and list(candidate.columns) == MODULE.PREDICTION_COLUMNS
    assert membership.empty and list(membership.columns) == MODULE.STATE_MEMBERSHIP_COLUMNS
    assert MODULE.validate_prediction_candidate(candidate)["passed"] is True
    assert MODULE.state_membership_summary(membership)["state_counts"] == {}


def test_finalize_abandonment_prediction_publishes_or_reuses_exact_bundle(tmp_path: Path):
    merged_root, features, _ = _build_small_inputs(tmp_path)
    detector_root = tmp_path / "detector"
    detector_root.mkdir()
    shutil.copy2(merged_root / "chunk_0_0.nc", detector_root / "chunk_0_0.nc")
    fingerprint = "f" * 64
    detector_manifest = detector_root / "run_manifest.json"
    merged_manifest = merged_root / "run_manifest.json"
    payload = {"run_fingerprint": fingerprint}
    detector_manifest.write_text(json.dumps(payload), encoding="utf-8")
    merged_manifest.write_text(json.dumps(payload), encoding="utf-8")
    states = gpd.GeoDataFrame(
        {"STATEFP": ["06"], "STUSPS": ["CA"], "NAME": ["California"]},
        geometry=[box(-101.0, 27.0, -96.0, 32.0)],
        crs="EPSG:4326",
    )
    state_path = tmp_path / "states.gpkg"
    states.to_file(state_path, driver="GPKG")
    csv_path = tmp_path / "data" / "us_abandon_clean_1km_mode_min5.csv"
    acceptance_path = tmp_path / "outputs" / "local_run_abandonment_detection_acceptance.json"
    receipt = {
        "status": "detector_chunks_complete_prediction_pending",
        "accepted": False,
        "feature": "1km_mode_min5",
        "parameters": {
            "target_nc": str(tmp_path / "reclass_lccs_1km.nc"),
            "window_year": 5,
            "start_year": 1992,
            "current_end_year": 2020,
            "init_cropland": 2,
            "extend_validation": True,
            "feature": "1km_mode_min5",
        },
        "parameter_sha256": "a" * 64,
        "run_fingerprint": fingerprint,
        "source_sha256": "b" * 64,
        "extension_years_available": True,
        "abandonment_chunk_root": str(detector_root),
        "merged_chunk_root": str(merged_root),
        "expected_chunk_count": 1,
        "verified_chunk_count": 1,
        "abandonment_manifest_path": str(detector_manifest),
        "merged_manifest_path": str(merged_manifest),
        "required_csv_path": str(csv_path),
        "required_acceptance_path": str(acceptance_path),
    }

    first = MODULE.finalize_abandonment_prediction(
        receipt,
        feature_root=features,
        state_path=state_path,
    )
    assert first["accepted"] is True
    assert "acceptance_pending" not in first
    assert first["publication_reused"] is False
    assert csv_path.is_file()
    assert acceptance_path.is_file()
    manifest_path = csv_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["state_membership"]["state_counts"] == {"06": 16}
    assert manifest["state_membership"]["columns"] == MODULE.STATE_MEMBERSHIP_COLUMNS
    assert manifest["california_subset"]["row_count"] == 16
    assert manifest["california_csv_published"] is False
    assert manifest["validated_through_year"] == 2022
    assert manifest["aoi"]["state_inventory"][0]["state_fips"] == "06"
    assert manifest["product_state"] == "local_product_accepted"

    second = MODULE.finalize_abandonment_prediction(
        receipt,
        feature_root=features,
        state_path=state_path,
    )
    assert second["publication_reused"] is True
    csv_path.write_text("different\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite different"):
        MODULE.finalize_abandonment_prediction(
            receipt,
            feature_root=features,
            state_path=state_path,
        )


def test_chunked_vector_extract_does_not_materialize_backend_array(tmp_path: Path):
    path = tmp_path / "chunked.nc"
    values = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    dataset = xr.Dataset(
        {"value": (("time", "lat", "lon"), values)},
        coords={
            "time": pd.to_datetime(["2018-01-01", "2020-01-01"]),
            "lat": [3.0, 2.0, 1.0, 0.0],
            "lon": [10.0, 11.0, 12.0, 13.0, 14.0],
        },
    )
    dataset.to_netcdf(path, encoding={"value": {"zlib": True, "chunksizes": (1, 2, 3)}})
    with xr.open_dataset(path) as opened:
        data = opened["value"]
        assert data.variable._in_memory is False
        assert MODULE._axis_chunk_lengths(data, "lat") == (2, 2)
        assert data.variable._in_memory is False
        observed = MODULE._vector_extract(
            data,
            np.array([0, 3, 1]),
            np.array([0, 4, 3]),
            pd.Series(pd.to_datetime(["2018-01-01", "2020-01-01", "2020-01-01"])),
        )
        assert data.variable._in_memory is False
    expected = np.array([values[0, 0, 0], values[1, 3, 4], values[1, 1, 3]])
    np.testing.assert_array_equal(observed, expected)


def test_vector_extract_uses_safe_dask_point_indexing():
    values = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    data = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": pd.to_datetime(["2018-01-01", "2020-01-01"])},
        name="value",
    ).chunk({"time": 1, "lat": 2, "lon": 3})
    observed = MODULE._vector_extract(
        data,
        np.array([0, 3, 1]),
        np.array([0, 4, 3]),
        pd.Series(pd.to_datetime(["2018-01-01", "2020-01-01", "2020-01-01"])),
    )
    expected = np.array([values[0, 0, 0], values[1, 3, 4], values[1, 1, 3]])
    np.testing.assert_array_equal(observed, expected)


def test_embedding_promotion_rolls_back_if_candidate_replace_fails(tmp_path: Path, monkeypatch):
    current = tmp_path / "training_embedding.csv"
    candidate = tmp_path / "training_embedding.next.csv"
    backup = tmp_path / "training_embedding_us_oldversion_v2.csv"
    current.write_text("old", encoding="utf-8")
    candidate.write_text("new", encoding="utf-8")
    real_replace = MODULE.os.replace
    calls = {"count": 0}

    def controlled_replace(source, destination):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(MODULE.os, "replace", controlled_replace)
    with pytest.raises(OSError, match="simulated"):
        MODULE.promote_training_embedding(candidate, current, backup)
    assert current.read_text(encoding="utf-8") == "old"
    assert not backup.exists()
    assert candidate.exists()
