from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_abandonment_legacy_events.py"
REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_abandonment_legacy_events", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_audit_legacy_event_dataset_uses_inclusive_2020_window():
    years = pd.to_datetime([f"{year}-01-01" for year in range(1992, 2023)])
    landcover = np.ones((31, 2, 2), dtype=np.float32)

    # Active event containing water.
    landcover[10:29, 0, 0] = 4
    landcover[20, 0, 0] = 9
    # Active event containing wetland and NoData.
    landcover[15:29, 0, 1] = 4
    landcover[18, 0, 1] = 6
    landcover[19, 0, 1] = np.nan
    # Event ended in 2019; must not enter the 2020 denominator.
    landcover[10:28, 1, 0] = 7

    dataset = xr.Dataset(
        {
            "abandonment_year": (("lat", "lon"), np.array([[2002, 2007], [2002, np.nan]], dtype=np.float32)),
            "abandonment_duration": (("lat", "lon"), np.array([[19, 14], [18, np.nan]], dtype=np.float32)),
            "landcover": (("time", "lat", "lon"), landcover),
        },
        coords={"time": years, "lat": [40.0, 39.0], "lon": [-100.0, -99.0]},
        attrs={"detector_contract": "legacy_equivalent_mode_resample_only"},
    )

    result = MODULE.audit_legacy_event_dataset(dataset, source_key="fixture", target_year=2020)
    counts = result.set_index("metric")["pixel_count"].to_dict()

    assert counts["current_active_2020"] == 2
    assert counts["event_contains_water"] == 1
    assert counts["event_contains_wetland"] == 1
    assert counts["event_contains_nodata"] == 1
    assert counts["event_contains_built"] == 0
    assert counts["event_contains_any_suspicious_class"] == 2
    assert np.isfinite(result["area_ha"]).all()

    allowed = pd.MultiIndex.from_tuples([(40.0, -100.0)], names=["lat", "lon"])
    filtered = MODULE.audit_legacy_event_dataset(
        dataset,
        source_key="fixture",
        target_year=2020,
        allowed_grid_keys=allowed,
    )
    filtered_counts = filtered.set_index("metric")["pixel_count"].to_dict()
    assert filtered_counts["current_active_2020"] == 1
    assert filtered_counts["event_contains_water"] == 1
    assert filtered_counts["event_contains_wetland"] == 0


def test_chunk_directory_aggregates_disjoint_chunks(tmp_path: Path):
    years = pd.to_datetime([f"{year}-01-01" for year in range(1992, 2023)])
    for index, lon in enumerate((-100.0, -99.0)):
        landcover = np.ones((31, 1, 1), dtype=np.float32)
        landcover[10:29, 0, 0] = 9 if index == 0 else 4
        dataset = xr.Dataset(
            {
                "abandonment_year": (("lat", "lon"), [[2002.0]]),
                "abandonment_duration": (("lat", "lon"), [[19.0]]),
                "landcover": (("time", "lat", "lon"), landcover),
            },
            coords={"time": years, "lat": [40.0], "lon": [lon]},
        )
        dataset.to_netcdf(tmp_path / f"chunk_{index}_0.nc")

    result, manifest = MODULE.audit_chunk_directory(tmp_path, source_key="fixture")
    counts = result.set_index("metric")["pixel_count"].to_dict()
    assert manifest["chunk_count"] == 2
    assert counts["current_active_2020"] == 2
    assert counts["event_contains_water"] == 1


def test_published_audit_reconciles_canonical_areas_and_xie_year():
    audit_path = REPO_ROOT / "outputs" / "s0_us_abandon" / "summaries" / "abandonment_legacy_event_class_audit.csv"
    comparison_path = REPO_ROOT / "outputs" / "s0_us_abandon" / "summaries" / "abandonment_resolution_comparison.csv"
    audit = pd.read_csv(audit_path)
    comparison = pd.read_csv(comparison_path)
    active = audit.loc[audit["metric"] == "current_active_2020"].set_index("source_key")
    expected = comparison.set_index("source_key")
    assert np.isclose(active.loc["esa_cci_1km_mode", "area_mha"], expected.loc["esa_cci_1km_mode", "area_mha"], atol=1e-9)
    assert np.isclose(active.loc["esa_cci_300m", "area_mha"], expected.loc["esa_cci_300m", "area_mha"], atol=1e-9)
    assert int(expected.loc["xie_2024_30m", "status_year"]) == 2018
