from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import rasterio.transform
import xarray as xr
from pyproj import CRS, Transformer
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from function import cropland_abandonment as abandonment
from function import embedding_pipeline as embedding
from tools import run_multiscale_abandonment as multiscale


FEATURE_ROOT = Path(r"D:\xarray\aligned2\Feature_all")
STATE_PATH = REPO_ROOT / "data" / "cb_2018_us_state_500k.shp"
DATASET_NAME = "aligned_for_contrasting0819"

BASE_COLUMNS = [
    "source_key",
    "source_valid_through_year",
    "target_year",
    "state_fips",
    "lat",
    "lon",
    "native_y",
    "native_x",
    "feature_grid_row",
    "feature_grid_col",
    "feature_match_distance_degrees",
    "pixel_area_m2",
    "abandonment_year",
    "abandonment_duration",
    "current_abandonment",
    "landcover",
    "landcover_at_abandonment",
]
OUTPUT_COLUMNS = [*BASE_COLUMNS, *embedding.FEATURE_2D_VARS, *embedding.FEATURE_3D_VARS]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _source_contract(source_key: str, run_id: str) -> dict[str, object]:
    subdirs = multiscale.build_source_subdirs(run_id)[source_key]
    if source_key == "esa_cci_300m":
        return {
            "chunk_dir": subdirs["merged_chunks"],
            "spatial_dims": ("lat", "lon"),
            "crs": "EPSG:4326",
            "target_year": 2020,
            "valid_through_year": 2022,
            "state_mask": subdirs["intermediate"] / "state_fips_300m_esa_cci.nc",
            "has_native_landcover": True,
        }
    if source_key == "lcmap_c13_30m":
        return {
            "chunk_dir": subdirs["merged_chunks"],
            "spatial_dims": ("y", "x"),
            "crs": None,
            "target_year": 2020,
            "valid_through_year": 2021,
            "state_mask": None,
            "has_native_landcover": True,
        }
    if source_key == "xie_2024_30m":
        return {
            "chunk_dir": subdirs["chunks"],
            "spatial_dims": ("y", "x"),
            "crs": "EPSG:4326",
            "target_year": 2018,
            "valid_through_year": 2018,
            "state_mask": None,
            "has_native_landcover": False,
        }
    raise ValueError(f"Unsupported source_key: {source_key}")


@functools.lru_cache(maxsize=4)
def _states_for_crs(crs_text: str):
    states = gpd.read_file(STATE_PATH)
    excluded = {"02", "15", "60", "66", "69", "72", "78"}
    states = states.loc[~states.STATEFP.astype(str).str.zfill(2).isin(excluded)].copy()
    return states.to_crs(crs_text)


def _rasterize_state_chunk(y_values: np.ndarray, x_values: np.ndarray, crs: str | CRS) -> np.ndarray:
    y_res = float(abs(y_values[1] - y_values[0])) if y_values.size > 1 else 30.0
    x_res = float(abs(x_values[1] - x_values[0])) if x_values.size > 1 else 30.0
    transform = rasterio.transform.from_origin(
        float(x_values[0] - x_res / 2.0),
        float(y_values[0] + y_res / 2.0),
        x_res,
        y_res,
    )
    states = _states_for_crs(CRS.from_user_input(crs).to_wkt())
    shapes = [(geometry, int(fips)) for geometry, fips in zip(states.geometry, states.STATEFP)]
    return rasterio.features.rasterize(
        shapes,
        out_shape=(y_values.size, x_values.size),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )


def _native_lon_lat(
    y_values: np.ndarray,
    x_values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    crs: str | CRS,
) -> tuple[np.ndarray, np.ndarray]:
    native_y = y_values[rows]
    native_x = x_values[cols]
    if CRS.from_user_input(crs).is_geographic:
        return native_x.astype(np.float64), native_y.astype(np.float64)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(native_x, native_y)
    return np.asarray(lon), np.asarray(lat)


def _feature_contexts(target_year: int):
    contexts = {}
    for variable in [*embedding.FEATURE_2D_VARS, *embedding.FEATURE_3D_VARS]:
        paths = embedding._feature_paths(FEATURE_ROOT, variable, (target_year,))
        context = xr.open_dataset(paths[0]) if len(paths) == 1 else xr.open_mfdataset(paths, combine="by_coords")
        contexts[variable] = context
    return contexts


def _attach_features(
    frame: pd.DataFrame,
    contexts: dict[str, xr.Dataset],
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    target_year: int,
) -> pd.DataFrame:
    tolerance = (1.0 / 120.0) / 2.0 + 1e-7
    row = embedding._nearest_indices(grid_lat, frame.lat.to_numpy(), tolerance)
    col = embedding._nearest_indices(grid_lon, frame.lon.to_numpy(), tolerance)
    valid = (row >= 0) & (col >= 0)
    if not valid.all():
        raise ValueError(f"{int((~valid).sum())} native pixels do not map to the Feature_all grid")
    frame = frame.loc[valid].reset_index(drop=True)
    row, col = row[valid], col[valid]
    frame["feature_grid_row"] = row
    frame["feature_grid_col"] = col
    frame["feature_match_distance_degrees"] = np.hypot(
        frame.lat.to_numpy() - grid_lat[row], frame.lon.to_numpy() - grid_lon[col]
    )
    pairs = pd.MultiIndex.from_arrays([row, col])
    unique_pairs = pairs.unique()
    unique_row = unique_pairs.get_level_values(0).to_numpy(dtype=np.int64)
    unique_col = unique_pairs.get_level_values(1).to_numpy(dtype=np.int64)
    lookup = pd.Series(np.arange(len(unique_pairs)), index=unique_pairs)
    inverse = lookup.loc[pairs].to_numpy(dtype=np.int64)
    times = pd.Series(pd.to_datetime([f"{target_year}-01-01"] * len(unique_pairs)))
    for variable, context in contexts.items():
        data = context[variable]
        values = embedding._vector_extract(
            data,
            unique_row,
            unique_col,
            times if "time" in data.dims else None,
        )
        frame[variable] = values[inverse]
    return frame


def _chunk_frame(
    dataset: xr.Dataset,
    source_key: str,
    contract: dict[str, object],
    state_values: np.ndarray,
) -> pd.DataFrame:
    y_dim, x_dim = contract["spatial_dims"]
    start = np.asarray(dataset.abandonment_year.values, dtype=np.float64)
    duration = np.asarray(dataset.abandonment_duration.values, dtype=np.float64)
    target_year = int(contract["target_year"])
    active = np.isfinite(start) & np.isfinite(duration) & (start <= target_year) & (
        start + duration - 1 >= target_year
    )
    active &= state_values > 0
    rows, cols = np.nonzero(active)
    if not rows.size:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    y_values = dataset[y_dim].values
    x_values = dataset[x_dim].values
    crs = contract["crs"] or dataset.attrs.get("crs")
    if not crs:
        raise ValueError(f"Missing native CRS for {source_key}")
    lon, lat = _native_lon_lat(y_values, x_values, rows, cols, crs)
    native_crs = CRS.from_user_input(crs)
    if native_crs.is_geographic:
        row_area = abandonment.cell_area_m2(
            dataset[y_dim],
            dataset[x_dim].isel({x_dim: slice(0, 2)}),
            spatial_dims=(y_dim, x_dim),
            crs=str(crs),
        ).values[:, 0]
        pixel_area_m2 = row_area[rows]
    else:
        projected_area = abandonment.cell_area_m2(
            dataset[y_dim].isel({y_dim: slice(0, 2)}),
            dataset[x_dim].isel({x_dim: slice(0, 2)}),
            spatial_dims=(y_dim, x_dim),
            crs=str(crs),
        ).values[0, 0]
        pixel_area_m2 = np.full(rows.size, projected_area, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "source_key": source_key,
            "source_valid_through_year": int(contract["valid_through_year"]),
            "target_year": target_year,
            "state_fips": [f"{int(value):02d}" for value in state_values[rows, cols]],
            "lat": lat,
            "lon": lon,
            "native_y": y_values[rows],
            "native_x": x_values[cols],
            "pixel_area_m2": pixel_area_m2,
            "abandonment_year": start[rows, cols].astype(np.float32),
            "abandonment_duration": duration[rows, cols].astype(np.float32),
            "current_abandonment": np.ones(rows.size, dtype=np.uint8),
        }
    )
    if contract["has_native_landcover"]:
        years = {int(year): idx for idx, year in enumerate(pd.to_datetime(dataset.time.values).year)}
        frame["landcover"] = dataset.landcover.isel(time=years[target_year]).values[rows, cols]
        event_values = np.full(rows.size, np.nan, dtype=np.float32)
        for year in np.unique(start[rows, cols].astype(np.int16)):
            positions = np.flatnonzero(start[rows, cols].astype(np.int16) == year)
            event_values[positions] = dataset.landcover.isel(time=years[int(year)]).values[
                rows[positions], cols[positions]
            ]
        frame["landcover_at_abandonment"] = event_values
    else:
        frame["landcover"] = np.nan
        frame["landcover_at_abandonment"] = np.nan
    return frame


def _sample_numeric_csv_rows(
    path: Path,
    count: int,
    rng: np.random.Generator,
) -> list[dict[str, str]]:
    with path.open("rb") as handle:
        header = handle.readline()
        header_values = next(csv.reader([header.decode("utf-8")]))
        data_start = handle.tell()
        file_size = path.stat().st_size
        if file_size <= data_start:
            raise ValueError(f"Embedding part has no data rows: {path}")
        records = []
        for _ in range(count):
            row = b""
            for _ in range(8):
                handle.seek(int(rng.integers(data_start, file_size)))
                if handle.tell() > data_start:
                    handle.readline()
                row = handle.readline()
                if row.strip():
                    break
            if not row.strip():
                handle.seek(data_start)
                row = handle.readline()
            if not row.strip():
                raise ValueError(f"Embedding part has no readable data row: {path}")
            row_values = next(csv.reader([row.decode("utf-8")]))
            records.append(dict(zip(header_values, row_values, strict=True)))
        return records


def validate_embedding_feature_mapping(
    source_key: str,
    *,
    sample_size: int = 2_000,
    feature_sample_size: int = 10,
    random_seed: int = 20_260_819,
) -> dict[str, object]:
    run_id = multiscale.CANONICAL_RUN_ID
    source_root = multiscale.build_source_run_roots(run_id)[source_key]
    index_path = source_root / "embedding_parts" / "embedding_index.csv"
    index = pd.read_csv(index_path, dtype={"state_fips": str})
    if index.empty or int(index.rows.sum()) <= 0:
        raise ValueError(f"Embedding index has no rows for {source_key}")
    rng = np.random.default_rng(random_seed)
    weights = index.rows.to_numpy(dtype=np.float64)
    weights /= weights.sum()
    selected = rng.choice(len(index), size=min(sample_size, int(index.rows.sum())), replace=True, p=weights)
    selected_counts = np.bincount(selected, minlength=len(index))
    sampled_records = []
    for position in np.flatnonzero(selected_counts):
        sampled_records.extend(
            _sample_numeric_csv_rows(
                Path(index.iloc[int(position)].path),
                int(selected_counts[position]),
                rng,
            )
        )
    samples = pd.DataFrame(sampled_records)
    for column in samples.columns.difference(["source_key", "state_fips"]):
        samples[column] = pd.to_numeric(samples[column], errors="coerce")
    with xr.open_dataset(FEATURE_ROOT / "DEM.nc") as reference:
        grid_lat = reference.lat.values
        grid_lon = reference.lon.values
    brute_rows = np.array([int(np.abs(grid_lat - value).argmin()) for value in samples.lat], dtype=np.int64)
    brute_cols = np.array([int(np.abs(grid_lon - value).argmin()) for value in samples.lon], dtype=np.int64)
    row_mismatch = int((brute_rows != samples.feature_grid_row.to_numpy(dtype=np.int64)).sum())
    col_mismatch = int((brute_cols != samples.feature_grid_col.to_numpy(dtype=np.int64)).sum())
    contract = _source_contract(source_key, run_id)
    contexts = _feature_contexts(int(contract["target_year"]))
    feature_mismatches = 0
    feature_checks = 0
    feature_samples = samples.iloc[: min(feature_sample_size, len(samples))].copy()
    feature_rows = brute_rows[: len(feature_samples)]
    feature_cols = brute_cols[: len(feature_samples)]
    try:
        for variable, context in contexts.items():
            data = context[variable]
            expected_values = []
            for row, col in zip(feature_rows, feature_cols, strict=True):
                selection = data.isel(lat=int(row), lon=int(col))
                if "time" in data.dims:
                    selection = selection.sel(time=f"{int(contract['target_year'])}-01-01")
                expected_values.append(float(selection.item()))
            expected = np.asarray(expected_values, dtype=np.float64)
            observed = feature_samples[variable].to_numpy(dtype=np.float64)
            matches = np.isclose(expected, observed, rtol=1e-6, atol=1e-6, equal_nan=True)
            feature_mismatches += int((~matches).sum())
            feature_checks += int(matches.size)
    finally:
        for context in contexts.values():
            context.close()
    samples["expected_feature_grid_row"] = brute_rows
    samples["expected_feature_grid_col"] = brute_cols
    diagnostic_path = source_root / "diagnostics" / "embedding_feature_mapping_sample.csv"
    temporary = diagnostic_path.with_suffix(".tmp.csv")
    samples.to_csv(temporary, index=False)
    os.replace(temporary, diagnostic_path)
    report = {
        "status": "complete" if row_mismatch == 0 and col_mismatch == 0 and feature_mismatches == 0 else "failed",
        "source_key": source_key,
        "sample_size": len(samples),
        "feature_sample_size": len(feature_samples),
        "random_seed": random_seed,
        "row_mismatches": row_mismatch,
        "col_mismatches": col_mismatch,
        "feature_checks": feature_checks,
        "feature_mismatches": feature_mismatches,
        "diagnostic_path": str(diagnostic_path),
    }
    _atomic_json(source_root / "diagnostics" / "embedding_feature_mapping_validation.json", report)
    if report["status"] != "complete":
        raise ValueError(f"Native-to-master feature mapping QA failed: {report}")
    return report


def validate_and_amend_embedding_manifest(source_key: str) -> dict[str, object]:
    run_id = multiscale.CANONICAL_RUN_ID
    source_root = multiscale.build_source_run_roots(run_id)[source_key]
    embedding_manifest_path = source_root / "embedding_parts" / "embedding_manifest.json"
    source_manifest_path = source_root / "manifests" / "source_manifest.json"
    if not embedding_manifest_path.exists() or not source_manifest_path.exists():
        raise FileNotFoundError(f"Embedding/source manifest missing for {source_key}")
    payload = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Embedding is not complete for {source_key}")
    report = validate_embedding_feature_mapping(source_key)
    payload["source_manifest"] = str(source_manifest_path)
    payload["source_manifest_sha256"] = multiscale.sha256_file(source_manifest_path)
    payload["feature_mapping_qa"] = report
    _atomic_json(embedding_manifest_path, payload)
    return payload


def build_multiscale_embedding(
    source_key: str,
    *,
    shard_index: int | None = None,
    shard_count: int | None = None,
    finalize: bool = True,
) -> dict[str, object]:
    run_snapshot = multiscale.build_parameter_snapshot("full")
    run_id, _ = multiscale.build_run_id(run_snapshot)
    contract = _source_contract(source_key, run_id)
    chunk_dir = Path(contract["chunk_dir"])
    chunk_paths = sorted(chunk_dir.glob("chunk_*.nc"))
    if not chunk_paths:
        raise FileNotFoundError(f"No validated chunks for {source_key}: {chunk_dir}")
    source_root = multiscale.build_source_run_roots(run_id)[source_key]
    source_manifest_path = source_root / "manifests" / "source_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Missing source acceptance manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete":
        raise RuntimeError(f"Source acceptance is incomplete for {source_key}")
    expected_chunks = int(
        source_manifest.get("candidate_chunk_count", source_manifest.get("chunk_count", len(chunk_paths)))
    )
    source_chunk_size = int(source_manifest.get("parameters", {}).get("chunk_size", 0))
    if contract["state_mask"]:
        if source_chunk_size <= 0:
            raise ValueError(f"Source manifest lacks authoritative chunk_size for {source_key}")
    if len(chunk_paths) != expected_chunks or list(chunk_dir.glob("*.tmp*")):
        raise ValueError(f"Validated source chunk set differs for {source_key}")
    if shard_count is None:
        processing_paths = chunk_paths
    else:
        if shard_count < 1 or shard_index is None or shard_index < 0 or shard_index >= shard_count:
            raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
        processing_paths = chunk_paths[shard_index::shard_count]
    output_root = multiscale.D_RUN_ROOT / run_id / "embedding" / DATASET_NAME / f"source_key={source_key}"
    manifest_root = source_root / "embedding_parts" / "chunks"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(FEATURE_ROOT / "DEM.nc") as reference:
        grid_lat, grid_lon = reference.lat.values, reference.lon.values
    contexts = _feature_contexts(int(contract["target_year"]))
    state_dataset = xr.open_dataset(contract["state_mask"]) if contract["state_mask"] else None
    index_records = []
    try:
        for path in tqdm(processing_paths, desc=f"embedding {source_key}", unit="chunk"):
            chunk_manifest = manifest_root / f"{path.stem}.json"
            if chunk_manifest.exists():
                payload = json.loads(chunk_manifest.read_text(encoding="utf-8"))
                if payload.get("status") == "complete" and all(Path(item["path"]).exists() for item in payload["parts"]):
                    index_records.extend(payload["parts"])
                    continue
            with xr.open_dataset(path) as opened:
                dataset = opened.load()
            y_dim, x_dim = contract["spatial_dims"]
            if state_dataset is not None:
                pieces = path.stem.split("_")
                row_key, col_key = int(pieces[-2]), int(pieces[-1])
                row_start = row_key * source_chunk_size
                col_start = col_key * source_chunk_size
                states = state_dataset.state_fips.isel(
                    lat=slice(row_start, row_start + dataset.sizes[y_dim]),
                    lon=slice(col_start, col_start + dataset.sizes[x_dim]),
                ).values
            else:
                native_crs = contract["crs"] or dataset.attrs.get("crs")
                states = _rasterize_state_chunk(dataset[y_dim].values, dataset[x_dim].values, native_crs)
            frame = _chunk_frame(dataset, source_key, contract, states)
            if not frame.empty:
                frame = _attach_features(
                    frame, contexts, grid_lat, grid_lon, int(contract["target_year"])
                )
            parts = []
            for fips, part in frame.groupby("state_fips", sort=True):
                destination_dir = output_root / f"state_fips={fips}"
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"part_{path.stem}.csv"
                temporary = destination.with_suffix(".tmp.csv")
                part[OUTPUT_COLUMNS].to_csv(temporary, index=False)
                os.replace(temporary, destination)
                record = {
                    "source_key": source_key,
                    "state_fips": str(fips),
                    "chunk": path.name,
                    "path": str(destination),
                    "rows": len(part),
                    "bytes": destination.stat().st_size,
                    "area_m2": float(part["pixel_area_m2"].sum()),
                    "sha256": multiscale.sha256_file(destination),
                }
                parts.append(record)
                index_records.append(record)
            _atomic_json(chunk_manifest, {"status": "complete", "chunk": path.name, "parts": parts})
    finally:
        if state_dataset is not None:
            state_dataset.close()
        for context in contexts.values():
            context.close()
    if not finalize:
        return {
            "status": "parts_complete",
            "source_key": source_key,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "processed_chunks": len(processing_paths),
            "parts": len(index_records),
        }
    completed_chunk_manifests = sorted(manifest_root.glob("chunk_*.json"))
    temporary_parts = sorted(output_root.rglob("*.tmp.csv"))
    if len(completed_chunk_manifests) != len(chunk_paths) or temporary_parts:
        raise ValueError(
            f"Embedding chunk completeness failed for {source_key}: "
            f"manifests={len(completed_chunk_manifests)}/{len(chunk_paths)} tmp={len(temporary_parts)}"
        )
    index = pd.DataFrame(index_records)
    index_path = source_root / "embedding_parts" / "embedding_index.csv"
    temporary_index = index_path.with_suffix(".tmp.csv")
    index.to_csv(temporary_index, index=False)
    os.replace(temporary_index, index_path)
    mapping_qa = validate_embedding_feature_mapping(source_key)
    manifest = {
        "status": "complete",
        "source_key": source_key,
        "run_id": run_id,
        "output_root": str(output_root),
        "parts": len(index),
        "chunk_count": len(chunk_paths),
        "rows": int(index.rows.sum()) if len(index) else 0,
        "bytes": int(index.bytes.sum()) if len(index) else 0,
        "area_m2": float(index.area_m2.sum()) if len(index) and "area_m2" in index else None,
        "index_path": str(index_path),
        "schema": OUTPUT_COLUMNS,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": multiscale.sha256_file(source_manifest_path),
        "feature_mapping_qa": mapping_qa,
    }
    _atomic_json(source_root / "embedding_parts" / "embedding_manifest.json", manifest)
    print(f"[RESULT] multiscale embedding complete source={source_key} rows={manifest['rows']}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-key", required=True, choices=("esa_cci_300m", "lcmap_c13_30m", "xie_2024_30m"))
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--parts-only", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    print(
        build_multiscale_embedding(
            args.source_key,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            finalize=not args.parts_only,
        )
    )
