from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr


ABANDONMENT_VARS = [
    "abandonment_year",
    "abandonment_duration",
    "recultivation",
    "current_abandonment",
]
FEATURE_3D_VARS = ["GDPpc", "GDPtot", "GURdist", "Population", "gdmp", "rsds", "tas", "wind"]
FEATURE_2D_VARS = ["DEM", "Powerdist", "PrimaryRoad", "SecondaryRoad", "Slope", "TertiaryRoad"]
EXPECTED_COLUMNS = [
    "time",
    "lat",
    "lon",
    "abandonment_year",
    "unique_id",
    "p_area",
    "capacity_m",
    "country",
    "year",
    "abandonment_duration",
    "recultivation",
    "current_abandonment",
    "landcover",
    "DEM",
    "gdmp",
    "GDPpc",
    "GDPtot",
    "GURdist",
    "landcover_at_abandonment",
    "Population",
    "Powerdist",
    "PrimaryRoad",
    "rsds",
    "SecondaryRoad",
    "Slope",
    "tas",
    "TertiaryRoad",
    "wind",
]
PREDICTION_COLUMNS = [
    "lat",
    "lon",
    "time",
    "abandonment_year",
    "abandonment_duration",
    "current_abandonment",
    "landcover",
    "DEM",
    "gdmp",
    "GDPpc",
    "GDPtot",
    "GURdist",
    "landcover_at_abandonment",
    "Population",
    "Powerdist",
    "PrimaryRoad",
    "rsds",
    "SecondaryRoad",
    "Slope",
    "tas",
    "TertiaryRoad",
    "wind",
]
INVARIANT_COLUMNS = [
    "unique_id",
    "p_area",
    "capacity_m",
    "country",
    "year",
    *FEATURE_2D_VARS,
    *FEATURE_3D_VARS,
]
CHANGED_COLUMNS = [*ABANDONMENT_VARS, "landcover", "landcover_at_abandonment"]
CHUNK_PATTERN = re.compile(r"chunk_(\d+)_(\d+)\.nc$")
DEFAULT_CONUS_BOUNDS = {
    "lon_min": -125.0,
    "lon_max": -65.0,
    "lat_min": 25.0,
    "lat_max": 49.0,
}
EXCLUDED_CONUS_STATE_FIPS = ("02", "15", "60", "66", "69", "72", "78")
STATE_MEMBERSHIP_COLUMNS = [
    "prediction_row_key",
    "lat",
    "lon",
    "state_fips",
    "state_code",
    "state_name",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _prediction_bounds(bounds: Mapping[str, float] | None) -> dict[str, float]:
    extent = dict(DEFAULT_CONUS_BOUNDS if bounds is None else bounds)
    if set(extent) != set(DEFAULT_CONUS_BOUNDS):
        raise ValueError(f"Prediction bounds must contain exactly {sorted(DEFAULT_CONUS_BOUNDS)}")
    extent = {key: float(value) for key, value in extent.items()}
    if not all(np.isfinite(list(extent.values()))):
        raise ValueError("Prediction bounds must be finite")
    if extent["lon_min"] > extent["lon_max"] or extent["lat_min"] > extent["lat_max"]:
        raise ValueError("Prediction bounds minima must not exceed maxima")
    return extent


def prediction_row_keys(frame: pd.DataFrame) -> pd.Series:
    """Return stable keys for the established prediction-table row domain."""
    required = {"time", "lat", "lon"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction rows are missing key columns: {sorted(missing)}")
    times = pd.to_datetime(frame["time"], errors="coerce")
    latitudes = pd.to_numeric(frame["lat"], errors="coerce")
    longitudes = pd.to_numeric(frame["lon"], errors="coerce")
    if times.isna().any() or latitudes.isna().any() or longitudes.isna().any():
        raise ValueError("Prediction row keys require finite time/latitude/longitude values")
    values = [
        f"{timestamp:%Y-%m-%d}|{latitude:.10f}|{longitude:.10f}"
        for timestamp, latitude, longitude in zip(times, latitudes, longitudes)
    ]
    keys = pd.Series(values, index=frame.index, name="prediction_row_key", dtype="string")
    if keys.duplicated().any():
        duplicates = sorted(keys.loc[keys.duplicated(keep=False)].unique().tolist())
        raise ValueError(f"Prediction row keys are not unique: {duplicates[:5]}")
    return keys


def build_state_membership_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the deterministic row-key-to-state companion for a CONUS table."""
    required = {"lat", "lon", "time", "state_fips", "state_code", "state_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"State-clipped prediction rows are missing: {sorted(missing)}")
    membership = frame[["lat", "lon", "state_fips", "state_code", "state_name"]].copy()
    membership.insert(0, "prediction_row_key", prediction_row_keys(frame).to_numpy())
    membership["state_fips"] = membership["state_fips"].astype(str).str.zfill(2)
    membership["state_code"] = membership["state_code"].astype(str)
    membership["state_name"] = membership["state_name"].astype(str)
    membership = membership.sort_values("prediction_row_key", kind="stable").reset_index(drop=True)
    return membership[STATE_MEMBERSHIP_COLUMNS]


def state_membership_summary(membership: pd.DataFrame) -> dict[str, object]:
    if list(membership.columns) != STATE_MEMBERSHIP_COLUMNS:
        raise ValueError("State-membership columns differ from the canonical schema")
    work = membership.copy()
    work["state_fips"] = work["state_fips"].astype(str).str.zfill(2)
    if work["prediction_row_key"].duplicated().any():
        raise ValueError("State-membership row keys must be unique")
    work = work.sort_values("prediction_row_key", kind="stable").reset_index(drop=True)
    records = work[["prediction_row_key", "state_fips", "state_code", "state_name"]].to_dict("records")
    state_counts = {
        str(key): int(value)
        for key, value in work.groupby("state_fips", sort=True).size().items()
    }
    row_keys = work["prediction_row_key"].astype(str).tolist()
    return {
        "schema_version": "prediction-state-membership.v1",
        "columns": list(STATE_MEMBERSHIP_COLUMNS),
        "row_count": int(len(work)),
        "state_counts": state_counts,
        "row_key_sha256": _canonical_json_sha256(row_keys),
        "membership_sha256": _canonical_json_sha256(records),
    }


def reconstruct_state_subset(
    candidate: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    state_fips: str = "06",
    expected_membership_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reconstruct a hash-bound state subset without publishing another CSV."""
    summary = state_membership_summary(membership)
    if expected_membership_sha256 is not None and (
        summary["membership_sha256"] != str(expected_membership_sha256).upper()
    ):
        raise ValueError("State-membership fingerprint differs from the accepted parent")
    candidate_keys = prediction_row_keys(candidate)
    membership_by_key = membership.set_index("prediction_row_key", verify_integrity=True)
    if set(candidate_keys) != set(membership_by_key.index.astype(str)):
        raise ValueError("Prediction CSV row keys and state-membership row keys differ")
    normalized_fips = str(state_fips).zfill(2)
    selected_keys = set(
        membership_by_key.loc[
            membership_by_key["state_fips"].astype(str).str.zfill(2).eq(normalized_fips)
        ].index.astype(str)
    )
    subset = candidate.loc[candidate_keys.isin(selected_keys)].copy().reset_index(drop=True)
    subset_keys = sorted(selected_keys)
    subset_csv = subset.to_csv(index=False, lineterminator="\n", float_format="%.10g", na_rep="")
    subset_summary = {
        "schema_version": "prediction-state-subset.v1",
        "state_fips": normalized_fips,
        "parent_membership_sha256": summary["membership_sha256"],
        "row_count": int(len(subset)),
        "row_key_sha256": _canonical_json_sha256(subset_keys),
        "subset_sha256": hashlib.sha256(subset_csv.encode("utf-8")).hexdigest().upper(),
    }
    return subset, subset_summary


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".tmp.csv", mode="w", encoding="utf-8", newline="") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _nearest_indices(grid: np.ndarray, values: np.ndarray, tolerance: float) -> np.ndarray:
    grid = np.asarray(grid)
    values = np.asarray(values)
    if grid[0] > grid[-1]:
        reversed_index = pd.Index(grid[::-1])
        found = reversed_index.get_indexer(values, method="nearest", tolerance=tolerance)
        return np.where(found >= 0, grid.size - 1 - found, -1)
    return pd.Index(grid).get_indexer(values, method="nearest", tolerance=tolerance)


def load_aligned_pv_sites(path: str | Path, *, years: Sequence[int] = (2018, 2020)) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"longitude": "lon", "latitude": "lat"})
    required = {"unique_id", "p_area", "capacity_m", "country", "year", "lon", "lat"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Aligned PV CSV is missing columns: {sorted(missing)}")
    frame = frame.loc[frame["year"].isin(years)].copy()
    frame["time"] = pd.to_datetime(frame["year"].astype(int).astype(str), format="%Y")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="raise").astype("float32")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="raise").astype("float32")
    return (
        frame.sort_values(["time"], kind="stable")
        .drop_duplicates(["time", "lat", "lon"], keep="last")
        .reset_index(drop=True)
    )


def _axis_chunk_lengths(data: xr.DataArray, dim: str) -> tuple[int, ...]:
    axis = data.get_axis_num(dim)
    # DataArray.data materializes backend arrays; DataArray.chunks is metadata-only.
    dask_chunks = data.chunks
    if dask_chunks is not None:
        return tuple(int(value) for value in dask_chunks[axis])
    stored_chunks = data.encoding.get("chunksizes")
    chunk_size = int(stored_chunks[axis]) if stored_chunks else min(500, data.sizes[dim])
    full, remainder = divmod(data.sizes[dim], chunk_size)
    return (chunk_size,) * full + ((remainder,) if remainder else ())


def _vector_extract(data: xr.DataArray, lat_indices: np.ndarray, lon_indices: np.ndarray, times: pd.Series | None = None) -> np.ndarray:
    lat_indices = np.asarray(lat_indices, dtype=np.int64)
    lon_indices = np.asarray(lon_indices, dtype=np.int64)
    if lat_indices.shape != lon_indices.shape:
        raise ValueError("Latitude and longitude point index arrays must have the same shape")
    if not lat_indices.size:
        return np.empty(0, dtype=data.dtype)
    if (lat_indices < 0).any() or (lat_indices >= data.sizes["lat"]).any():
        raise IndexError("Latitude point index exceeds data bounds")
    if (lon_indices < 0).any() or (lon_indices >= data.sizes["lon"]).any():
        raise IndexError("Longitude point index exceeds data bounds")

    time_index = None
    if "time" in data.dims:
        if times is None:
            raise ValueError(f"Time values are required for {data.name}")
        time_index = pd.Index(pd.to_datetime(data.time.values)).get_indexer(pd.to_datetime(times))
        if (time_index < 0).any():
            missing = sorted(set(pd.to_datetime(times[time_index < 0]).astype(str)))
            raise ValueError(f"Missing time values for {data.name}: {missing}")

    if data.chunks is not None:
        point_dim = "points"
        indexers = {
            "lat": xr.DataArray(lat_indices, dims=point_dim),
            "lon": xr.DataArray(lon_indices, dims=point_dim),
        }
        if time_index is not None:
            indexers["time"] = xr.DataArray(time_index, dims=point_dim)
        return np.asarray(data.isel(indexers).compute().values)

    lat_lengths = _axis_chunk_lengths(data, "lat")
    lon_lengths = _axis_chunk_lengths(data, "lon")
    lat_starts = np.concatenate(([0], np.cumsum(lat_lengths[:-1], dtype=np.int64)))
    lon_starts = np.concatenate(([0], np.cumsum(lon_lengths[:-1], dtype=np.int64)))
    lat_ends = np.cumsum(lat_lengths, dtype=np.int64)
    lon_ends = np.cumsum(lon_lengths, dtype=np.int64)
    lat_groups = np.searchsorted(lat_ends, lat_indices, side="right")
    lon_groups = np.searchsorted(lon_ends, lon_indices, side="right")

    grouped: dict[tuple[int, int, int], list[int]] = {}
    for point, (lat_group, lon_group) in enumerate(zip(lat_groups, lon_groups)):
        time_group = -1 if time_index is None else int(time_index[point])
        grouped.setdefault((time_group, int(lat_group), int(lon_group)), []).append(point)

    output = np.empty(lat_indices.size, dtype=data.dtype)
    for (time_group, lat_group, lon_group), positions_list in grouped.items():
        lat_start = int(lat_starts[lat_group])
        lon_start = int(lon_starts[lon_group])
        indexers = {
            "lat": slice(lat_start, int(lat_ends[lat_group])),
            "lon": slice(lon_start, int(lon_ends[lon_group])),
        }
        if time_group >= 0:
            indexers["time"] = time_group
        block = data.isel(indexers).transpose("lat", "lon").compute()
        positions = np.asarray(positions_list, dtype=np.int64)
        output[positions] = np.asarray(block.values)[
            lat_indices[positions] - lat_start,
            lon_indices[positions] - lon_start,
        ]
    return output


def _feature_paths(feature_root: Path, variable: str, years: Sequence[int]) -> list[Path]:
    if variable in FEATURE_2D_VARS:
        paths = [feature_root / f"{variable}.nc"]
    else:
        paths = [feature_root / f"{variable}_{year}.nc" for year in years]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature files for {variable}: {missing}")
    return paths


def build_training_embedding(
    merged_chunk_pattern: str,
    feature_root: str | Path,
    pv_csv: str | Path,
    *,
    landcover_path: str | Path | None = None,
    row_domain_csv: str | Path | None = None,
    years: Sequence[int] = (2018, 2020),
    chunk_size: int = 500,
    progress_callback=None,
) -> pd.DataFrame:
    merged_files = sorted(Path(path) for path in glob.glob(merged_chunk_pattern))
    if not merged_files:
        raise FileNotFoundError(f"No merged chunks matched: {merged_chunk_pattern}")
    feature_root = Path(feature_root)
    reference_path = feature_root / "DEM.nc"
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    pv = load_aligned_pv_sites(pv_csv, years=years)
    with xr.open_dataset(reference_path) as reference:
        grid_lat = reference.lat.values
        grid_lon = reference.lon.values
    tolerance = (1.0 / 120.0) / 2.0 + 1e-7
    lat_index = _nearest_indices(grid_lat, pv["lat"].to_numpy(), tolerance)
    lon_index = _nearest_indices(grid_lon, pv["lon"].to_numpy(), tolerance)
    valid_grid = (lat_index >= 0) & (lon_index >= 0)
    pv = pv.loc[valid_grid].reset_index(drop=True)
    lat_index = lat_index[valid_grid]
    lon_index = lon_index[valid_grid]
    pv["lat"] = grid_lat[lat_index]
    pv["lon"] = grid_lon[lon_index]

    if row_domain_csv is not None:
        reference_columns = ["time", "lat", "lon", "unique_id", "p_area", "capacity_m", "country", "year"]
        reference = pd.read_csv(
            row_domain_csv,
            usecols=lambda name: name in reference_columns,
        )
        missing_domain_columns = {"time", "lat", "lon"} - set(reference.columns)
        if missing_domain_columns:
            raise ValueError(f"Reference embedding row domain is missing columns: {sorted(missing_domain_columns)}")
        reference_time = pd.to_datetime(reference["time"], errors="raise").dt.year.to_numpy()
        reference_lat = _nearest_indices(grid_lat, reference["lat"].to_numpy(), tolerance)
        reference_lon = _nearest_indices(grid_lon, reference["lon"].to_numpy(), tolerance)
        if (reference_lat < 0).any() or (reference_lon < 0).any():
            raise ValueError("Reference embedding row domain contains coordinates outside the standard grid")
        reference_keys = pd.MultiIndex.from_arrays(
            [reference_time, reference_lat, reference_lon], names=["year", "lat_index", "lon_index"]
        )
        if reference_keys.has_duplicates:
            raise ValueError("Reference embedding row domain contains duplicate grid keys")
        candidate_keys = pd.MultiIndex.from_arrays(
            [pd.to_datetime(pv["time"]).dt.year.to_numpy(), lat_index, lon_index],
            names=reference_keys.names,
        )
        if candidate_keys.has_duplicates:
            raise ValueError("Current PV input contains duplicate standard-grid keys")
        missing_reference = reference_keys.difference(candidate_keys)
        if len(missing_reference):
            raise ValueError(f"Current PV input is missing {len(missing_reference)} reference grid keys")
        candidate_positions = pd.Series(np.arange(len(candidate_keys)), index=candidate_keys)
        ordered_positions = candidate_positions.loc[reference_keys].to_numpy(dtype=np.int64)
        pv = pv.iloc[ordered_positions].reset_index(drop=True)
        lat_index = reference_lat.astype(np.int64, copy=False)
        lon_index = reference_lon.astype(np.int64, copy=False)
        pv["time"] = pd.to_datetime(reference["time"]).to_numpy()
        pv["lat"] = grid_lat[lat_index]
        pv["lon"] = grid_lon[lon_index]
        for column in ("unique_id", "p_area", "capacity_m", "country", "year"):
            if column in reference:
                pv[column] = reference[column].to_numpy()

    chunk_map: dict[tuple[int, int], Path] = {}
    for path in merged_files:
        match = CHUNK_PATTERN.fullmatch(path.name)
        if match:
            chunk_map[(int(match.group(1)), int(match.group(2)))] = path
    row_chunk = lat_index // chunk_size
    col_chunk = lon_index // chunk_size
    if landcover_path is None:
        # Backward-compatible sparse behavior for callers that do not provide
        # the authoritative global land-cover grid.
        exact_coverage = np.array(
            [(int(row), int(col)) in chunk_map for row, col in zip(row_chunk, col_chunk)],
            dtype=bool,
        )
        pv = pv.loc[exact_coverage].reset_index(drop=True)
        lat_index = lat_index[exact_coverage]
        lon_index = lon_index[exact_coverage]
        row_chunk = row_chunk[exact_coverage]
        col_chunk = col_chunk[exact_coverage]

    result = pv[["time", "lat", "lon", "unique_id", "p_area", "capacity_m", "country", "year"]].copy()
    for name in [*ABANDONMENT_VARS, "landcover", "landcover_at_abandonment", *FEATURE_2D_VARS, *FEATURE_3D_VARS]:
        result[name] = np.nan

    processed = 0
    for key, path in sorted(chunk_map.items()):
        rows = np.flatnonzero((row_chunk == key[0]) & (col_chunk == key[1]))
        if rows.size == 0:
            continue
        with xr.open_dataset(path) as chunk:
            local_lat = lat_index[rows] - key[0] * chunk_size
            local_lon = lon_index[rows] - key[1] * chunk_size
            if (local_lat >= chunk.sizes["lat"]).any() or (local_lon >= chunk.sizes["lon"]).any():
                raise IndexError(f"Point index exceeds chunk bounds: {path}")
            for variable in [*ABANDONMENT_VARS, "landcover"]:
                if variable not in chunk:
                    raise KeyError(f"{path.name} missing {variable}")
                values = _vector_extract(
                    chunk[variable],
                    local_lat,
                    local_lon,
                    result.loc[rows, "time"] if "time" in chunk[variable].dims else None,
                )
                result.loc[rows, variable] = values

            valid_abandonment = rows[result.loc[rows, "abandonment_year"].notna().to_numpy()]
            if valid_abandonment.size:
                local_lat = lat_index[valid_abandonment] - key[0] * chunk_size
                local_lon = lon_index[valid_abandonment] - key[1] * chunk_size
                abandonment_times = pd.to_datetime(
                    result.loc[valid_abandonment, "abandonment_year"].astype(int).astype(str), format="%Y"
                )
                result.loc[valid_abandonment, "landcover_at_abandonment"] = _vector_extract(
                    chunk["landcover"], local_lat, local_lon, abandonment_times
                )
        processed += 1
        if progress_callback:
            progress_callback({"stage": "merged_chunks", "done": processed, "total": len(chunk_map), "path": str(path)})

    if landcover_path is not None:
        with xr.open_dataset(landcover_path) as landcover_dataset:
            source_landcover = landcover_dataset["lccs_class"]
            if not np.array_equal(source_landcover.lat.values, grid_lat) or not np.array_equal(
                source_landcover.lon.values, grid_lon
            ):
                raise ValueError("Authoritative land-cover grid does not exactly match Feature_all grid")
            authoritative_landcover = _vector_extract(
                source_landcover,
                lat_index,
                lon_index,
                result["time"],
            )
        existing_landcover = pd.to_numeric(result["landcover"], errors="coerce").to_numpy()
        comparable = np.isfinite(existing_landcover) & np.isfinite(authoritative_landcover)
        if comparable.any() and not np.allclose(
            existing_landcover[comparable], authoritative_landcover[comparable], atol=0, rtol=0
        ):
            raise ValueError("Merged chunk landcover differs from authoritative mode-resampled grid")
        result["landcover"] = authoritative_landcover

    result = project_abandonment_to_sample_year(result)

    for variable in [*FEATURE_2D_VARS, *FEATURE_3D_VARS]:
        paths = _feature_paths(feature_root, variable, years)
        if len(paths) == 1:
            context = xr.open_dataset(paths[0])
        else:
            context = xr.open_mfdataset(paths, combine="by_coords")
        with context as dataset:
            data = dataset[variable]
            result[variable] = _vector_extract(
                data,
                lat_index,
                lon_index,
                result["time"] if "time" in data.dims else None,
            )
        if progress_callback:
            progress_callback({"stage": "features", "variable": variable})

    result["time"] = pd.to_datetime(result["time"]).dt.strftime("%Y-%m-%d")
    for column in EXPECTED_COLUMNS:
        if column not in result:
            result[column] = np.nan
    return result[EXPECTED_COLUMNS]


def _year_indices(time_values: np.ndarray) -> dict[int, int]:
    years = pd.to_datetime(time_values).year.to_numpy()
    if len(np.unique(years)) != len(years):
        raise ValueError("Time coordinate contains duplicate years")
    return {int(year): int(index) for index, year in enumerate(years)}


def _clip_prediction_to_conus(
    frame: pd.DataFrame,
    state_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import geopandas as gpd

    states = gpd.read_file(state_path)
    required = {"STATEFP", "STUSPS", "NAME"}
    missing = required - set(states.columns)
    if missing:
        raise ValueError(f"State boundary file is missing columns: {sorted(missing)}")
    if states.crs is None:
        raise ValueError("State boundary file has no declared CRS")
    states = states.to_crs("EPSG:4326")
    states["STATEFP"] = states["STATEFP"].astype(str).str.zfill(2)
    states = states.loc[~states["STATEFP"].isin(EXCLUDED_CONUS_STATE_FIPS)].copy()
    states = states.sort_values("STATEFP", kind="stable").reset_index(drop=True)
    points = gpd.GeoDataFrame(
        frame.assign(_source_order=np.arange(len(frame), dtype=np.int64)),
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        points,
        states[["STATEFP", "STUSPS", "NAME", "geometry"]],
        how="inner",
        predicate="within",
    )
    joined = joined.rename(
        columns={"STATEFP": "state_fips", "STUSPS": "state_code", "NAME": "state_name"}
    ).drop(columns=["geometry", "index_right"], errors="ignore")
    joined["state_fips"] = joined["state_fips"].astype(str).str.zfill(2)
    joined = joined.sort_values("_source_order", kind="stable").drop(columns="_source_order")
    duplicate_columns = ["time", "lat", "lon"] if "time" in joined else ["lat", "lon"]
    if joined.duplicated(duplicate_columns).any():
        raise ValueError("A prediction point joined to more than one retained state polygon")
    state_lookup = states[["STATEFP", "STUSPS", "NAME"]].rename(
        columns={"STATEFP": "state_fips", "STUSPS": "state_code", "NAME": "state_name"}
    )
    return joined.reset_index(drop=True), state_lookup.reset_index(drop=True)


def prediction_aoi_contract(
    state_path: str | Path,
    *,
    bounds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Fingerprint the confirmed bbox and state-polygon membership contract."""
    import geopandas as gpd

    path = Path(state_path)
    states = gpd.read_file(path)
    required = {"STATEFP", "STUSPS", "NAME"}
    missing = required - set(states.columns)
    if missing:
        raise ValueError(f"State boundary file is missing columns: {sorted(missing)}")
    if states.crs is None:
        raise ValueError("State boundary file has no declared CRS")
    source_crs = str(states.crs)
    states = states.to_crs("EPSG:4326")
    states["STATEFP"] = states["STATEFP"].astype(str).str.zfill(2)
    states = states.loc[~states["STATEFP"].isin(EXCLUDED_CONUS_STATE_FIPS)].copy()
    states = states.sort_values("STATEFP", kind="stable").reset_index(drop=True)
    state_inventory = [
        {
            "state_fips": str(row.STATEFP),
            "state_code": str(row.STUSPS),
            "state_name": str(row.NAME),
        }
        for row in states.itertuples(index=False)
    ]
    geometry_inventory = [
        {
            "state_fips": str(row.STATEFP),
            "state_code": str(row.STUSPS),
            "state_name": str(row.NAME),
            "geometry_wkb_hex": row.geometry.wkb_hex,
        }
        for row in states.itertuples(index=False)
    ]
    components = [path]
    if path.suffix.lower() == ".shp":
        components = sorted(
            candidate
            for candidate in path.parent.glob(f"{path.stem}.*")
            if candidate.is_file()
        )
    component_inventory = [
        {
            "name": component.name,
            "size_bytes": int(component.stat().st_size),
            "sha256": sha256_file(component),
        }
        for component in components
    ]
    payload = {
        "schema_version": "prediction-conus-aoi.v1",
        "state_path": str(path.expanduser().resolve()),
        "bounds": _prediction_bounds(bounds),
        "bounds_inclusive": True,
        "source_crs": source_crs,
        "target_crs": "EPSG:4326",
        "excluded_state_fips": list(EXCLUDED_CONUS_STATE_FIPS),
        "join": {"how": "inner", "predicate": "within", "point": "pixel_center"},
        "state_polygon_count": int(len(states)),
        "state_inventory": state_inventory,
        "state_inventory_sha256": _canonical_json_sha256(state_inventory),
        "state_components": component_inventory,
        "geometry_inventory_sha256": _canonical_json_sha256(geometry_inventory),
    }
    payload["aoi_sha256"] = _canonical_json_sha256(payload)
    return payload


def build_prediction_embedding(
    merged_chunk_pattern: str,
    feature_root: str | Path,
    state_path: str | Path,
    *,
    target_year: int = 2020,
    bounds: Mapping[str, float] | None = None,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a CONUS prediction table from validated mode-resampled chunks."""
    merged_files = sorted(Path(path) for path in glob.glob(merged_chunk_pattern))
    if not merged_files:
        raise FileNotFoundError(f"No merged chunks matched: {merged_chunk_pattern}")
    extent = _prediction_bounds(bounds)
    fragments: list[pd.DataFrame] = []
    for done, path in enumerate(merged_files, start=1):
        with xr.open_dataset(path) as dataset:
            required = {*ABANDONMENT_VARS, "landcover"}
            missing = required - set(dataset.data_vars)
            if missing:
                raise KeyError(f"{path.name} missing variables: {sorted(missing)}")
            lat_values = np.asarray(dataset.lat.values)
            lon_values = np.asarray(dataset.lon.values)
            lat_positions = np.flatnonzero(
                (lat_values >= extent["lat_min"]) & (lat_values <= extent["lat_max"])
            )
            lon_positions = np.flatnonzero(
                (lon_values >= extent["lon_min"]) & (lon_values <= extent["lon_max"])
            )
            if not lat_positions.size or not lon_positions.size:
                if progress_callback:
                    progress_callback({"done": done, "total": len(merged_files), "rows": 0})
                continue
            subset = dataset.isel(lat=lat_positions, lon=lon_positions).load()
        start = np.asarray(subset["abandonment_year"].values, dtype=np.float64)
        duration = np.asarray(subset["abandonment_duration"].values, dtype=np.float64)
        end = start + duration - 1
        active = np.isfinite(start) & np.isfinite(duration) & (start <= target_year) & (end >= target_year)
        rows, cols = np.nonzero(active)
        if rows.size:
            year_index = _year_indices(subset.time.values)
            if target_year not in year_index:
                raise ValueError(f"Target year {target_year} missing from {path.name}")
            target_landcover = np.asarray(subset["landcover"].isel(time=year_index[target_year]).values)
            event_landcover = np.full(rows.size, np.nan, dtype=np.float32)
            event_years = start[rows, cols].astype(np.int16)
            for event_year in np.unique(event_years):
                if int(event_year) not in year_index:
                    raise ValueError(f"Abandonment year {event_year} missing from {path.name}")
                positions = np.flatnonzero(event_years == event_year)
                event_slice = np.asarray(subset["landcover"].isel(time=year_index[int(event_year)]).values)
                event_landcover[positions] = event_slice[rows[positions], cols[positions]]
            fragments.append(
                pd.DataFrame(
                    {
                        "lat": subset.lat.values[rows],
                        "lon": subset.lon.values[cols],
                        "time": pd.Timestamp(f"{target_year}-01-01"),
                        "abandonment_year": start[rows, cols].astype(np.float32),
                        "abandonment_duration": duration[rows, cols].astype(np.float32),
                        "current_abandonment": np.ones(rows.size, dtype=np.uint8),
                        "landcover": target_landcover[rows, cols],
                        "landcover_at_abandonment": event_landcover,
                    }
                )
            )
        if progress_callback:
            progress_callback({"done": done, "total": len(merged_files), "rows": int(rows.size)})
    if not fragments:
        prediction = pd.DataFrame(columns=PREDICTION_COLUMNS)
        membership = pd.DataFrame(columns=STATE_MEMBERSHIP_COLUMNS)
        membership_summary = state_membership_summary(membership)
        prediction.attrs["state_membership_sha256"] = membership_summary["membership_sha256"]
        membership.attrs.update(membership_summary)
        return prediction, membership
    candidate = pd.concat(fragments, ignore_index=True)
    candidate, _ = _clip_prediction_to_conus(candidate, state_path)
    if candidate.empty:
        prediction = pd.DataFrame(columns=PREDICTION_COLUMNS)
        membership = pd.DataFrame(columns=STATE_MEMBERSHIP_COLUMNS)
        membership_summary = state_membership_summary(membership)
        prediction.attrs["state_membership_sha256"] = membership_summary["membership_sha256"]
        membership.attrs.update(membership_summary)
        return prediction, membership
    membership = build_state_membership_frame(candidate)

    feature_root = Path(feature_root)
    with xr.open_dataset(feature_root / "DEM.nc") as reference:
        grid_lat = reference.lat.values
        grid_lon = reference.lon.values
    tolerance = (1.0 / 120.0) / 2.0 + 1e-7
    lat_index = _nearest_indices(grid_lat, candidate["lat"].to_numpy(), tolerance)
    lon_index = _nearest_indices(grid_lon, candidate["lon"].to_numpy(), tolerance)
    if (lat_index < 0).any() or (lon_index < 0).any():
        raise ValueError("Prediction pixels do not align to the Feature_all master grid")
    if not np.allclose(candidate["lat"], grid_lat[lat_index], atol=1e-7, rtol=0) or not np.allclose(
        candidate["lon"], grid_lon[lon_index], atol=1e-7, rtol=0
    ):
        raise ValueError("Prediction and Feature_all coordinates are not exact master-grid matches")
    for variable in [*FEATURE_2D_VARS, *FEATURE_3D_VARS]:
        paths = _feature_paths(feature_root, variable, (target_year,))
        context = xr.open_dataset(paths[0]) if len(paths) == 1 else xr.open_mfdataset(paths, combine="by_coords")
        with context as feature_dataset:
            data = feature_dataset[variable]
            candidate[variable] = _vector_extract(
                data,
                lat_index,
                lon_index,
                candidate["time"] if "time" in data.dims else None,
            )
        if progress_callback:
            progress_callback({"stage": "feature", "variable": variable})
    candidate["time"] = pd.to_datetime(candidate["time"]).dt.strftime("%Y-%m-%d")
    prediction = candidate[PREDICTION_COLUMNS].copy()
    membership_summary = state_membership_summary(membership)
    prediction.attrs["state_membership_sha256"] = membership_summary["membership_sha256"]
    membership.attrs.update(membership_summary)
    return prediction, membership


def _stage_prediction_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=destination.parent,
        suffix=".next.csv",
        mode="w",
        encoding="utf-8",
        newline="",
    ) as handle:
        temporary = Path(handle.name)
    frame.to_csv(temporary, index=False)
    return temporary


def _stage_prediction_json(payload: Mapping[str, object], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=destination.parent,
        suffix=".next.json",
        mode="w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        return Path(handle.name)


def _commit_exact_prediction_outputs(staged: Mapping[Path, Path]) -> bool:
    """Publish a new output set, or reuse it only when every byte is identical."""
    existing = [destination for destination in staged if destination.exists()]
    if existing:
        if len(existing) != len(staged):
            missing = sorted(str(path) for path in staged if not path.exists())
            raise FileExistsError(f"Prediction output set is partial; missing={missing}")
        mismatches = [
            str(destination)
            for destination, temporary in staged.items()
            if sha256_file(destination) != sha256_file(temporary)
        ]
        if mismatches:
            raise FileExistsError(f"Refusing to overwrite different prediction outputs: {mismatches}")
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        return True

    promoted: list[Path] = []
    try:
        for destination, temporary in staged.items():
            if destination.exists():
                raise FileExistsError(f"Prediction output appeared during publication: {destination}")
            os.replace(temporary, destination)
            promoted.append(destination)
    except Exception:
        for destination in reversed(promoted):
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return False


def finalize_abandonment_prediction(
    detector_receipt: Mapping[str, object],
    *,
    feature_root: str | Path,
    state_path: str | Path,
    bounds: Mapping[str, float] | None = None,
    progress_callback=None,
) -> dict[str, object]:
    """Complete and fingerprint the CONUS CSV for a public detector receipt."""
    required_receipt = {
        "feature",
        "parameters",
        "parameter_sha256",
        "run_fingerprint",
        "source_sha256",
        "abandonment_chunk_root",
        "merged_chunk_root",
        "expected_chunk_count",
        "verified_chunk_count",
        "abandonment_manifest_path",
        "merged_manifest_path",
        "required_csv_path",
        "required_acceptance_path",
    }
    missing = required_receipt - set(detector_receipt)
    if missing:
        raise ValueError(f"Detector receipt is missing fields: {sorted(missing)}")
    run_fingerprint = str(detector_receipt["run_fingerprint"])
    expected_count = int(detector_receipt["expected_chunk_count"])
    detector_root = Path(str(detector_receipt["abandonment_chunk_root"]))
    merged_root = Path(str(detector_receipt["merged_chunk_root"]))
    detector_manifest_path = Path(str(detector_receipt["abandonment_manifest_path"]))
    merged_manifest_path = Path(str(detector_receipt["merged_manifest_path"]))
    for path in (detector_manifest_path, merged_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    detector_manifest = json.loads(detector_manifest_path.read_text(encoding="utf-8"))
    merged_manifest = json.loads(merged_manifest_path.read_text(encoding="utf-8"))
    if detector_manifest.get("run_fingerprint") != run_fingerprint:
        raise ValueError("Detector manifest fingerprint differs from the public receipt")
    if merged_manifest.get("run_fingerprint") != run_fingerprint:
        raise ValueError("Merged manifest fingerprint differs from the public receipt")
    detector_files = sorted(detector_root.glob("chunk_*.nc"))
    merged_files = sorted(merged_root.glob("chunk_*.nc"))
    detector_keys = [path.name for path in detector_files]
    merged_keys = [path.name for path in merged_files]
    if len(detector_keys) != expected_count or len(merged_keys) != expected_count:
        raise ValueError(
            f"Expected {expected_count} detector and merged chunks; "
            f"found detector={len(detector_keys)} merged={len(merged_keys)}"
        )
    if detector_keys != merged_keys:
        raise ValueError("Detector and merged chunk-key inventories differ")
    if int(detector_receipt["verified_chunk_count"]) != len(detector_keys):
        raise ValueError("Detector receipt verified count differs from the physical inventory")

    parameters = dict(detector_receipt["parameters"])
    target_year = int(parameters["current_end_year"])
    candidate, membership = build_prediction_embedding(
        str(merged_root / "chunk_*.nc"),
        feature_root,
        state_path,
        target_year=target_year,
        bounds=bounds,
        progress_callback=progress_callback,
    )
    validation = validate_prediction_candidate(candidate, target_year=target_year)
    if not validation["passed"]:
        raise ValueError(validation)
    membership_report = state_membership_summary(membership)
    _, california_report = reconstruct_state_subset(
        candidate,
        membership,
        state_fips="06",
        expected_membership_sha256=str(membership_report["membership_sha256"]),
    )
    aoi_contract = prediction_aoi_contract(state_path, bounds=bounds)
    if Path(state_path).name == "cb_2018_us_state_500k.shp" and aoi_contract["state_polygon_count"] != 49:
        raise ValueError("Canonical CONUS state boundary must retain exactly 49 state/DC polygons")

    csv_path = Path(str(detector_receipt["required_csv_path"]))
    manifest_path = csv_path.with_suffix(".manifest.json")
    acceptance_path = Path(str(detector_receipt["required_acceptance_path"]))
    staged: dict[Path, Path] = {}
    try:
        staged[csv_path] = _stage_prediction_csv(candidate, csv_path)
        roundtrip = pd.read_csv(staged[csv_path])
        roundtrip_validation = validate_prediction_candidate(roundtrip, target_year=target_year)
        if not roundtrip_validation["passed"] or len(roundtrip) != len(candidate):
            raise ValueError(roundtrip_validation)
        csv_sha256 = sha256_file(staged[csv_path])
        chunk_key_sha256 = _canonical_json_sha256(detector_keys)
        manifest = {
            "schema_version": "us-abandon-clean-feature-manifest.v1",
            "status": "complete",
            "feature": detector_receipt["feature"],
            "parameters": parameters,
            "parameter_sha256": detector_receipt["parameter_sha256"],
            "run_fingerprint": run_fingerprint,
            "source_sha256": detector_receipt["source_sha256"],
            "validated_through_year": (
                int(parameters["current_end_year"]) + 2
                if bool(parameters["extend_validation"])
                and bool(detector_receipt.get("extension_years_available"))
                else int(parameters["current_end_year"])
            ),
            "detector": {
                "root": str(detector_root),
                "manifest_path": str(detector_manifest_path),
                "manifest_sha256": sha256_file(detector_manifest_path),
                "chunk_count": len(detector_keys),
                "chunk_key_sha256": chunk_key_sha256,
            },
            "merged": {
                "root": str(merged_root),
                "manifest_path": str(merged_manifest_path),
                "manifest_sha256": sha256_file(merged_manifest_path),
                "chunk_count": len(merged_keys),
                "chunk_key_sha256": chunk_key_sha256,
            },
            "aoi": aoi_contract,
            "aoi_sha256": aoi_contract["aoi_sha256"],
            "prediction_validation": validation,
            "roundtrip_validation": roundtrip_validation,
            "csv": {
                "path": str(csv_path),
                "sha256": csv_sha256,
                "size_bytes": int(staged[csv_path].stat().st_size),
                "rows": int(len(candidate)),
                "columns": list(candidate.columns),
            },
            "state_membership": membership_report,
            "california_subset": california_report,
            "california_csv_published": False,
            "product_state": (
                "local_zero_event_accepted" if candidate.empty else "local_product_accepted"
            ),
            "zero_event_explanation": (
                "No active abandonment pixels remained after the confirmed CONUS AOI filters."
                if candidate.empty
                else None
            ),
        }
        staged[manifest_path] = _stage_prediction_json(manifest, manifest_path)
        manifest_sha256 = sha256_file(staged[manifest_path])
        acceptance = {
            "schema_version": "local-run-abandonment-detection-acceptance.v1",
            "status": "accepted",
            "accepted": True,
            "feature": detector_receipt["feature"],
            "parameter_sha256": detector_receipt["parameter_sha256"],
            "aoi_sha256": aoi_contract["aoi_sha256"],
            "run_fingerprint": run_fingerprint,
            "expected_chunk_count": expected_count,
            "verified_chunk_count": len(detector_keys),
            "csv_path": str(csv_path),
            "csv_sha256": csv_sha256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "state_membership_sha256": membership_report["membership_sha256"],
            "california_subset_sha256": california_report["subset_sha256"],
            "product_state": (
                "local_zero_event_accepted" if candidate.empty else "local_product_accepted"
            ),
        }
        staged[acceptance_path] = _stage_prediction_json(acceptance, acceptance_path)
        reused = _commit_exact_prediction_outputs(staged)
    except Exception:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise
    finalized = dict(detector_receipt)
    finalized.pop("acceptance_pending", None)
    finalized.update(acceptance)
    finalized.update(
        {
            "publication_reused": reused,
            "required_manifest_path": str(manifest_path),
            "required_acceptance_path": str(acceptance_path),
        }
    )
    return finalized


def validate_prediction_candidate(
    candidate: pd.DataFrame,
    *,
    target_year: int = 2020,
) -> dict[str, object]:
    errors: list[str] = []
    if list(candidate.columns) != PREDICTION_COLUMNS:
        errors.append("prediction column order differs from the canonical schema")
    if not candidate.empty:
        parsed_time = pd.to_datetime(candidate["time"], errors="coerce")
        if parsed_time.isna().any() or set(parsed_time.dt.year.unique()) != {target_year}:
            errors.append("prediction time values do not match the target year")
        start = pd.to_numeric(candidate["abandonment_year"], errors="coerce")
        duration = pd.to_numeric(candidate["abandonment_duration"], errors="coerce")
        active = start.le(target_year) & (start + duration - 1).ge(target_year)
        if not active.all():
            errors.append("prediction contains pixels not active in the target year")
        current = pd.to_numeric(candidate["current_abandonment"], errors="coerce")
        if not current.eq(1).all():
            errors.append("current_abandonment must equal one for every prediction row")
    duplicates = int(candidate.duplicated(["time", "lat", "lon"]).sum())
    if duplicates:
        errors.append(f"prediction contains {duplicates} duplicate grid keys")
    return {
        "rows": len(candidate),
        "columns": list(candidate.columns),
        "target_year": target_year,
        "duplicate_grid_keys": duplicates,
        "errors": errors,
        "passed": not errors,
    }


def build_prediction_pool(
    candidate: pd.DataFrame,
    pv_embedding: pd.DataFrame,
    *,
    fill_columns: Sequence[str] = (
        "GDPpc",
        "GDPtot",
        "GURdist",
        "Population",
        "PrimaryRoad",
        "SecondaryRoad",
        "TertiaryRoad",
        "gdmp",
    ),
) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    result = candidate.copy()
    for column in fill_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        positive = values.gt(0) & values.notna()
        missing = ~positive
        if missing.any():
            if not positive.any():
                raise ValueError(f"No positive values are available to fill {column}")
            tree = cKDTree(result.loc[positive, ["lat", "lon"]].to_numpy())
            _, indices = tree.query(result.loc[missing, ["lat", "lon"]].to_numpy(), k=1)
            result.loc[missing, column] = values.loc[positive].to_numpy()[indices]
    resolution = 1.0 / 120.0
    prediction_keys = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(result["time"]).dt.year,
            np.rint((90.0 - result["lat"]) / resolution - 0.5).astype(np.int64),
            np.rint((result["lon"] + 180.0) / resolution - 0.5).astype(np.int64),
        ]
    )
    pv_keys = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(pv_embedding["time"]).dt.year,
            np.rint((90.0 - pv_embedding["lat"]) / resolution - 0.5).astype(np.int64),
            np.rint((pv_embedding["lon"] + 180.0) / resolution - 0.5).astype(np.int64),
        ]
    )
    return result.loc[~prediction_keys.isin(pv_keys)].reset_index(drop=True)


def project_abandonment_to_sample_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Project a 1992-2022 event interval onto each embedding row's year."""
    result = frame.copy()
    sample_year = pd.to_datetime(result["time"], errors="coerce").dt.year
    start_year = pd.to_numeric(result["abandonment_year"], errors="coerce")
    duration = pd.to_numeric(result["abandonment_duration"], errors="coerce")
    end_year = start_year + duration - 1
    active = start_year.notna() & duration.notna() & sample_year.notna()
    active &= start_year.le(sample_year) & end_year.ge(sample_year)

    event_columns = [*ABANDONMENT_VARS, "landcover_at_abandonment"]
    result.loc[~active, event_columns] = np.nan
    result.loc[active, "recultivation"] = 0
    result.loc[active, "current_abandonment"] = 1
    return result


def validate_embedding_candidate(
    candidate: pd.DataFrame,
    previous: pd.DataFrame,
    *,
    years: Sequence[int] = (2018, 2020),
    conus_bounds: Mapping[str, float] | None = None,
    expected_rows: int | None = None,
) -> dict[str, object]:
    errors = []
    if list(candidate.columns) != EXPECTED_COLUMNS:
        errors.append("candidate column order differs from expected 28-column schema")
    if expected_rows is not None and len(candidate) != expected_rows:
        errors.append(f"candidate row count {len(candidate)} differs from expected {expected_rows}")
    parsed_time = pd.to_datetime(candidate["time"], errors="coerce")
    if parsed_time.isna().any() or not set(parsed_time.dt.year.unique()).issubset(set(years)):
        errors.append("candidate time values are invalid")
    duplicate_count = int(candidate.duplicated(["time", "lat", "lon", "unique_id"]).sum())
    if duplicate_count:
        errors.append(f"candidate contains {duplicate_count} duplicate keys")
    sample_year = parsed_time.dt.year
    start_year = pd.to_numeric(candidate["abandonment_year"], errors="coerce")
    duration = pd.to_numeric(candidate["abandonment_duration"], errors="coerce")
    event_rows = start_year.notna() | duration.notna()
    active = (
        start_year.notna()
        & duration.notna()
        & start_year.le(sample_year)
        & (start_year + duration - 1).ge(sample_year)
    )
    if (event_rows & ~active).any():
        errors.append("abandonment fields include events that are not active in the sample year")
    current = pd.to_numeric(candidate["current_abandonment"], errors="coerce")
    if not current.loc[active].eq(1).all() or current.loc[~active].notna().any():
        errors.append("current_abandonment is not projected to the sample year")

    def with_grid_keys(frame: pd.DataFrame) -> pd.DataFrame:
        keyed = frame.copy()
        keyed["time"] = pd.to_datetime(keyed["time"]).dt.strftime("%Y-%m-%d")
        resolution = 1.0 / 120.0
        keyed["__lat_key"] = np.rint((90.0 - pd.to_numeric(keyed["lat"])) / resolution - 0.5).astype(np.int64)
        keyed["__lon_key"] = np.rint((pd.to_numeric(keyed["lon"]) + 180.0) / resolution - 0.5).astype(np.int64)
        return keyed

    keys = ["time", "__lat_key", "__lon_key"]
    candidate_common = with_grid_keys(candidate)
    previous_common = with_grid_keys(previous)
    if candidate_common.duplicated(keys).any() or previous_common.duplicated(keys).any():
        errors.append("candidate or previous embedding contains duplicate standard-grid keys")
    common = candidate_common.merge(previous_common, on=keys, how="inner", suffixes=("_new", "_old"))
    if len(common) != len(candidate) or len(common) != len(previous):
        errors.append("candidate and previous embedding row domains differ on standard-grid keys")
    invariant_differences = {}
    for column in INVARIANT_COLUMNS:
        new_name, old_name = f"{column}_new", f"{column}_old"
        if new_name not in common or old_name not in common:
            continue
        if pd.api.types.is_numeric_dtype(common[new_name]) and pd.api.types.is_numeric_dtype(common[old_name]):
            difference = ~np.isclose(common[new_name], common[old_name], equal_nan=True, atol=1e-6, rtol=1e-6)
        else:
            difference = common[new_name].fillna("<NA>") != common[old_name].fillna("<NA>")
        invariant_differences[column] = int(difference.sum())
    if any(invariant_differences.values()):
        errors.append("invariant PV/environment columns changed on common keys")

    changed_summary = {}
    for column in CHANGED_COLUMNS:
        new_name, old_name = f"{column}_new", f"{column}_old"
        if new_name in common and old_name in common:
            if pd.api.types.is_numeric_dtype(common[new_name]) and pd.api.types.is_numeric_dtype(common[old_name]):
                changed = ~np.isclose(common[new_name], common[old_name], equal_nan=True)
            else:
                changed = common[new_name].fillna("<NA>") != common[old_name].fillna("<NA>")
            changed_summary[column] = int(changed.sum())

    bounds = dict(conus_bounds or {"lon_min": -125, "lon_max": -65, "lat_min": 25, "lat_max": 49})
    conus = candidate.loc[
        candidate["lon"].between(bounds["lon_min"], bounds["lon_max"])
        & candidate["lat"].between(bounds["lat_min"], bounds["lat_max"])
    ].copy()
    report = {
        "candidate_rows": len(candidate),
        "previous_rows": len(previous),
        "common_rows": len(common),
        "candidate_only_rows": len(candidate) - len(common),
        "previous_only_rows": len(previous) - len(common),
        "duplicate_keys": duplicate_count,
        "invariant_differences": invariant_differences,
        "allowed_changed_columns": changed_summary,
        "conus_rows": len(conus),
        "conus_bounds": bounds,
        "max_coordinate_shift_degrees": {
            "lat": float(np.abs(common["lat_new"] - common["lat_old"]).max()) if len(common) else None,
            "lon": float(np.abs(common["lon_new"] - common["lon_old"]).max()) if len(common) else None,
        },
        "errors": errors,
        "passed": not errors,
    }
    return {"report": report, "conus": conus}


def promote_training_embedding(
    candidate_path: str | Path,
    current_path: str | Path,
    backup_path: str | Path,
    *,
    expected_current_sha256: str | None = None,
) -> dict[str, object]:
    candidate = Path(candidate_path)
    current = Path(current_path)
    backup = Path(backup_path)
    if not candidate.exists() or not current.exists():
        raise FileNotFoundError(f"candidate={candidate.exists()} current={current.exists()}")
    if backup.exists():
        raise FileExistsError(f"Backup already exists and will not be overwritten: {backup}")
    current_hash = sha256_file(current)
    if expected_current_sha256 and current_hash.upper() != expected_current_sha256.upper():
        raise ValueError(f"Current embedding hash changed: {current_hash}")
    os.replace(current, backup)
    try:
        os.replace(candidate, current)
    except Exception:
        os.replace(backup, current)
        raise
    return {
        "current_path": str(current),
        "backup_path": str(backup),
        "backup_sha256": sha256_file(backup),
        "current_sha256": sha256_file(current),
    }


def publish_embedding_candidate(
    candidate_path: str | Path,
    final_path: str | Path,
) -> dict[str, object]:
    """Atomically publish a validated candidate without modifying legacy CSVs."""
    candidate = Path(candidate_path)
    final = Path(final_path)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    candidate_hash = sha256_file(candidate)
    if final.exists():
        final_hash = sha256_file(final)
        if final_hash != candidate_hash:
            raise FileExistsError(f"Refusing to overwrite a different embedding: {final}")
        candidate.unlink()
        return {
            "final_path": str(final),
            "final_sha256": final_hash,
            "reused": True,
        }
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate, final)
    return {
        "final_path": str(final),
        "final_sha256": sha256_file(final),
        "reused": False,
    }


def write_embedding_outputs(
    candidate: pd.DataFrame,
    previous: pd.DataFrame,
    *,
    candidate_path: str | Path,
    conus_path: str | Path,
    report_path: str | Path,
    expected_rows: int | None = None,
) -> dict[str, object]:
    validation = validate_embedding_candidate(candidate, previous, expected_rows=expected_rows)
    if not validation["report"]["passed"]:
        raise ValueError(validation["report"])
    _atomic_csv(candidate, Path(candidate_path))
    _atomic_csv(validation["conus"][EXPECTED_COLUMNS], Path(conus_path))
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(validation["report"], indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, report_file)
    return validation["report"]


__all__ = [
    "ABANDONMENT_VARS",
    "CHANGED_COLUMNS",
    "EXPECTED_COLUMNS",
    "FEATURE_2D_VARS",
    "FEATURE_3D_VARS",
    "INVARIANT_COLUMNS",
    "PREDICTION_COLUMNS",
    "DEFAULT_CONUS_BOUNDS",
    "EXCLUDED_CONUS_STATE_FIPS",
    "STATE_MEMBERSHIP_COLUMNS",
    "build_prediction_embedding",
    "build_prediction_pool",
    "build_state_membership_frame",
    "build_training_embedding",
    "finalize_abandonment_prediction",
    "load_aligned_pv_sites",
    "prediction_aoi_contract",
    "prediction_row_keys",
    "project_abandonment_to_sample_year",
    "publish_embedding_candidate",
    "promote_training_embedding",
    "sha256_file",
    "state_membership_summary",
    "reconstruct_state_subset",
    "validate_embedding_candidate",
    "validate_prediction_candidate",
    "write_embedding_outputs",
]
