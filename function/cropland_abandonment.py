from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod


SIMPLIFIED_ESA_CLASSES: dict[int, tuple[int, ...]] = {
    1: (10, 11, 12, 20, 30, 40),
    2: (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170),
    3: (100, 110),
    4: (120, 121, 122),
    5: (130, 140, 150, 151, 152, 153),
    6: (180,),
    7: (190,),
    8: (200, 201, 202, 220),
    9: (210,),
}
LCMAP_LCPRI_TO_SIMPLIFIED9: dict[int, int] = {
    0: 0,
    1: 7,
    2: 1,
    3: 4,
    4: 2,
    5: 9,
    6: 6,
    7: 8,
    8: 5,
}
SIMPLIFIED_CLASS_MEANINGS = {
    0: "nodata",
    1: "cropland",
    2: "forest_and_flooded_vegetation",
    3: "shrubland",
    4: "grassland",
    5: "lichens_and_sparse_vegetation",
    6: "wetland",
    7: "settlement",
    8: "other_land",
    9: "water",
}
DEFAULT_RAW_CROPLAND_CODES = frozenset(SIMPLIFIED_ESA_CLASSES[1])
DEFAULT_RAW_BUILT_UP_CODES = frozenset({190})
DEFAULT_RAW_WETLAND_CODES = frozenset({180})
DEFAULT_SIMPLIFIED_CROPLAND_CODES = frozenset({1})
DEFAULT_SIMPLIFIED_BUILT_UP_CODES = frozenset({7})
DEFAULT_SIMPLIFIED_WETLAND_CODES = frozenset({6})
DEFAULT_BINARY_VARS = ("recultivation", "current_abandonment", "qualifies_at_cutoff")
WGS84_GEOD = Geod(ellps="WGS84")
REFERENCE_SCAN_MAX_PIXELS = 5_000_000
LEGACY_OUTPUT_VARS = (
    "abandonment_year",
    "abandonment_duration",
    "recultivation",
    "current_abandonment",
)
CUTOFF_SAFE_OUTPUT_VARS = (
    "abandonment_year",
    "abandonment_duration",
    "recultivation",
    "current_abandonment",
    "abandonment_end_year",
    "qualifies_at_cutoff",
    "eligible_cropland",
    "persistence_validated",
)
CUTOFF_SAFE_BINARY_VARS = (
    "recultivation",
    "current_abandonment",
    "qualifies_at_cutoff",
    "eligible_cropland",
    "persistence_validated",
)
DEFAULT_LOCAL_TARGET_NC = Path(r"D:\xarray\reclass_lccs_1km.nc")
DEFAULT_LOCAL_CHUNK_SIZE = 500
_LOCAL_OUTPUT_ROOT = Path(r"D:\xarray")
_LOCAL_FEATURE_ROOT = Path(r"D:\xarray\aligned2\Feature_all")
_LOCAL_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "cb_2018_us_state_500k.shp"
_LOCAL_PREDICTION_ENABLED = True


def reclassify_esa_cci(
    landcover: xr.DataArray,
    *,
    mode: str = "simplified9",
    output_name: str | None = None,
    cropland_codes: Sequence[int] = tuple(sorted(DEFAULT_RAW_CROPLAND_CODES)),
) -> xr.DataArray:
    if mode not in {"simplified9", "binary_crop"}:
        raise ValueError("mode must be 'simplified9' or 'binary_crop'.")

    if mode == "simplified9":
        result = xr.apply_ufunc(
            _simplified9_kernel,
            landcover,
            dask="parallelized",
            output_dtypes=[np.uint8],
        )
        result.name = output_name or "esa_cci_simplified9"
        result.attrs.update(
            {
                "classification_schema": "esa_cci_simplified9",
                "flag_values": list(range(10)),
                "flag_meanings": " ".join(
                    SIMPLIFIED_CLASS_MEANINGS[idx] for idx in range(10)
                ),
            }
        )
        result.encoding = dict(result.encoding)
        result.encoding["_FillValue"] = None
        return result

    result = xr.apply_ufunc(
        _binary_crop_kernel,
        landcover,
        input_core_dims=[[]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        kwargs={"cropland_codes": tuple(int(code) for code in cropland_codes)},
        output_dtypes=[np.uint8],
    )
    result.name = output_name or "cropland_mask"
    result.attrs.update(
        {
            "classification_schema": "binary_crop",
            "flag_values": [0, 1],
            "flag_meanings": "non_cropland_or_nodata cropland",
            "cropland_codes": list(map(int, cropland_codes)),
        }
    )
    result.encoding = dict(result.encoding)
    result.encoding["_FillValue"] = None
    return result


def reclassify_lcmap_lcpri(
    landcover: xr.DataArray,
    *,
    output_name: str = "lcmap_simplified9",
    fail_on_unknown: bool = True,
) -> xr.DataArray:
    result = xr.apply_ufunc(
        _lcmap_to_simplified9_kernel,
        landcover,
        dask="parallelized",
        output_dtypes=[np.uint8],
    )
    if fail_on_unknown:
        invalid = xr.apply_ufunc(
            _lcmap_unknown_mask_kernel,
            landcover,
            dask="parallelized",
            output_dtypes=[np.uint8],
        )
        invalid_flag = invalid.max()
        if bool(invalid_flag.compute().item() if hasattr(invalid_flag.data, "compute") else invalid_flag.item()):
            raise ValueError("LCMAP LCPRI contains unknown values outside the supported mapping.")
    result.name = output_name
    result.attrs.update(
        {
            "classification_schema": "lcmap_lcpri_to_simplified9",
            "flag_values": list(range(10)),
            "flag_meanings": " ".join(SIMPLIFIED_CLASS_MEANINGS[idx] for idx in range(10)),
            "source_mapping": json.dumps(LCMAP_LCPRI_TO_SIMPLIFIED9, sort_keys=True),
        }
    )
    result.encoding = dict(result.encoding)
    result.encoding["_FillValue"] = None
    return result


def aggregate_categorical_mode(
    landcover: xr.DataArray,
    *,
    lat_factor: int = 3,
    lon_factor: int = 3,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    output_name: str | None = None,
) -> xr.DataArray:
    if lat_factor < 1 or lon_factor < 1:
        raise ValueError("Aggregation factors must be positive integers.")

    aligned = landcover
    if landcover.chunks is not None:
        rechunk = {}
        for dimension, factor in ((lat_dim, lat_factor), (lon_dim, lon_factor)):
            current_max = max(landcover.chunks[landcover.get_axis_num(dimension)])
            target = max(factor, (current_max // factor) * factor)
            rechunk[dimension] = target
        aligned = landcover.chunk(rechunk)

    coarse = aligned.coarsen(
        {lat_dim: lat_factor, lon_dim: lon_factor},
        boundary="trim",
        coord_func={lat_dim: "mean", lon_dim: "mean"},
    ).construct(
        {
            lat_dim: (lat_dim, f"{lat_dim}_window"),
            lon_dim: (lon_dim, f"{lon_dim}_window"),
        }
    )

    result = xr.apply_ufunc(
        _window_mode_block,
        coarse,
        input_core_dims=[[f"{lat_dim}_window", f"{lon_dim}_window"]],
        output_core_dims=[[]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[landcover.dtype],
    )
    result = result.assign_coords(
        {
            lat_dim: landcover[lat_dim]
            .coarsen({lat_dim: lat_factor}, boundary="trim")
            .mean()
            .values,
            lon_dim: landcover[lon_dim]
            .coarsen({lon_dim: lon_factor}, boundary="trim")
            .mean()
            .values,
        }
    )
    result.name = output_name or landcover.name
    result.attrs = dict(landcover.attrs)
    result.attrs.update(
        {
            "aggregation": "categorical_mode",
            "aggregation_factors": f"{lat_dim}={lat_factor},{lon_dim}={lon_factor}",
        }
    )
    return result


def temporal_majority_filter(
    landcover: xr.DataArray,
    *,
    window: int = 5,
    dim: str = "time",
    output_name: str | None = None,
) -> xr.DataArray:
    if window < 1 or window % 2 == 0:
        raise ValueError("Window must be a positive odd integer.")
    if dim not in landcover.dims:
        raise ValueError(f"Dimension '{dim}' not found.")

    result = xr.apply_ufunc(
        _temporal_majority_block,
        landcover,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=False,
        dask="parallelized",
        kwargs={"window": int(window)},
        output_dtypes=[landcover.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    result = result.transpose(*landcover.dims)
    result.name = output_name or landcover.name
    result.attrs = dict(landcover.attrs)
    result.attrs.update({"temporal_filter_window": int(window), "temporal_filter": "majority"})
    return result


def detect_abandonment(
    landcover: xr.DataArray,
    *,
    schema: str = "raw_esa",
    cropland_codes: Sequence[int] | None = None,
    built_up_codes: Sequence[int] | None = None,
    wetland_codes: Sequence[int] | None = None,
    baseline_years: tuple[int, int] = (1992, 1997),
    min_abandonment_years: int = 5,
    required_through_year: int = 2020,
    analysis_end_year: int | None = None,
    recultivation_years: int = 1,
    detector: str = "stable_crop",
    pre_window: int = 5,
    post_window: int = 5,
    max_noncrop_pre_years: int = 0,
    max_crop_post_years: int = 0,
    time_dim: str = "time",
) -> xr.Dataset:
    if time_dim not in landcover.dims:
        raise ValueError(f"Dimension '{time_dim}' not found.")
    if min_abandonment_years < 1 or recultivation_years < 1:
        raise ValueError("Minimum abandonment and recultivation years must be positive.")
    if detector not in {"stable_crop", "xie_window"}:
        raise ValueError("detector must be 'stable_crop' or 'xie_window'.")

    years = _as_year_index(landcover[time_dim].values)
    if years.size == 0:
        raise ValueError("Landcover time dimension is empty.")
    if not np.all(np.diff(years) == 1):
        raise ValueError("Landcover years must be continuous with a step of 1 year.")

    analysis_end_year = int(analysis_end_year or years[-1])
    if analysis_end_year not in years:
        raise ValueError("analysis_end_year must exist in the provided time axis.")
    if required_through_year not in years:
        raise ValueError("required_through_year must exist in the provided time axis.")

    crop_codes, built_codes, wetland_class_codes, nodata_codes = _resolve_schema_codes(
        schema=schema,
        cropland_codes=cropland_codes,
        built_up_codes=built_up_codes,
        wetland_codes=wetland_codes,
    )

    outputs = xr.apply_ufunc(
        _detect_abandonment_series,
        landcover,
        input_core_dims=[[time_dim]],
        output_core_dims=[[], [], [], [], [], []],
        vectorize=True,
        dask="parallelized",
        kwargs={
            "years": years.astype(np.int64),
            "crop_codes": tuple(int(code) for code in crop_codes),
            "built_codes": tuple(int(code) for code in built_codes),
            "wetland_codes": tuple(int(code) for code in wetland_class_codes),
            "nodata_codes": tuple(int(code) for code in nodata_codes),
            "baseline_years": tuple(int(value) for value in baseline_years),
            "min_abandonment_years": int(min_abandonment_years),
            "required_through_year": int(required_through_year),
            "analysis_end_year": int(analysis_end_year),
            "recultivation_years": int(recultivation_years),
            "detector": detector,
            "pre_window": int(pre_window),
            "post_window": int(post_window),
            "max_noncrop_pre_years": int(max_noncrop_pre_years),
            "max_crop_post_years": int(max_crop_post_years),
        },
        output_dtypes=[np.float32, np.int16, np.uint8, np.uint8, np.float32, np.uint8],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    spatial_dims = tuple(dim for dim in landcover.dims if dim != time_dim)
    coords = {dim: landcover.coords[dim] for dim in spatial_dims}
    dataset = xr.Dataset(
        data_vars={
            "abandonment_year": xr.DataArray(outputs[0], coords=coords, dims=spatial_dims),
            "abandonment_duration": xr.DataArray(outputs[1], coords=coords, dims=spatial_dims),
            "recultivation": xr.DataArray(outputs[2], coords=coords, dims=spatial_dims),
            "current_abandonment": xr.DataArray(outputs[3], coords=coords, dims=spatial_dims),
            "abandonment_end_year": xr.DataArray(outputs[4], coords=coords, dims=spatial_dims),
            "qualifies_at_cutoff": xr.DataArray(outputs[5], coords=coords, dims=spatial_dims),
        },
        attrs={
            "classification_schema": schema,
            "baseline_years": list(map(int, baseline_years)),
            "min_abandonment_years": int(min_abandonment_years),
            "required_through_year": int(required_through_year),
            "analysis_end_year": int(analysis_end_year),
            "recultivation_years": int(recultivation_years),
            "detector": detector,
            "pre_window": int(pre_window),
            "post_window": int(post_window),
            "max_noncrop_pre_years": int(max_noncrop_pre_years),
            "max_crop_post_years": int(max_crop_post_years),
        },
    )
    for name in DEFAULT_BINARY_VARS:
        dataset[name].attrs.update({"flag_values": [0, 1], "flag_meanings": "false true"})
        dataset[name].encoding = dict(dataset[name].encoding)
        dataset[name].encoding["_FillValue"] = None
    dataset.attrs["crs"] = landcover.attrs.get("crs", "EPSG:4326")
    return dataset


def cell_area_m2(
    lat: xr.DataArray | np.ndarray | Sequence[float],
    lon: xr.DataArray | np.ndarray | Sequence[float],
    *,
    spatial_dims: tuple[str, str] = ("lat", "lon"),
    crs: str = "EPSG:4326",
) -> xr.DataArray:
    lat_values = np.asarray(lat, dtype=np.float64)
    lon_values = np.asarray(lon, dtype=np.float64)
    if lat_values.ndim != 1 or lon_values.ndim != 1:
        raise ValueError("lat and lon must be one-dimensional coordinate arrays.")

    lat_edges = _coordinate_edges(lat_values)
    lon_edges = _coordinate_edges(lon_values)
    if str(crs).upper() in {"EPSG:4326", "WGS84"}:
        lon_widths = np.abs(np.diff(lon_edges))
        unique_widths, inverse = np.unique(np.round(lon_widths, 15), return_inverse=True)
        width_lookup = {width: raw for width, raw in zip(unique_widths, _restore_widths(unique_widths, lon_widths))}
        band_matrix = np.empty((lat_values.size, unique_widths.size), dtype=np.float64)
        for lat_index in range(lat_values.size):
            south = lat_edges[lat_index]
            north = lat_edges[lat_index + 1]
            for width_index, width_key in enumerate(unique_widths):
                width = width_lookup[width_key]
                polygon_lon = [0.0, width, width, 0.0, 0.0]
                polygon_lat = [south, south, north, north, south]
                polygon_area, _ = WGS84_GEOD.polygon_area_perimeter(polygon_lon, polygon_lat)
                band_matrix[lat_index, width_index] = abs(polygon_area)
        area = band_matrix[:, inverse]
    else:
        lat_heights = np.abs(np.diff(lat_edges))
        lon_widths = np.abs(np.diff(lon_edges))
        area = lat_heights[:, None] * lon_widths[None, :]
    lat_coords = lat.values if isinstance(lat, xr.DataArray) else lat_values
    lon_coords = lon.values if isinstance(lon, xr.DataArray) else lon_values
    return xr.DataArray(
        area,
        coords={spatial_dims[0]: lat_coords, spatial_dims[1]: lon_coords},
        dims=spatial_dims,
        name="cell_area_m2",
        attrs={
            "units": "m2",
            "crs": crs,
            "geodetic_model": "WGS84" if str(crs).upper() in {"EPSG:4326", "WGS84"} else "projected_rectilinear",
        },
    )


def write_abandonment_chunks(
    dataset: xr.Dataset,
    output_dir: str | Path,
    *,
    chunk_slices: Sequence[Mapping[str, slice]] | None = None,
    chunk_indices: Sequence[tuple[int, int]] | None = None,
    spatial_dims: tuple[str, str] | None = None,
    chunk_size: int | None = None,
    resume: bool = False,
    manifest: Mapping[str, object] | None = None,
    prefix: str = "chunk",
) -> list[Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    spatial_dims = spatial_dims or _infer_spatial_dims(dataset)
    selections = list(chunk_slices or _auto_chunk_slices(dataset, spatial_dims=spatial_dims, chunk_size=chunk_size))
    manifest_path = target_dir / "run_manifest.json"
    if resume and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("spatial_dims") != list(spatial_dims):
            raise ValueError("Existing chunk manifest spatial_dims do not match current request.")
    if chunk_indices is not None and len(chunk_indices) != len(selections):
        raise ValueError("chunk_indices and chunk_slices must have the same length.")

    written: list[Path] = []
    generated = skipped = failed = 0
    for index, selection in enumerate(selections):
        subset = dataset.isel(selection) if selection else dataset
        if chunk_indices is None:
            filename = f"{prefix}_{index:04d}.nc"
        else:
            row_idx, col_idx = chunk_indices[index]
            filename = f"{prefix}_{int(row_idx):03d}_{int(col_idx):03d}.nc"
        destination = target_dir / filename
        if resume and destination.exists():
            try:
                with xr.open_dataset(destination) as existing:
                    if set(existing.data_vars) == set(dataset.data_vars):
                        written.append(destination)
                        skipped += 1
                        continue
            except Exception:
                pass
        _atomic_write_netcdf(subset, destination)
        written.append(destination)
        generated += 1
    payload = {
        "output_dir": str(target_dir),
        "spatial_dims": list(spatial_dims),
        "chunk_size": chunk_size,
        "chunk_paths": [str(path) for path in written],
        "stats": {"total": len(selections), "generated": generated, "skipped": skipped, "failed": failed},
    }
    if manifest is not None:
        payload.update(dict(manifest))
    _atomic_write_text(manifest_path, json.dumps(payload, indent=2, ensure_ascii=False))
    return written


def run_abandonment_detection(
    target_nc: str | Path = DEFAULT_LOCAL_TARGET_NC,
    window_year: int = 5,
    start_year: int = 1992,
    current_end_year: int = 2020,
    init_cropland: int = 2,
    extend_validation: bool = True,
) -> dict[str, object]:
    """Run the local cutoff-safe detector and retain its chunk inventories.

    This entry point owns the scientific event semantics.  The downstream
    complete-CONUS prediction CSV and final acceptance receipt are added by the
    prediction orchestration layer; until those products are verified this
    function reports ``accepted=False`` rather than claiming full delivery.
    """

    _validate_cutoff_safe_parameters(
        window_year=window_year,
        start_year=start_year,
        current_end_year=current_end_year,
        init_cropland=init_cropland,
        extend_validation=extend_validation,
    )
    source = Path(target_nc).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    feature = _cutoff_safe_feature_id(source, window_year)
    output_root = Path(_LOCAL_OUTPUT_ROOT).expanduser().resolve()
    abandonment_root = output_root / f"abandon_2_{feature}"
    merged_root = output_root / f"merged_chunk_2_{feature}"
    if abandonment_root == merged_root:
        raise ValueError("Abandonment and merged chunk roots must be distinct.")

    with xr.open_dataset(source) as metadata_ds:
        landcover_var, time_dim, spatial_dims = _select_cutoff_safe_landcover(metadata_ds)
        landcover_meta = metadata_ds[landcover_var]
        years = _as_year_index(landcover_meta[time_dim].values)
        _validate_cutoff_safe_year_axis(
            years,
            start_year=start_year,
            current_end_year=current_end_year,
            init_cropland=init_cropland,
        )
        source_metadata = {
            "landcover_var": landcover_var,
            "time_dim": time_dim,
            "spatial_dims": list(spatial_dims),
            "dimensions": {name: int(size) for name, size in landcover_meta.sizes.items()},
            "years": [int(years[0]), int(years[-1])],
            "dtype": str(landcover_meta.dtype),
            "crs": landcover_meta.attrs.get("crs", metadata_ds.attrs.get("crs", "EPSG:4326")),
            "coordinate_sha256": {
                name: _array_sha256(np.asarray(landcover_meta[name].values))
                for name in (*spatial_dims, time_dim)
            },
        }

    source_stat = source.stat()
    source_sha256 = _sha256_file(source).upper()
    code_sha256 = _sha256_file(Path(__file__).resolve()).upper()
    parameter_payload = {
        "target_nc": str(source),
        "window_year": int(window_year),
        "start_year": int(start_year),
        "current_end_year": int(current_end_year),
        "init_cropland": int(init_cropland),
        "extend_validation": bool(extend_validation),
        "feature": feature,
    }
    parameter_sha256 = _canonical_json_sha256(parameter_payload)

    core_indices = np.flatnonzero((years >= start_year) & (years <= current_end_year))
    extension_years = (current_end_year + 1, current_end_year + 2)
    extension_indices = [int(np.flatnonzero(years == year)[0]) for year in extension_years if np.any(years == year)]
    extension_available = len(extension_indices) == 2
    selected_indices = core_indices.tolist()
    if extend_validation and extension_available:
        selected_indices.extend(extension_indices)

    open_chunks = {
        time_dim: -1,
        spatial_dims[0]: DEFAULT_LOCAL_CHUNK_SIZE,
        spatial_dims[1]: DEFAULT_LOCAL_CHUNK_SIZE,
    }
    try:
        dataset_context = xr.open_dataset(source, chunks=open_chunks)
    except (ImportError, ValueError):
        dataset_context = xr.open_dataset(source)

    with dataset_context as source_ds:
        landcover = source_ds[landcover_var]
        selected = landcover.isel({time_dim: selected_indices})
        initial = landcover.isel({time_dim: core_indices[:init_cropland]})
        eligibility_mask = (initial == 1).all(time_dim).astype(np.uint8)
        eligibility_mask.name = "eligible_cropland"
        expected_keys = candidate_chunk_keys(
            eligibility_mask,
            chunk_size=DEFAULT_LOCAL_CHUNK_SIZE,
            spatial_dims=spatial_dims,
        )
        expected_names = {_chunk_path(Path("."), key).name for key in expected_keys}
        request_payload = {
            "schema_version": "cutoff-safe-local-run-request.v1",
            "parameter_sha256": parameter_sha256,
            "parameters": parameter_payload,
            "source": {
                "path": str(source),
                "size_bytes": int(source_stat.st_size),
                "mtime_ns": int(source_stat.st_mtime_ns),
                "sha256": source_sha256,
                **source_metadata,
            },
            "code_sha256": code_sha256,
            "output_roots": {
                "abandonment": str(abandonment_root),
                "merged": str(merged_root),
            },
            "class_contract": {
                "crop": 1,
                "built": 7,
                "valid_classes": list(range(1, 10)),
                "nodata_and_unmapped": "invalid",
            },
            "validated_through_year": int(
                current_end_year + 2
                if extend_validation and extension_available
                else current_end_year
            ),
            "extension_years_available": bool(extension_available),
            "chunk_size": DEFAULT_LOCAL_CHUNK_SIZE,
            "expected_keys": [list(key) for key in expected_keys],
        }
        run_fingerprint = _canonical_json_sha256(request_payload)
        request_payload["run_fingerprint"] = run_fingerprint
        _prepare_cutoff_safe_root(abandonment_root, request_payload, expected_names)
        _prepare_cutoff_safe_root(merged_root, request_payload, expected_names)

        generated = skipped = 0
        y_dim, x_dim = spatial_dims
        for key in expected_keys:
            row_start = key[0] * DEFAULT_LOCAL_CHUNK_SIZE
            col_start = key[1] * DEFAULT_LOCAL_CHUNK_SIZE
            selection = {
                y_dim: slice(row_start, min(row_start + DEFAULT_LOCAL_CHUNK_SIZE, selected.sizes[y_dim])),
                x_dim: slice(col_start, min(col_start + DEFAULT_LOCAL_CHUNK_SIZE, selected.sizes[x_dim])),
            }
            selected_chunk = selected.isel(selection).load()
            result = _detect_cutoff_safe_chunk(
                selected_chunk,
                window_year=window_year,
                start_year=start_year,
                current_end_year=current_end_year,
                init_cropland=init_cropland,
                extend_validation=extend_validation,
                extension_available=extension_available,
                time_dim=time_dim,
                spatial_dims=spatial_dims,
            )
            result.attrs.update(
                {
                    "feature": feature,
                    "parameter_sha256": parameter_sha256,
                    "run_fingerprint": run_fingerprint,
                    "source_sha256": source_sha256,
                }
            )
            detector_path = _chunk_path(abandonment_root, key)
            merged_path = _chunk_path(merged_root, key)
            merged = xr.merge([result, selected_chunk.rename("landcover")], compat="no_conflicts", join="exact")
            merged.attrs = dict(result.attrs)

            detector_exists = detector_path.exists()
            merged_exists = merged_path.exists()
            if detector_exists:
                _validate_cutoff_safe_chunk_file(
                    detector_path,
                    expected=result,
                    run_fingerprint=run_fingerprint,
                    merged=False,
                )
            else:
                _atomic_write_netcdf(result, detector_path)
            if merged_exists:
                _validate_cutoff_safe_chunk_file(
                    merged_path,
                    expected=merged,
                    run_fingerprint=run_fingerprint,
                    merged=True,
                )
            else:
                _atomic_write_netcdf(merged, merged_path)
            if detector_exists and merged_exists:
                skipped += 1
            else:
                generated += 1

    detector_inventory, event_counts = _inventory_cutoff_safe_chunks(
        abandonment_root,
        expected_keys=expected_keys,
        run_fingerprint=run_fingerprint,
        merged=False,
    )
    merged_inventory, _ = _inventory_cutoff_safe_chunks(
        merged_root,
        expected_keys=expected_keys,
        run_fingerprint=run_fingerprint,
        merged=True,
    )
    manifest_common = {
        **request_payload,
        "stats": {
            "total": len(expected_keys),
            "verified": len(detector_inventory),
            "failed": 0,
        },
        "event_counts": event_counts,
    }
    detector_manifest = {**manifest_common, "kind": "abandonment", "files": detector_inventory}
    merged_manifest = {**manifest_common, "kind": "merged", "files": merged_inventory}
    _retain_cutoff_safe_manifest(abandonment_root / "run_manifest.json", detector_manifest)
    _retain_cutoff_safe_manifest(merged_root / "run_manifest.json", merged_manifest)

    repository_root = Path(__file__).resolve().parents[1]
    required_csv_path = repository_root / "data" / f"us_abandon_clean_{feature}.csv"
    required_acceptance_path = (
        repository_root
        / "outputs"
        / "s0_us_abandon"
        / feature
        / "local_run_abandonment_detection_acceptance.json"
    )
    detector_receipt = {
        "status": "detector_chunks_complete_prediction_pending",
        "accepted": False,
        "acceptance_pending": "complete-CONUS CSV construction and full physical-product verification",
        "feature": feature,
        "parameters": parameter_payload,
        "parameter_sha256": parameter_sha256,
        "run_fingerprint": run_fingerprint,
        "source_sha256": source_sha256,
        "extension_years_available": bool(extension_available),
        "abandonment_chunk_root": str(abandonment_root),
        "merged_chunk_root": str(merged_root),
        "expected_chunk_count": len(expected_keys),
        "verified_chunk_count": len(detector_inventory),
        "operation_chunk_counts": {"generated": generated, "skipped": skipped},
        "abandonment_manifest_path": str(abandonment_root / "run_manifest.json"),
        "merged_manifest_path": str(merged_root / "run_manifest.json"),
        "required_csv_path": str(required_csv_path),
        "required_acceptance_path": str(required_acceptance_path),
        "event_counts": event_counts,
    }
    if not _LOCAL_PREDICTION_ENABLED:
        return detector_receipt
    from function.embedding_pipeline import finalize_abandonment_prediction

    return finalize_abandonment_prediction(
        detector_receipt,
        feature_root=_LOCAL_FEATURE_ROOT,
        state_path=_LOCAL_STATE_PATH,
    )


def _detect_cutoff_safe_chunk(
    landcover: xr.DataArray,
    *,
    window_year: int,
    start_year: int,
    current_end_year: int,
    init_cropland: int,
    extend_validation: bool,
    extension_available: bool | None = None,
    time_dim: str = "time",
    spatial_dims: tuple[str, str] | None = None,
) -> xr.Dataset:
    _validate_cutoff_safe_parameters(
        window_year=window_year,
        start_year=start_year,
        current_end_year=current_end_year,
        init_cropland=init_cropland,
        extend_validation=extend_validation,
    )
    if time_dim not in landcover.dims:
        raise ValueError(f"Missing time dimension: {time_dim}")
    spatial_dims = spatial_dims or tuple(dim for dim in landcover.dims if dim != time_dim)
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected two spatial dimensions, got {spatial_dims}")
    years = _as_year_index(landcover[time_dim].values)
    core_mask = (years >= start_year) & (years <= current_end_year)
    core_years = years[core_mask]
    if core_years.size != current_end_year - start_year + 1 or not np.array_equal(
        core_years, np.arange(start_year, current_end_year + 1)
    ):
        raise ValueError("The selected chunk must contain every core analysis year exactly once.")
    extension_years = (current_end_year + 1, current_end_year + 2)
    inferred_extension_available = all(np.count_nonzero(years == year) == 1 for year in extension_years)
    if extension_available is None:
        extension_available = inferred_extension_available
    if extension_available and not inferred_extension_available:
        raise ValueError("extension_available=True but the two exact extension years are absent.")

    selected_indices = np.flatnonzero(core_mask).tolist()
    if extend_validation and extension_available:
        selected_indices.extend(int(np.flatnonzero(years == year)[0]) for year in extension_years)
    values = np.asarray(landcover.isel({time_dim: selected_indices}).transpose(time_dim, *spatial_dims).values)
    time_count = values.shape[0]
    flat = np.ascontiguousarray(values.reshape(time_count, -1).T.astype(np.float32, copy=False))
    kernel = _cutoff_safe_detect_many_numba or _cutoff_safe_detect_many_python
    outputs = kernel(
        flat,
        int(core_years.size),
        int(start_year),
        int(current_end_year),
        int(window_year),
        int(init_cropland),
        bool(extend_validation),
        bool(extension_available),
    )
    shape = tuple(landcover.sizes[dim] for dim in spatial_dims)
    coords = {dim: landcover.coords[dim] for dim in spatial_dims}
    data_vars = {
        name: (spatial_dims, np.asarray(array).reshape(shape))
        for name, array in zip(CUTOFF_SAFE_OUTPUT_VARS, outputs)
    }
    result = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "detector_contract": "cutoff_safe_local_v1",
            "classification_schema": "simplified9",
            "window_year": int(window_year),
            "minimum_abandonment_years": int(window_year),
            "start_year": int(start_year),
            "current_end_year": int(current_end_year),
            "init_cropland": int(init_cropland),
            "extend_validation": int(bool(extend_validation)),
            "extension_years": list(extension_years) if extend_validation else [],
            "extension_years_available": int(bool(extension_available)),
            "validated_through_year": int(
                current_end_year + 2
                if extend_validation and extension_available
                else current_end_year
            ),
            "duration_definition": "inclusive_through_current_end_year",
            "extension_effect": "persistence_validated_only",
            "spatial_dims": list(spatial_dims),
        },
    )
    for name in CUTOFF_SAFE_BINARY_VARS:
        result[name].attrs.update({"flag_values": [0, 1], "flag_meanings": "false true"})
        result[name].encoding["_FillValue"] = None
    if landcover.attrs.get("crs") is not None:
        result.attrs["crs"] = landcover.attrs["crs"]
    return result


def _validate_cutoff_safe_parameters(
    *,
    window_year: int,
    start_year: int,
    current_end_year: int,
    init_cropland: int,
    extend_validation: bool,
) -> None:
    if type(window_year) is not int or window_year < 1:
        raise ValueError("window_year must be a positive integer.")
    if type(init_cropland) is not int or init_cropland < 1:
        raise ValueError("init_cropland must be a positive integer.")
    if type(start_year) is not int or type(current_end_year) is not int:
        raise ValueError("start_year and current_end_year must be integers.")
    if current_end_year < start_year:
        raise ValueError("current_end_year must be greater than or equal to start_year.")
    if type(extend_validation) is not bool:
        raise ValueError("extend_validation must be a boolean.")


def _validate_cutoff_safe_year_axis(
    years: np.ndarray,
    *,
    start_year: int,
    current_end_year: int,
    init_cropland: int,
) -> None:
    if years.size == 0:
        raise ValueError("Landcover time dimension is empty.")
    if np.unique(years).size != years.size or not np.all(np.diff(years) == 1):
        raise ValueError("Landcover years must be unique and consecutive with a one-year step.")
    required = np.arange(start_year, current_end_year + 1)
    available = years[(years >= start_year) & (years <= current_end_year)]
    if not np.array_equal(available, required):
        raise ValueError("The source must contain every year from start_year through current_end_year.")
    if init_cropland > required.size:
        raise ValueError("init_cropland exceeds the core analysis period.")


def _select_cutoff_safe_landcover(dataset: xr.Dataset) -> tuple[str, str, tuple[str, str]]:
    candidates = []
    for name, variable in dataset.data_vars.items():
        if "time" in variable.dims and variable.ndim == 3:
            candidates.append(name)
    if len(candidates) != 1:
        raise ValueError(
            "The source must contain exactly one unambiguous three-dimensional annual land-cover variable; "
            f"found {candidates}."
        )
    name = candidates[0]
    variable = dataset[name]
    spatial_dims = _infer_spatial_dims(variable)
    return name, "time", spatial_dims


def _cutoff_safe_feature_id(source: Path, window_year: int) -> str:
    stem = source.stem.lower()
    if "300m" in stem:
        feature = f"300m_min{window_year}"
        allowed = {"300m_min4", "300m_min6"}
    elif "1km" in stem:
        feature = f"1km_mode_min{window_year}"
        allowed = {"1km_mode_min4", "1km_mode_min5", "1km_mode_min6"}
    else:
        raise ValueError("Unable to infer the declared 300m or 1km_mode feature from target_nc.")
    if feature not in allowed:
        raise ValueError(f"Feature '{feature}' is outside the sealed five-feature family.")
    return feature


def _canonical_json_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def _prepare_cutoff_safe_root(root: Path, request: Mapping[str, object], expected_names: set[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "run_request.json"
    manifest_path = root / "run_manifest.json"
    temporary = sorted(path.name for path in root.glob("*.tmp*"))
    if temporary:
        raise ValueError(f"Temporary files block safe resume in {root}: {temporary}")
    existing_chunks = {path.name for path in root.glob("chunk_*.nc")}
    unexpected = existing_chunks - expected_names
    if unexpected:
        raise ValueError(f"Unexpected chunk files in {root}: {sorted(unexpected)}")
    if request_path.exists():
        existing = json.loads(request_path.read_text(encoding="utf-8"))
        if existing != dict(request):
            raise ValueError(f"Existing run request fingerprint does not match {root}.")
    else:
        if existing_chunks or manifest_path.exists():
            raise FileExistsError(f"Unfingerprinted outputs already exist in {root}; refusing to overwrite.")
        _atomic_write_text(request_path, json.dumps(request, indent=2, ensure_ascii=False))
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("run_fingerprint") != request.get("run_fingerprint"):
            raise ValueError(f"Existing run manifest fingerprint does not match {root}.")


def _validate_cutoff_safe_chunk_file(
    path: Path,
    *,
    expected: xr.Dataset,
    run_fingerprint: str,
    merged: bool,
) -> None:
    required = set(CUTOFF_SAFE_OUTPUT_VARS) | ({"landcover"} if merged else set())
    with xr.open_dataset(path) as observed:
        if set(observed.data_vars) != required:
            raise ValueError(f"{path} has unexpected variables: {sorted(observed.data_vars)}")
        if observed.attrs.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"{path} has a mismatched run fingerprint.")
        if dict(observed.sizes) != dict(expected.sizes):
            raise ValueError(f"{path} has unexpected dimensions.")
        for name in expected.coords:
            if name not in observed.coords or not _array_equal_or_allclose(observed[name].values, expected[name].values):
                raise ValueError(f"{path} has a coordinate mismatch for {name}.")
        for name in expected.data_vars:
            if observed[name].dtype != expected[name].dtype:
                raise ValueError(f"{path} has a dtype mismatch for {name}.")
            if not _array_equal_or_allclose(observed[name].values, expected[name].values):
                raise ValueError(f"{path} has a data mismatch for {name}.")


def _inventory_cutoff_safe_chunks(
    root: Path,
    *,
    expected_keys: Sequence[tuple[int, int]],
    run_fingerprint: str,
    merged: bool,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    expected_names = [_chunk_path(Path("."), key).name for key in expected_keys]
    actual_names = sorted(path.name for path in root.glob("chunk_*.nc"))
    if set(actual_names) != set(expected_names):
        raise ValueError(
            f"Chunk inventory mismatch in {root}: expected {sorted(expected_names)}, got {actual_names}."
        )
    required = set(CUTOFF_SAFE_OUTPUT_VARS) | ({"landcover"} if merged else set())
    records: list[dict[str, object]] = []
    counts = {
        "eligible_cropland": 0,
        "qualifies_at_cutoff": 0,
        "current_abandonment": 0,
        "persistence_validated": 0,
    }
    for name in expected_names:
        path = root / name
        with xr.open_dataset(path) as dataset:
            if set(dataset.data_vars) != required:
                raise ValueError(f"Unexpected variables in {path}.")
            if dataset.attrs.get("run_fingerprint") != run_fingerprint:
                raise ValueError(f"Run fingerprint mismatch in {path}.")
            if not merged:
                for field in counts:
                    counts[field] += int(np.count_nonzero(np.asarray(dataset[field].values) == 1))
            dimensions = {dim: int(size) for dim, size in dataset.sizes.items()}
            dtypes = {field: str(dataset[field].dtype) for field in dataset.data_vars}
        records.append(
            {
                "name": name,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path).upper(),
                "dimensions": dimensions,
                "dtypes": dtypes,
            }
        )
    return records, counts


def _retain_cutoff_safe_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    payload = dict(manifest)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Existing manifest content does not match verified files: {path}")
        return
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def validate_mode_reclass_file(
    path: str | Path,
    *,
    expected_years: int = 31,
    expected_lat: int = 21_600,
    expected_lon: int = 43_200,
    expected_resolution: float = 1.0 / 120.0,
) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    with xr.open_dataset(source) as dataset:
        if "lccs_class" not in dataset:
            raise KeyError("lccs_class")
        landcover = dataset["lccs_class"]
        expected_dims = {"time": expected_years, "lat": expected_lat, "lon": expected_lon}
        if dict(landcover.sizes) != expected_dims:
            raise ValueError(f"Unexpected dimensions: {dict(landcover.sizes)}")
        lat_step = float(abs(landcover.lat.values[1] - landcover.lat.values[0]))
        lon_step = float(abs(landcover.lon.values[1] - landcover.lon.values[0]))
        if not np.isclose(lat_step, expected_resolution, atol=1e-10, rtol=0.0):
            raise ValueError(f"Unexpected latitude resolution: {lat_step}")
        if not np.isclose(lon_step, expected_resolution, atol=1e-10, rtol=0.0):
            raise ValueError(f"Unexpected longitude resolution: {lon_step}")
        if dataset.attrs.get("spatial_aggregation") != "3x3_categorical_mode":
            raise ValueError("Missing spatial_aggregation=3x3_categorical_mode")

        sample_records = []
        for time_index in (0, expected_years // 2, expected_years - 1):
            for lat_index, lon_index in ((0, 0), (expected_lat // 2, expected_lon // 2), (expected_lat - 1, expected_lon - 1)):
                value = float(landcover.isel(time=time_index, lat=lat_index, lon=lon_index).values)
                if not (np.isnan(value) or value in range(10)):
                    raise ValueError(f"Invalid categorical sample: {value}")
                sample_records.append(
                    {
                        "time_index": time_index,
                        "lat_index": lat_index,
                        "lon_index": lon_index,
                        "value": value,
                    }
                )
        metadata = {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
            "dimensions": expected_dims,
            "lat_step": lat_step,
            "lon_step": lon_step,
            "dtype": str(landcover.dtype),
            "encoding": {
                key: landcover.encoding.get(key)
                for key in ("dtype", "zlib", "complevel", "shuffle", "chunksizes", "_FillValue")
            },
            "attrs": dict(dataset.attrs),
            "samples": sample_records,
            "passed": True,
        }
    return metadata


def monitor_mode_reclass_file(
    path: str | Path,
    *,
    poll_seconds: int = 60,
    stable_seconds: int = 600,
    timeout_seconds: int = 86_400,
    progress_callback=None,
    validation_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = Path(path)
    started = time.time()
    stable_started: float | None = None
    last_signature: tuple[int, int] | None = None
    probes: list[dict[str, object]] = []
    while time.time() - started <= timeout_seconds:
        if not source.exists():
            stable_started = None
            last_signature = None
            probe = {"exists": False, "elapsed_seconds": time.time() - started}
        else:
            stat = source.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if signature == last_signature:
                stable_started = stable_started or time.time()
            else:
                stable_started = None
            last_signature = signature
            stable_for = 0.0 if stable_started is None else time.time() - stable_started
            probe = {
                "exists": True,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "stable_seconds": stable_for,
                "elapsed_seconds": time.time() - started,
            }
            if stable_for >= stable_seconds:
                try:
                    validation = validate_mode_reclass_file(source, **dict(validation_kwargs or {}))
                    validation["probes"] = probes + [probe]
                    return validation
                except Exception as error:
                    probe["validation_error"] = repr(error)
        probes.append(probe)
        if progress_callback is not None:
            progress_callback(probe)
        time.sleep(poll_seconds)
    raise TimeoutError(f"Mode reclass file did not become ready within {timeout_seconds} seconds: {source}")


def build_legacy_candidate_mask(
    landcover: xr.DataArray,
    *,
    time_dim: str = "time",
    crop_code: int = 1,
    built_code: int = 7,
) -> xr.DataArray:
    if time_dim not in landcover.dims:
        raise ValueError(f"Missing time dimension: {time_dim}")
    is_crop = landcover == crop_code
    is_built = landcover == built_code
    # Preserve the legacy classification exactly: every value other than crop=1
    # and built=7, including decoded NoData, is mapped to the regex "0" class.
    is_nonartificial = ~(is_crop | is_built)
    condition_1 = is_nonartificial.sum(time_dim) >= 5
    condition_2 = is_crop.any(time_dim)
    condition_3 = (
        is_crop
        & is_nonartificial.shift({time_dim: -1}, fill_value=False)
        & is_nonartificial.shift({time_dim: -2}, fill_value=False)
    ).any(time_dim)
    condition_4 = (
        is_nonartificial.rolling({time_dim: 5}, min_periods=5)
        .construct("legacy_window")
        .all("legacy_window")
        .any(time_dim)
    )
    result = (condition_1 & condition_2 & condition_3 & condition_4).astype(np.uint8)
    result.name = "final_mask"
    result.attrs.update(
        {
            "detector_contract": "legacy_equivalent_candidate_mask",
            "cropland_code": crop_code,
            "built_code": built_code,
            "minimum_nonartificial_years": 5,
        }
    )
    result.encoding["_FillValue"] = None
    return result


def legacy_equivalent_detect_series(
    values: Sequence[float],
    years: Sequence[int],
) -> tuple[float, float, int, int]:
    sequence = []
    for value in values:
        if np.isfinite(value) and int(round(float(value))) == 1:
            sequence.append("1")
        elif np.isfinite(value) and int(round(float(value))) == 7:
            sequence.append("7")
        else:
            sequence.append("0")
    text = "".join(sequence)
    if not text.startswith("11"):
        return np.nan, np.nan, 0, 0
    matches = list(re.finditer(r"1{2,}(0{5,})", text))
    if not matches:
        return np.nan, np.nan, 0, 0
    match = matches[-1]
    start_index = match.start(1)
    duration = len(match.group(1))
    after = text[match.end() :]
    recultivation = int("11" in after)
    built = int("7" in after)
    current = int(not (recultivation or built))
    return float(years[start_index]), float(duration), recultivation, current


def _legacy_detect_many_python(series: np.ndarray, years: np.ndarray) -> tuple[np.ndarray, ...]:
    count = series.shape[0]
    year = np.full(count, np.nan, dtype=np.float32)
    duration = np.full(count, np.nan, dtype=np.float32)
    recultivation = np.zeros(count, dtype=np.uint8)
    current = np.zeros(count, dtype=np.uint8)
    for index in range(count):
        y0, dur, rec, curr = legacy_equivalent_detect_series(series[index], years)
        year[index] = y0
        duration[index] = dur
        recultivation[index] = rec
        current[index] = curr
    return year, duration, recultivation, current


try:
    import numba as _numba
except ImportError:  # pragma: no cover - fallback for minimal environments
    _numba = None


if _numba is not None:
    @_numba.njit(parallel=True, cache=False)
    def _legacy_detect_many_numba(series: np.ndarray, years: np.ndarray) -> tuple[np.ndarray, ...]:
        count, time_count = series.shape
        out_year = np.full(count, np.nan, dtype=np.float32)
        out_duration = np.full(count, np.nan, dtype=np.float32)
        out_recultivation = np.zeros(count, dtype=np.uint8)
        out_current = np.zeros(count, dtype=np.uint8)
        for pixel in _numba.prange(count):
            codes = np.zeros(time_count, dtype=np.uint8)
            for time_index in range(time_count):
                value = series[pixel, time_index]
                if np.isfinite(value) and int(round(value)) == 1:
                    codes[time_index] = 1
                elif np.isfinite(value) and int(round(value)) == 7:
                    codes[time_index] = 7
            if time_count < 2 or codes[0] != 1 or codes[1] != 1:
                continue
            latest_start = -1
            latest_duration = 0
            cursor = 0
            while cursor < time_count:
                if codes[cursor] != 1:
                    cursor += 1
                    continue
                crop_start = cursor
                while cursor < time_count and codes[cursor] == 1:
                    cursor += 1
                if cursor - crop_start < 2:
                    continue
                zero_start = cursor
                while cursor < time_count and codes[cursor] == 0:
                    cursor += 1
                zero_duration = cursor - zero_start
                if zero_duration >= 5:
                    latest_start = zero_start
                    latest_duration = zero_duration
            if latest_start < 0:
                continue
            event_end = latest_start + latest_duration
            recultivation = 0
            built = 0
            for index in range(event_end, time_count):
                if codes[index] == 7:
                    built = 1
                if index + 1 < time_count and codes[index] == 1 and codes[index + 1] == 1:
                    recultivation = 1
            out_year[pixel] = years[latest_start]
            out_duration[pixel] = latest_duration
            out_recultivation[pixel] = recultivation
            out_current[pixel] = 1 if recultivation == 0 and built == 0 else 0
        return out_year, out_duration, out_recultivation, out_current
else:
    _legacy_detect_many_numba = None


def _cutoff_safe_detect_many_python(
    series: np.ndarray,
    core_count: int,
    start_year: int,
    current_end_year: int,
    window_year: int,
    init_cropland: int,
    extend_validation: bool,
    extension_available: bool,
) -> tuple[np.ndarray, ...]:
    count = series.shape[0]
    abandonment_year = np.full(count, np.nan, dtype=np.float32)
    abandonment_duration = np.full(count, np.nan, dtype=np.float32)
    recultivation = np.zeros(count, dtype=np.uint8)
    current = np.zeros(count, dtype=np.uint8)
    abandonment_end_year = np.full(count, np.nan, dtype=np.float32)
    qualifies = np.zeros(count, dtype=np.uint8)
    eligible = np.zeros(count, dtype=np.uint8)
    persistence = np.zeros(count, dtype=np.uint8)

    for pixel in range(count):
        values = series[pixel]
        initial_ok = core_count >= init_cropland
        for index in range(min(init_cropland, core_count)):
            value = float(values[index])
            if not np.isfinite(value) or abs(value - round(value)) > 1e-6 or int(round(value)) != 1:
                initial_ok = False
                break
        if not initial_ok:
            continue
        eligible[pixel] = 1

        latest_start = -1
        for candidate in range(core_count - window_year, init_cropland - 1, -1):
            prefix_ok = True
            for index in range(candidate - init_cropland, candidate):
                value = float(values[index])
                if not np.isfinite(value) or abs(value - round(value)) > 1e-6 or int(round(value)) != 1:
                    prefix_ok = False
                    break
            if not prefix_ok:
                continue
            event_ok = True
            for index in range(candidate, core_count):
                value = float(values[index])
                if not np.isfinite(value) or abs(value - round(value)) > 1e-6:
                    event_ok = False
                    break
                code = int(round(value))
                if code < 1 or code > 9 or code == 1 or code == 7:
                    event_ok = False
                    break
            if event_ok:
                latest_start = candidate
                break
        if latest_start < 0:
            continue

        abandonment_year[pixel] = np.float32(start_year + latest_start)
        abandonment_duration[pixel] = np.float32(core_count - latest_start)
        current[pixel] = 1
        abandonment_end_year[pixel] = np.float32(current_end_year)
        qualifies[pixel] = 1
        if not extend_validation:
            persistence[pixel] = 1
        elif extension_available and values.size >= core_count + 2:
            extension_ok = True
            for index in range(core_count, core_count + 2):
                value = float(values[index])
                if not np.isfinite(value) or abs(value - round(value)) > 1e-6:
                    extension_ok = False
                    break
                code = int(round(value))
                if code < 1 or code > 9 or code == 1 or code == 7:
                    extension_ok = False
                    break
            persistence[pixel] = np.uint8(extension_ok)

    return (
        abandonment_year,
        abandonment_duration,
        recultivation,
        current,
        abandonment_end_year,
        qualifies,
        eligible,
        persistence,
    )


if _numba is not None:
    @_numba.njit(parallel=True, cache=False)
    def _cutoff_safe_detect_many_numba(
        series: np.ndarray,
        core_count: int,
        start_year: int,
        current_end_year: int,
        window_year: int,
        init_cropland: int,
        extend_validation: bool,
        extension_available: bool,
    ) -> tuple[np.ndarray, ...]:
        count = series.shape[0]
        abandonment_year = np.full(count, np.nan, dtype=np.float32)
        abandonment_duration = np.full(count, np.nan, dtype=np.float32)
        recultivation = np.zeros(count, dtype=np.uint8)
        current = np.zeros(count, dtype=np.uint8)
        abandonment_end_year = np.full(count, np.nan, dtype=np.float32)
        qualifies = np.zeros(count, dtype=np.uint8)
        eligible = np.zeros(count, dtype=np.uint8)
        persistence = np.zeros(count, dtype=np.uint8)
        for pixel in _numba.prange(count):
            initial_ok = core_count >= init_cropland
            for index in range(min(init_cropland, core_count)):
                value = series[pixel, index]
                rounded = int(round(value)) if np.isfinite(value) else -1
                if not np.isfinite(value) or abs(value - rounded) > 1e-6 or rounded != 1:
                    initial_ok = False
                    break
            if not initial_ok:
                continue
            eligible[pixel] = 1

            latest_start = -1
            for candidate in range(core_count - window_year, init_cropland - 1, -1):
                prefix_ok = True
                for index in range(candidate - init_cropland, candidate):
                    value = series[pixel, index]
                    rounded = int(round(value)) if np.isfinite(value) else -1
                    if not np.isfinite(value) or abs(value - rounded) > 1e-6 or rounded != 1:
                        prefix_ok = False
                        break
                if not prefix_ok:
                    continue
                event_ok = True
                for index in range(candidate, core_count):
                    value = series[pixel, index]
                    rounded = int(round(value)) if np.isfinite(value) else -1
                    if (
                        not np.isfinite(value)
                        or abs(value - rounded) > 1e-6
                        or rounded < 1
                        or rounded > 9
                        or rounded == 1
                        or rounded == 7
                    ):
                        event_ok = False
                        break
                if event_ok:
                    latest_start = candidate
                    break
            if latest_start < 0:
                continue

            abandonment_year[pixel] = start_year + latest_start
            abandonment_duration[pixel] = core_count - latest_start
            current[pixel] = 1
            abandonment_end_year[pixel] = current_end_year
            qualifies[pixel] = 1
            if not extend_validation:
                persistence[pixel] = 1
            elif extension_available and series.shape[1] >= core_count + 2:
                extension_ok = True
                for index in range(core_count, core_count + 2):
                    value = series[pixel, index]
                    rounded = int(round(value)) if np.isfinite(value) else -1
                    if (
                        not np.isfinite(value)
                        or abs(value - rounded) > 1e-6
                        or rounded < 1
                        or rounded > 9
                        or rounded == 1
                        or rounded == 7
                    ):
                        extension_ok = False
                        break
                persistence[pixel] = 1 if extension_ok else 0
        return (
            abandonment_year,
            abandonment_duration,
            recultivation,
            current,
            abandonment_end_year,
            qualifies,
            eligible,
            persistence,
        )
else:
    _cutoff_safe_detect_many_numba = None


def detect_legacy_equivalent_chunk(
    landcover: xr.DataArray,
    candidate_mask: xr.DataArray,
    *,
    time_dim: str = "time",
    spatial_dims: tuple[str, str] | None = None,
) -> xr.Dataset:
    years = _as_year_index(landcover[time_dim].values).astype(np.int16)
    spatial_dims = spatial_dims or tuple(dim for dim in landcover.dims if dim != time_dim)
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected two spatial dimensions, got {spatial_dims}")
    array = np.asarray(landcover.transpose(time_dim, *spatial_dims).values)
    mask = np.asarray(candidate_mask.transpose(*spatial_dims).values).astype(bool)
    positions = np.flatnonzero(mask.reshape(-1))
    shape = mask.shape
    abandonment_year = np.full(shape, np.nan, dtype=np.float32)
    abandonment_duration = np.full(shape, np.nan, dtype=np.float32)
    recultivation = np.zeros(shape, dtype=np.uint8)
    current = np.zeros(shape, dtype=np.uint8)
    if positions.size:
        series = np.ascontiguousarray(array.reshape(array.shape[0], -1)[:, positions].T.astype(np.float32, copy=False))
        kernel = _legacy_detect_many_numba or _legacy_detect_many_python
        year_values, duration_values, rec_values, current_values = kernel(series, years)
        abandonment_year.reshape(-1)[positions] = year_values
        abandonment_duration.reshape(-1)[positions] = duration_values
        recultivation.reshape(-1)[positions] = rec_values
        current.reshape(-1)[positions] = current_values
    coords = {dim: landcover.coords[dim] for dim in spatial_dims}
    dataset = xr.Dataset(
        {
            "abandonment_year": (spatial_dims, abandonment_year),
            "abandonment_duration": (spatial_dims, abandonment_duration),
            "recultivation": (spatial_dims, recultivation),
            "current_abandonment": (spatial_dims, current),
        },
        coords=coords,
        attrs={
            "detector_contract": "legacy_equivalent_mode_resample_only",
            "years": [int(years[0]), int(years[-1])],
            "recultivation_rule": "two_consecutive_crop_years_after_latest_event",
            "minimum_abandonment_years": 5,
            "spatial_dims": list(spatial_dims),
        },
    )
    if landcover.attrs.get("crs") is not None:
        dataset.attrs["crs"] = landcover.attrs.get("crs")
    for name in ("recultivation", "current_abandonment"):
        dataset[name].attrs.update({"flag_values": [0, 1], "flag_meanings": "false true"})
        dataset[name].encoding["_FillValue"] = None
    return dataset


def candidate_chunk_keys(
    mask: xr.DataArray,
    *,
    chunk_size: int = 500,
    spatial_dims: tuple[str, str] | None = None,
) -> list[tuple[int, int]]:
    spatial_dims = spatial_dims or _infer_spatial_dims(mask)
    y_dim, x_dim = spatial_dims
    keys = []
    for row_index, row_start in enumerate(range(0, mask.sizes[y_dim], chunk_size)):
        for col_index, col_start in enumerate(range(0, mask.sizes[x_dim], chunk_size)):
            subset = mask.isel(
                {
                    y_dim: slice(row_start, min(row_start + chunk_size, mask.sizes[y_dim])),
                    x_dim: slice(col_start, min(col_start + chunk_size, mask.sizes[x_dim])),
                }
            )
            if bool(subset.any().compute().item() if hasattr(subset.data, "compute") else subset.any().item()):
                keys.append((row_index, col_index))
    return keys


def _chunk_path(directory: Path, key: tuple[int, int]) -> Path:
    return directory / f"chunk_{key[0]}_{key[1]}.nc"


def write_legacy_equivalent_chunks(
    landcover_path: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_size: int = 500,
    spatial_dims: tuple[str, str] = ("lat", "lon"),
    time_dim: str = "time",
    landcover_var: str = "lccs_class",
    mask_var: str = "final_mask",
    expected_keys: Sequence[tuple[int, int]] | None = None,
    resume: bool = True,
    progress_callback=None,
) -> dict[str, object]:
    landcover_source = Path(landcover_path)
    mask_source = Path(mask_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    landcover_sha256 = _sha256_file(landcover_source).upper()
    mask_sha256 = _sha256_file(mask_source).upper()
    existing_manifest_path = target / "run_manifest.json"
    resume_allowed = resume and existing_manifest_path.exists()
    if resume_allowed:
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("landcover_sha256") != landcover_sha256 or existing_manifest.get("mask_sha256") != mask_sha256:
            raise ValueError("Existing chunk manifest source fingerprints do not match current inputs")
    y_dim, x_dim = spatial_dims
    with xr.open_dataset(landcover_source, chunks={time_dim: -1, y_dim: chunk_size, x_dim: chunk_size}) as land_ds, xr.open_dataset(
        mask_source, chunks={y_dim: chunk_size, x_dim: chunk_size}
    ) as mask_ds:
        landcover = land_ds[landcover_var]
        mask = mask_ds[mask_var]
        keys = list(expected_keys or candidate_chunk_keys(mask, chunk_size=chunk_size, spatial_dims=spatial_dims))
        generated = skipped = failed = 0
        paths = []
        for key in keys:
            row_start = key[0] * chunk_size
            col_start = key[1] * chunk_size
            selection = {
                y_dim: slice(row_start, min(row_start + chunk_size, landcover.sizes[y_dim])),
                x_dim: slice(col_start, min(col_start + chunk_size, landcover.sizes[x_dim])),
            }
            destination = _chunk_path(target, key)
            if resume_allowed and destination.exists():
                try:
                    with xr.open_dataset(destination) as existing:
                        if set(existing.data_vars) == set(LEGACY_OUTPUT_VARS) and existing.sizes[y_dim] == selection[y_dim].stop - selection[y_dim].start and existing.sizes[x_dim] == selection[x_dim].stop - selection[x_dim].start:
                            paths.append(destination)
                            skipped += 1
                            if progress_callback:
                                progress_callback({"key": key, "generated": generated, "skipped": skipped, "failed": failed})
                            continue
                except Exception:
                    pass
            try:
                chunk_landcover = landcover.isel(selection).load()
                chunk_mask = mask.isel(selection).load()
                result = detect_legacy_equivalent_chunk(chunk_landcover, chunk_mask, time_dim=time_dim, spatial_dims=spatial_dims)
                encoding = {
                    "abandonment_year": {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32", "_FillValue": np.nan},
                    "abandonment_duration": {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32", "_FillValue": np.nan},
                    "recultivation": {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "uint8", "_FillValue": None},
                    "current_abandonment": {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "uint8", "_FillValue": None},
                }
                with tempfile.NamedTemporaryFile(delete=False, dir=target, suffix=".tmp.nc") as handle:
                    temporary = Path(handle.name)
                try:
                    result.to_netcdf(temporary, encoding=encoding)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                paths.append(destination)
                generated += 1
            except Exception:
                failed += 1
                if progress_callback:
                    progress_callback({"key": key, "generated": generated, "skipped": skipped, "failed": failed})
                raise
            if progress_callback:
                progress_callback({"key": key, "generated": generated, "skipped": skipped, "failed": failed})
    manifest = {
        "landcover_path": str(landcover_source),
        "landcover_size_bytes": landcover_source.stat().st_size,
        "landcover_mtime_ns": landcover_source.stat().st_mtime_ns,
        "landcover_sha256": landcover_sha256,
        "mask_path": str(mask_source),
        "mask_size_bytes": mask_source.stat().st_size,
        "mask_sha256": mask_sha256,
        "output_dir": str(target),
        "chunk_size": chunk_size,
        "spatial_dims": list(spatial_dims),
        "time_dim": time_dim,
        "landcover_var": landcover_var,
        "mask_var": mask_var,
        "crs": landcover.attrs.get("crs"),
        "expected_keys": [list(key) for key in keys],
        "chunk_paths": [str(path) for path in paths],
        "stats": {"total": len(keys), "generated": generated, "skipped": skipped, "failed": failed},
    }
    if (
        resume_allowed
        and generated == 0
        and failed == 0
        and skipped == len(keys)
        and existing_manifest.get("stats", {}).get("total") == len(keys)
    ):
        return existing_manifest
    _atomic_write_text(target / "run_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def validate_legacy_chunk_set(
    landcover_path: str | Path,
    mask_path: str | Path,
    chunk_dir: str | Path,
    *,
    chunk_size: int = 500,
    spatial_dims: tuple[str, str] = ("lat", "lon"),
    time_dim: str = "time",
    landcover_var: str = "lccs_class",
    mask_var: str = "final_mask",
    sample_pixels: int = 2_000,
    random_seed: int = 20_260_818,
    expected_keys: Sequence[tuple[int, int]] | None = None,
) -> dict[str, object]:
    source = Path(landcover_path)
    mask_source = Path(mask_path)
    directory = Path(chunk_dir)
    y_dim, x_dim = spatial_dims
    with xr.open_dataset(source, chunks={time_dim: -1, y_dim: chunk_size, x_dim: chunk_size}) as land_ds, xr.open_dataset(
        mask_source, chunks={y_dim: chunk_size, x_dim: chunk_size}
    ) as mask_ds:
        landcover = land_ds[landcover_var]
        mask = mask_ds[mask_var]
        expected = list(expected_keys) if expected_keys is not None else candidate_chunk_keys(
            mask,
            chunk_size=chunk_size,
            spatial_dims=spatial_dims,
        )
        pattern = re.compile(r"chunk_(\d+)_(\d+)\.nc$")
        actual_map = {}
        for path in directory.glob("chunk_*.nc"):
            match = pattern.fullmatch(path.name)
            if match:
                actual_map[(int(match.group(1)), int(match.group(2)))] = path
        expected_set = set(expected)
        actual_set = set(actual_map)
        errors = []
        binary_checks = {"recultivation": True, "current_abandonment": True}
        range_sampled_points = 0
        for key in sorted(actual_set):
            path = actual_map[key]
            row_start, col_start = key[0] * chunk_size, key[1] * chunk_size
            expected_y = landcover[y_dim].isel({y_dim: slice(row_start, min(row_start + chunk_size, landcover.sizes[y_dim]))}).values
            expected_x = landcover[x_dim].isel({x_dim: slice(col_start, min(col_start + chunk_size, landcover.sizes[x_dim]))}).values
            try:
                with xr.open_dataset(path) as dataset:
                    if set(dataset.data_vars) != set(LEGACY_OUTPUT_VARS):
                        errors.append(f"{path.name}: variables")
                    if not np.array_equal(dataset[y_dim].values, expected_y) or not np.array_equal(dataset[x_dim].values, expected_x):
                        errors.append(f"{path.name}: coordinates")
                    sample_positions = (
                        (0, 0),
                        (dataset.sizes[y_dim] // 2, dataset.sizes[x_dim] // 2),
                        (dataset.sizes[y_dim] - 1, dataset.sizes[x_dim] - 1),
                    )
                    valid_years = np.asarray(
                        [dataset.abandonment_year.isel({y_dim: row, x_dim: col}).item() for row, col in sample_positions],
                        dtype=np.float64,
                    )
                    valid_years = valid_years[np.isfinite(valid_years)]
                    valid_durations = np.asarray(
                        [dataset.abandonment_duration.isel({y_dim: row, x_dim: col}).item() for row, col in sample_positions],
                        dtype=np.float64,
                    )
                    valid_durations = valid_durations[np.isfinite(valid_durations)]
                    range_sampled_points += len(sample_positions)
                    if valid_years.size and (valid_years.min() < 1994 or valid_years.max() > 2018):
                        errors.append(f"{path.name}: abandonment_year range")
                    if valid_durations.size and valid_durations.min() < 5:
                        errors.append(f"{path.name}: duration range")
                    for name in binary_checks:
                        values = np.asarray(
                            [dataset[name].isel({y_dim: row, x_dim: col}).item() for row, col in sample_positions]
                        )
                        if not np.isin(values[np.isfinite(values)], [0, 1]).all() or dataset[name].encoding.get("_FillValue") is not None:
                            binary_checks[name] = False
            except Exception as error:
                errors.append(f"{path.name}: {error!r}")

        rng = np.random.default_rng(random_seed)
        sampled = checked = mismatched = 0
        candidate_keys = sorted(expected_set & actual_set)
        if candidate_keys:
            selected_keys = rng.choice(len(candidate_keys), size=min(len(candidate_keys), max(1, math.ceil(sample_pixels / 20))), replace=False)
            for selected_index in np.atleast_1d(selected_keys):
                key = candidate_keys[int(selected_index)]
                row_start, col_start = key[0] * chunk_size, key[1] * chunk_size
                selection = {
                    y_dim: slice(row_start, min(row_start + chunk_size, landcover.sizes[y_dim])),
                    x_dim: slice(col_start, min(col_start + chunk_size, landcover.sizes[x_dim])),
                }
                chunk_mask = mask.isel(selection).load().values.astype(bool)
                positions = np.argwhere(chunk_mask)
                if positions.size == 0:
                    continue
                positions = positions[rng.choice(len(positions), size=min(20, len(positions)), replace=False)]
                chunk_land = landcover.isel(selection).load()
                with xr.open_dataset(actual_map[key]) as output:
                    years = _as_year_index(chunk_land[time_dim].values)
                    for local_lat, local_lon in positions:
                        expected_values = legacy_equivalent_detect_series(chunk_land[:, local_lat, local_lon].values, years)
                        observed_values = (
                            float(output.abandonment_year.values[local_lat, local_lon]),
                            float(output.abandonment_duration.values[local_lat, local_lon]),
                            int(output.recultivation.values[local_lat, local_lon]),
                            int(output.current_abandonment.values[local_lat, local_lon]),
                        )
                        for expected_value, observed_value in zip(expected_values, observed_values):
                            if np.isnan(expected_value) and np.isnan(observed_value):
                                continue
                            if expected_value != observed_value:
                                mismatched += 1
                                break
                        checked += 1
                        sampled += 1
        temporary_files = [str(path) for path in directory.glob("*.tmp*")]
    report = {
        "expected_count": len(expected_set),
        "actual_count": len(actual_set),
        "missing_keys": [list(key) for key in sorted(expected_set - actual_set)],
        "unexpected_keys": [list(key) for key in sorted(actual_set - expected_set)],
        "errors": errors,
        "binary_checks": binary_checks,
        "sampled_pixels": sampled,
        "range_sampled_points": range_sampled_points,
        "sample_mismatches": mismatched,
        "temporary_files": temporary_files,
    }
    report["passed"] = bool(
        not report["missing_keys"]
        and not report["unexpected_keys"]
        and not errors
        and all(binary_checks.values())
        and mismatched == 0
        and not temporary_files
    )
    _atomic_write_text(directory / "validation_report.json", json.dumps(report, indent=2, ensure_ascii=False))
    return report


def merge_landcover_into_legacy_chunks(
    landcover_path: str | Path,
    chunk_dir: str | Path,
    merged_dir: str | Path,
    *,
    spatial_dims: tuple[str, str] = ("lat", "lon"),
    time_dim: str = "time",
    landcover_var: str = "lccs_class",
    chunk_size: int = 500,
    compression_level: int = 4,
    resume: bool = False,
    progress_callback=None,
) -> dict[str, object]:
    source = Path(landcover_path)
    input_dir = Path(chunk_dir)
    output_dir = Path(merged_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = sorted(input_dir.glob("chunk_*.nc"))
    written = []
    generated = skipped = 0
    expected_time_count = None
    y_dim, x_dim = spatial_dims
    with xr.open_dataset(source, chunks={time_dim: 1, y_dim: chunk_size, x_dim: chunk_size}) as land_ds:
        landcover = land_ds[landcover_var].rename("landcover")
        expected_time_count = landcover.sizes[time_dim]
        for index, chunk_path in enumerate(chunk_paths, start=1):
            destination = output_dir / chunk_path.name
            if resume and destination.exists():
                try:
                    with xr.open_dataset(chunk_path) as chunk, xr.open_dataset(destination) as existing:
                        if (
                            set(existing.data_vars) == set(LEGACY_OUTPUT_VARS) | {"landcover"}
                            and existing.sizes.get(time_dim) == expected_time_count
                            and np.array_equal(existing[y_dim].values, chunk[y_dim].values)
                            and np.array_equal(existing[x_dim].values, chunk[x_dim].values)
                        ):
                            written.append(destination)
                            skipped += 1
                            if progress_callback:
                                progress_callback(
                                    {
                                        "done": index,
                                        "total": len(chunk_paths),
                                        "generated": generated,
                                        "skipped": skipped,
                                        "path": str(destination),
                                    }
                                )
                            continue
                except Exception:
                    pass
            with xr.open_dataset(chunk_path) as chunk:
                land_sub = landcover.sel({y_dim: chunk[y_dim].values, x_dim: chunk[x_dim].values})
                if not np.array_equal(land_sub[y_dim].values, chunk[y_dim].values) or not np.array_equal(land_sub[x_dim].values, chunk[x_dim].values):
                    raise ValueError(f"Exact coordinate selection failed for {chunk_path.name}")
                left, right = xr.align(chunk.load(), land_sub.load(), join="exact")
                merged = xr.merge([left, right])
            _atomic_write_netcdf(
                merged,
                destination,
                compression_level=compression_level,
            )
            written.append(destination)
            generated += 1
            if progress_callback:
                progress_callback(
                    {
                        "done": index,
                        "total": len(chunk_paths),
                        "generated": generated,
                        "skipped": skipped,
                        "path": str(destination),
                    }
                )
    errors = []
    required_variables = set(LEGACY_OUTPUT_VARS) | {"landcover"}
    for path in written:
        try:
            with xr.open_dataset(path) as dataset:
                if set(dataset.data_vars) != required_variables:
                    errors.append(f"{path.name}: variables")
                if dataset.sizes.get(time_dim) != expected_time_count:
                    errors.append(f"{path.name}: time dimension")
                samples = dataset.landcover.isel(
                    {
                        time_dim: [0, dataset.sizes[time_dim] // 2, dataset.sizes[time_dim] - 1],
                        y_dim: [0, dataset.sizes[y_dim] // 2, dataset.sizes[y_dim] - 1],
                        x_dim: [0, dataset.sizes[x_dim] // 2, dataset.sizes[x_dim] - 1],
                    }
                ).values
                finite = samples[np.isfinite(samples)]
                if finite.size and not np.isin(np.rint(finite).astype(int), range(10)).all():
                    errors.append(f"{path.name}: landcover range")
        except Exception as error:
            errors.append(f"{path.name}: {error!r}")
    temporary_files = [str(path) for path in output_dir.glob("*.tmp*")]
    report = {
        "source_landcover": str(source),
        "input_chunk_dir": str(input_dir),
        "merged_dir": str(output_dir),
        "input_count": len(chunk_paths),
        "output_count": len(written),
        "generated": generated,
        "skipped": skipped,
        "chunk_size": chunk_size,
        "compression_level": compression_level,
        "errors": errors,
        "temporary_files": temporary_files,
        "passed": len(chunk_paths) == len(written) and not errors and not temporary_files,
    }
    _atomic_write_text(output_dir / "run_manifest.json", json.dumps(report, indent=2, ensure_ascii=False))
    return report


def validate_roundtrip(
    expected: xr.Dataset,
    observed: xr.Dataset,
    *,
    binary_vars: Sequence[str] = DEFAULT_BINARY_VARS,
) -> dict[str, object]:
    if any(hasattr(expected[name].data, "compute") for name in expected.data_vars):
        expected = expected.compute()
    if any(hasattr(observed[name].data, "compute") for name in observed.data_vars):
        observed = observed.compute()

    report: dict[str, object] = {
        "same_variables": set(expected.data_vars) == set(observed.data_vars),
        "same_coords": set(expected.coords) == set(observed.coords),
        "dataset_attrs_equal": _mapping_equal(dict(expected.attrs), dict(observed.attrs)),
        "coord_equal": {},
        "dtype_equal": {},
        "value_equal": {},
        "var_attrs_equal": {},
        "binary_0_1": {},
        "encoding_fillvalue_equal": {},
        "crs_equal": expected.attrs.get("crs") == observed.attrs.get("crs"),
    }

    for coord_name in expected.coords:
        report["coord_equal"][coord_name] = coord_name in observed.coords and np.array_equal(
            expected[coord_name].values, observed[coord_name].values
        )

    for name in expected.data_vars:
        if name not in observed:
            report["dtype_equal"][name] = False
            report["value_equal"][name] = False
            report["var_attrs_equal"][name] = False
            report["encoding_fillvalue_equal"][name] = False
            continue
        report["dtype_equal"][name] = expected[name].dtype == observed[name].dtype
        report["value_equal"][name] = _array_equal_or_allclose(expected[name].values, observed[name].values)
        report["var_attrs_equal"][name] = _mapping_equal(dict(expected[name].attrs), dict(observed[name].attrs))
        expected_fill = expected[name].encoding.get("_FillValue")
        observed_fill = observed[name].encoding.get("_FillValue")
        if name in binary_vars and expected_fill is None:
            report["encoding_fillvalue_equal"][name] = observed_fill is None
        else:
            report["encoding_fillvalue_equal"][name] = True if expected_fill is None else expected_fill == observed_fill

    for name in binary_vars:
        if name not in observed:
            report["binary_0_1"][name] = False
            continue
        values = np.asarray(observed[name].values)
        finite = values[np.isfinite(values)]
        report["binary_0_1"][name] = np.isin(finite, [0, 1]).all()

    report["passed"] = bool(
        report["same_variables"]
        and report["same_coords"]
        and report["dataset_attrs_equal"]
        and report["crs_equal"]
        and all(report["coord_equal"].values())
        and all(report["dtype_equal"].values())
        and all(report["value_equal"].values())
        and all(report["var_attrs_equal"].values())
        and all(report["binary_0_1"].values())
        and all(report["encoding_fillvalue_equal"].values())
    )
    return report


def summarize_key_numbers(
    dataset: xr.Dataset,
    area_m2: xr.DataArray,
    *,
    state_codes: xr.DataArray | None = None,
) -> dict[str, pd.DataFrame]:
    metric_names = ("qualifies_at_cutoff", "current_abandonment", "recultivation")
    materialized = dataset[["abandonment_year", *metric_names]]
    if any(hasattr(materialized[name].data, "compute") for name in materialized.data_vars):
        materialized = materialized.compute()

    abandonment_year = np.asarray(materialized["abandonment_year"].values)
    area = np.asarray(area_m2.broadcast_like(materialized["abandonment_year"]).values, dtype=np.float64)
    detected = np.isfinite(abandonment_year)
    qualifies = np.asarray(materialized["qualifies_at_cutoff"].fillna(0).values, dtype=np.uint8) == 1
    current = np.asarray(materialized["current_abandonment"].fillna(0).values, dtype=np.uint8) == 1
    recultivated = np.asarray(materialized["recultivation"].fillna(0).values, dtype=np.uint8) == 1

    def overall_record(metric: str, mask: np.ndarray) -> dict[str, object]:
        return {
            "metric": metric,
            "pixels": int(np.count_nonzero(mask)),
            "area_ha": float(area[mask].sum(dtype=np.float64) / 10_000.0),
        }

    overall = pd.DataFrame(
        [
            overall_record("detected", detected),
            overall_record("cutoff_qualified", qualifies),
            overall_record("current_abandonment", current),
            overall_record("recultivated", recultivated),
        ]
    )

    valid_years = np.unique(abandonment_year[detected].astype(np.int64, copy=False))
    by_year_records: list[dict[str, object]] = []
    if valid_years.size:
        year_groups = abandonment_year[detected].astype(np.int64, copy=False)
        year_area = area[detected]
        year_qualifies = qualifies[detected]
        year_current = current[detected]
        year_recultivated = recultivated[detected]
        detected_pixels = np.bincount(year_groups)
        detected_area = np.bincount(year_groups, weights=year_area)
        qualifies_pixels = np.bincount(year_groups, weights=year_qualifies.astype(np.uint8))
        qualifies_area = np.bincount(year_groups, weights=year_area * year_qualifies)
        current_pixels = np.bincount(year_groups, weights=year_current.astype(np.uint8))
        current_area = np.bincount(year_groups, weights=year_area * year_current)
        recultivated_pixels = np.bincount(year_groups, weights=year_recultivated.astype(np.uint8))
        recultivated_area = np.bincount(year_groups, weights=year_area * year_recultivated)
        for year_value in valid_years:
            by_year_records.append(
                {
                    "abandonment_year": int(year_value),
                    "detected_pixels": int(detected_pixels[year_value]),
                    "detected_area_ha": float(detected_area[year_value] / 10_000.0),
                    "cutoff_qualified_pixels": int(qualifies_pixels[year_value]),
                    "cutoff_qualified_area_ha": float(qualifies_area[year_value] / 10_000.0),
                    "current_pixels": int(current_pixels[year_value]),
                    "current_area_ha": float(current_area[year_value] / 10_000.0),
                    "recultivated_pixels": int(recultivated_pixels[year_value]),
                    "recultivated_area_ha": float(recultivated_area[year_value] / 10_000.0),
                }
            )
    by_year = pd.DataFrame(
        by_year_records,
        columns=[
            "abandonment_year",
            "detected_pixels",
            "detected_area_ha",
            "cutoff_qualified_pixels",
            "cutoff_qualified_area_ha",
            "current_pixels",
            "current_area_ha",
            "recultivated_pixels",
            "recultivated_area_ha",
        ],
    )

    if state_codes is None:
        by_state = pd.DataFrame(
            columns=[
                "state_code",
                "detected_pixels",
                "detected_area_ha",
                "cutoff_qualified_pixels",
                "cutoff_qualified_area_ha",
                "current_pixels",
                "current_area_ha",
                "recultivated_pixels",
                "recultivated_area_ha",
            ]
        )
    else:
        state_aligned = np.asarray(
            state_codes.broadcast_like(materialized["current_abandonment"]).values
        )
        by_state_records: list[dict[str, object]] = []
        valid_state_mask = np.isfinite(state_aligned) & (state_aligned != 0)
        valid_states = np.unique(state_aligned[valid_state_mask].astype(np.int64, copy=False))
        if valid_states.size:
            state_groups = state_aligned[valid_state_mask].astype(np.int64, copy=False)
            state_area = area[valid_state_mask]
            state_detected = detected[valid_state_mask]
            state_qualifies = qualifies[valid_state_mask]
            state_current = current[valid_state_mask]
            state_recultivated = recultivated[valid_state_mask]

            def grouped(mask: np.ndarray, *, weighted_area: bool = False) -> np.ndarray:
                weights = state_area * mask if weighted_area else mask.astype(np.uint8)
                return np.bincount(state_groups, weights=weights)

            state_metrics = {
                "detected_pixels": grouped(state_detected),
                "detected_area_ha": grouped(state_detected, weighted_area=True) / 10_000.0,
                "cutoff_qualified_pixels": grouped(state_qualifies),
                "cutoff_qualified_area_ha": grouped(state_qualifies, weighted_area=True) / 10_000.0,
                "current_pixels": grouped(state_current),
                "current_area_ha": grouped(state_current, weighted_area=True) / 10_000.0,
                "recultivated_pixels": grouped(state_recultivated),
                "recultivated_area_ha": grouped(state_recultivated, weighted_area=True) / 10_000.0,
            }
            for state_value in valid_states:
                by_state_records.append(
                    {
                        "state_code": int(state_value),
                        **{
                            metric: int(values[state_value]) if metric.endswith("pixels") else float(values[state_value])
                            for metric, values in state_metrics.items()
                        },
                    }
                )
        by_state = pd.DataFrame(
            by_state_records,
            columns=[
                "state_code",
                "detected_pixels",
                "detected_area_ha",
                "cutoff_qualified_pixels",
                "cutoff_qualified_area_ha",
                "current_pixels",
                "current_area_ha",
                "recultivated_pixels",
                "recultivated_area_ha",
            ],
        )

    return {"overall": overall, "by_state": by_state, "by_year": by_year}


def publish_validation_results(
    *,
    run_id: str,
    repo_result_root: str | Path,
    manifest: Mapping[str, object],
    summaries: Mapping[str, pd.DataFrame],
    report_text: str,
) -> dict[str, Path]:
    root = Path(repo_result_root)
    manifest_dir = root / "manifests"
    summary_dir = root / "summaries"
    report_dir = root / "reports"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f"{run_id}.json"
    report_path = report_dir / "validation_report.md"
    _atomic_write_text(manifest_path, json.dumps(dict(manifest), indent=2, ensure_ascii=False))
    _atomic_write_text(report_path, report_text)

    written: dict[str, Path] = {"manifest": manifest_path, "report": report_path}
    for name, frame in summaries.items():
        path = summary_dir / f"{name}.csv"
        _atomic_write_csv(frame, path)
        written[name] = path
    return written


def fetch_reference_raster(
    source: str | Path | None = None,
    target_path: str | Path | None = None,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    archive_source: str | Path | None = None,
    archive_path: str | Path | None = None,
    archive_member_name: str | None = None,
    extracted_raster_path: str | Path | None = None,
    expected_archive_sha256: str | None = None,
    expected_archive_size: int | None = None,
    expected_raster_sha256: str | None = None,
    expected_raster_size: int | None = None,
    chunk_size: int = 1 << 20,
) -> Path:
    if archive_source is not None or archive_path is not None or extracted_raster_path is not None:
        if archive_path is None or extracted_raster_path is None or archive_member_name is None:
            raise ValueError("archive_path, extracted_raster_path, and archive_member_name are required for archive flow.")
        archive_target = Path(archive_path)
        archive_target.parent.mkdir(parents=True, exist_ok=True)
        if archive_target.exists():
            _validate_file(archive_target, expected_size=expected_archive_size, expected_sha256=expected_archive_sha256)
        else:
            if archive_source is None:
                raise ValueError("archive_source is required when archive_path does not exist.")
            _fetch_file(archive_source, archive_target, chunk_size=chunk_size)
            _validate_file(archive_target, expected_size=expected_archive_size, expected_sha256=expected_archive_sha256)

        raster_target = Path(extracted_raster_path)
        raster_target.parent.mkdir(parents=True, exist_ok=True)
        if raster_target.exists():
            _validate_file(raster_target, expected_size=expected_raster_size, expected_sha256=expected_raster_sha256)
            return raster_target
        _extract_archive_member(archive_target, archive_member_name, raster_target)
        _validate_file(raster_target, expected_size=expected_raster_size, expected_sha256=expected_raster_sha256)
        return raster_target

    if source is None or target_path is None:
        raise ValueError("source and target_path are required for direct fetch flow.")
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _validate_file(target, expected_size=expected_size, expected_sha256=expected_sha256)
        return target
    _fetch_file(source, target, chunk_size=chunk_size)
    _validate_file(target, expected_size=expected_size, expected_sha256=expected_sha256)
    return target


def validate_reference_raster(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    expected_crs: str | None = None,
    expected_count: int | None = None,
    expected_dtype: str | None = None,
    expected_nodata: float | int | None = None,
    expected_year_min: int | None = None,
    expected_year_max: int | None = None,
    max_pixels_for_scan: int = REFERENCE_SCAN_MAX_PIXELS,
) -> dict[str, object]:
    import rasterio

    raster_path = Path(path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Reference raster not found: {raster_path}")
    _validate_file(raster_path, expected_size=expected_size, expected_sha256=expected_sha256)

    with rasterio.open(raster_path) as src:
        metadata = {
            "path": str(raster_path),
            "size_bytes": raster_path.stat().st_size,
            "sha256": _sha256_file(raster_path) if expected_sha256 is not None else None,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": src.crs.to_string() if src.crs else None,
            "transform": tuple(src.transform),
            "nodata": src.nodata,
        }
        if expected_nodata is not None and src.nodata != expected_nodata:
            raise ValueError(f"Reference raster nodata mismatch: expected {expected_nodata}, got {src.nodata}")
        if expected_crs is not None and metadata["crs"] != expected_crs:
            raise ValueError(f"Reference raster CRS mismatch: expected {expected_crs}, got {metadata['crs']}")
        if expected_count is not None and metadata["count"] != expected_count:
            raise ValueError(f"Reference raster band-count mismatch: expected {expected_count}, got {metadata['count']}")
        if expected_dtype is not None and metadata["dtype"] != expected_dtype:
            raise ValueError(f"Reference raster dtype mismatch: expected {expected_dtype}, got {metadata['dtype']}")

        tags = {key.upper(): value for key, value in src.tags(1).items()}
        stats_min = _parse_float(tags.get("STATISTICS_MINIMUM") or tags.get("STATISTICS_VALID_MINIMUM"))
        stats_max = _parse_float(tags.get("STATISTICS_MAXIMUM") or tags.get("STATISTICS_VALID_MAXIMUM"))

        if stats_min is None or stats_max is None:
            pixel_count = int(src.width) * int(src.height)
            if pixel_count <= max_pixels_for_scan:
                band = src.read(1, masked=True)
                valid = band.compressed()
                if valid.size:
                    stats_min = float(valid.min())
                    stats_max = float(valid.max())
            elif expected_year_min is not None or expected_year_max is not None:
                raise ValueError("Raster statistics tags are required for large reference rasters.")

        metadata["year_min"] = int(stats_min) if stats_min is not None else None
        metadata["year_max"] = int(stats_max) if stats_max is not None else None
        if expected_year_min is not None and metadata["year_min"] != expected_year_min:
            raise ValueError("Reference raster minimum year mismatch.")
        if expected_year_max is not None and metadata["year_max"] != expected_year_max:
            raise ValueError("Reference raster maximum year mismatch.")
    return metadata


def open_reference_abandonment_year(
    path: str | Path,
    *,
    nodata: float | int = 0,
    output_name: str = "reference_abandonment_year",
    chunks: str | int | dict[str, int] | None = None,
) -> xr.DataArray:
    import rioxarray

    raster_path = Path(path)
    chunk_spec = {"x": 2048, "y": 2048} if chunks is None else chunks
    opened = rioxarray.open_rasterio(raster_path, masked=True, chunks=chunk_spec)
    result = opened.squeeze(drop=True).astype(np.float32)
    if nodata is not None:
        result = result.where(result != nodata)
    result.name = output_name
    result.attrs.update(
        {
            "source_path": str(raster_path),
            "source_nodata": nodata,
            "source_crs": result.rio.crs.to_string() if result.rio.crs else None,
            "source_transform": tuple(result.rio.transform()),
        }
    )
    return result


def align_reference_abandonment(
    reference_year: xr.DataArray,
    target_like: xr.DataArray | xr.Dataset,
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    method: str = "nearest",
) -> xr.DataArray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    source_path = reference_year.attrs.get("source_path")
    if not source_path:
        raise ValueError("reference_year must include source_path for bounded raster reprojection.")

    if isinstance(target_like, xr.Dataset):
        target_lat = target_like[lat_dim]
        target_lon = target_like[lon_dim]
    else:
        target_lat = target_like[lat_dim]
        target_lon = target_like[lon_dim]

    process_lat = target_lat.sortby(target_lat, ascending=False)
    process_lon = target_lon.sortby(target_lon, ascending=True)
    destination = np.full((process_lat.size, process_lon.size), np.nan, dtype=np.float32)
    dst_transform = _regular_grid_transform(process_lat.values, process_lon.values)
    dst_crs = "EPSG:4326"
    resampling = {"nearest": Resampling.nearest, "mode": Resampling.mode}[method]

    with rasterio.open(source_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    aligned = xr.DataArray(
        destination,
        coords={lat_dim: process_lat.values, lon_dim: process_lon.values},
        dims=(lat_dim, lon_dim),
        name="reference_abandonment_year_aligned",
        attrs=dict(reference_year.attrs),
    )
    aligned.attrs["alignment_method"] = method
    return aligned.reindex({lat_dim: target_lat.values, lon_dim: target_lon.values})


def compare_reference_abandonment(
    dataset: xr.Dataset,
    reference_year: xr.DataArray,
    area_m2: xr.DataArray,
    *,
    tolerance_years: int = 0,
    comparison_years: tuple[int, int] | None = None,
    state_codes: xr.DataArray | None = None,
) -> dict[str, pd.DataFrame]:
    ours_da = dataset["abandonment_year"]
    reference_da = reference_year.broadcast_like(ours_da)
    if hasattr(ours_da.data, "compute"):
        ours_da = ours_da.compute()
    if hasattr(reference_da.data, "compute"):
        reference_da = reference_da.compute()
    ours = np.asarray(ours_da.values)
    reference_aligned = np.asarray(reference_da.values)
    area = np.asarray(area_m2.broadcast_like(ours_da).values, dtype=np.float64)

    interval_start: int | None = None
    interval_end: int | None = None
    if comparison_years is not None:
        interval_start, interval_end = (int(comparison_years[0]), int(comparison_years[1]))
        if interval_start > interval_end:
            raise ValueError("comparison_years must satisfy start <= end.")
    ours_detected = np.isfinite(ours)
    ref_detected = np.isfinite(reference_aligned)
    if interval_start is not None:
        ours_detected &= ours >= interval_start
        ref_detected &= reference_aligned >= interval_start
    if interval_end is not None:
        ours_detected &= ours <= interval_end
        ref_detected &= reference_aligned <= interval_end
    overlap = ours_detected & ref_detected
    exact = overlap & (np.abs(ours - reference_aligned) <= tolerance_years)
    only_ours = ours_detected & ~ref_detected
    only_reference = ref_detected & ~ours_detected

    def comparison_record(metric: str, mask: np.ndarray) -> dict[str, object]:
        return {
            "metric": metric,
            "pixels": int(np.count_nonzero(mask)),
            "area_ha": float(area[mask].sum(dtype=np.float64) / 10_000.0),
        }

    overall = pd.DataFrame(
        [
            comparison_record("overlap_detected", overlap),
            comparison_record("year_match_within_tolerance", exact),
            comparison_record("only_ours", only_ours),
            comparison_record("only_reference", only_reference),
        ]
    )
    overall["comparison_start_year"] = interval_start
    overall["comparison_end_year"] = interval_end

    ours_years = ours[ours_detected].astype(np.int64, copy=False)
    ref_years = reference_aligned[ref_detected].astype(np.int64, copy=False)
    years_union = np.unique(np.concatenate([ours_years, ref_years])) if (ours_years.size or ref_years.size) else np.array([], dtype=int)
    ours_counts = np.bincount(ours_years) if ours_years.size else np.array([], dtype=np.int64)
    ref_counts = np.bincount(ref_years) if ref_years.size else np.array([], dtype=np.int64)
    exact_years = ours[exact].astype(np.int64, copy=False)
    exact_counts = np.bincount(exact_years) if exact_years.size else np.array([], dtype=np.int64)

    def count_for(counts: np.ndarray, year_value: int) -> int:
        return int(counts[year_value]) if year_value < counts.size else 0

    by_year = pd.DataFrame(
        [
            {
                "abandonment_year": int(year_value),
                "ours_pixels": count_for(ours_counts, int(year_value)),
                "reference_pixels": count_for(ref_counts, int(year_value)),
                "match_pixels": count_for(exact_counts, int(year_value)),
                "comparison_start_year": interval_start,
                "comparison_end_year": interval_end,
            }
            for year_value in years_union
        ],
        columns=[
            "abandonment_year",
            "ours_pixels",
            "reference_pixels",
            "match_pixels",
            "comparison_start_year",
            "comparison_end_year",
        ],
    )

    if state_codes is None:
        by_state = pd.DataFrame(
            columns=[
                "state_code",
                "overlap_pixels",
                "only_ours_pixels",
                "only_reference_pixels",
                "overlap_area_ha",
                "only_ours_area_ha",
                "only_reference_area_ha",
                "comparison_start_year",
                "comparison_end_year",
            ]
        )
    else:
        states = np.asarray(state_codes.broadcast_like(ours_da).values)
        records: list[dict[str, object]] = []
        valid_state_mask = np.isfinite(states) & (states != 0)
        valid_states = np.unique(states[valid_state_mask].astype(np.int64, copy=False))
        state_groups = states[valid_state_mask].astype(np.int64, copy=False)

        def state_counts(mask: np.ndarray) -> np.ndarray:
            return np.bincount(state_groups, weights=mask[valid_state_mask].astype(np.uint8))

        overlap_counts = state_counts(overlap)
        only_ours_counts = state_counts(only_ours)
        only_reference_counts = state_counts(only_reference)

        def state_areas(mask: np.ndarray) -> np.ndarray:
            return np.bincount(
                state_groups,
                weights=area[valid_state_mask] * mask[valid_state_mask],
            ) / 10_000.0

        overlap_areas = state_areas(overlap)
        only_ours_areas = state_areas(only_ours)
        only_reference_areas = state_areas(only_reference)
        for state_value in valid_states:
            records.append(
                {
                    "state_code": int(state_value),
                    "overlap_pixels": int(overlap_counts[state_value]),
                    "only_ours_pixels": int(only_ours_counts[state_value]),
                    "only_reference_pixels": int(only_reference_counts[state_value]),
                    "overlap_area_ha": float(overlap_areas[state_value]),
                    "only_ours_area_ha": float(only_ours_areas[state_value]),
                    "only_reference_area_ha": float(only_reference_areas[state_value]),
                    "comparison_start_year": interval_start,
                    "comparison_end_year": interval_end,
                }
            )
        by_state = pd.DataFrame(
            records,
            columns=[
                "state_code",
                "overlap_pixels",
                "only_ours_pixels",
                "only_reference_pixels",
                "overlap_area_ha",
                "only_ours_area_ha",
                "only_reference_area_ha",
                "comparison_start_year",
                "comparison_end_year",
            ],
        )
    return {"overall": overall, "by_year": by_year, "by_state": by_state}


def xie_reference_to_event_dataset(
    reference: xr.DataArray | str | Path,
    *,
    current_year: int,
    cutoff_year: int | None = None,
    min_abandonment_years: int = 1,
    spatial_dims: tuple[str, str] | None = None,
    align_to: xr.DataArray | xr.Dataset | None = None,
    alignment_method: str = "nearest",
) -> xr.Dataset:
    reference_year = reference if isinstance(reference, xr.DataArray) else open_reference_abandonment_year(reference)
    if align_to is not None:
        target_dims = spatial_dims or _infer_spatial_dims(align_to)
        reference_year = align_reference_abandonment(
            reference_year,
            align_to,
            lat_dim=target_dims[0],
            lon_dim=target_dims[1],
            method=alignment_method,
        )
    spatial_dims = spatial_dims or _infer_spatial_dims(reference_year)
    detected = reference_year.notnull()
    duration = xr.where(detected, current_year - reference_year + 1, np.nan).astype(np.float32)
    if cutoff_year is None:
        qualifies_logic = detected & (duration >= min_abandonment_years)
    else:
        qualifies_logic = detected & ((cutoff_year - reference_year + 1) >= min_abandonment_years)
    qualifies = xr.where(qualifies_logic, 1, 0).astype(np.uint8)
    current = detected.astype(np.uint8)
    recultivation = xr.zeros_like(current, dtype=np.uint8)
    abandonment_end_year = xr.where(detected, float(current_year), np.nan).astype(np.float32)
    dataset = xr.Dataset(
        {
            "abandonment_year": reference_year.astype(np.float32),
            "abandonment_duration": duration,
            "recultivation": recultivation,
            "current_abandonment": current,
            "abandonment_end_year": abandonment_end_year,
            "qualifies_at_cutoff": qualifies,
        }
    )
    dataset.attrs.update(
        {
            "source_contract": "xie_event_geotiff",
            "current_year": int(current_year),
            "cutoff_year": int(cutoff_year) if cutoff_year is not None else None,
            "min_abandonment_years": int(min_abandonment_years),
            "spatial_dims": list(spatial_dims),
            "crs": reference_year.attrs.get("source_crs") or reference_year.attrs.get("crs"),
        }
    )
    for name in DEFAULT_BINARY_VARS:
        dataset[name].attrs.update({"flag_values": [0, 1], "flag_meanings": "false true"})
        dataset[name].encoding = dict(dataset[name].encoding)
        dataset[name].encoding["_FillValue"] = None
    return dataset


def write_xie_event_chunks(
    reference: xr.DataArray | str | Path,
    output_dir: str | Path,
    *,
    current_year: int,
    cutoff_year: int | None = None,
    min_abandonment_years: int = 1,
    spatial_dims: tuple[str, str] | None = None,
    align_to: xr.DataArray | xr.Dataset | None = None,
    alignment_method: str = "nearest",
    chunk_size: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    dataset = xie_reference_to_event_dataset(
        reference,
        current_year=current_year,
        cutoff_year=cutoff_year,
        min_abandonment_years=min_abandonment_years,
        spatial_dims=spatial_dims,
        align_to=align_to,
        alignment_method=alignment_method,
    )
    resolved_dims = spatial_dims or _infer_spatial_dims(dataset)
    paths = write_abandonment_chunks(
        dataset,
        output_dir,
        spatial_dims=resolved_dims,
        chunk_size=chunk_size,
        resume=resume,
        manifest={
            "source_contract": "xie_event_geotiff",
            "current_year": int(current_year),
            "cutoff_year": int(cutoff_year) if cutoff_year is not None else None,
            "min_abandonment_years": int(min_abandonment_years),
            "crs": dataset.attrs.get("crs"),
        },
    )
    return {
        "output_dir": str(output_dir),
        "chunk_paths": [str(path) for path in paths],
        "spatial_dims": list(resolved_dims),
        "chunk_size": chunk_size,
    }


def _simplified9_kernel(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.uint8)
    finite = np.isfinite(values)
    rounded = np.rint(values[finite]).astype(np.int32, copy=False)
    for target_class, raw_codes in SIMPLIFIED_ESA_CLASSES.items():
        result[finite] = np.where(np.isin(rounded, raw_codes), np.uint8(target_class), result[finite])
    return result


def _lcmap_to_simplified9_kernel(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, 255, dtype=np.uint8)
    finite = np.isfinite(values)
    rounded = np.zeros(values.shape, dtype=np.int16)
    rounded[finite] = np.rint(values[finite]).astype(np.int16, copy=False)
    for source_code, target_code in LCMAP_LCPRI_TO_SIMPLIFIED9.items():
        result[finite & (rounded == source_code)] = np.uint8(target_code)
    result[~finite] = 0
    return result


def _lcmap_unknown_mask_kernel(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    rounded = np.zeros(values.shape, dtype=np.int16)
    rounded[finite] = np.rint(values[finite]).astype(np.int16, copy=False)
    valid = np.isin(rounded, np.array(sorted(LCMAP_LCPRI_TO_SIMPLIFIED9), dtype=np.int16))
    return (finite & ~valid).astype(np.uint8)


def _binary_crop_kernel(value: float, *, cropland_codes: Sequence[int]) -> np.uint8:
    if not np.isfinite(value):
        return np.uint8(0)
    return np.uint8(int(round(float(value))) in set(cropland_codes))


def _window_mode_block(window: np.ndarray) -> np.ndarray:
    array = np.asarray(window)
    if array.ndim < 2:
        raise ValueError("Mode windows must include two trailing spatial axes.")
    output_shape = array.shape[:-2]
    n_win = int(array.shape[-2] * array.shape[-1])
    n_windows = int(np.prod(output_shape, dtype=np.int64)) if output_shape else 1
    if n_windows == 0 or n_win == 0:
        if np.issubdtype(array.dtype, np.floating):
            return np.full(output_shape, np.nan, dtype=array.dtype)
        return np.zeros(output_shape, dtype=array.dtype)
    if np.issubdtype(array.dtype, np.integer):
        return _integer_histogram_mode(array, output_shape, n_windows, n_win)
    return _generic_unique_mode(array, output_shape)


def _integer_histogram_mode(
    array: np.ndarray,
    output_shape: tuple[int, ...],
    n_windows: int,
    n_win: int,
) -> np.ndarray:
    flat = np.ascontiguousarray(array).reshape(n_windows, n_win)
    vmin = int(flat.min())
    vmax = int(flat.max())
    n_bins = vmax - vmin + 1
    if n_bins > 64:
        return _generic_unique_mode(array, output_shape)
    codes = flat.astype(np.intp, copy=False) - vmin
    offsets = np.arange(n_windows, dtype=np.intp)[:, None] * n_bins
    counts = np.bincount((codes + offsets).ravel(), minlength=n_windows * n_bins).reshape(n_windows, n_bins)
    max_counts = counts.max(axis=1)
    first_winner = counts.argmax(axis=1).astype(np.intp, copy=False) + vmin
    center_value = array[..., array.shape[-2] // 2, array.shape[-1] // 2].reshape(n_windows)
    center_index = center_value.astype(np.intp, copy=False) - vmin
    valid_center = (center_index >= 0) & (center_index < n_bins)
    safe_center = np.clip(center_index, 0, n_bins - 1)
    center_counts = counts[np.arange(n_windows), safe_center]
    center_wins = valid_center & (center_counts == max_counts) & (max_counts > 0)
    selected = np.where(center_wins, center_value, first_winner)
    result = np.where(max_counts > 0, selected, np.zeros(n_windows, dtype=array.dtype))
    return result.astype(array.dtype, copy=False).reshape(output_shape)


def _generic_unique_mode(array: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
    finite_values = array[np.isfinite(array)]
    if np.issubdtype(array.dtype, np.floating):
        result = np.full(output_shape, np.nan, dtype=array.dtype)
    else:
        result = np.zeros(output_shape, dtype=array.dtype)
    if finite_values.size == 0:
        return result

    classes = np.unique(finite_values)
    counts = np.empty((*output_shape, classes.size), dtype=np.uint8)
    for class_index, class_value in enumerate(classes):
        counts[..., class_index] = np.count_nonzero(array == class_value, axis=(-2, -1))
    max_counts = counts.max(axis=-1)
    first_winner = classes[counts.argmax(axis=-1)]
    center_value = array[..., array.shape[-2] // 2, array.shape[-1] // 2]
    center_class_index = np.searchsorted(classes, center_value)
    center_in_classes = center_class_index < classes.size
    safe_center_index = np.minimum(center_class_index, classes.size - 1)
    center_in_classes &= classes[safe_center_index] == center_value
    center_counts = np.take_along_axis(counts, safe_center_index[..., None], axis=-1)[..., 0]
    center_wins = center_in_classes & (center_counts == max_counts) & (max_counts > 0)
    selected = np.where(center_wins, center_value, first_winner)
    return np.where(max_counts > 0, selected, result).astype(array.dtype, copy=False)


def _temporal_majority_block(values: np.ndarray, *, window: int) -> np.ndarray:
    array = np.asarray(values)
    finite_values = array[np.isfinite(array)]
    if np.issubdtype(array.dtype, np.floating):
        result = np.full(array.shape, np.nan, dtype=array.dtype)
    else:
        result = np.zeros(array.shape, dtype=array.dtype)
    if finite_values.size == 0:
        return result

    classes = np.unique(finite_values)
    half_window = window // 2
    spatial_shape = array.shape[:-1]
    for time_index in range(array.shape[-1]):
        start = max(0, time_index - half_window)
        stop = min(array.shape[-1], time_index + half_window + 1)
        time_window = array[..., start:stop]
        counts = np.empty((*spatial_shape, classes.size), dtype=np.uint8)
        for class_index, class_value in enumerate(classes):
            counts[..., class_index] = np.count_nonzero(time_window == class_value, axis=-1)

        max_counts = counts.max(axis=-1)
        first_winner = classes[counts.argmax(axis=-1)]
        center_value = array[..., time_index]
        center_class_index = np.searchsorted(classes, center_value)
        center_in_classes = center_class_index < classes.size
        safe_center_index = np.minimum(center_class_index, classes.size - 1)
        center_in_classes &= classes[safe_center_index] == center_value
        center_counts = np.take_along_axis(counts, safe_center_index[..., None], axis=-1)[..., 0]
        center_wins = center_in_classes & (center_counts == max_counts) & (max_counts > 0)
        selected = np.where(center_wins, center_value, first_winner)
        result[..., time_index] = np.where(max_counts > 0, selected, result[..., time_index])
    return result


def _mode_from_values(valid_values: np.ndarray, center_value: float) -> np.generic:
    unique, counts = np.unique(valid_values, return_counts=True)
    winners = unique[counts == counts.max()]
    if np.isfinite(center_value) and np.any(winners == center_value):
        return center_value.item() if hasattr(center_value, "item") else center_value
    winner = winners.min()
    return winner.item() if hasattr(winner, "item") else winner


def _detect_abandonment_series(
    values: np.ndarray,
    *,
    years: np.ndarray,
    crop_codes: Sequence[int],
    built_codes: Sequence[int],
    wetland_codes: Sequence[int],
    nodata_codes: Sequence[int],
    baseline_years: tuple[int, int],
    min_abandonment_years: int,
    required_through_year: int,
    analysis_end_year: int,
    recultivation_years: int,
    detector: str,
    pre_window: int,
    post_window: int,
    max_noncrop_pre_years: int,
    max_crop_post_years: int,
) -> tuple[np.float32, np.int16, np.uint8, np.uint8, np.float32, np.uint8]:
    series = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(series)
    rounded = np.zeros(series.shape, dtype=np.int32)
    rounded[finite] = np.rint(series[finite]).astype(np.int32, copy=False)
    if nodata_codes:
        finite = finite & ~np.isin(rounded, np.asarray(tuple(nodata_codes), dtype=np.int32))

    is_crop = finite & np.isin(rounded, np.asarray(tuple(crop_codes), dtype=np.int32))
    is_excluded = finite & np.isin(rounded, np.asarray(tuple(set(built_codes) | set(wetland_codes)), dtype=np.int32))

    analysis_end_index = int(np.where(years == analysis_end_year)[0][0])
    cutoff_index = int(np.where(years == required_through_year)[0][0])
    baseline_mask = (years >= baseline_years[0]) & (years <= baseline_years[1])

    if detector == "stable_crop":
        if not baseline_mask.any() or not np.all(is_crop[baseline_mask]):
            return _empty_detection_result()
        candidate_start = int(np.where(baseline_mask)[0][-1] + 1)
        events = _find_stable_crop_events(
            is_crop=is_crop,
            is_excluded=is_excluded,
            finite=finite,
            years=years,
            start_index=candidate_start,
            analysis_end_index=analysis_end_index,
            cutoff_index=cutoff_index,
            min_abandonment_years=min_abandonment_years,
            recultivation_years=recultivation_years,
        )
    else:
        events = _find_xie_events(
            is_crop=is_crop,
            is_excluded=is_excluded,
            finite=finite,
            years=years,
            analysis_end_index=analysis_end_index,
            cutoff_index=cutoff_index,
            pre_window=pre_window,
            post_window=post_window,
            max_noncrop_pre_years=max_noncrop_pre_years,
            max_crop_post_years=max_crop_post_years,
            min_abandonment_years=min_abandonment_years,
            recultivation_years=recultivation_years,
        )

    if not events:
        return _empty_detection_result()
    latest = max(events, key=lambda item: item["start_index"])
    return (
        np.float32(latest["start_year"]),
        np.int16(latest["duration"]),
        np.uint8(latest["recultivation"]),
        np.uint8(latest["current_abandonment"]),
        np.float32(latest["end_year"]),
        np.uint8(latest["qualifies_at_cutoff"]),
    )


def _find_stable_crop_events(
    *,
    is_crop: np.ndarray,
    is_excluded: np.ndarray,
    finite: np.ndarray,
    years: np.ndarray,
    start_index: int,
    analysis_end_index: int,
    cutoff_index: int,
    min_abandonment_years: int,
    recultivation_years: int,
) -> list[dict[str, int]]:
    events: list[dict[str, int]] = []
    idx = start_index
    while idx <= analysis_end_index - min_abandonment_years + 1:
        if is_crop[idx] or not finite[idx]:
            idx += 1
            continue
        if idx > 0 and not is_crop[idx - 1]:
            idx += 1
            continue
        qualify_stop = idx + min_abandonment_years
        if qualify_stop > analysis_end_index + 1:
            break
        candidate_finite = finite[idx:qualify_stop]
        candidate_crop = is_crop[idx:qualify_stop]
        candidate_excluded = is_excluded[idx:qualify_stop]
        if (not candidate_finite.all()) or candidate_crop.any() or candidate_excluded.any():
            idx += 1
            continue
        event = _materialize_event(
            start_index=idx,
            is_crop=is_crop,
            is_excluded=is_excluded,
            finite=finite,
            years=years,
            analysis_end_index=analysis_end_index,
            cutoff_index=cutoff_index,
            min_abandonment_years=min_abandonment_years,
            recultivation_years=recultivation_years,
            qualification_crop_allowance=0,
            early_event_crop_allowance=0,
        )
        if event is not None:
            events.append(event)
            idx = event["end_index"] + 1
        else:
            idx += 1
    return events


def _find_xie_events(
    *,
    is_crop: np.ndarray,
    is_excluded: np.ndarray,
    finite: np.ndarray,
    years: np.ndarray,
    analysis_end_index: int,
    cutoff_index: int,
    pre_window: int,
    post_window: int,
    max_noncrop_pre_years: int,
    max_crop_post_years: int,
    min_abandonment_years: int,
    recultivation_years: int,
) -> list[dict[str, int]]:
    events: list[dict[str, int]] = []
    for idx in range(pre_window, analysis_end_index - post_window + 2):
        pre_slice = slice(idx - pre_window, idx)
        post_slice = slice(idx, idx + post_window)
        if idx + post_window > analysis_end_index + 1:
            break
        if not finite[pre_slice].all() or not finite[post_slice].all():
            continue
        if int((~is_crop[pre_slice]).sum()) > max_noncrop_pre_years:
            continue
        if int(is_crop[post_slice].sum()) > max_crop_post_years:
            continue
        if int(is_excluded[idx : idx + min_abandonment_years].sum()) > 0:
            continue
        if idx > 0 and not is_crop[idx - 1]:
            continue
        event = _materialize_event(
            start_index=idx,
            is_crop=is_crop,
            is_excluded=is_excluded,
            finite=finite,
            years=years,
            analysis_end_index=analysis_end_index,
            cutoff_index=cutoff_index,
            min_abandonment_years=min_abandonment_years,
            recultivation_years=recultivation_years,
            qualification_crop_allowance=max_crop_post_years,
            early_event_crop_allowance=max_crop_post_years,
            early_event_window=post_window,
        )
        if event is not None:
            events.append(event)
    return events


def _materialize_event(
    *,
    start_index: int,
    is_crop: np.ndarray,
    is_excluded: np.ndarray,
    finite: np.ndarray,
    years: np.ndarray,
    analysis_end_index: int,
    cutoff_index: int,
    min_abandonment_years: int,
    recultivation_years: int,
    qualification_crop_allowance: int,
    early_event_crop_allowance: int,
    early_event_window: int = 0,
) -> dict[str, int] | None:
    qualification_stop = start_index + min_abandonment_years
    if qualification_stop > analysis_end_index + 1:
        return None
    qualification_slice = slice(start_index, qualification_stop)
    if not finite[qualification_slice].all():
        return None
    if int(is_crop[qualification_slice].sum()) > qualification_crop_allowance:
        return None
    if is_excluded[qualification_slice].any():
        return None

    end_index = analysis_end_index
    recultivation = 0
    anomaly_budget = early_event_crop_allowance
    anomaly_stop = start_index + max(early_event_window, min_abandonment_years)
    for idx in range(qualification_stop, analysis_end_index + 1):
        if not finite[idx] or is_excluded[idx]:
            end_index = idx - 1
            break
        if idx < anomaly_stop and is_crop[idx] and anomaly_budget > 0:
            anomaly_budget -= 1
            continue
        if idx + recultivation_years <= analysis_end_index + 1 and is_crop[idx : idx + recultivation_years].all():
            end_index = idx - 1
            recultivation = 1
            break

    if end_index < start_index:
        return None
    duration = end_index - start_index + 1
    qualifies_at_cutoff = int(start_index <= cutoff_index <= end_index and (cutoff_index - start_index + 1) >= min_abandonment_years)
    return {
        "start_index": int(start_index),
        "end_index": int(end_index),
        "start_year": int(years[start_index]),
        "end_year": int(years[end_index]),
        "duration": int(duration),
        "recultivation": int(recultivation),
        "current_abandonment": int(end_index == analysis_end_index and recultivation == 0),
        "qualifies_at_cutoff": int(qualifies_at_cutoff),
    }


def _empty_detection_result() -> tuple[np.float32, np.int16, np.uint8, np.uint8, np.float32, np.uint8]:
    return (np.float32(np.nan), np.int16(0), np.uint8(0), np.uint8(0), np.float32(np.nan), np.uint8(0))


def _coordinate_edges(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=np.float64)
    diffs = np.diff(values)
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = values[:-1] + diffs / 2.0
    edges[0] = values[0] - diffs[0] / 2.0
    edges[-1] = values[-1] + diffs[-1] / 2.0
    return edges


def _restore_widths(unique_widths: np.ndarray, original_widths: np.ndarray) -> list[float]:
    restored: list[float] = []
    for width in unique_widths:
        matches = original_widths[np.isclose(original_widths, width, atol=1e-15, rtol=0.0)]
        restored.append(float(matches[0] if matches.size else width))
    return restored


def _as_year_index(values: Sequence[object]) -> np.ndarray:
    index = pd.Index(values)
    if np.issubdtype(index.dtype, np.datetime64):
        return index.year.to_numpy(dtype=np.int64)
    return index.astype(np.int64).to_numpy()


def _resolve_schema_codes(
    *,
    schema: str,
    cropland_codes: Sequence[int] | None,
    built_up_codes: Sequence[int] | None,
    wetland_codes: Sequence[int] | None,
) -> tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    if schema == "raw_esa":
        return (
            tuple(cropland_codes or tuple(sorted(DEFAULT_RAW_CROPLAND_CODES))),
            tuple(built_up_codes or tuple(sorted(DEFAULT_RAW_BUILT_UP_CODES))),
            tuple(wetland_codes or tuple(sorted(DEFAULT_RAW_WETLAND_CODES))),
            (0,),
        )
    if schema == "simplified9":
        return (
            tuple(cropland_codes or tuple(sorted(DEFAULT_SIMPLIFIED_CROPLAND_CODES))),
            tuple(built_up_codes or tuple(sorted(DEFAULT_SIMPLIFIED_BUILT_UP_CODES))),
            tuple(wetland_codes or tuple(sorted(DEFAULT_SIMPLIFIED_WETLAND_CODES))),
            (0,),
        )
    raise ValueError("schema must be 'raw_esa' or 'simplified9'.")


def _parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _array_equal_or_allclose(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind in {"f", "c"} or right.dtype.kind in {"f", "c"}:
        return np.allclose(left, right, equal_nan=True)
    return np.array_equal(left, right)


def _mapping_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
            if not np.array_equal(np.asarray(left_value), np.asarray(right_value)):
                return False
            continue
        if left_value != right_value:
            return False
    return True


def _regular_grid_transform(lat_values: np.ndarray, lon_values: np.ndarray):
    from rasterio.transform import Affine

    lat_edges = _coordinate_edges(np.asarray(lat_values, dtype=np.float64))
    lon_edges = _coordinate_edges(np.asarray(lon_values, dtype=np.float64))
    x_res = float(abs(lon_edges[1] - lon_edges[0]))
    y_res = float(abs(lat_edges[1] - lat_edges[0]))
    west = float(lon_edges.min())
    north = float(lat_edges.max())
    return Affine(x_res, 0.0, west, 0.0, -y_res, north)


def _atomic_write_netcdf(
    dataset: xr.Dataset,
    destination: Path,
    *,
    compression_level: int = 4,
) -> None:
    if compression_level < 0 or compression_level > 9:
        raise ValueError("compression_level must be between 0 and 9")
    encoding = {}
    for name in dataset.data_vars:
        entry = {
            "zlib": compression_level > 0,
            "complevel": compression_level,
            "shuffle": True,
        }
        if name in set(DEFAULT_BINARY_VARS) | set(CUTOFF_SAFE_BINARY_VARS):
            entry["_FillValue"] = None
        encoding[name] = entry

    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".tmp.nc") as handle:
        temp_path = Path(handle.name)
    try:
        dataset.to_netcdf(temp_path, encoding=encoding)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _atomic_write_text(destination: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".tmp", mode="w", encoding="utf-8") as handle:
        temp_path = Path(handle.name)
        handle.write(text)
    try:
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".tmp.csv", mode="w", encoding="utf-8", newline="") as handle:
        temp_path = Path(handle.name)
    try:
        frame.to_csv(temp_path, index=False)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _validate_file(path: Path, *, expected_size: int | None, expected_sha256: str | None) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"File size mismatch for {path}")
    if expected_sha256 is not None and _sha256_file(path).upper() != expected_sha256.upper():
        raise ValueError(f"SHA256 mismatch for {path}")


def _fetch_file(source: str | Path, destination: Path, *, chunk_size: int) -> None:
    parsed = urllib.parse.urlparse(str(source))
    if parsed.scheme in {"http", "https"}:
        _download_with_resume(str(source), destination, chunk_size=chunk_size)
        return
    _atomic_copy(Path(source), destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".part") as handle:
        temp_path = Path(handle.name)
    try:
        shutil.copyfile(source, temp_path)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _download_with_resume(source_url: str, target_path: Path, *, chunk_size: int) -> None:
    partial_path = target_path.with_suffix(target_path.suffix + ".part")
    existing_bytes = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
    request = urllib.request.Request(source_url, headers=headers)
    with urllib.request.urlopen(request) as response:
        mode = "ab" if response.status == 206 and existing_bytes > 0 else "wb"
        with partial_path.open(mode) as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
    os.replace(partial_path, target_path)


def _extract_archive_member(archive_path: Path, archive_member_name: str, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        if archive_member_name not in members:
            raise FileNotFoundError(f"{archive_member_name} not found in archive {archive_path}")
        member = members[archive_member_name]
        with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".tmp") as handle:
            temp_path = Path(handle.name)
        try:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(member.name)
            with extracted, temp_path.open("wb") as writer:
                shutil.copyfileobj(extracted, writer, length=1 << 20)
            os.replace(temp_path, destination)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _infer_spatial_dims(data: xr.DataArray | xr.Dataset) -> tuple[str, str]:
    dims = tuple(data.dims) if isinstance(data, xr.DataArray) else tuple(next(iter(data.data_vars.values())).dims)
    for candidate in (("lat", "lon"), ("y", "x")):
        if all(dim in dims for dim in candidate):
            return candidate
    non_time = tuple(dim for dim in dims if dim != "time")
    if len(non_time) >= 2:
        return non_time[-2], non_time[-1]
    raise ValueError(f"Unable to infer spatial dimensions from {dims}")


def _auto_chunk_slices(
    dataset: xr.Dataset,
    *,
    spatial_dims: tuple[str, str],
    chunk_size: int | None,
) -> list[dict[str, slice]]:
    if chunk_size is None:
        return [{}]
    y_dim, x_dim = spatial_dims
    slices: list[dict[str, slice]] = []
    for row_start in range(0, dataset.sizes[y_dim], chunk_size):
        for col_start in range(0, dataset.sizes[x_dim], chunk_size):
            slices.append(
                {
                    y_dim: slice(row_start, min(row_start + chunk_size, dataset.sizes[y_dim])),
                    x_dim: slice(col_start, min(col_start + chunk_size, dataset.sizes[x_dim])),
                }
            )
    return slices


__all__ = [
    "CUTOFF_SAFE_OUTPUT_VARS",
    "LEGACY_OUTPUT_VARS",
    "LCMAP_LCPRI_TO_SIMPLIFIED9",
    "SIMPLIFIED_ESA_CLASSES",
    "aggregate_categorical_mode",
    "align_reference_abandonment",
    "build_legacy_candidate_mask",
    "candidate_chunk_keys",
    "cell_area_m2",
    "compare_reference_abandonment",
    "detect_legacy_equivalent_chunk",
    "detect_abandonment",
    "fetch_reference_raster",
    "legacy_equivalent_detect_series",
    "merge_landcover_into_legacy_chunks",
    "monitor_mode_reclass_file",
    "open_reference_abandonment_year",
    "publish_validation_results",
    "reclassify_esa_cci",
    "reclassify_lcmap_lcpri",
    "run_abandonment_detection",
    "summarize_key_numbers",
    "temporal_majority_filter",
    "validate_reference_raster",
    "validate_legacy_chunk_set",
    "validate_mode_reclass_file",
    "validate_roundtrip",
    "write_abandonment_chunks",
    "write_legacy_equivalent_chunks",
    "write_xie_event_chunks",
    "xie_reference_to_event_dataset",
]
