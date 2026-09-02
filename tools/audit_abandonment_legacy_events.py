from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from function.cropland_abandonment import cell_area_m2


DEFAULT_RUN_ID = "s0-us-abandon-multiscale-2240f3ec40e9"
DEFAULT_SOURCES = {
    "esa_cci_1km_mode": Path(r"D:\xarray\merged_chunk_2"),
    "esa_cci_300m": (
        Path(r"D:\xarray\s0_us_abandon")
        / DEFAULT_RUN_ID
        / "sources"
        / "esa_cci_300m"
        / "merged_chunks"
    ),
}
DEFAULT_ALLOWED_POINTS = {
    "esa_cci_1km_mode": REPO_ROOT / "data" / "us_abandon_clean.csv",
}
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "s0_us_abandon"
    / "summaries"
    / "abandonment_legacy_event_class_audit.csv"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "outputs"
    / "s0_us_abandon"
    / "reports"
    / "comparability_limits.md"
)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".tmp", dir=path.parent, delete=False, encoding="utf-8", newline=""
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".tmp", dir=path.parent, delete=False, encoding="utf-8", newline=""
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _year_values(time: xr.DataArray) -> np.ndarray:
    values = np.asarray(time.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.DatetimeIndex(values).year.to_numpy(dtype=np.int16)
    return values.astype(np.int16)


def audit_legacy_event_dataset(
    dataset: xr.Dataset,
    *,
    source_key: str,
    target_year: int = 2020,
    landcover_name: str = "landcover",
    class_codes: Mapping[str, int] | None = None,
    allowed_grid_keys: pd.MultiIndex | None = None,
) -> pd.DataFrame:
    """Audit questionable land-cover classes inside events active at ``target_year``.

    Metrics are deliberately overlapping.  For example, one event can contain both
    wetland and water years.  This is an audit of the frozen legacy detector, not a
    reclassification or a replacement detector.
    """

    class_codes = dict(class_codes or {"wetland": 6, "built": 7, "water": 9})
    required = {"abandonment_year", "abandonment_duration", landcover_name}
    missing = required - set(dataset.data_vars)
    if missing:
        raise ValueError(f"Missing audit variables: {sorted(missing)}")
    if "time" not in dataset.coords:
        raise ValueError("A time coordinate is required for event-class auditing")

    years = _year_values(dataset["time"])
    matches = np.flatnonzero(years == int(target_year))
    if matches.size != 1:
        raise ValueError(f"Expected exactly one target year {target_year}; found {matches.size}")
    target_index = int(matches[0])

    start = np.asarray(dataset["abandonment_year"].values, dtype=np.float64)
    duration = np.asarray(dataset["abandonment_duration"].values, dtype=np.float64)
    end = start + duration - 1.0
    active = np.isfinite(start) & np.isfinite(duration) & (start <= target_year) & (end >= target_year)

    lat_name = "lat" if "lat" in dataset.coords else "latitude"
    lon_name = "lon" if "lon" in dataset.coords else "longitude"
    if allowed_grid_keys is not None and active.any():
        rows, cols = np.nonzero(active)
        candidate_keys = pd.MultiIndex.from_arrays(
            [
                np.round(np.asarray(dataset[lat_name].values)[rows], 7),
                np.round(np.asarray(dataset[lon_name].values)[cols], 7),
            ],
            names=["lat", "lon"],
        )
        keep = candidate_keys.isin(allowed_grid_keys)
        filtered_active = np.zeros_like(active, dtype=bool)
        filtered_active[rows[keep], cols[keep]] = True
        active = filtered_active

    landcover = np.asarray(dataset[landcover_name].isel(time=slice(0, target_index + 1)).values)
    audit_years = years[: target_index + 1]
    event_window = active[None, :, :] & (audit_years[:, None, None] >= start[None, :, :])

    nodata_values = (~np.isfinite(landcover)) | (landcover == 255)
    flags: dict[str, np.ndarray] = {
        "current_active_2020": active,
        "event_contains_nodata": np.any(event_window & nodata_values, axis=0),
    }
    for label, code in class_codes.items():
        flags[f"event_contains_{label}"] = np.any(event_window & (landcover == code), axis=0)
        flags[f"target_is_{label}"] = active & (landcover[target_index] == code)

    suspicious_names = [name for name in flags if name.startswith("event_contains_")]
    flags["event_contains_any_suspicious_class"] = np.logical_or.reduce(
        [flags[name] for name in suspicious_names]
    )

    area = np.asarray(cell_area_m2(dataset[lat_name], dataset[lon_name]).values, dtype=np.float64)
    active_count = int(active.sum())
    active_area = float(area[active].sum())

    rows: list[dict[str, object]] = []
    for metric, mask in flags.items():
        pixel_count = int(mask.sum())
        area_m2 = float(area[mask].sum())
        rows.append(
            {
                "source_key": source_key,
                "target_year": int(target_year),
                "detector_contract": dataset.attrs.get("detector_contract", "unknown"),
                "metric": metric,
                "pixel_count": pixel_count,
                "area_ha": area_m2 / 10_000.0,
                "area_mha": area_m2 / 1.0e10,
                "percent_of_active_pixels": (100.0 * pixel_count / active_count) if active_count else 0.0,
                "percent_of_active_area": (100.0 * area_m2 / active_area) if active_area else 0.0,
            }
        )
    return pd.DataFrame(rows)


def audit_chunk_directory(
    directory: Path,
    *,
    source_key: str,
    target_year: int = 2020,
    pattern: str = "chunk_*.nc",
    allowed_grid_keys: pd.MultiIndex | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for path in files:
        try:
            with xr.open_dataset(path, mask_and_scale=True) as dataset:
                frames.append(
                    audit_legacy_event_dataset(
                        dataset,
                        source_key=source_key,
                        target_year=target_year,
                        allowed_grid_keys=allowed_grid_keys,
                    )
                )
        except Exception as exc:  # retain exact failed path in the audit manifest
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    if failures:
        raise RuntimeError(f"Legacy event audit failed for {len(failures)} chunks: {failures[:3]}")
    combined = pd.concat(frames, ignore_index=True)
    numeric = [
        "pixel_count",
        "area_ha",
        "area_mha",
    ]
    grouped = combined.groupby(
        ["source_key", "target_year", "detector_contract", "metric"], as_index=False
    )[numeric].sum()
    active = grouped.loc[grouped["metric"] == "current_active_2020"].set_index("source_key")
    active_pixels = active["pixel_count"].to_dict()
    active_area = active["area_ha"].to_dict()
    grouped["percent_of_active_pixels"] = grouped.apply(
        lambda row: 100.0 * row.pixel_count / active_pixels.get(row.source_key, 1), axis=1
    )
    grouped["percent_of_active_area"] = grouped.apply(
        lambda row: 100.0 * row.area_ha / active_area.get(row.source_key, 1.0), axis=1
    )
    manifest = {
        "source_key": source_key,
        "directory": str(directory),
        "target_year": int(target_year),
        "chunk_count": len(files),
        "failed_chunks": 0,
        "first_chunk": str(files[0]),
        "last_chunk": str(files[-1]),
        "allowed_grid_key_count": int(len(allowed_grid_keys)) if allowed_grid_keys is not None else None,
    }
    return grouped, manifest


def _comparability_report(audit: pd.DataFrame, output_csv: Path) -> str:
    comparison_path = (
        REPO_ROOT
        / "outputs"
        / "s0_us_abandon"
        / "summaries"
        / "abandonment_resolution_comparison.csv"
    )
    comparison = pd.read_csv(comparison_path)
    columns = [column for column in ["source_key", "target_year", "area_mha", "delta_mha_vs_1km_mode"] if column in comparison]
    comparison_md = comparison[columns].to_markdown(index=False)
    suspicious = audit[audit["metric"].isin(["current_active_2020", "event_contains_any_suspicious_class"])]
    suspicious_md = suspicious[
        ["source_key", "metric", "pixel_count", "area_mha", "percent_of_active_area"]
    ].to_markdown(index=False)
    return f"""# Abandonment comparability limits

Generated at `{datetime.now(timezone.utc).isoformat()}` from `{output_csv.name}`.

## Frozen detector boundary

The production ESA chain remains the legacy-equivalent detector with categorical-mode
aggregation at 1 km.  This audit does not mutate event years, durations, or current status.

## Published area comparison

{comparison_md}

## Legacy event-class audit

{suspicious_md}

## Comparability limits

- ESA-CCI 1 km versus ESA-CCI 300 m is a same-source resolution-sensitivity comparison.
- LCMAP is a different annual classification system and grid; its area difference is source sensitivity, not a truth ranking.
- Xie is an event product valid through 2018.  It must not be represented as a 2020 current-abandonment estimate.
- Event-class flags overlap.  They identify cases requiring strict-S0 sensitivity analysis and are not corrections to the frozen production result.
"""


def run_audit(
    sources: Mapping[str, Path],
    *,
    target_year: int,
    output_csv: Path,
    report_path: Path,
    allowed_point_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    for source_key, directory in sources.items():
        allowed_grid_keys = None
        allowed_path = (allowed_point_paths or {}).get(source_key)
        if allowed_path is not None:
            allowed_points = pd.read_csv(allowed_path, usecols=["lat", "lon"])
            allowed_grid_keys = pd.MultiIndex.from_arrays(
                [
                    allowed_points["lat"].round(7),
                    allowed_points["lon"].round(7),
                ],
                names=["lat", "lon"],
            ).drop_duplicates()
        print(f"[STAGE] auditing source={source_key} directory={directory}")
        frame, manifest = audit_chunk_directory(
            directory,
            source_key=source_key,
            target_year=target_year,
            allowed_grid_keys=allowed_grid_keys,
        )
        frames.append(frame)
        manifests.append(manifest)
        print(
            f"[RESULT] source={source_key} chunks={manifest['chunk_count']} "
            f"active_area_mha={frame.loc[frame.metric == 'current_active_2020', 'area_mha'].sum():.6f}"
        )
    audit = pd.concat(frames, ignore_index=True)
    _atomic_csv(output_csv, audit)
    _atomic_text(report_path, _comparability_report(audit, output_csv))
    payload = {
        "passed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_year": target_year,
        "sources": manifests,
        "output_csv": str(output_csv),
        "output_sha256": _sha256(output_csv),
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    _atomic_text(manifest_path, json.dumps(payload, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit suspicious land-cover classes in frozen legacy abandonment events.")
    parser.add_argument("--target-year", type=int, default=2020)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source", action="append", default=[], metavar="KEY=DIR")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    sources = dict(DEFAULT_SOURCES)
    if args.source:
        sources = {}
        for item in args.source:
            key, separator, raw_path = item.partition("=")
            if not separator:
                raise ValueError(f"Invalid --source {item!r}; expected KEY=DIR")
            sources[key] = Path(raw_path)
    print(f"[CONFIG] target_year={args.target_year} sources={list(sources)}")
    print(f"[PATH] output={args.output} report={args.report}")
    result = run_audit(
        sources,
        target_year=args.target_year,
        output_csv=args.output,
        report_path=args.report,
        allowed_point_paths=DEFAULT_ALLOWED_POINTS if not args.source else None,
    )
    print(f"[QA] audit={'PASS' if result['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
