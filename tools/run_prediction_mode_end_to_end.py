from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from function import cropland_abandonment as abandonment
from function import embedding_pipeline as embedding


MODE_LANDCOVER = Path(r"D:\xarray\reclass_lccs_1km.nc")
MODE_ABANDON_DIR = Path(r"D:\xarray\abandon_2")
MODE_MERGED_DIR = Path(r"D:\xarray\merged_chunk_2")
MEAN_MERGED_DIR = Path(r"D:\xarray\history_data\merged_chunk_2")
FEATURE_ROOT = Path(r"D:\xarray\aligned2\Feature_all")
STATE_PATH = REPO_ROOT / "data" / "cb_2018_us_state_500k.shp"
PV_EMBEDDING = REPO_ROOT / "data" / "aligned_for_training0819.csv"
CURRENT_US_CSV = REPO_ROOT / "data" / "us_abandon_clean.csv"
VERSIONED_US_CSV = REPO_ROOT / "data" / "us_abandon_clean_mode0819.csv"
PREDICTION_POOL = REPO_ROOT / "data" / "us_abandon_for_prediction0819.csv"
US_MANIFEST = REPO_ROOT / "data" / "us_abandon_clean_mode0819.manifest.json"
ARCHIVE_ROOT = Path(r"D:\xarray\history_data\prediction_mean_legacy_20260819")
AREA_SUMMARY = REPO_ROOT / "outputs" / "s0_us_abandon" / "summaries" / "abandonment_mean_vs_mode_conus.csv"
AREA_STATE_SUMMARY = REPO_ROOT / "outputs" / "s0_us_abandon" / "summaries" / "abandonment_mean_vs_mode_by_state.csv"
TARGET_YEAR = 2020
EXPECTED_CHUNKS = 1022
WGS84 = Geod(ellps="WGS84")
DEFAULT_WINDOW_YEAR = 5
DEFAULT_START_YEAR = 1992
DEFAULT_CURRENT_END_YEAR = 2020
DEFAULT_INIT_CROPLAND = 2
DEFAULT_EXTEND_VALIDATION = True
CONUS_BOUNDS = dict(embedding.DEFAULT_CONUS_BOUNDS)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _stage_csv(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=destination.parent, suffix=".next.csv", delete=False
    ) as handle:
        temporary = Path(handle.name)
    frame.to_csv(temporary, index=False)
    return temporary


def _stage_json(payload: dict[str, object], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, suffix=".next.json", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
        return Path(handle.name)


def _commit_staged_files(staged: dict[Path, Path]) -> None:
    """Publish only a new or byte-identical set; never replace different outputs."""
    existing = [destination for destination in staged if destination.exists()]
    if existing:
        if len(existing) != len(staged):
            raise FileExistsError("Refusing a partial legacy prediction output set")
        mismatches = [
            str(destination)
            for destination, temporary in staged.items()
            if sha256_file(destination) != sha256_file(temporary)
        ]
        if mismatches:
            raise FileExistsError(f"Refusing to overwrite different prediction outputs: {mismatches}")
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        return
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


def _reject_legacy_featureless_publication() -> None:
    raise RuntimeError(
        "Legacy featureless prediction publication is disabled; "
        "call run_prediction_mode_end_to_end() and consume the public feature receipt."
    )


def preflight(abandonment_receipt: dict[str, object]) -> dict[str, object]:
    mode_abandon_dir = Path(str(abandonment_receipt["abandonment_chunk_root"]))
    mode_merged_dir = Path(str(abandonment_receipt["merged_chunk_root"]))
    expected_chunks = int(abandonment_receipt["expected_chunk_count"])
    required = [
        MODE_LANDCOVER,
        mode_abandon_dir / "run_manifest.json",
        mode_merged_dir / "run_manifest.json",
        FEATURE_ROOT / "DEM.nc",
        STATE_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prediction inputs: {missing}")
    detection = json.loads((mode_abandon_dir / "run_manifest.json").read_text(encoding="utf-8"))
    merge_qa = json.loads((mode_merged_dir / "run_manifest.json").read_text(encoding="utf-8"))
    mode_files = sorted(mode_merged_dir.glob("chunk_*.nc"))
    abandon_files = sorted(mode_abandon_dir.glob("chunk_*.nc"))
    if len(mode_files) != expected_chunks or len(abandon_files) != expected_chunks:
        raise ValueError(f"Expected {expected_chunks} validated chunks")
    if {path.name for path in mode_files} != {path.name for path in abandon_files}:
        raise ValueError("Detection and merged chunk file keys differ")
    if detection.get("run_fingerprint") != abandonment_receipt.get("run_fingerprint"):
        raise ValueError("Detection manifest fingerprint differs from the public API receipt")
    if merge_qa.get("run_fingerprint") != abandonment_receipt.get("run_fingerprint"):
        raise ValueError("Merged manifest fingerprint differs from the public API receipt")
    landcover_hash = str(abandonment_receipt["source_sha256"])
    return {
        "landcover_sha256": landcover_hash,
        "detection_manifest_sha256": sha256_file(mode_abandon_dir / "run_manifest.json"),
        "merge_manifest_sha256": sha256_file(mode_merged_dir / "run_manifest.json"),
        "chunk_count": len(mode_files),
        "chunk_keys": [path.name for path in mode_files],
        "detection_manifest": detection,
        "merge_manifest": merge_qa,
    }


def archive_current_us_csv() -> dict[str, object]:
    _reject_legacy_featureless_publication()
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = ARCHIVE_ROOT / CURRENT_US_CSV.name
    manifest_path = ARCHIVE_ROOT / "archive_manifest.json"
    source_hash = sha256_file(CURRENT_US_CSV)
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete" and destination.exists():
            archive_hash = sha256_file(destination)
            if archive_hash != prior.get("archive_sha256"):
                raise RuntimeError("Existing prediction archive no longer matches its manifest")
            if source_hash == prior.get("source_sha256"):
                return prior
            if VERSIONED_US_CSV.exists() and source_hash == sha256_file(VERSIONED_US_CSV):
                return prior
            raise FileExistsError("Current US CSV differs from both the archived legacy and published mode versions")
    if destination.exists():
        if sha256_file(destination) != source_hash:
            raise FileExistsError(f"Archive collision with different content: {destination}")
    else:
        with tempfile.NamedTemporaryFile(dir=ARCHIVE_ROOT, suffix=".copying", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copy2(CURRENT_US_CSV, temporary)
            if sha256_file(temporary) != source_hash:
                raise RuntimeError("Staged US CSV archive hash differs from the source")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    archive_hash = sha256_file(destination)
    if archive_hash != source_hash:
        raise RuntimeError("Archived US CSV hash differs from the source")
    header = pd.read_csv(CURRENT_US_CSV, nrows=0)
    rows = sum(1 for _ in CURRENT_US_CSV.open("r", encoding="utf-8", errors="ignore")) - 1
    manifest = {
        "status": "complete",
        "source": str(CURRENT_US_CSV),
        "destination": str(destination),
        "source_sha256": source_hash,
        "archive_sha256": archive_hash,
        "source_size_bytes": CURRENT_US_CSV.stat().st_size,
        "source_mtime_ns": CURRENT_US_CSV.stat().st_mtime_ns,
        "rows": rows,
        "columns": list(header.columns),
        "provenance_note": "legacy mean-resampled prediction CSV; exact writer absent from repository",
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _collect_active_points(directory: Path, target_year: int) -> pd.DataFrame:
    records = []
    files = sorted(directory.glob("chunk_*.nc"))
    for path in tqdm(files, desc=f"active pixels {directory.name}", unit="chunk"):
        with xr.open_dataset(path) as dataset:
            lat = dataset.lat.values
            lon = dataset.lon.values
            lat_positions = np.flatnonzero((lat >= 25.0) & (lat <= 49.0))
            lon_positions = np.flatnonzero((lon >= -125.0) & (lon <= -65.0))
            if not lat_positions.size or not lon_positions.size:
                continue
            subset = dataset[["abandonment_year", "abandonment_duration"]].isel(
                lat=lat_positions, lon=lon_positions
            ).load()
        start = np.asarray(subset.abandonment_year.values, dtype=np.float64)
        duration = np.asarray(subset.abandonment_duration.values, dtype=np.float64)
        active = np.isfinite(start) & np.isfinite(duration) & (start <= target_year) & (
            start + duration - 1 >= target_year
        )
        rows, cols = np.nonzero(active)
        if rows.size:
            records.append(pd.DataFrame({"lat": subset.lat.values[rows], "lon": subset.lon.values[cols]}))
    if not records:
        return pd.DataFrame(columns=["lat", "lon"])
    frame = pd.concat(records, ignore_index=True).drop_duplicates(["lat", "lon"])
    clipped, _ = embedding._clip_prediction_to_conus(frame, STATE_PATH)
    return clipped


def _cell_area_for_latitudes(latitudes: pd.Series, lon_resolution: float = 1.0 / 120.0) -> np.ndarray:
    values = latitudes.to_numpy(dtype=np.float64)
    unique, inverse = np.unique(values, return_inverse=True)
    lookup = np.empty(unique.size, dtype=np.float64)
    half_lat = (1.0 / 120.0) / 2.0
    for index, latitude in enumerate(unique):
        south, north = latitude - half_lat, latitude + half_lat
        area, _ = WGS84.polygon_area_perimeter(
            [0.0, lon_resolution, lon_resolution, 0.0, 0.0],
            [south, south, north, north, south],
        )
        lookup[index] = abs(area)
    return lookup[inverse]


def compare_mean_mode_area() -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = _collect_active_points(MEAN_MERGED_DIR, TARGET_YEAR)
    mode = _collect_active_points(MODE_MERGED_DIR, TARGET_YEAR)
    for frame in (mean, mode):
        frame["area_m2"] = _cell_area_for_latitudes(frame["lat"])
        frame["grid_key"] = list(zip(np.round(frame["lat"], 8), np.round(frame["lon"], 8)))
    mean_keys, mode_keys = set(mean.grid_key), set(mode.grid_key)

    def record(scope: str, state_code: str, mean_part: pd.DataFrame, mode_part: pd.DataFrame) -> dict[str, object]:
        mean_part_keys = set(mean_part.grid_key)
        mode_part_keys = set(mode_part.grid_key)
        intersection = mean_part_keys & mode_part_keys
        union = mean_part_keys | mode_part_keys
        mean_ha = float(mean_part.area_m2.sum() / 10_000.0)
        mode_ha = float(mode_part.area_m2.sum() / 10_000.0)
        return {
            "scope": scope,
            "state_code": state_code,
            "target_year": TARGET_YEAR,
            "mean_pixels": len(mean_part),
            "mode_pixels": len(mode_part),
            "mean_area_ha": mean_ha,
            "mode_area_ha": mode_ha,
            "mean_area_mha": mean_ha / 1_000_000.0,
            "mode_area_mha": mode_ha / 1_000_000.0,
            "delta_area_ha": mode_ha - mean_ha,
            "delta_percent": ((mode_ha - mean_ha) / mean_ha * 100.0) if mean_ha else np.nan,
            "intersection_pixels": len(intersection),
            "union_pixels": len(union),
            "jaccard": len(intersection) / len(union) if union else np.nan,
        }

    overall = pd.DataFrame([record("CONUS", "ALL", mean, mode)])
    state_records = []
    for state_code in sorted(set(mean.state_code) | set(mode.state_code)):
        state_records.append(
            record(
                "state",
                state_code,
                mean.loc[mean.state_code == state_code],
                mode.loc[mode.state_code == state_code],
            )
        )
    states = pd.DataFrame(state_records)
    if not np.isclose(states.mean_area_ha.sum(), overall.mean_area_ha.iloc[0], rtol=0, atol=1e-6):
        raise ValueError("Mean CONUS area does not reconcile with state totals")
    if not np.isclose(states.mode_area_ha.sum(), overall.mode_area_ha.iloc[0], rtol=0, atol=1e-6):
        raise ValueError("Mode CONUS area does not reconcile with state totals")
    return overall, states


class PredictionProgress:
    def __init__(self, total: int) -> None:
        self.progress = tqdm(total=total, desc="prediction chunks", unit="chunk")

    def __call__(self, status: dict[str, object]) -> None:
        if "done" in status:
            self.progress.n = int(status["done"])
            self.progress.set_postfix(rows=status.get("rows", 0), failed=0)
            self.progress.refresh()
        elif status.get("stage") == "feature":
            self.progress.set_postfix(feature=status.get("variable"), failed=0)

    def close(self) -> None:
        self.progress.close()


def combine_mode_prediction_features(
    abandonment_receipt: dict[str, object],
    *,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Readable successor to the legacy notebook combine_feature loop."""
    merged_root = Path(str(abandonment_receipt["merged_chunk_root"]))
    progress = PredictionProgress(int(abandonment_receipt["expected_chunk_count"])) if show_progress else None
    try:
        candidate, membership = embedding.build_prediction_embedding(
            str(merged_root / "chunk_*.nc"),
            FEATURE_ROOT,
            STATE_PATH,
            target_year=TARGET_YEAR,
            bounds=CONUS_BOUNDS,
            progress_callback=progress,
        )
    finally:
        if progress is not None:
            progress.close()
    return candidate, membership


def publish_feature_prediction_bundle(
    candidate: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    abandonment_receipt: dict[str, object],
    parameters: dict[str, object],
    preflight_report: dict[str, object],
    elapsed_seconds: float,
) -> dict[str, object]:
    _reject_legacy_featureless_publication()
    validation = embedding.validate_prediction_candidate(candidate, target_year=TARGET_YEAR)
    if not validation["passed"]:
        raise ValueError(validation)
    membership_report = embedding.state_membership_summary(membership)
    _, california_report = embedding.reconstruct_state_subset(
        candidate,
        membership,
        state_fips="06",
        expected_membership_sha256=str(membership_report["membership_sha256"]),
    )
    aoi_contract = embedding.prediction_aoi_contract(STATE_PATH, bounds=CONUS_BOUNDS)
    csv_path = Path(str(abandonment_receipt["required_csv_path"]))
    manifest_path = csv_path.with_suffix(".manifest.json")
    acceptance_path = Path(str(abandonment_receipt["required_acceptance_path"]))
    staged: dict[Path, Path] = {}
    try:
        staged[csv_path] = _stage_csv(candidate, csv_path)
        roundtrip = pd.read_csv(staged[csv_path])
        roundtrip_validation = embedding.validate_prediction_candidate(roundtrip, target_year=TARGET_YEAR)
        if not roundtrip_validation["passed"] or len(roundtrip) != len(candidate):
            raise ValueError(roundtrip_validation)
        csv_sha256 = sha256_file(staged[csv_path])
        manifest = {
            "schema_version": "us-abandon-clean-feature-manifest.v1",
            "status": "complete",
            "feature": abandonment_receipt["feature"],
            "parameters": parameters,
            "parameter_sha256": abandonment_receipt["parameter_sha256"],
            "run_fingerprint": abandonment_receipt["run_fingerprint"],
            "source_sha256": abandonment_receipt["source_sha256"],
            "aoi": aoi_contract,
            "aoi_sha256": aoi_contract["aoi_sha256"],
            "preflight": preflight_report,
            "prediction_validation": validation,
            "roundtrip_validation": roundtrip_validation,
            "csv_path": str(csv_path),
            "csv_sha256": csv_sha256,
            "csv_size_bytes": int(staged[csv_path].stat().st_size),
            "csv_rows": int(len(candidate)),
            "csv_columns": list(candidate.columns),
            "state_membership": membership_report,
            "california_subset": california_report,
            "california_csv_published": False,
            "elapsed_seconds": float(elapsed_seconds),
        }
        staged[manifest_path] = _stage_json(manifest, manifest_path)
        manifest_sha256 = sha256_file(staged[manifest_path])
        acceptance = {
            "schema_version": "local-run-abandonment-detection-acceptance.v1",
            "status": "accepted",
            "accepted": True,
            "feature": abandonment_receipt["feature"],
            "parameter_sha256": abandonment_receipt["parameter_sha256"],
            "aoi_sha256": aoi_contract["aoi_sha256"],
            "run_fingerprint": abandonment_receipt["run_fingerprint"],
            "expected_chunk_count": abandonment_receipt["expected_chunk_count"],
            "verified_chunk_count": abandonment_receipt["verified_chunk_count"],
            "csv_path": str(csv_path),
            "csv_sha256": csv_sha256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "state_membership_sha256": membership_report["membership_sha256"],
            "california_subset_sha256": california_report["subset_sha256"],
        }
        staged[acceptance_path] = _stage_json(acceptance, acceptance_path)
        _commit_staged_files(staged)
        return acceptance
    except Exception:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise


def publish_prediction_outputs(candidate: pd.DataFrame) -> dict[str, object]:
    _reject_legacy_featureless_publication()
    validation = embedding.validate_prediction_candidate(candidate, target_year=TARGET_YEAR)
    if not validation["passed"]:
        raise ValueError(validation)
    staged: dict[Path, Path] = {}
    try:
        staged[VERSIONED_US_CSV] = _stage_csv(candidate, VERSIONED_US_CSV)
        roundtrip = pd.read_csv(staged[VERSIONED_US_CSV])
        roundtrip_validation = embedding.validate_prediction_candidate(roundtrip, target_year=TARGET_YEAR)
        if not roundtrip_validation["passed"] or len(roundtrip) != len(candidate):
            raise ValueError(roundtrip_validation)
        staged[CURRENT_US_CSV] = _stage_csv(candidate, CURRENT_US_CSV)
        if sha256_file(staged[CURRENT_US_CSV]) != sha256_file(staged[VERSIONED_US_CSV]):
            raise RuntimeError("Canonical staging hash differs from versioned prediction CSV")
        pv = pd.read_csv(PV_EMBEDDING)
        pool = embedding.build_prediction_pool(candidate, pv)
        staged[PREDICTION_POOL] = _stage_csv(pool, PREDICTION_POOL)
        _commit_staged_files(staged)
    except Exception:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise
    published_hash = sha256_file(CURRENT_US_CSV)
    if published_hash != sha256_file(VERSIONED_US_CSV):
        raise RuntimeError("Published canonical and versioned prediction CSV hashes differ")
    return {
        "prediction_validation": validation,
        "roundtrip_validation": roundtrip_validation,
        "versioned_csv": str(VERSIONED_US_CSV),
        "canonical_csv": str(CURRENT_US_CSV),
        "csv_sha256": published_hash,
        "prediction_pool": str(PREDICTION_POOL),
        "prediction_pool_rows": len(pool),
    }


def publish_mean_mode_area_comparison() -> dict[str, object]:
    _reject_legacy_featureless_publication()
    overall, states = compare_mean_mode_area()
    _commit_staged_files(
        {
            AREA_SUMMARY: _stage_csv(overall, AREA_SUMMARY),
            AREA_STATE_SUMMARY: _stage_csv(states, AREA_STATE_SUMMARY),
        }
    )
    return {
        "area_summary": str(AREA_SUMMARY),
        "area_state_summary": str(AREA_STATE_SUMMARY),
        "overall": overall.to_dict("records"),
    }


def publish_prediction_bundle(
    candidate: pd.DataFrame,
    overall: pd.DataFrame,
    states: pd.DataFrame,
    *,
    preflight_report: dict[str, object],
    archive_report: dict[str, object],
    elapsed_seconds: float,
) -> dict[str, object]:
    _reject_legacy_featureless_publication()
    """Stage, validate and transactionally publish the complete prediction result set."""
    validation = embedding.validate_prediction_candidate(candidate, target_year=TARGET_YEAR)
    if not validation["passed"]:
        raise ValueError(validation)
    pool = embedding.build_prediction_pool(candidate, pd.read_csv(PV_EMBEDDING))
    staged: dict[Path, Path] = {}
    try:
        staged.update(
            {
                VERSIONED_US_CSV: _stage_csv(candidate, VERSIONED_US_CSV),
                CURRENT_US_CSV: _stage_csv(candidate, CURRENT_US_CSV),
                PREDICTION_POOL: _stage_csv(pool, PREDICTION_POOL),
                AREA_SUMMARY: _stage_csv(overall, AREA_SUMMARY),
                AREA_STATE_SUMMARY: _stage_csv(states, AREA_STATE_SUMMARY),
            }
        )
        versioned_hash = sha256_file(staged[VERSIONED_US_CSV])
        if sha256_file(staged[CURRENT_US_CSV]) != versioned_hash:
            raise RuntimeError("Canonical and versioned staging hashes differ")
        roundtrip = pd.read_csv(staged[VERSIONED_US_CSV])
        roundtrip_validation = embedding.validate_prediction_candidate(roundtrip, target_year=TARGET_YEAR)
        if not roundtrip_validation["passed"] or len(roundtrip) != len(candidate):
            raise ValueError(roundtrip_validation)
        manifest = {
            "status": "complete",
            "target_year": TARGET_YEAR,
            "preflight": preflight_report,
            "archive": archive_report,
            "prediction_validation": validation,
            "roundtrip_validation": roundtrip_validation,
            "versioned_csv": str(VERSIONED_US_CSV),
            "canonical_csv": str(CURRENT_US_CSV),
            "csv_sha256": versioned_hash,
            "prediction_pool": str(PREDICTION_POOL),
            "prediction_pool_rows": len(pool),
            "area_summary": str(AREA_SUMMARY),
            "area_state_summary": str(AREA_STATE_SUMMARY),
            "overall": overall.to_dict("records"),
            "elapsed_seconds": elapsed_seconds,
        }
        staged[US_MANIFEST] = _stage_json(manifest, US_MANIFEST)
        _commit_staged_files(staged)
        return manifest
    except Exception:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise


def run_prediction_mode_end_to_end(
    target_nc: str | Path = MODE_LANDCOVER,
    window_year: int = DEFAULT_WINDOW_YEAR,
    start_year: int = DEFAULT_START_YEAR,
    current_end_year: int = DEFAULT_CURRENT_END_YEAR,
    init_cropland: int = DEFAULT_INIT_CROPLAND,
    extend_validation: bool = DEFAULT_EXTEND_VALIDATION,
) -> dict[str, object]:
    os.chdir(REPO_ROOT)
    started = time.time()
    parameters = {
        "target_nc": str(Path(target_nc).expanduser().resolve()),
        "window_year": int(window_year),
        "start_year": int(start_year),
        "current_end_year": int(current_end_year),
        "init_cropland": int(init_cropland),
        "extend_validation": bool(extend_validation),
    }
    print("[STAGE] invoke public abandonment API")
    acceptance = abandonment.run_abandonment_detection(**parameters)
    if not acceptance.get("accepted"):
        raise RuntimeError(f"Public abandonment API did not finalize prediction: {acceptance}")
    print(
        f"[RESULT] feature={acceptance['feature']} "
        f"csv={acceptance['csv_path']} manifest={acceptance['manifest_path']} "
        f"elapsed_seconds={time.time() - started:.1f}"
    )
    return acceptance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cutoff-safe local detection and CONUS publication")
    parser.add_argument("--target-nc", type=Path, default=MODE_LANDCOVER)
    parser.add_argument("--window-year", type=int, default=DEFAULT_WINDOW_YEAR)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--current-end-year", type=int, default=DEFAULT_CURRENT_END_YEAR)
    parser.add_argument("--init-cropland", type=int, default=DEFAULT_INIT_CROPLAND)
    parser.add_argument(
        "--extend-validation",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_EXTEND_VALIDATION,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_prediction_mode_end_to_end(
        target_nc=arguments.target_nc,
        window_year=arguments.window_year,
        start_year=arguments.start_year,
        current_end_year=arguments.current_end_year,
        init_cropland=arguments.init_cropland,
        extend_validation=arguments.extend_validation,
    )
