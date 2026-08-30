from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
MODE_RECLASS_PATH = Path(r"D:\xarray\reclass_lccs_1km.nc")
HISTORY_RECLASS_PATH = Path(r"D:\xarray\history_data\reclass_lccs_1km.nc")
FEATURE_ROOT = Path(r"D:\xarray\aligned2\Feature_all")
PV_ALIGNMENT_CSV = REPO_ROOT / "data" / "aligned_for_training0519.csv"
CURRENT_EMBEDDING = REPO_ROOT / "data" / "training_embedding.csv"
FINAL_EMBEDDING = REPO_ROOT / "data" / "aligned_for_training0819.csv"
NEXT_EMBEDDING = REPO_ROOT / "data" / "aligned_for_training0819.next.csv"
CONUS_EMBEDDING = REPO_ROOT / "data" / "aligned_for_training0819_conus.csv"
EMBEDDING_REPORT = REPO_ROOT / "data" / "aligned_for_training0819_validation.json"
EMBEDDING_MANIFEST = REPO_ROOT / "data" / "aligned_for_training0819_manifest.json"
PIPELINE_MANIFEST = Path(r"D:\xarray\abandon_0819_pipeline_manifest.json")
PIPELINE_LOG = Path(r"D:\xarray\abandon_0819_pipeline.log")
REPO_MANIFEST = REPO_ROOT / "outputs" / "s0_us_abandon" / "manifests" / "abandon_0819_pipeline.json"
EXPECTED_OLD_EMBEDDING_SHA256 = "965269A0B8710CFA32208C804DBF64E9CFEE05D504879F9D66E28EE011D6B20A"
EXPECTED_EMBEDDING_ROWS = 90_821
CHUNK_SIZE = 500


def load_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def atomic_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, destination)


def main() -> None:
    os.chdir(REPO_ROOT)
    log_handle = PIPELINE_LOG.open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)
    abandonment = load_module("mode2_abandonment", "function/cropland_abandonment.py")
    embedding = load_module("mode2_embedding", "function/embedding_pipeline.py")
    state = {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}
    atomic_json(state, PIPELINE_MANIFEST)

    print("[STAGE] validate completed corrected 1 km output")
    readiness = abandonment.validate_mode_reclass_file(MODE_RECLASS_PATH)
    readiness["history_size_bytes"] = HISTORY_RECLASS_PATH.stat().st_size if HISTORY_RECLASS_PATH.exists() else None
    readiness["size_comparable"] = False
    readiness["size_comparison_note"] = "new compression level 4; history compression level 9"
    state["stages"]["readiness"] = readiness
    atomic_json(state, PIPELINE_MANIFEST)
    print(f"[QA] readiness PASS bytes={readiness['size_bytes']}")

    print("[STAGE] invoke public abandonment API")
    abandonment_receipt = abandonment.run_abandonment_detection(
        target_nc=MODE_RECLASS_PATH,
        window_year=5,
        start_year=1992,
        current_end_year=2020,
        init_cropland=2,
        extend_validation=True,
    )
    MODE_ABANDON_DIR = Path(abandonment_receipt["abandonment_chunk_root"])
    MODE_MERGED_DIR = Path(abandonment_receipt["merged_chunk_root"])
    state["stages"]["abandonment_receipt"] = abandonment_receipt
    atomic_json(state, PIPELINE_MANIFEST)
    print(
        f"[RESULT] public_abandonment_feature={abandonment_receipt['feature']} "
        f"accepted={abandonment_receipt['accepted']}"
    )
    print(f"[PATH] abandon_chunks={MODE_ABANDON_DIR}")
    print(f"[PATH] merged_chunks={MODE_MERGED_DIR}")
    print(f"[PATH] required_csv={abandonment_receipt['required_csv_path']}")

    print("[STAGE] rebuild global and CONUS embeddings")
    old_hash = embedding.sha256_file(CURRENT_EMBEDDING)
    if old_hash != EXPECTED_OLD_EMBEDDING_SHA256:
        raise ValueError(f"Current embedding hash changed: {old_hash}")
    previous = pd.read_csv(CURRENT_EMBEDDING)
    embedding_progress = tqdm(desc="mode2 embedding", unit="stage")

    def embedding_update(status):
        embedding_progress.update(1)
        embedding_progress.set_postfix(**{key: value for key, value in status.items() if key != "path"})

    try:
        candidate = embedding.build_training_embedding(
            str(MODE_MERGED_DIR / "chunk_*.nc"),
            FEATURE_ROOT,
            PV_ALIGNMENT_CSV,
            landcover_path=MODE_RECLASS_PATH,
            row_domain_csv=CURRENT_EMBEDDING,
            years=(2018, 2020),
            chunk_size=CHUNK_SIZE,
            progress_callback=embedding_update,
        )
    finally:
        embedding_progress.close()
    validation = embedding.write_embedding_outputs(
        candidate,
        previous,
        candidate_path=NEXT_EMBEDDING,
        conus_path=CONUS_EMBEDDING,
        report_path=EMBEDDING_REPORT,
        expected_rows=EXPECTED_EMBEDDING_ROWS,
    )
    state["stages"]["embedding_validation"] = validation
    atomic_json(state, PIPELINE_MANIFEST)
    publication = embedding.publish_embedding_candidate(
        NEXT_EMBEDDING,
        FINAL_EMBEDDING,
    )
    observed_old_hash = embedding.sha256_file(CURRENT_EMBEDDING)
    if observed_old_hash != old_hash:
        raise RuntimeError("Legacy data/training_embedding.csv changed during 0819 publication")
    embedding_manifest = {
        "source_merged_chunks": str(MODE_MERGED_DIR),
        "global_rows": len(candidate),
        "columns": list(candidate.columns),
        "validation": validation,
        "publication": publication,
        "conus_path": str(CONUS_EMBEDDING),
        "legacy_training_embedding_sha256_before": old_hash,
        "legacy_training_embedding_sha256_after": observed_old_hash,
    }
    state["stages"]["embedding_publication"] = publication
    state["status"] = "complete"
    state["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["final_embedding"] = publication
    try:
        atomic_json(embedding_manifest, EMBEDDING_MANIFEST)
        atomic_json(state, PIPELINE_MANIFEST)
        atomic_json(state, REPO_MANIFEST)
    except Exception as metadata_error:
        state["status"] = "complete_with_metadata_error"
        state["metadata_sync_error"] = repr(metadata_error)
        try:
            atomic_json(state, PIPELINE_MANIFEST)
        except Exception:
            pass
        print(f"[QA] data_published=True metadata_sync_failed={metadata_error!r}")
        return
    print(f"[RESULT] MODE2_PIPELINE_COMPLETE manifest={PIPELINE_MANIFEST}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        try:
            failure = json.loads(PIPELINE_MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            failure = {"stages": {}}
        failure.update(
            {
                "status": "failed",
                "error": repr(error),
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        atomic_json(failure, PIPELINE_MANIFEST)
        print(f"[QA] MODE2_PIPELINE_FAILED error={error!r}")
        raise
