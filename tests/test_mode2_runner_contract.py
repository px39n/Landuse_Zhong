from __future__ import annotations

import ast
from pathlib import Path


RUNNER = Path("tools/run_mode2_end_to_end.py")
ARCHIVER = Path("tools/archive_mode2_outputs.py")
PREDICTION_RUNNER = Path("tools/run_prediction_mode_end_to_end.py")
ABANDONMENT_MODULE = Path("function/cropland_abandonment.py")
EMBEDDING_MODULE = Path("function/embedding_pipeline.py")


def test_runner_uses_original_d_drive_names_and_0819_publication() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "run_abandonment_detection(" in source
    assert "abandonment_receipt = abandonment.run_abandonment_detection(" in source
    assert "abandonment_chunk_root" in source
    assert "merged_chunk_root" in source
    assert "required_csv_path" in source
    assert "build_legacy_candidate_mask" not in source
    assert "write_legacy_equivalent_chunks" not in source
    assert "merge_landcover_into_legacy_chunks" not in source
    assert 'FINAL_EMBEDDING = REPO_ROOT / "data" / "aligned_for_training0819.csv"' in source
    assert 'NEXT_EMBEDDING = REPO_ROOT / "data" / "aligned_for_training0819.next.csv"' in source
    assert "publish_embedding_candidate" in source
    assert "landcover_path=MODE_RECLASS_PATH" in source
    assert "row_domain_csv=CURRENT_EMBEDDING" in source
    assert "EXPECTED_EMBEDDING_ROWS = 90_821" in source
    assert 'PIPELINE_MANIFEST = Path(r"D:\\xarray\\abandon_0819_pipeline_manifest.json")' in source
    assert 'PIPELINE_LOG = Path(r"D:\\xarray\\abandon_0819_pipeline.log")' in source


def test_runner_keeps_legacy_training_embedding_read_only() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "promote_training_embedding" not in source
    assert "BACKUP_EMBEDDING" not in source
    assert "Legacy data/training_embedding.csv changed" in source
    assert 'state["status"] = "complete_with_metadata_error"' in source


def test_archiver_writes_recovery_journal_before_each_move() -> None:
    source = ARCHIVER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "archive_manifest_20260819.in_progress.json" in source
    first_journal = source.index("_atomic_json(journal_path, manifest)")
    first_move = source.index("os.replace(source, destination)")
    assert first_journal < first_move
    assert 'records[index]["moved"] = True' in source
    assert 'record["verified"] = True' in source


def test_prediction_runner_uses_feature_receipt_paths_instead_of_featureless_hardcoded_targets() -> None:
    source = PREDICTION_RUNNER.read_text(encoding="utf-8")
    ast.parse(source)
    active = source[source.index("def run_prediction_mode_end_to_end(") :]
    assert "abandonment.run_abandonment_detection(**parameters)" in active
    assert "if not acceptance.get(\"accepted\")" in active
    assert "preflight(" not in active
    assert "combine_mode_prediction_features(" not in active
    assert "publish_feature_prediction_bundle(" not in active
    assert "archive_current_us_csv()" not in active
    assert "MODE_MERGED_DIR" not in active
    assert "CURRENT_US_CSV" not in active
    for option in (
        "--target-nc",
        "--window-year",
        "--start-year",
        "--current-end-year",
        "--init-cropland",
        "--extend-validation",
    ):
        assert option in active


def test_prediction_runner_consumes_companion_state_membership_frame_and_binds_california_subset_identity() -> None:
    detector = ABANDONMENT_MODULE.read_text(encoding="utf-8")
    publisher = EMBEDDING_MODULE.read_text(encoding="utf-8")
    assert "finalize_abandonment_prediction(" in detector
    assert "candidate, membership = build_prediction_embedding(" in publisher
    assert "membership_report = state_membership_summary(membership)" in publisher
    assert 'state_fips="06"' in publisher
    assert '"state_membership_sha256"' in publisher
    assert '"california_subset_sha256"' in publisher
    assert '"california_csv_published": False' in publisher
    assert "_commit_exact_prediction_outputs(staged)" in publisher
    assert "_reject_legacy_featureless_publication()" in PREDICTION_RUNNER.read_text(encoding="utf-8")
