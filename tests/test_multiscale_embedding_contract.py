from __future__ import annotations

import ast
from pathlib import Path


TOOL = Path("tools/build_multiscale_embedding.py")


def test_multiscale_embedding_has_all_source_partitions_and_trace_columns():
    source = TOOL.read_text(encoding="utf-8")
    ast.parse(source)
    for key in ("esa_cci_300m", "lcmap_c13_30m", "xie_2024_30m"):
        assert key in source
    for column in (
        "source_key",
        "source_valid_through_year",
        "state_fips",
        "feature_grid_row",
        "feature_grid_col",
        "pixel_area_m2",
    ):
        assert f'"{column}"' in source
    assert 'DATASET_NAME = "aligned_for_contrasting0819"' in source
    assert 'f"state_fips={fips}"' in source
    assert "embedding_index.csv" in source
    assert "embedding_manifest.json" in source


def test_xie_uses_2018_features_and_lcmap_uses_native_crs():
    source = TOOL.read_text(encoding="utf-8")
    assert '"target_year": 2018' in source
    assert '"valid_through_year": 2018' in source
    assert "Transformer.from_crs" in source
    assert "rasterize" in source


def test_embedding_requires_source_acceptance_and_random_feature_mapping_qa():
    source = TOOL.read_text(encoding="utf-8")
    assert 'source_root / "manifests" / "source_manifest.json"' in source
    assert "validate_embedding_feature_mapping(source_key)" in source
    assert 'sample_size: int = 2_000' in source
    assert 'feature_sample_size: int = 10' in source
    assert "expected_feature_grid_row" in source
    assert "feature_mismatches" in source
    assert 'index_path.with_suffix(".tmp.csv")' in source
    assert 'output_root.rglob("*.tmp.csv")' in source
    assert "Embedding chunk completeness failed" in source
    assert 'processing_paths = chunk_paths[shard_index::shard_count]' in source
    assert '"--parts-only"' in source
    assert '"status": "parts_complete"' in source
    assert "row_key * source_chunk_size" in source
    assert "col_key * source_chunk_size" in source
    assert "row_key * 1024" not in source
