from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import nbformat

from tools.update_process_notebook_contract import (
    LEGACY_CODE_INDICES,
    MODE_RESAMPLE_CELL_ID,
    MODE_RESAMPLE_INDEX,
    NEW_CELL_ORDER,
    NOTEBOOK_PATH,
    build_mode_resample_cell,
    comment_legacy_source_text,
    load_head_notebook,
    transform_notebook,
)


MODULE_PATH = Path("function") / "cropland_abandonment.py"
SPEC = importlib.util.spec_from_file_location("cropland_abandonment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


REQUIRED_PARAMETERS = {
    "INPUT_SOURCE_KEY": "esa_cci_300m",
    "INPUT_YEAR_RANGE": "(1992, 2022)",
    "CALIFORNIA_BOUNDS": "32.5",
    "PRIMARY_GRID": "1km_mode",
    "BASELINE_CROP_YEARS": "(1992, 1997)",
    "MIN_ABANDONMENT_YEARS": "5",
    "REQUIRED_THROUGH_YEAR": "2020",
    "ANALYSIS_END_YEAR": "2020",
    "RECULTIVATION_YEARS": "1",
    "TEMPORAL_FILTER_WINDOW": "5",
    "NETCDF_OPEN_CHUNKS": "{}",
    "EXECUTION_MODE": "full",
    "SHOW_PROGRESS": "True",
    "PRINT_STAGE_SUMMARY": "True",
    "PV_DEDUP_RETAIN_MASK_SOURCE": "None",
    "FEATURE_COMPLETE_RETAIN_MASK_SOURCE": "None",
    "XIE_SOURCE_PAGE_URL": "https://www.glbrc.org/data-and-tools/glbrc-data-sets/cropland-abandonment-between-1986-and-2018-across-united-states",
    "XIE_DIRECT_DOWNLOAD_URL": "https://atlas.glbrc.org/files/public/abanMaps_5yr_v4__abanDef2_noRec_noUr.tar.gz",
    "XIE_ARCHIVE_LOCAL_PATH": r'D:\xarray\project_temdata\xie_2024_conus_abandonment_30m.tar.gz',
    "XIE_ARCHIVE_SHA256": "B2EC7094B20A497259618CD5F3EF5CBD4AB3A06D4C12127D006D61D3F6CCEAB8",
    "XIE_ARCHIVE_SIZE_BYTES": "198383647",
    "XIE_ARCHIVE_MEMBER": "abanMaps_5yr_v4__abanDef2_noRec_noUr/abanMaps_5yr_v4__abanDef2_noRec_noUr.tif",
    "XIE_TIFF_LOCAL_PATH": r'D:\xarray\project_temdata\abanMaps_5yr_v4__abanDef2_noRec_noUr\abanMaps_5yr_v4__abanDef2_noRec_noUr.tif',
    "XIE_TIFF_SHA256": "97936653440C3F288B2D8E2F851DBE16904212767EE46C73991FAA5451BDC404",
    "XIE_TIFF_SIZE_BYTES": "244084474",
    "XIE_VALID_YEAR_MIN": "1991",
    "XIE_VALID_YEAR_MAX": "2014",
}


def load_notebook():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cells_by_id = {cell["id"]: cell for cell in nb.cells}
    return nb, cells_by_id


def test_notebook_validates_and_has_unique_ids() -> None:
    nb, _ = load_notebook()
    assert nb.nbformat_minor >= 5
    ids = [cell.get("id") for cell in nb.cells]
    assert all(ids)
    assert len(ids) == len(set(ids))
    nbformat.validate(nb)


def test_new_cells_exist_once_and_in_exact_order() -> None:
    nb, _ = load_notebook()
    ids = [cell["id"] for cell in nb.cells]
    assert ids[-len(NEW_CELL_ORDER) :] == NEW_CELL_ORDER
    assert ids.index("proc-abandon-section") == 43
    assert not ids[44].startswith("s0-")
    for cell_id in NEW_CELL_ORDER:
        assert ids.count(cell_id) == 1


def test_s0_tail_has_readable_heading_hierarchy_and_structure_table() -> None:
    nb, cells = load_notebook()
    assert str(cells["s0-abandon-overview"]["source"]).startswith("## 3、美国本土耕地撂荒审计")
    for cell_id in [
        "s0-title-config",
        "s0-title-preprocess",
        "s0-title-detection",
        "s0-title-statistics",
        "s0-title-external",
    ]:
        assert cells[cell_id]["cell_type"] == "markdown"
        assert str(cells[cell_id]["source"]).startswith("### （")
    overview = str(cells["s0-abandon-overview"]["source"])
    assert "| 主要结构 | 对应代码 Cell | 核心职责 |" in overview
    assert "s0-abandon-imports" in overview
    assert "s0-abandon-publish" in overview
    assert nb.cells[-1]["id"] == "public-abandonment-run"


def test_mode_chain_tail_is_end_to_end_and_uses_isolated_paths() -> None:
    nb, cells = load_notebook()
    assert str(cells["mode-chain-overview"]["source"]).startswith("## 4、Mode修正后的全球旧版等价撂荒重建")
    for cell_id in [
        "mode-chain-monitor-title",
        "mode-chain-detection-title",
        "mode-chain-qa-title",
        "mode-chain-merge-title",
        "mode-chain-handoff-title",
    ]:
        assert cells[cell_id]["cell_type"] == "markdown"
        assert str(cells[cell_id]["source"]).startswith("### （")
    params = str(cells["mode-chain-parameters"]["source"])
    assert 'MODE_RECLASS_PATH = Path(r"D:\\xarray\\reclass_lccs_1km.nc")' in params
    assert 'MODE_MASK_PATH = Path(r"D:\\xarray\\final_mask_1km_new.nc")' in params
    assert 'MODE_ABANDON_DIR = Path(r"D:\\xarray\\abandon_2")' in params
    assert 'MODE_MERGED_DIR = Path(r"D:\\xarray\\merged_chunk_2")' in params
    assert "MODE_CHUNK_SIZE = 500" in params
    assert "from tqdm.auto import tqdm" in params
    monitor = str(cells["mode-chain-monitor"]["source"])
    assert "stable_seconds=MODE_MONITOR_STABLE_SECONDS" in monitor
    merge = str(cells["mode-chain-merge"]["source"])
    assert "merge_landcover_into_legacy_chunks" in merge
    run_all = str(cells["mode-chain-run-all"]["source"])
    for function_name in [
        "run_mode_readiness()",
        "run_mode_mask()",
        "run_mode_detection()",
        "run_mode_chunk_qa()",
        "run_mode_merge()",
    ]:
        assert function_name in run_all
    assert nb.cells[-1]["id"] == "public-abandonment-run"


def test_multiscale_tail_exposes_300m_30m_entry_only() -> None:
    nb, cells = load_notebook()
    assert str(cells["multiscale-overview"]["source"]).startswith("## 5、300m/30m多源运行入口")
    assert str(cells["multiscale-parameters-title"]["source"]).startswith("### （1）")
    assert str(cells["multiscale-run-title"]["source"]).startswith("### （2）")
    params = str(cells["multiscale-parameters"]["source"])
    assert 'from tqdm.auto import tqdm' in params
    assert 'MULTISCALE_PRIMARY_300M_SOURCE = Path(r"D:\\xarray\\reclass_lccs_300m_esa_cci.nc")' in params
    assert 'MULTISCALE_LCMAP_30M_SOURCE = Path(r"D:\\xarray\\reclass_lccs_30m_lcmap_c13.nc")' in params
    assert "from tools.run_multiscale_abandonment import (" in params
    assert 'MULTISCALE_XIE_30M_SOURCE = Path(r"D:\\xarray\\project_temdata\\abanMaps_5yr_v4__abanDef2_noRec_noUr\\abanMaps_5yr_v4__abanDef2_noRec_noUr.tif")' in params
    assert "run_multiscale_abandonment_end_to_end" in params
    run_all = str(cells["multiscale-run-all"]["source"])
    assert "run_multiscale_abandonment_end_to_end(" in run_all
    assert "request" not in run_all
    assert "execution_mode=MULTISCALE_EXECUTION_MODE" in run_all
    assert nb.cells[-3]["id"] == "public-abandonment-overview"


def test_public_abandonment_tail_uses_the_public_api_only() -> None:
    nb, cells = load_notebook()
    assert str(cells["public-abandonment-overview"]["source"]).startswith("## 6、公共本地撂荒入口")
    assert str(cells["public-abandonment-run-title"]["source"]).startswith("### （1）")
    source = str(cells["public-abandonment-run"]["source"])
    assert 'from function.cropland_abandonment import run_abandonment_detection' in source
    assert 'PUBLIC_TARGET_NC = Path(r"D:\\xarray\\reclass_lccs_1km.nc")' in source
    assert "PUBLIC_WINDOW_YEAR = 5" in source
    assert "PUBLIC_START_YEAR = 1992" in source
    assert "PUBLIC_CURRENT_END_YEAR = 2020" in source
    assert "PUBLIC_INIT_CROPLAND = 2" in source
    assert "PUBLIC_EXTEND_VALIDATION = True" in source
    assert "run_abandonment_detection(" in source
    assert "abandonment_chunk_root" in source
    assert "merged_chunk_root" in source
    assert "required_csv_path" in source
    assert "required_acceptance_path" in source
    assert nb.cells[-1]["id"] == "public-abandonment-run"


def test_all_new_code_cells_compile() -> None:
    _, cells = load_notebook()
    for cell_id in NEW_CELL_ORDER:
        cell = cells[cell_id]
        if cell["cell_type"] == "code":
            compile(str(cell["source"]), f"Process.ipynb#{cell_id}", "exec")


def test_global_resampling_cell_is_active_mode_and_preserves_wrong_lines_as_comments() -> None:
    nb, cells = load_notebook()
    cell = cells[MODE_RESAMPLE_CELL_ID]
    source = str(cell["source"])
    assert nb.cells[MODE_RESAMPLE_INDEX]["id"] == MODE_RESAMPLE_CELL_ID
    assert cell["cell_type"] == "code"
    assert not source.startswith("# LEGACY_DISABLED:")
    assert 'input_path = Path("output/recalss_lccs.nc").resolve()' in source
    assert 'output_dir = Path(r"D:\\xarray")' in source
    assert 'reclassify_esa_cci(' not in source
    assert 'aggregate_categorical_mode(' in source
    assert '# reduced_ds = ds.coarsen(lat=factor, lon=factor, boundary="trim").reduce(np.nanmean)' in source
    assert '# reduced_ds = reduced_ds.round()' in source
    active_lines = [line.strip() for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert not any("np.nanmean" in line for line in active_lines)
    assert not any("reduced_ds.round" in line for line in active_lines)
    compile(source, f"Process.ipynb#{MODE_RESAMPLE_CELL_ID}", "exec")


def test_imports_and_external_check_use_module_api_only() -> None:
    _, cells = load_notebook()
    imports_source = str(cells["s0-abandon-imports"]["source"])
    assert "from tqdm.auto import tqdm" in imports_source
    assert "SIMPLIFIED_ESA_CLASSES" in imports_source
    assert "spec_from_file_location" in imports_source
    assert "from function.cropland_abandonment import" not in imports_source
    for func_name in [
        "fetch_reference_raster",
        "validate_reference_raster",
        "open_reference_abandonment_year",
        "align_reference_abandonment",
        "compare_reference_abandonment",
    ]:
        assert func_name in imports_source

    external_source = str(cells["s0-abandon-external-check"]["source"])
    for func_name in [
        "fetch_reference_raster(",
        "validate_reference_raster(",
        "open_reference_abandonment_year(",
        "align_reference_abandonment(",
        "compare_reference_abandonment(",
    ]:
        assert func_name in external_source
    for forbidden in ["requests.get", "src.read(1)", "xr.open_dataarray", ".interp(", ".rio.reproject_match("]:
        assert forbidden not in external_source


def test_parameter_cell_contains_exact_xie_contract_values() -> None:
    _, cells = load_notebook()
    params_source = str(cells["s0-abandon-parameters"]["source"])
    for parameter, expected in REQUIRED_PARAMETERS.items():
        assert parameter in params_source
        assert expected in params_source
    assert '"esa_cci_300m": {' in params_source
    assert '"xie_2024_30m": {' in params_source
    assert 'D_ACTIVE_RUN / "sources" / source_key' in params_source
    assert "NetCDF/xarray" in params_source
    assert "GeoTIFF/rasterio-rioxarray" in params_source
    assert "IMPLEMENTATION_SHA256" in params_source
    assert "NOTEBOOK_CONTRACT_SHA256" in params_source


def test_notebook_calls_match_actual_module_signatures() -> None:
    _, cells = load_notebook()
    signatures = {
        "write_abandonment_chunks": inspect.signature(MODULE.write_abandonment_chunks),
        "summarize_key_numbers": inspect.signature(MODULE.summarize_key_numbers),
        "publish_validation_results": inspect.signature(MODULE.publish_validation_results),
        "fetch_reference_raster": inspect.signature(MODULE.fetch_reference_raster),
        "validate_reference_raster": inspect.signature(MODULE.validate_reference_raster),
        "open_reference_abandonment_year": inspect.signature(MODULE.open_reference_abandonment_year),
        "align_reference_abandonment": inspect.signature(MODULE.align_reference_abandonment),
        "compare_reference_abandonment": inspect.signature(MODULE.compare_reference_abandonment),
    }
    assert "output_dir" in signatures["write_abandonment_chunks"].parameters
    assert "run_id" in signatures["publish_validation_results"].parameters
    assert "repo_result_root" in signatures["publish_validation_results"].parameters

    write_source = str(cells["s0-abandon-write-chunks"]["source"])
    publish_source = str(cells["s0-abandon-publish"]["source"])
    keynum_source = str(cells["s0-abandon-key-numbers"]["source"])
    external_source = str(cells["s0-abandon-external-check"]["source"])
    detection_source = str(cells["s0-abandon-detection"]["source"])
    poisson_source = str(cells["s0-abandon-poisson"]["source"])

    assert "output_dir=" in write_source
    assert "chunk_slices=" in write_source
    assert "run_root=" not in write_source
    assert "repo_result_root=" in publish_source
    assert "run_id=" in publish_source
    assert "repo_root=" not in publish_source
    assert "state_names=" not in keynum_source
    assert "eligible_cropland" in detection_source
    assert "schema=\"simplified9\"" in detection_source
    assert "fetch_reference_raster(" in external_source

    preflight_source = str(cells["s0-abandon-preflight"]["source"])
    reclass_source = str(cells["s0-abandon-reclass"]["source"])
    grid_source = str(cells["s0-abandon-grid-branches"]["source"])
    roundtrip_source = str(cells["s0-abandon-roundtrip"]["source"])
    merge_source = str(cells["s0-abandon-merge-chunks"]["source"])
    assert "raw_var.isel(time=0).values" not in preflight_source
    assert "reclass_year.values" not in reclass_source
    assert "annual_mode.values" not in grid_source
    assert ".eq(1)" not in detection_source
    assert "merged_dataset" in roundtrip_source
    assert "chunk_paths\"]" not in roundtrip_source
    assert '.reindex(' in merge_source
    assert 'S0_CONTEXT["primary_detection"]["lat"]' in merge_source
    assert "merged_ds.load()" in merge_source
    assert "merged_ds.close()" in merge_source
    assert "merged_tmp.unlink(missing_ok=True)" in merge_source

    assert "archive_source=" in external_source
    assert "archive_member_name=XIE_ARCHIVE_MEMBER" in external_source
    assert "extracted_raster_path=XIE_TIFF_LOCAL_PATH" in external_source
    assert "D_RUN_SUBDIRS[\"intermediate\"] / XIE_TIFF_LOCAL_PATH.name" not in external_source
    assert "D_PRIMARY_SUBDIRS[\"chunks\"]" in write_source
    assert 'D_XIE_SUBDIRS["diagnostics"] / "xie_external_comparison_overall.csv"' in external_source
    assert 'D_XIE_SUBDIRS["diagnostics"] / "xie_external_comparison_by_state.csv"' in external_source
    assert 'D_XIE_SUBDIRS["diagnostics"] / "xie_external_comparison_by_year.csv"' in external_source
    assert 'D_PRIMARY_SUBDIRS["intermediate"] / f"{INPUT_SOURCE_KEY}_ca_lccs.nc"' in reclass_source
    assert 'D_PRIMARY_SUBDIRS["intermediate"] / f"{INPUT_SOURCE_KEY}_ca_lccs_1km_mode.nc"' in grid_source
    assert "california_1km_ds.load()" in grid_source
    assert 'D_XIE_SUBDIRS["intermediate"] / "xie_2024_30m_ca_lccs.nc"' in external_source
    assert 'california_validation_chain.json' in external_source
    assert "default=json_default" in external_source
    assert 'EXECUTION_MODE == "california"' in reclass_source
    assert "comparison_years=comparison_years" in external_source
    assert 'method="mode"' in external_source
    assert "external_comparison_years" in publish_source
    assert "source_runs" in publish_source
    assert "source_grid_contracts" in publish_source
    assert '"abandonment_external_comparison"' not in publish_source
    assert '"comparison_summary"' in publish_source
    assert 'xr.open_dataset(input_path, chunks=NETCDF_OPEN_CHUNKS)' in preflight_source
    assert "set(CROPLAND_CODES) != set(SIMPLIFIED_ESA_CLASSES[1])" in preflight_source
    assert "key_number_reconciliation" in keynum_source
    assert "pv_deduplicated" in keynum_source
    assert "complete_features" in keynum_source
    assert "downstream_stage_complete" in keynum_source
    assert ".sum().compute().item()" in poisson_source
    assert "qa_gate" in publish_source
    assert '"downstream_stage_complete"' in publish_source
    assert "manifest_identity" in publish_source
    assert "Smoke runs are diagnostic-only" in publish_source


def test_source_isolation_and_grid_contracts_are_exposed_across_cells() -> None:
    _, cells = load_notebook()
    overview_source = str(cells["s0-abandon-overview"]["source"])
    params_source = str(cells["s0-abandon-parameters"]["source"])
    preflight_source = str(cells["s0-abandon-preflight"]["source"])
    external_source = str(cells["s0-abandon-external-check"]["source"])
    publish_source = str(cells["s0-abandon-publish"]["source"])

    assert 'D_ACTIVE_RUN / "sources"' in overview_source
    assert "ESA-CCI" in overview_source
    assert "GLBRC/Xie" in overview_source
    assert "D_SOURCE_RUN_ROOTS" in params_source
    assert "D_PRIMARY_SUBDIRS" in params_source
    assert "D_XIE_SUBDIRS" in params_source
    assert "SOURCE_GRID_CONTRACTS" in params_source
    assert "netcdf_grid_contract" in preflight_source
    assert "geotiff_grid_contract" in external_source
    assert "source_runs" in publish_source
    assert "source_grid_contracts" in publish_source
    assert "External comparison years" in publish_source


def test_print_prefixes_and_no_hidden_d_drive_literals_outside_parameter_cell() -> None:
    _, cells = load_notebook()
    combined = "\n".join(str(cells[cell_id]["source"]) for cell_id in NEW_CELL_ORDER)
    for prefix in ["[CONFIG]", "[PATH]", "[INPUT]", "[AOI]", "[STAGE]", "[CHUNK]", "[QA]", "[RESULT]", "[PUBLISH]"]:
        assert prefix in combined
    for cell_id in NEW_CELL_ORDER:
        if cell_id in {
            "s0-abandon-parameters",
            "mode-chain-parameters",
            "multiscale-parameters",
            "public-abandonment-run",
        }:
            continue
        source = str(cells[cell_id]["source"])
        assert r"D:\xarray" not in source


def test_progress_contract_reports_counts_and_area_without_pixel_logging() -> None:
    _, cells = load_notebook()
    for cell_id in [
        "s0-abandon-reclass",
        "s0-abandon-grid-branches",
        "s0-abandon-write-chunks",
        "s0-abandon-merge-chunks",
        "s0-abandon-robustness",
        "s0-abandon-external-check",
    ]:
        source = str(cells[cell_id]["source"])
        assert "set_postfix" in source
        assert "failed=" in source
        assert "valid_pixels=" in source
        assert "area_ha=" in source
    assert "for pixel" not in "\n".join(str(cells[cell_id]["source"]) for cell_id in NEW_CELL_ORDER)


def test_transformer_is_idempotent_and_restores_legacy_outputs(tmp_path: Path) -> None:
    head_nb = load_head_notebook(NOTEBOOK_PATH)
    assert head_nb is not None
    temp_notebook = tmp_path / "Process.ipynb"
    nbformat.write(head_nb, temp_notebook)

    first = transform_notebook(temp_notebook, head_reference=head_nb)
    second = transform_notebook(temp_notebook, head_reference=head_nb)
    first_ids = [cell["id"] for cell in first.cells]
    second_ids = [cell["id"] for cell in second.cells]
    assert first_ids == second_ids
    assert second_ids[-len(NEW_CELL_ORDER) :] == NEW_CELL_ORDER
    for cell_id in NEW_CELL_ORDER:
        assert second_ids.count(cell_id) == 1

    for original_index, legacy_id in LEGACY_CODE_INDICES.items():
        legacy_cell = next(cell for cell in second.cells if cell.get("id") == legacy_id)
        assert str(legacy_cell["source"]).startswith("# LEGACY_DISABLED:")
        head_outputs = head_nb.cells[original_index].get("outputs", [])
        if head_outputs:
            assert legacy_cell.get("outputs", []) == head_outputs
            assert legacy_cell.get("execution_count") == head_nb.cells[original_index].get("execution_count")


def test_transformer_recovers_empty_notebook_atomically(tmp_path: Path) -> None:
    head_nb = load_head_notebook(NOTEBOOK_PATH)
    assert head_nb is not None
    empty_notebook = tmp_path / "Process.ipynb"
    empty_notebook.write_text("", encoding="utf-8")
    recovered = transform_notebook(empty_notebook, head_reference=head_nb)
    assert len(recovered.cells) == 72 + len(NEW_CELL_ORDER)
    assert recovered.cells[MODE_RESAMPLE_INDEX]["id"] == MODE_RESAMPLE_CELL_ID
    assert recovered.cells[-1]["id"] == "public-abandonment-run"
    assert empty_notebook.stat().st_size > 0
    assert not empty_notebook.with_suffix(".tmp.ipynb").exists()
    nbformat.validate(nbformat.read(empty_notebook, as_version=4))


def test_current_notebook_legacy_outputs_match_head_reference() -> None:
    nb, cells = load_notebook()
    head_nb = load_head_notebook(NOTEBOOK_PATH)
    assert head_nb is not None
    for original_index, legacy_id in LEGACY_CODE_INDICES.items():
        legacy_cell = cells[legacy_id]
        head_cell = head_nb.cells[original_index]
        assert str(legacy_cell["source"]).startswith("# LEGACY_DISABLED:")
        if head_cell.get("outputs"):
            assert legacy_cell.get("outputs", []) == head_cell.get("outputs", [])
            assert legacy_cell.get("execution_count") == head_cell.get("execution_count")


def test_all_original_cells_are_preserved_in_order_and_content() -> None:
    nb, _ = load_notebook()
    head_nb = load_head_notebook(NOTEBOOK_PATH)
    assert head_nb is not None
    existing_cells = nb.cells[: -len(NEW_CELL_ORDER)]
    extra_cells = [cell for cell in existing_cells if cell.get("id") == "5c334980"]
    assert len(extra_cells) == 1
    assert str(extra_cells[0]["source"]).strip() == "ds"
    comparable_cells = [cell for cell in existing_cells if cell.get("id") != "5c334980"]
    assert len(comparable_cells) == len(head_nb.cells) == 72
    for index, (current, original) in enumerate(zip(comparable_cells, head_nb.cells)):
        if index == MODE_RESAMPLE_INDEX:
            expected_source = str(build_mode_resample_cell()["source"])
        elif index in LEGACY_CODE_INDICES:
            expected_source = comment_legacy_source_text(str(original["source"]))
        else:
            expected_source = str(original["source"])
        assert str(current["source"]) == expected_source
        if index == MODE_RESAMPLE_INDEX:
            assert current.get("outputs", []) == []
            assert current.get("execution_count") is None
        else:
            assert current.get("outputs", []) == original.get("outputs", [])
            assert current.get("execution_count") == original.get("execution_count")


def test_raw_json_shape_is_expected() -> None:
    payload = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert payload["nbformat_minor"] >= 5
    assert len(payload["cells"]) == 73 + len(NEW_CELL_ORDER)
    assert payload["cells"][43]["id"] == "proc-abandon-section"
    assert payload["cells"][MODE_RESAMPLE_INDEX]["id"] == MODE_RESAMPLE_CELL_ID
    assert [cell["id"] for cell in payload["cells"][-len(NEW_CELL_ORDER) :]] == NEW_CELL_ORDER
