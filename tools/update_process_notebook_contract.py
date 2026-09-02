from __future__ import annotations

import copy
import hashlib
import re
import subprocess
import unicodedata
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


NOTEBOOK_PATH = Path("Process.ipynb")
PROC_INDEX = 43
MODE_RESAMPLE_INDEX = 49
MODE_RESAMPLE_CELL_ID = "mode-abandon-categorical-resample"
PRESERVED_EXTRA_CELL_IDS = {"5c334980"}
LEGACY_CODE_INDICES = {
    45: "legacy-abandon-mask-code",
    47: "legacy-abandon-mask-check",
    51: "legacy-abandon-regex-detect",
    52: "legacy-abandon-chunk-preview",
    54: "legacy-abandon-sample-check",
    56: "legacy-abandon-integrity-check",
    58: "legacy-abandon-spatial-check",
    60: "legacy-abandon-temporal-check",
    62: "legacy-abandon-duration-stats",
}
NEW_CELL_ORDER = [
    "s0-abandon-overview",
    "s0-title-config",
    "s0-abandon-imports",
    "s0-abandon-parameters",
    "s0-abandon-preflight",
    "s0-title-preprocess",
    "s0-abandon-conus-mask",
    "s0-abandon-reclass",
    "s0-abandon-grid-branches",
    "s0-title-detection",
    "s0-abandon-detection",
    "s0-abandon-write-chunks",
    "s0-abandon-merge-chunks",
    "s0-abandon-roundtrip",
    "s0-title-statistics",
    "s0-abandon-key-numbers",
    "s0-abandon-robustness",
    "s0-abandon-poisson",
    "s0-title-external",
    "s0-abandon-external-check",
    "s0-abandon-publish",
    "mode-chain-overview",
    "mode-chain-monitor-title",
    "mode-chain-parameters",
    "mode-chain-monitor",
    "mode-chain-detection-title",
    "mode-chain-mask",
    "mode-chain-detect-chunks",
    "mode-chain-qa-title",
    "mode-chain-qa",
    "mode-chain-merge-title",
    "mode-chain-merge",
    "mode-chain-handoff-title",
    "mode-chain-run-all",
    "multiscale-overview",
    "multiscale-parameters-title",
    "multiscale-parameters",
    "multiscale-run-title",
    "multiscale-run-all",
    "public-abandonment-overview",
    "public-abandonment-run-title",
    "public-abandonment-run",
]


def stable_id(index: int, cell: nbformat.NotebookNode) -> str:
    if index == PROC_INDEX:
        return "proc-abandon-section"
    if index == MODE_RESAMPLE_INDEX:
        return MODE_RESAMPLE_CELL_ID
    if index in LEGACY_CODE_INDICES:
        return LEGACY_CODE_INDICES[index]

    source = "".join(cell.get("source", []))
    first_line = next((line.strip() for line in source.splitlines() if line.strip()), cell["cell_type"])
    slug = unicodedata.normalize("NFKD", first_line).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-").lower() or cell["cell_type"]
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:40]}-{index:02d}-{digest}"[:64]


def _normalize_stream_outputs(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    # The historical notebook contains one pre-nbformat-4.5 stream output
    # without the required name field. Preserve its text and add only the
    # standard stdout metadata needed for nbformat validation.
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream" and not output.get("name"):
                output["name"] = "stdout"
            if output_type in {"display_data", "execute_result"} and "metadata" not in output:
                output["metadata"] = {}
            if output_type == "execute_result" and "execution_count" not in output:
                output["execution_count"] = cell.get("execution_count")
    return notebook


def load_head_notebook(repo_path: Path = NOTEBOOK_PATH) -> nbformat.NotebookNode | None:
    relative = repo_path.as_posix()
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    notebook = _normalize_stream_outputs(nbformat.reads(result.stdout, as_version=4))
    # After the generated notebook is committed, HEAD contains the managed
    # appendix as well as the one preserved exploratory cell.  The generator's
    # immutable comparison baseline is still the 72 original cells, so derive
    # that baseline explicitly instead of assuming HEAD predates generation.
    managed_ids = set(NEW_CELL_ORDER) | PRESERVED_EXTRA_CELL_IDS
    legacy_ids = set(LEGACY_CODE_INDICES.values())
    baseline_cells = []
    for cell in notebook.cells:
        cell_id = cell.get("id")
        if cell_id in managed_ids:
            continue
        baseline_cell = copy.deepcopy(cell)
        if cell_id in legacy_ids:
            baseline_cell["source"] = restore_legacy_source_text(
                str(baseline_cell.get("source", ""))
            )
        baseline_cells.append(baseline_cell)
    notebook.cells = baseline_cells
    return notebook


def comment_legacy_source_text(source_text: str) -> str:
    lines = source_text.splitlines(keepends=True)
    commented = ["# LEGACY_DISABLED: original content retained for side-by-side comparison\n"]
    for line in lines:
        body = line[:-1] if line.endswith("\n") else line
        suffix = "\n" if line.endswith("\n") else ""
        commented.append(f"# {body}{suffix}" if body else f"#{suffix}")
    return "".join(commented)


def restore_legacy_source_text(source_text: str) -> str:
    """Reverse ``comment_legacy_source_text`` for a committed legacy cell."""
    marker = "# LEGACY_DISABLED: original content retained for side-by-side comparison"
    lines = source_text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != marker:
        return source_text

    restored: list[str] = []
    for line in lines[1:]:
        if line.startswith("# "):
            restored.append(line[2:])
        elif line.startswith("#"):
            restored.append(line[1:])
        else:
            raise ValueError("Committed legacy cell contains a non-commented line")
    return "".join(restored)


def markdown(text: str, cell_id: str) -> nbformat.NotebookNode:
    cell = new_markdown_cell(dedent(text).strip("\n") + "\n")
    cell["id"] = cell_id
    return cell


def code(text: str, cell_id: str) -> nbformat.NotebookNode:
    cell = new_code_cell(dedent(text).strip("\n") + "\n")
    cell["id"] = cell_id
    return cell


def build_mode_resample_cell() -> nbformat.NotebookNode:
    return code(
        r'''
        import importlib.util
        import os
        from pathlib import Path

        import numpy as np
        import xarray as xr

        # === 参数：旧版全球重采样的众数修正版 ===
        input_path = Path("output/recalss_lccs.nc").resolve()
        output_dir = Path(r"D:\xarray")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "reclass_lccs_1km_mode.nc"
        factor = 3  # 10 arc-second (~300 m) → 30 arc-second (~1 km)
        source_chunks = {"time": 1, "lat": 2592, "lon": 5184}

        # 直接加载经过单元测试的分类众数函数，避免触发 function/__init__.py 的其它流程。
        module_path = Path("function/cropland_abandonment.py").resolve()
        module_spec = importlib.util.spec_from_file_location("mode_resample_module", module_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Cannot load categorical resampling module: {module_path}")
        mode_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(mode_module)
        aggregate_categorical_mode = mode_module.aggregate_categorical_mode

        print(f"[PATH] input={input_path}")
        print(f"[PATH] output={output_path}")
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        ds = xr.open_dataset(input_path, chunks=source_chunks)
        print(
            f"[INPUT] source_resolution_deg="
            f"({float(abs(ds.lat[1] - ds.lat[0])):.7f}, {float(abs(ds.lon[1] - ds.lon[0])):.7f})"
        )
        print(f"[INPUT] chunks={ds['lccs_class'].chunks}")

        # 错误实现仅作为审计证据保留：分类编码不能做均值后四舍五入。
        # reduced_ds = ds.coarsen(lat=factor, lon=factor, boundary="trim").reduce(np.nanmean)
        # reduced_ds = reduced_ds.round()

        # input_path 已是正确的0–9简化分类；这里只修正空间聚合方法。
        # 正确实现：每个3×3分类窗口取众数；平票时优先中心像元，否则取最小类别码。
        landcover_1km = aggregate_categorical_mode(
            ds["lccs_class"],
            lat_factor=factor,
            lon_factor=factor,
            output_name="lccs_class",
        )
        if landcover_1km.dtype != np.uint8:
            landcover_1km = landcover_1km.astype(np.uint8)
        reduced_ds = landcover_1km.to_dataset(name="lccs_class")
        reduced_ds.attrs.update(
            {
                "source_path": str(input_path),
                "spatial_aggregation": "3x3_categorical_mode",
                "source_resolution": "10 arc-second (~300 m)",
                "target_resolution": "30 arc-second (~1 km)",
            }
        )

        encoding = {
            "lccs_class": {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "uint8",
                "_FillValue": 255,
                "chunksizes": (
                    1,
                    source_chunks["lat"] // factor,
                    source_chunks["lon"] // factor,
                ),
            }
        }
        from dask.diagnostics import ProgressBar

        tmp_path = output_path.with_name(output_path.name + ".tmp")
        with ProgressBar():
            reduced_ds.to_netcdf(tmp_path, encoding=encoding, compute=True)
        os.replace(tmp_path, output_path)
        print(f"[STAGE] categorical mode resampling completed: {output_path}")
        ''',
        MODE_RESAMPLE_CELL_ID,
    )


def build_new_cells() -> list[nbformat.NotebookNode]:
    return [
        markdown(
            """
            ## 3、美国本土耕地撂荒审计（S0 参数化流程）

            本节统一承接前文 legacy 撂荒流程的模块化替代方案，并固定放置在 Notebook 末尾，避免与原始探索代码交错。

            | 主要结构 | 对应代码 Cell | 核心职责 |
            |---|---|---|
            | （1）环境、参数与预检 | `s0-abandon-imports` → `s0-abandon-preflight` | 显式科学参数、路径、run_id、31年时间轴与输入追踪 |
            | （2）研究区与分类预处理 | `s0-abandon-conus-mask` → `s0-abandon-grid-branches` | CONUS/California范围、ESA重分类、五年滤波、300 m与1 km众数分支 |
            | （3）检测、分块与落盘QA | `s0-abandon-detection` → `s0-abandon-roundtrip` | 2020状态检测、原子chunk、合并与逐变量重读校验 |
            | （4）Key numbers与稳健性 | `s0-abandon-key-numbers` → `s0-abandon-poisson` | 测地面积、州/年份守恒、尺度/阈值情景与计数模型 |
            | （5）外部验证与发布 | `s0-abandon-external-check` → `s0-abandon-publish` | GLBRC/Xie可比年份验证、manifest与验收摘要 |

            科学主口径为：`BASELINE_CROP_YEARS` 内稳定耕地、五年时间多数滤波、至少 `MIN_ABANDONMENT_YEARS` 连续退出，并在 `REQUIRED_THROUGH_YEAR` 判断是否仍维持撂荒。分类面积按各目标网格的WGS84测地单元面积计算，禁止使用类别均值或固定平方公里近似。

            大型结果统一写入 `D_ACTIVE_RUN / "sources" / <source_key>`，ESA-CCI NetCDF与GLBRC/Xie GeoTIFF在同一 `RUN_ID` 下按数据源隔离；仓库内只发布通过验收的小型manifest、CSV和报告。
            """,
            "s0-abandon-overview",
        ),
        markdown(
            """
            ### （1）环境、参数接口与输入预检

            先加载模块与 `tqdm.auto`，再集中声明所有科学参数、运行模式和数据源，最后执行文件、时间轴、分辨率、类别与追踪ID预检。
            """,
            "s0-title-config",
        ),
        code(
            """
            from __future__ import annotations

            import hashlib
            import importlib.util
            import json
            import os
            import re
            import time
            from pathlib import Path

            import geopandas as gpd
            import numpy as np
            import pandas as pd
            import rasterio
            import xarray as xr
            from rasterio.features import rasterize
            from rasterio.transform import from_bounds
            from tqdm.auto import tqdm

            try:
                import statsmodels.api as sm
                from statsmodels.discrete.discrete_model import NegativeBinomial
            except ImportError:
                sm = None
                NegativeBinomial = None

            _ABANDONMENT_MODULE_PATH = Path("function/cropland_abandonment.py").resolve()
            _ABANDONMENT_SPEC = importlib.util.spec_from_file_location("s0_cropland_abandonment", _ABANDONMENT_MODULE_PATH)
            if _ABANDONMENT_SPEC is None or _ABANDONMENT_SPEC.loader is None:
                raise ImportError(f"Cannot load abandonment module: {_ABANDONMENT_MODULE_PATH}")
            _ABANDONMENT_MODULE = importlib.util.module_from_spec(_ABANDONMENT_SPEC)
            _ABANDONMENT_SPEC.loader.exec_module(_ABANDONMENT_MODULE)
            for _api_name in (
                "SIMPLIFIED_ESA_CLASSES",
                "aggregate_categorical_mode",
                "align_reference_abandonment",
                "cell_area_m2",
                "compare_reference_abandonment",
                "detect_abandonment",
                "fetch_reference_raster",
                "open_reference_abandonment_year",
                "publish_validation_results",
                "reclassify_esa_cci",
                "summarize_key_numbers",
                "temporal_majority_filter",
                "validate_reference_raster",
                "validate_roundtrip",
                "write_abandonment_chunks",
            ):
                globals()[_api_name] = getattr(_ABANDONMENT_MODULE, _api_name)

            def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    while True:
                        chunk = handle.read(chunk_size)
                        if not chunk:
                            break
                        digest.update(chunk)
                return digest.hexdigest().upper()

            def json_default(value):
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, Path):
                    return str(value)
                raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

            def regular_transform(lat_values: np.ndarray, lon_values: np.ndarray):
                lon_values = np.asarray(lon_values, dtype=float)
                lat_values = np.asarray(lat_values, dtype=float)
                lon_res = float(np.abs(np.diff(lon_values[:2]))[0])
                lat_res = float(np.abs(np.diff(lat_values[:2]))[0])
                left = float(lon_values.min() - lon_res / 2.0)
                right = float(lon_values.max() + lon_res / 2.0)
                bottom = float(lat_values.min() - lat_res / 2.0)
                top = float(lat_values.max() + lat_res / 2.0)
                return from_bounds(left, bottom, right, top, len(lon_values), len(lat_values))

            def rasterize_match(gdf: gpd.GeoDataFrame, lat_values: np.ndarray, lon_values: np.ndarray, value_column: str | None = None) -> xr.DataArray:
                transform = regular_transform(lat_values, lon_values)
                if value_column is None:
                    shapes = ((geom, 1) for geom in gdf.geometry)
                    fill = 0
                    dtype = "uint8"
                else:
                    shapes = ((geom, int(value)) for geom, value in zip(gdf.geometry, gdf[value_column]))
                    fill = 0
                    dtype = "int32"
                raster = rasterize(
                    shapes=shapes,
                    out_shape=(len(lat_values), len(lon_values)),
                    transform=transform,
                    fill=fill,
                    dtype=dtype,
                )
                if lat_values[0] < lat_values[-1]:
                    raster = np.flipud(raster)
                return xr.DataArray(raster, dims=("lat", "lon"), coords={"lat": lat_values, "lon": lon_values})

            S0_CONTEXT = {}
            print("[STAGE] import layer ready for s0_us_abandon")
            """,
            "s0-abandon-imports",
        ),
        code(
            """
            # REPO_RESULT_ROOT
            # - Unit: repository-relative directory path.
            # - Allowed values: any Path under outputs/.
            # - Default reason: isolates compact acceptance artifacts in outputs/s0_us_abandon.
            # - Key-number impact: none directly; affects publication location only.
            # - Run ID: no.
            # - Interactive-only: no.
            REPO_RESULT_ROOT = Path("outputs/s0_us_abandon")

            # D_RUN_ROOT
            # - Unit: absolute filesystem directory path.
            # - Allowed values: any dedicated scratch root outside Git.
            # - Default reason: required by the run contract for chunk/intermediate storage.
            # - Key-number impact: none directly; affects recoverability and audit trace only.
            # - Run ID: no.
            # - Interactive-only: no.
            D_RUN_ROOT = Path(r"D:\\xarray\\s0_us_abandon")

            # INPUT_SOURCES
            # - Unit: absolute file path per named source.
            # - Allowed values: paths to annual ESA-CCI-like land-cover cubes.
            # - Default reason: use the merged ESA-CCI stack already staged under D:\\xarray\\project_temdata.
            # - Key-number impact: yes; changing the input source changes all scientific outputs.
            # - Run ID: yes.
            # - Interactive-only: no.
            INPUT_SOURCES = {
                "esa_cci_300m": Path(r"D:\\xarray\\project_temdata\\merged_lccs.nc"),
            }

            # INPUT_SOURCE_KEY
            # - Unit: dictionary key.
            # - Allowed values: keys present in INPUT_SOURCES.
            # - Default reason: keeps the existing ESA-CCI 300 m stack as the primary source.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            INPUT_SOURCE_KEY = "esa_cci_300m"

            # INPUT_VARIABLE
            # - Unit: variable name in the NetCDF source.
            # - Allowed values: categorical land-cover variable names.
            # - Default reason: legacy merged_lccs.nc stores ESA classes in lccs_class.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            INPUT_VARIABLE = "lccs_class"

            # INPUT_YEAR_RANGE
            # - Unit: inclusive calendar years.
            # - Allowed values: a two-integer range fully covered by the source time axis.
            # - Default reason: the staged ESA-CCI cube contains the complete 1992-2022 series.
            # - Key-number impact: yes; changes the observable abandonment and recultivation window.
            # - Run ID: yes.
            # - Interactive-only: no.
            INPUT_YEAR_RANGE = (1992, 2022)

            # CALIFORNIA_BOUNDS
            # - Unit: decimal degrees in EPSG:4326.
            # - Allowed values: mapping with lat_min/lat_max/lon_min/lon_max and increasing min/max values.
            # - Default reason: reproduces the legacy California rectangular validation extent exactly.
            # - Key-number impact: yes in EXECUTION_MODE="california"; it defines every California area total.
            # - Run ID: yes.
            # - Interactive-only: no.
            CALIFORNIA_BOUNDS = {
                "lat_min": 32.5,
                "lat_max": 42.0,
                "lon_min": -124.5,
                "lon_max": -114.0,
            }

            # NETCDF_OPEN_CHUNKS
            # - Unit: xarray/dask chunk mapping.
            # - Allowed values: {}, "auto", or a dimension-to-positive-integer mapping.
            # - Default reason: {} reuses the NetCDF preferred chunks (1 x 2025 x 2025) without loading the global cube eagerly.
            # - Key-number impact: none; this changes execution scheduling only.
            # - Run ID: no.
            # - Interactive-only: yes.
            NETCDF_OPEN_CHUNKS = {}

            # PRIMARY_GRID
            # - Unit: branch label.
            # - Allowed values: "300m_native", "1km_mode", "1km_mean_round_legacy".
            # - Default reason: 1 km categorical mode is the main audited branch.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            PRIMARY_GRID = "1km_mode"

            # BASELINE_CROP_YEARS
            # - Unit: inclusive calendar years.
            # - Allowed values: two integers inside the source time axis.
            # - Default reason: matches the requested 1992-1997 stable cropland baseline.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            BASELINE_CROP_YEARS = (1992, 1997)

            # MIN_ABANDONMENT_YEARS
            # - Unit: years.
            # - Allowed values: positive integers, typically 5/7/10 in robustness tests.
            # - Default reason: five-year sustained exit is the primary definition.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            MIN_ABANDONMENT_YEARS = 5

            # REQUIRED_THROUGH_YEAR
            # - Unit: calendar year.
            # - Allowed values: years inside the source time axis.
            # - Default reason: 2020 matches the paper-facing key-number checkpoint.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            REQUIRED_THROUGH_YEAR = 2020

            # ANALYSIS_END_YEAR
            # - Unit: calendar year.
            # - Allowed values: years inside the source time axis.
            # - Default reason: primary audited endpoint equals the 2020 cutoff before 2022 robustness expansion.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            ANALYSIS_END_YEAR = 2020

            # RECULTIVATION_YEARS
            # - Unit: years.
            # - Allowed values: positive integers.
            # - Default reason: one-year return is enough to mark recultivation in the main audit.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            RECULTIVATION_YEARS = 1

            # TEMPORAL_FILTER_WINDOW
            # - Unit: years.
            # - Allowed values: positive odd integers.
            # - Default reason: five-year majority filter aligns with the requested main detector.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            TEMPORAL_FILTER_WINDOW = 5

            # MODE_AGG_FACTOR
            # - Unit: source pixels per target edge.
            # - Allowed values: positive integers.
            # - Default reason: 3x3 categorical mode maps 10 arc-second source cells to 30 arc-second target cells.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            MODE_AGG_FACTOR = 3

            # CROPLAND_CODES
            # - Unit: ESA-CCI categorical codes.
            # - Allowed values: exactly {10, 11, 12, 20, 30, 40}; another set fails preflight because this audit fixes the ESA legend contract.
            # - Default reason: fixed cropland definition requested for the audit and encoded by the simplified9 module schema.
            # - Key-number impact: yes; a future contract revision would change baseline eligibility and all abandonment totals.
            # - Run ID: yes.
            # - Interactive-only: no.
            CROPLAND_CODES = {10, 11, 12, 20, 30, 40}

            # EXCLUDED_TRANSITION_CODES
            # - Unit: simplified9 categorical codes produced by reclassify_esa_cci.
            # - Allowed values: iterable drawn from 1..9 (6=wetland, 7=settlement).
            # - Default reason: exclude wetland and settlement transitions in the primary detector.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            EXCLUDED_TRANSITION_CODES = {6, 7}

            # EXECUTION_MODE
            # - Unit: execution profile name.
            # - Allowed values: "smoke", "california", "full".
            # - Default reason: full keeps scientific branches enabled; california is the retained regional validation; smoke is interface-only.
            # - Key-number impact: no when the same inputs are used.
            # - Run ID: no.
            # - Interactive-only: yes.
            EXECUTION_MODE = "full"

            # SMOKE_WINDOW_PIXELS
            # - Unit: native-grid pixels per spatial edge.
            # - Allowed values: positive integers.
            # - Default reason: 128 x 128 around a CONUS representative point is large enough for interface QA and quick enough for review.
            # - Key-number impact: none; used only when EXECUTION_MODE="smoke", whose outputs are never publishable scientific results.
            # - Run ID: no; smoke receives a visible run-id suffix instead.
            # - Interactive-only: yes.
            SMOKE_WINDOW_PIXELS = 128

            # SHOW_PROGRESS
            # - Unit: boolean flag.
            # - Allowed values: True/False.
            # - Default reason: keep tqdm visible for notebook inspection.
            # - Key-number impact: none.
            # - Run ID: no.
            # - Interactive-only: yes.
            SHOW_PROGRESS = True

            # PRINT_STAGE_SUMMARY
            # - Unit: boolean flag.
            # - Allowed values: True/False.
            # - Default reason: emit concise audit logs without dumping entire objects.
            # - Key-number impact: none.
            # - Run ID: no.
            # - Interactive-only: yes.
            PRINT_STAGE_SUMMARY = True

            # ALLOW_NETWORK_DOWNLOADS
            # - Unit: boolean flag.
            # - Allowed values: True/False.
            # - Default reason: False prevents tests and ordinary notebook inspection from downloading external rasters.
            # - Key-number impact: none unless a missing external benchmark must be fetched.
            # - Run ID: no.
            # - Interactive-only: yes.
            ALLOW_NETWORK_DOWNLOADS = False

            # STATE_SHAPEFILE
            # - Unit: repository-relative path.
            # - Allowed values: vector boundary files containing U.S. states.
            # - Default reason: existing Census state boundaries are already tracked in data/.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            STATE_SHAPEFILE = Path("data/cb_2018_us_state_500k.shp")

            # COUNTY_SHAPEFILE
            # - Unit: repository-relative path.
            # - Allowed values: county-level boundary files.
            # - Default reason: existing Census county boundaries support county-year Poisson checks.
            # - Key-number impact: no for the national key number; yes for robustness diagnostics.
            # - Run ID: yes.
            # - Interactive-only: no.
            COUNTY_SHAPEFILE = Path("data/cb_2018_us_county_5m.shp")

            # PV_DEDUP_RETAIN_MASK_SOURCE
            # - Unit: NetCDF file path or None.
            # - Allowed values: None, or a grid-exact boolean/0-1 retain mask aligned to the primary abandonment grid.
            # - Default reason: no audited PV-dedup mask is identified in the current repository, so the stage must remain explicitly unevaluated.
            # - Key-number impact: yes; controls the post-PV-dedup retained area.
            # - Run ID: yes.
            # - Interactive-only: no.
            PV_DEDUP_RETAIN_MASK_SOURCE = None

            # PV_DEDUP_RETAIN_MASK_VARIABLE
            # - Unit: NetCDF variable name.
            # - Allowed values: a variable containing only 0 (discard) and 1 (retain).
            # - Default reason: gives the downstream contract a stable, explicit field name.
            # - Key-number impact: yes; selecting another field changes the retained PV-deduplicated area.
            # - Run ID: yes.
            # - Interactive-only: no.
            PV_DEDUP_RETAIN_MASK_VARIABLE = "retain_after_pv_dedup"

            # FEATURE_COMPLETE_RETAIN_MASK_SOURCE
            # - Unit: NetCDF file path or None.
            # - Allowed values: None, or a grid-exact boolean/0-1 retain mask aligned to the primary abandonment grid.
            # - Default reason: feature-completeness must be supplied by the downstream preprocessing audit, not inferred inside this detector.
            # - Key-number impact: yes; controls the complete-feature retained area.
            # - Run ID: yes.
            # - Interactive-only: no.
            FEATURE_COMPLETE_RETAIN_MASK_SOURCE = None

            # FEATURE_COMPLETE_RETAIN_MASK_VARIABLE
            # - Unit: NetCDF variable name.
            # - Allowed values: a variable containing only 0 (incomplete) and 1 (complete).
            # - Default reason: gives the downstream completeness contract a stable, explicit field name.
            # - Key-number impact: yes; selecting another field changes the complete-feature area.
            # - Run ID: yes.
            # - Interactive-only: no.
            FEATURE_COMPLETE_RETAIN_MASK_VARIABLE = "retain_complete_features"

            # CONUS_EXCLUDED_STUSPS
            # - Unit: USPS state/territory abbreviations.
            # - Allowed values: iterable of strings present in STATE_SHAPEFILE.
            # - Default reason: exclude Alaska, Hawaii, and U.S. territories from the audit.
            # - Key-number impact: yes.
            # - Run ID: yes.
            # - Interactive-only: no.
            CONUS_EXCLUDED_STUSPS = ("AK", "HI", "PR", "VI", "GU", "MP", "AS")

            # XIE_SOURCE_PAGE_URL
            # - Unit: HTTPS URL string.
            # - Allowed values: provider landing pages for the benchmark dataset.
            # - Default reason: records the public source page users can inspect before using the benchmark.
            # - Key-number impact: none on the primary estimate; yes on validation traceability.
            # - Run ID: no.
            # - Interactive-only: no.
            XIE_SOURCE_PAGE_URL = "https://www.glbrc.org/data-and-tools/glbrc-data-sets/cropland-abandonment-between-1986-and-2018-across-united-states"

            # XIE_DIRECT_DOWNLOAD_URL
            # - Unit: HTTPS URL string.
            # - Allowed values: direct archive download links for the benchmark dataset.
            # - Default reason: exact provider archive required by the validation contract.
            # - Key-number impact: none on the primary estimate; yes on validation traceability.
            # - Run ID: no.
            # - Interactive-only: no.
            XIE_DIRECT_DOWNLOAD_URL = "https://atlas.glbrc.org/files/public/abanMaps_5yr_v4__abanDef2_noRec_noUr.tar.gz"

            # XIE_ARCHIVE_LOCAL_PATH
            # - Unit: absolute filesystem path.
            # - Allowed values: dedicated local cache paths for the benchmark archive.
            # - Default reason: keep the downloaded archive beside merged_lccs.nc under project_temdata.
            # - Key-number impact: none.
            # - Run ID: no.
            # - Interactive-only: no.
            XIE_ARCHIVE_LOCAL_PATH = Path(r"D:\\xarray\\project_temdata\\xie_2024_conus_abandonment_30m.tar.gz")
            XIE_ARCHIVE_SHA256 = "B2EC7094B20A497259618CD5F3EF5CBD4AB3A06D4C12127D006D61D3F6CCEAB8"
            XIE_ARCHIVE_SIZE_BYTES = 198383647
            XIE_ARCHIVE_MEMBER = "abanMaps_5yr_v4__abanDef2_noRec_noUr/abanMaps_5yr_v4__abanDef2_noRec_noUr.tif"

            # XIE_TIFF_LOCAL_PATH
            # - Unit: absolute filesystem path.
            # - Allowed values: extracted GeoTIFF paths for the benchmark raster.
            # - Default reason: local validated TIFF already exists under project_temdata for offline reuse.
            # - Key-number impact: none.
            # - Run ID: no.
            # - Interactive-only: no.
            XIE_TIFF_LOCAL_PATH = Path(r"D:\\xarray\\project_temdata\\abanMaps_5yr_v4__abanDef2_noRec_noUr\\abanMaps_5yr_v4__abanDef2_noRec_noUr.tif")
            XIE_TIFF_SHA256 = "97936653440C3F288B2D8E2F851DBE16904212767EE46C73991FAA5451BDC404"
            XIE_TIFF_SIZE_BYTES = 244084474
            XIE_VALID_YEAR_MIN = 1991
            XIE_VALID_YEAR_MAX = 2014

            SOURCE_GRID_CONTRACTS = {
                "esa_cci_300m": {
                    "storage": "NetCDF/xarray",
                    "crs": "EPSG:4326 coordinate centers from CF-style lat/lon axes",
                    "grid": "native 10 arc-second (~300 m) yearly categorical cube",
                    "resampling": "3x3 categorical mode to 30 arc-second for the 1 km branch; no class mean in the primary branch",
                    "area": "WGS84 geodesic cell area from target grid coordinates",
                },
                "xie_2024_30m": {
                    "storage": "GeoTIFF/rasterio-rioxarray",
                    "crs": "EPSG:4326 affine x/y raster geometry with explicit nodata",
                    "grid": "native ~0.0002694946 degree (~30 m) abandonment-year raster",
                    "resampling": "explicit raster reprojection/alignment to the audited target grid at comparison time",
                    "area": "comparison area inherited from the audited target grid, not from raw TIFF pixel area",
                },
            }

            IMPLEMENTATION_SHA256 = sha256_file(Path("function/cropland_abandonment.py"))
            NOTEBOOK_CONTRACT_SHA256 = sha256_file(Path("tools/update_process_notebook_contract.py"))

            PARAMETER_SNAPSHOT = {
                "input_source_key": INPUT_SOURCE_KEY,
                "input_variable": INPUT_VARIABLE,
                "input_year_range": INPUT_YEAR_RANGE,
                "california_bounds": CALIFORNIA_BOUNDS,
                "primary_grid": PRIMARY_GRID,
                "baseline_crop_years": BASELINE_CROP_YEARS,
                "min_abandonment_years": MIN_ABANDONMENT_YEARS,
                "required_through_year": REQUIRED_THROUGH_YEAR,
                "analysis_end_year": ANALYSIS_END_YEAR,
                "recultivation_years": RECULTIVATION_YEARS,
                "temporal_filter_window": TEMPORAL_FILTER_WINDOW,
                "mode_agg_factor": MODE_AGG_FACTOR,
                "cropland_codes": sorted(CROPLAND_CODES),
                "excluded_transition_codes": sorted(EXCLUDED_TRANSITION_CODES),
                "state_shapefile": str(STATE_SHAPEFILE),
                "county_shapefile": str(COUNTY_SHAPEFILE),
                "pv_dedup_retain_mask_source": str(PV_DEDUP_RETAIN_MASK_SOURCE) if PV_DEDUP_RETAIN_MASK_SOURCE else None,
                "pv_dedup_retain_mask_variable": PV_DEDUP_RETAIN_MASK_VARIABLE,
                "feature_complete_retain_mask_source": str(FEATURE_COMPLETE_RETAIN_MASK_SOURCE) if FEATURE_COMPLETE_RETAIN_MASK_SOURCE else None,
                "feature_complete_retain_mask_variable": FEATURE_COMPLETE_RETAIN_MASK_VARIABLE,
                "conus_excluded_stusps": list(CONUS_EXCLUDED_STUSPS),
                "xie_source_page_url": XIE_SOURCE_PAGE_URL,
                "xie_direct_download_url": XIE_DIRECT_DOWNLOAD_URL,
                "xie_archive_local_path": str(XIE_ARCHIVE_LOCAL_PATH),
                "xie_archive_sha256": XIE_ARCHIVE_SHA256,
                "xie_archive_size_bytes": XIE_ARCHIVE_SIZE_BYTES,
                "xie_archive_member": XIE_ARCHIVE_MEMBER,
                "xie_tiff_local_path": str(XIE_TIFF_LOCAL_PATH),
                "xie_tiff_sha256": XIE_TIFF_SHA256,
                "xie_tiff_size_bytes": XIE_TIFF_SIZE_BYTES,
                "xie_valid_year_min": XIE_VALID_YEAR_MIN,
                "xie_valid_year_max": XIE_VALID_YEAR_MAX,
                "source_grid_contracts": SOURCE_GRID_CONTRACTS,
                "implementation_sha256": IMPLEMENTATION_SHA256,
                "notebook_contract_sha256": NOTEBOOK_CONTRACT_SHA256,
            }
            PARAMETER_HASH = hashlib.sha256(json.dumps(PARAMETER_SNAPSHOT, sort_keys=True).encode("utf-8")).hexdigest()
            RUN_ID_BASE = f"s0-us-abandon-{PRIMARY_GRID}-{PARAMETER_HASH[:12]}"
            if EXECUTION_MODE not in {"smoke", "california", "full"}:
                raise ValueError(f"Unsupported EXECUTION_MODE: {EXECUTION_MODE}")
            RUN_ID = RUN_ID_BASE if EXECUTION_MODE == "full" else f"{RUN_ID_BASE}-{EXECUTION_MODE}"
            D_ACTIVE_RUN = D_RUN_ROOT / RUN_ID
            D_SOURCE_RUN_ROOTS = {
                source_key: D_ACTIVE_RUN / "sources" / source_key
                for source_key in sorted({*INPUT_SOURCES, "xie_2024_30m"})
            }
            if INPUT_SOURCE_KEY not in INPUT_SOURCES:
                raise KeyError(f"INPUT_SOURCE_KEY is not configured in INPUT_SOURCES: {INPUT_SOURCE_KEY}")
            if INPUT_SOURCE_KEY not in SOURCE_GRID_CONTRACTS:
                raise KeyError(f"Missing CRS/grid/area contract for input source: {INPUT_SOURCE_KEY}")
            D_PRIMARY_RUN_ROOT = D_SOURCE_RUN_ROOTS[INPUT_SOURCE_KEY]
            D_PRIMARY_SUBDIRS = {
                "intermediate": D_PRIMARY_RUN_ROOT / "intermediate",
                "chunks": D_PRIMARY_RUN_ROOT / "chunks",
                "merged_chunks": D_PRIMARY_RUN_ROOT / "merged_chunks",
                "diagnostics": D_PRIMARY_RUN_ROOT / "diagnostics",
                "tmp": D_PRIMARY_RUN_ROOT / "tmp",
            }
            D_XIE_RUN_ROOT = D_SOURCE_RUN_ROOTS["xie_2024_30m"]
            D_XIE_SUBDIRS = {
                "intermediate": D_XIE_RUN_ROOT / "intermediate",
                "diagnostics": D_XIE_RUN_ROOT / "diagnostics",
                "tmp": D_XIE_RUN_ROOT / "tmp",
            }

            print(f"[CONFIG] run_id={RUN_ID} execution_mode={EXECUTION_MODE} primary_grid={PRIMARY_GRID}")
            print(f"[CONFIG] parameter_hash={PARAMETER_HASH}")
            print(f"[PATH] input_source={INPUT_SOURCES[INPUT_SOURCE_KEY]}")
            print(f"[PATH] d_run_root={D_ACTIVE_RUN}")
            print(f"[PATH] esa_source_root={D_SOURCE_RUN_ROOTS['esa_cci_300m']}")
            print(f"[PATH] xie_source_root={D_SOURCE_RUN_ROOTS['xie_2024_30m']}")
            print(f"[PATH] repo_result_root={REPO_RESULT_ROOT}")
            print(f"[PATH] xie_archive={XIE_ARCHIVE_LOCAL_PATH}")
            print(f"[PATH] xie_tiff={XIE_TIFF_LOCAL_PATH}")
            print(f"[CONFIG] source_grid_contracts={json.dumps(SOURCE_GRID_CONTRACTS, ensure_ascii=False)}")
            """,
            "s0-abandon-parameters",
        ),
        code(
            """
            stage_started = time.time()
            for directory in [REPO_RESULT_ROOT, *D_PRIMARY_SUBDIRS.values(), *D_XIE_SUBDIRS.values()]:
                directory.mkdir(parents=True, exist_ok=True)

            input_path = INPUT_SOURCES[INPUT_SOURCE_KEY]
            if not input_path.exists():
                raise FileNotFoundError(input_path)
            input_stat = input_path.stat()
            print(
                f"[INPUT] source_exists=True size_gb={input_stat.st_size / (1024 ** 3):.3f} "
                f"mtime={pd.Timestamp(input_stat.st_mtime, unit='s')}"
            )

            raw_ds = xr.open_dataset(input_path, chunks=NETCDF_OPEN_CHUNKS)
            raw_var = raw_ds[INPUT_VARIABLE]
            if set(CROPLAND_CODES) != set(SIMPLIFIED_ESA_CLASSES[1]):
                raise ValueError(
                    f"CROPLAND_CODES must match the fixed ESA-CCI simplified9 contract: {SIMPLIFIED_ESA_CLASSES[1]}"
                )
            year_values = pd.to_datetime(raw_var["time"].values).year.to_numpy()
            expected_years = np.arange(INPUT_YEAR_RANGE[0], INPUT_YEAR_RANGE[1] + 1)
            if not np.array_equal(year_values, expected_years):
                raise ValueError(
                    f"Expected continuous years {expected_years[0]}-{expected_years[-1]}, "
                    f"got {year_values[0]}-{year_values[-1]}"
                )

            lat_name = "lat" if "lat" in raw_var.dims else "y"
            lon_name = "lon" if "lon" in raw_var.dims else "x"
            lat_values = raw_var[lat_name].values
            lon_values = raw_var[lon_name].values
            lon_res_deg = float(np.abs(np.diff(lon_values[:2]))[0])
            classes_preview = list(map(int, raw_var.attrs.get("flag_values", [])))
            if not classes_preview:
                preview = raw_var.isel(time=0, **{lat_name: slice(0, 64), lon_name: slice(0, 64)}).compute()
                classes_preview = np.unique(preview.values[np.isfinite(preview.values)]).astype(int).tolist()
            print(
                f"[INPUT] years={year_values[0]}-{year_values[-1]} count={len(year_values)} "
                f"shape={tuple(raw_var.shape)} variable={INPUT_VARIABLE}"
            )
            input_metadata = {
                "path": str(input_path),
                "size_bytes": int(input_stat.st_size),
                "mtime": pd.Timestamp(input_stat.st_mtime, unit="s").isoformat(),
                "product_version": raw_ds.attrs.get("product_version"),
                "tracking_id": raw_ds.attrs.get("tracking_id"),
                "dataset_id": raw_ds.attrs.get("id"),
                "preferred_chunks": raw_var.encoding.get("preferred_chunks"),
            }
            print(
                f"[INPUT] product_version={input_metadata['product_version']} "
                f"tracking_id={input_metadata['tracking_id']} dataset_id={input_metadata['dataset_id']}"
            )
            print(f"[INPUT] xarray_chunks={raw_var.chunks} preferred_chunks={input_metadata['preferred_chunks']}")
            print(f"[INPUT] source_resolution≈{lon_res_deg:.7f} degree (300 m / 10 arc-second) target_resolution=30 arc-second")
            print(f"[INPUT] classes_preview={classes_preview}")
            print(f"[INPUT] netcdf_grid_contract={json.dumps(SOURCE_GRID_CONTRACTS[INPUT_SOURCE_KEY], ensure_ascii=False)}")

            S0_CONTEXT.update(
                {
                    "raw_ds": raw_ds,
                    "raw_var": raw_var,
                    "lat_name": lat_name,
                    "lon_name": lon_name,
                    "year_values": year_values,
                    "input_metadata": input_metadata,
                }
            )
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] preflight completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-preflight",
        ),
        markdown(
            """
            ### （2）研究区、重分类与尺度分支

            构建CONUS或California矩形范围，将ESA-CCI重分类为简化类别，执行五年时间多数滤波，并分别保留原生300 m与30角秒分类众数结果。
            """,
            "s0-title-preprocess",
        ),
        code(
            """
            stage_started = time.time()
            states = gpd.read_file(STATE_SHAPEFILE).to_crs("EPSG:4326")
            state_code_col = "STUSPS" if "STUSPS" in states.columns else "STUSPS10"
            conus_states = states.loc[~states[state_code_col].isin(CONUS_EXCLUDED_STUSPS)].copy()
            conus_geometry = conus_states.union_all()
            excluded = sorted(set(states[state_code_col]) - set(conus_states[state_code_col]))
            bbox = tuple(float(v) for v in conus_states.total_bounds)
            conus_fips_col = "STATEFP" if "STATEFP" in conus_states.columns else "STATEFP10"
            conus_fips = sorted(conus_states[conus_fips_col].astype(str).tolist())
            conus_vector_area_ha = float(conus_states.to_crs("EPSG:5070").geometry.area.sum() / 10_000.0)
            print(f"[AOI] excluded_regions={excluded}")
            print(f"[AOI] conus_states={len(conus_states)} fips={conus_fips} bbox={bbox}")
            print(f"[AOI] vector_area_ha={conus_vector_area_ha:.2f}")
            S0_CONTEXT.update(
                {
                    "states_gdf": states,
                    "conus_states_gdf": conus_states,
                    "state_code_col": state_code_col,
                    "state_fips_col": conus_fips_col,
                    "conus_geometry": conus_geometry,
                    "conus_bbox": bbox,
                    "conus_vector_area_ha": conus_vector_area_ha,
                }
            )
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] conus mask prep completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-conus-mask",
        ),
        code(
            """
            stage_started = time.time()
            raw_var = S0_CONTEXT["raw_var"]
            lat_name = S0_CONTEXT["lat_name"]
            lon_name = S0_CONTEXT["lon_name"]
            if EXECUTION_MODE == "california":
                bbox = (
                    CALIFORNIA_BOUNDS["lon_min"],
                    CALIFORNIA_BOUNDS["lat_min"],
                    CALIFORNIA_BOUNDS["lon_max"],
                    CALIFORNIA_BOUNDS["lat_max"],
                )
                print(f"[AOI] california_bbox={bbox}")
            else:
                bbox = S0_CONTEXT["conus_bbox"]
            lat_values = raw_var[lat_name].values
            lat_slice = slice(bbox[3], bbox[1]) if lat_values[0] > lat_values[-1] else slice(bbox[1], bbox[3])
            lon_slice = slice(bbox[0], bbox[2])
            subset = raw_var.sel(
                time=slice(f"{INPUT_YEAR_RANGE[0]}-01-01", f"{INPUT_YEAR_RANGE[1]}-12-31"),
                **{lat_name: lat_slice, lon_name: lon_slice},
            )
            if EXECUTION_MODE == "smoke":
                representative = S0_CONTEXT["conus_geometry"].representative_point()
                center_lat_idx = int(np.abs(subset[lat_name].values - representative.y).argmin())
                center_lon_idx = int(np.abs(subset[lon_name].values - representative.x).argmin())
                half_window = SMOKE_WINDOW_PIXELS // 2
                subset = subset.isel(
                    **{
                        lat_name: slice(max(0, center_lat_idx - half_window), min(subset.sizes[lat_name], center_lat_idx + half_window)),
                        lon_name: slice(max(0, center_lon_idx - half_window), min(subset.sizes[lon_name], center_lon_idx + half_window)),
                    }
                )
                subset = subset.load()
                print(f"[CONFIG] smoke_window_shape={tuple(subset.shape)} representative_point=({representative.y:.4f},{representative.x:.4f})")
            conus_mask = rasterize_match(S0_CONTEXT["conus_states_gdf"], subset[lat_name].values, subset[lon_name].values).astype(bool)
            conus_pixel_count = int(conus_mask.sum().item())
            print(
                f"[AOI] preclip_pixels={raw_var.sizes[lat_name] * raw_var.sizes[lon_name]} "
                f"post_bbox_pixels={subset.sizes[lat_name] * subset.sizes[lon_name]} conus_pixels={conus_pixel_count}"
            )

            year_bar = tqdm(S0_CONTEXT["year_values"], disable=not SHOW_PROGRESS, desc="ESA years")
            reclass_slices = []
            for year in year_bar:
                year_slice = subset.sel(time=f"{int(year)}-01-01")
                reclass_year = reclassify_esa_cci(year_slice, mode="simplified9").where(conus_mask, other=0)
                reclass_year = reclass_year.expand_dims(time=[np.datetime64(f"{int(year)}-01-01")])
                reclass_slices.append(reclass_year)
                if SHOW_PROGRESS:
                    year_bar.set_postfix(
                        done=len(reclass_slices), skipped=0, failed=0,
                        valid_pixels=conus_pixel_count,
                        area_ha=f"{S0_CONTEXT['conus_vector_area_ha']:.0f}",
                    )

            landcover_300m = xr.concat(reclass_slices, dim="time")
            california_lccs_path = None
            if EXECUTION_MODE == "california":
                california_lccs_path = D_PRIMARY_SUBDIRS["intermediate"] / f"{INPUT_SOURCE_KEY}_ca_lccs.nc"
                california_lccs_tmp = D_PRIMARY_SUBDIRS["tmp"] / f"{INPUT_SOURCE_KEY}_ca_lccs.tmp.nc"
                california_lccs_tmp.unlink(missing_ok=True)
                california_ds = landcover_300m.rename("lccs_class").to_dataset()
                california_ds.attrs.update(
                    {
                        "source_key": INPUT_SOURCE_KEY,
                        "source_path": str(INPUT_SOURCES[INPUT_SOURCE_KEY]),
                        "aoi": "california_approximate_bbox",
                        "aoi_bounds_epsg4326": json.dumps(CALIFORNIA_BOUNDS, sort_keys=True),
                        "classification_schema": "esa_cci_simplified9",
                        "temporal_filter_applied": "false",
                        "parameter_hash": PARAMETER_HASH,
                    }
                )
                california_encoding = {
                    "lccs_class": {
                        "dtype": "uint8",
                        "zlib": True,
                        "complevel": 4,
                        "shuffle": True,
                        "_FillValue": None,
                        "chunksizes": (
                            1,
                            min(512, california_ds.sizes[lat_name]),
                            min(512, california_ds.sizes[lon_name]),
                        ),
                    }
                }
                try:
                    california_ds.to_netcdf(california_lccs_tmp, encoding=california_encoding)
                    os.replace(california_lccs_tmp, california_lccs_path)
                finally:
                    california_lccs_tmp.unlink(missing_ok=True)
                landcover_300m = xr.open_dataset(
                    california_lccs_path,
                    chunks={"time": -1, lat_name: 512, lon_name: 512},
                )["lccs_class"]
                print(f"[PATH] california_lccs={california_lccs_path}")
            landcover_300m_filtered = temporal_majority_filter(landcover_300m, window=TEMPORAL_FILTER_WINDOW)
            S0_CONTEXT.update(
                {
                    "subset_var": subset,
                    "conus_mask": conus_mask,
                    "landcover_300m": landcover_300m,
                    "landcover_300m_filtered": landcover_300m_filtered,
                    "active_bbox": bbox,
                    "california_lccs_path": california_lccs_path,
                }
            )
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] reclassification completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-reclass",
        ),
        code(
            """
            stage_started = time.time()
            landcover_300m_filtered = S0_CONTEXT["landcover_300m_filtered"]
            branch_outputs = {"300m_native": landcover_300m_filtered}
            branch_bar = tqdm(["1km_mode"], disable=not SHOW_PROGRESS, desc="grid branches")
            for branch_name in branch_bar:
                annual_outputs = []
                year_bar = tqdm(
                    pd.to_datetime(landcover_300m_filtered["time"].values).year.to_list(),
                    disable=not SHOW_PROGRESS,
                    desc=f"{branch_name} years",
                    leave=False,
                )
                for year in year_bar:
                    annual_mask = landcover_300m_filtered.sel(time=f"{int(year)}-01-01")
                    annual_mode = aggregate_categorical_mode(annual_mask, lat_factor=MODE_AGG_FACTOR, lon_factor=MODE_AGG_FACTOR)
                    annual_outputs.append(annual_mode.expand_dims(time=[np.datetime64(f"{int(year)}-01-01")]))
                    if SHOW_PROGRESS:
                        year_bar.set_postfix(
                            done=len(annual_outputs), skipped=0, failed=0,
                            valid_pixels=annual_mode.sizes["lat"] * annual_mode.sizes["lon"],
                            area_ha="deferred",
                        )
                branch_outputs[branch_name] = xr.concat(annual_outputs, dim="time")
                if SHOW_PROGRESS:
                    branch_bar.set_postfix(
                        done=branch_name, skipped=0, failed=0,
                        valid_pixels=branch_outputs[branch_name].sizes["lat"] * branch_outputs[branch_name].sizes["lon"],
                        area_ha="deferred",
                    )

            california_1km_lccs_path = None
            if EXECUTION_MODE == "california":
                california_1km_lccs_path = D_PRIMARY_SUBDIRS["intermediate"] / f"{INPUT_SOURCE_KEY}_ca_lccs_1km_mode.nc"
                california_1km_lccs_tmp = D_PRIMARY_SUBDIRS["tmp"] / f"{INPUT_SOURCE_KEY}_ca_lccs_1km_mode.tmp.nc"
                california_1km_lccs_tmp.unlink(missing_ok=True)
                california_1km_ds = branch_outputs["1km_mode"].rename("lccs_class").to_dataset()
                california_1km_ds.attrs.update(
                    {
                        "source_key": INPUT_SOURCE_KEY,
                        "source_path": str(S0_CONTEXT["california_lccs_path"]),
                        "aoi": "california_approximate_bbox",
                        "aoi_bounds_epsg4326": json.dumps(CALIFORNIA_BOUNDS, sort_keys=True),
                        "classification_schema": "esa_cci_simplified9",
                        "temporal_filter_window_years": TEMPORAL_FILTER_WINDOW,
                        "spatial_aggregation": "3x3_categorical_mode_to_30_arc_second",
                        "parameter_hash": PARAMETER_HASH,
                    }
                )
                california_1km_ds.load()
                try:
                    california_1km_ds.to_netcdf(
                        california_1km_lccs_tmp,
                        encoding={
                            "lccs_class": {
                                "dtype": "uint8",
                                "zlib": True,
                                "complevel": 4,
                                "shuffle": True,
                                "_FillValue": None,
                                "chunksizes": (
                                    1,
                                    min(256, california_1km_ds.sizes["lat"]),
                                    min(256, california_1km_ds.sizes["lon"]),
                                ),
                            }
                        },
                    )
                    os.replace(california_1km_lccs_tmp, california_1km_lccs_path)
                finally:
                    california_1km_lccs_tmp.unlink(missing_ok=True)
                branch_outputs["1km_mode"] = xr.open_dataset(
                    california_1km_lccs_path,
                    chunks={"time": -1, "lat": 256, "lon": 256},
                )["lccs_class"]
                print(f"[PATH] california_1km_lccs={california_1km_lccs_path}")

            S0_CONTEXT["grid_branches"] = branch_outputs
            S0_CONTEXT["california_1km_lccs_path"] = california_1km_lccs_path
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] grid branches completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-grid-branches",
        ),
        markdown(
            """
            ### （3）撂荒检测、分块落盘与Round-trip QA

            按显式截止年运行同一检测器；以空间chunk写入数据源隔离目录，随后合并并检查坐标、变量、dtype、0/1编码、attrs和数值一致性。
            """,
            "s0-title-detection",
        ),
        code(
            """
            stage_started = time.time()
            detection_outputs = {}
            for branch_name, branch_da in S0_CONTEXT["grid_branches"].items():
                print(f"[STAGE] detecting abandonment for branch={branch_name}")
                result_ds = detect_abandonment(
                    branch_da,
                    schema="simplified9",
                    built_up_codes=tuple(sorted({7} & EXCLUDED_TRANSITION_CODES)),
                    wetland_codes=tuple(sorted(EXCLUDED_TRANSITION_CODES - {7})),
                    baseline_years=BASELINE_CROP_YEARS,
                    min_abandonment_years=MIN_ABANDONMENT_YEARS,
                    required_through_year=REQUIRED_THROUGH_YEAR,
                    analysis_end_year=ANALYSIS_END_YEAR,
                    recultivation_years=RECULTIVATION_YEARS,
                    detector="stable_crop",
                    time_dim="time",
                )
                baseline_year_index = pd.to_datetime(branch_da["time"].values).year
                baseline_mask = (baseline_year_index >= BASELINE_CROP_YEARS[0]) & (baseline_year_index <= BASELINE_CROP_YEARS[1])
                eligible_mask = (
                    branch_da.isel(time=np.where(baseline_mask)[0]).fillna(0).astype(np.uint8) == 1
                ).all("time").astype(np.uint8)
                result_ds["eligible_cropland"] = eligible_mask
                detection_outputs[branch_name] = result_ds
                print(
                    f"[RESULT] branch={branch_name} graph_ready=True "
                    f"shape={tuple(result_ds['abandonment_year'].shape)} metrics_deferred_to_chunk_write=True"
                )

            S0_CONTEXT["detection_outputs"] = detection_outputs
            S0_CONTEXT["primary_detection"] = detection_outputs[PRIMARY_GRID]
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] detection completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-detection",
        ),
        code(
            """
            stage_started = time.time()
            primary_ds = S0_CONTEXT["primary_detection"]
            chunk_size_lat = 256
            chunk_size_lon = 256
            chunk_jobs = []
            for row_idx, lat_start in enumerate(range(0, primary_ds.sizes["lat"], chunk_size_lat)):
                lat_stop = min(primary_ds.sizes["lat"], lat_start + chunk_size_lat)
                for col_idx, lon_start in enumerate(range(0, primary_ds.sizes["lon"], chunk_size_lon)):
                    lon_stop = min(primary_ds.sizes["lon"], lon_start + chunk_size_lon)
                    chunk_jobs.append(
                        ((row_idx, col_idx), {"lat": slice(lat_start, lat_stop), "lon": slice(lon_start, lon_stop)})
                    )

            chunk_paths = []
            generated = skipped = failed = 0
            cumulative_valid_pixels = 0
            cumulative_area_ha = 0.0
            existing_manifest_path = D_ACTIVE_RUN / "run_manifest.json"
            resume_allowed = False
            if existing_manifest_path.exists():
                existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
                if existing_manifest.get("parameter_hash") != PARAMETER_HASH:
                    raise ValueError("Existing run manifest parameter hash does not match the active run.")
                resume_allowed = True
            chunk_bar = tqdm(chunk_jobs, disable=not SHOW_PROGRESS, desc=f"{RUN_ID} chunks")
            for (row_idx, col_idx), selection in chunk_bar:
                destination = D_PRIMARY_SUBDIRS["chunks"] / f"chunk_{row_idx:03d}_{col_idx:03d}.nc"
                if destination.exists() and resume_allowed:
                    chunk_paths.append(destination)
                    skipped += 1
                else:
                    try:
                        paths = write_abandonment_chunks(
                            primary_ds,
                            output_dir=D_PRIMARY_SUBDIRS["chunks"],
                            chunk_slices=[selection],
                            chunk_indices=[(row_idx, col_idx)],
                            prefix="chunk",
                        )
                        chunk_paths.extend(paths)
                        generated += 1
                    except Exception:
                        failed += 1
                        if SHOW_PROGRESS:
                            chunk_bar.set_postfix(
                                done=generated, skipped=skipped, failed=failed,
                                valid_pixels=cumulative_valid_pixels,
                                area_ha=f"{cumulative_area_ha:.2f}",
                            )
                        raise
                with xr.open_dataset(destination) as audit_chunk:
                    qualifies = audit_chunk["qualifies_at_cutoff"].fillna(0).astype(np.uint8)
                    chunk_area = cell_area_m2(audit_chunk["lat"], audit_chunk["lon"])
                    cumulative_valid_pixels += int(qualifies.sum().item())
                    cumulative_area_ha += float(chunk_area.where(qualifies == 1, 0).sum().item() / 10_000.0)
                if SHOW_PROGRESS:
                    chunk_bar.set_postfix(
                        done=generated, skipped=skipped, failed=failed,
                        valid_pixels=cumulative_valid_pixels,
                        area_ha=f"{cumulative_area_ha:.2f}",
                    )
            manifest = {
                "run_id": RUN_ID,
                "parameter_hash": PARAMETER_HASH,
                "parameter_snapshot": PARAMETER_SNAPSHOT,
                "input_source": str(INPUT_SOURCES[INPUT_SOURCE_KEY]),
                "source_runs": {key: str(path) for key, path in D_SOURCE_RUN_ROOTS.items()},
                "source_grid_contracts": SOURCE_GRID_CONTRACTS,
                "input_metadata": S0_CONTEXT["input_metadata"],
                "chunk_paths": [str(path) for path in chunk_paths],
                "chunk_stats": {
                    "total": len(chunk_jobs),
                    "generated": generated,
                    "skipped": skipped,
                    "failed": failed,
                    "cutoff_qualified_pixels": cumulative_valid_pixels,
                    "cutoff_qualified_area_ha": cumulative_area_ha,
                },
            }
            manifest_path = D_ACTIVE_RUN / "run_manifest.json"
            manifest_tmp = D_PRIMARY_SUBDIRS["tmp"] / "run_manifest.json.tmp"
            manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(manifest_tmp, manifest_path)
            S0_CONTEXT["chunk_paths"] = chunk_paths
            S0_CONTEXT["manifest"] = manifest
            S0_CONTEXT["chunk_stats"] = manifest["chunk_stats"]
            print(
                f"[CHUNK] total={len(chunk_jobs)} generated={generated} "
                f"skipped={skipped} failed={failed}"
            )
            print(f"[RESULT] cutoff_qualified_pixels={cumulative_valid_pixels} cutoff_qualified_area_ha={cumulative_area_ha:.2f}")
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] chunk writing completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-write-chunks",
        ),
        code(
            """
            stage_started = time.time()
            chunk_paths = [str(path) for path in S0_CONTEXT["chunk_paths"]]
            merge_bar = tqdm(chunk_paths, disable=not SHOW_PROGRESS, desc="merge chunks")
            for file_index, path_str in enumerate(merge_bar, start=1):
                if SHOW_PROGRESS:
                    merge_bar.set_postfix(
                        done=file_index, skipped=0, failed=0,
                        valid_pixels=S0_CONTEXT["chunk_stats"]["cutoff_qualified_pixels"],
                        area_ha=f"{S0_CONTEXT['chunk_stats']['cutoff_qualified_area_ha']:.2f}",
                    )
            merged_ds = xr.open_mfdataset(chunk_paths, combine="by_coords").reindex(
                lat=S0_CONTEXT["primary_detection"]["lat"],
                lon=S0_CONTEXT["primary_detection"]["lon"],
            )
            merged_ds.load()
            merged_ds.close()
            merged_path = D_PRIMARY_SUBDIRS["merged_chunks"] / "abandonment_primary_merged.nc"
            merged_tmp = D_PRIMARY_SUBDIRS["tmp"] / "abandonment_primary_merged.tmp.nc"
            merged_tmp.unlink(missing_ok=True)
            try:
                merged_ds.to_netcdf(merged_tmp)
                os.replace(merged_tmp, merged_path)
            finally:
                merged_tmp.unlink(missing_ok=True)
            merged_ds = xr.open_dataset(merged_path, chunks={})
            S0_CONTEXT["merged_dataset"] = merged_ds
            S0_CONTEXT["merged_path"] = merged_path
            print(f"[CHUNK] merged_file={merged_path}")
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] chunk merge completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-merge-chunks",
        ),
        code(
            """
            stage_started = time.time()
            roundtrip = validate_roundtrip(
                S0_CONTEXT["primary_detection"],
                S0_CONTEXT["merged_dataset"],
            )
            S0_CONTEXT["roundtrip"] = roundtrip
            S0_CONTEXT["primary_materialized"] = S0_CONTEXT["merged_dataset"]
            print(f"[QA] roundtrip={roundtrip}")
            print("[QA] PASS roundtrip validation passed all checks" if roundtrip["passed"] else "[QA] FAIL roundtrip validation did not pass all checks")
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] roundtrip validation completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-roundtrip",
        ),
        markdown(
            """
            ### （4）Key numbers、尺度稳健性与计数模型

            使用WGS84测地面积汇总全国、州级和年份级结果；比较300 m、1 km众数及legacy均值方案，并执行Poisson/Negative Binomial诊断。
            """,
            "s0-title-statistics",
        ),
        code(
            """
            stage_started = time.time()
            primary_ds = S0_CONTEXT["primary_materialized"]
            area = cell_area_m2(primary_ds["lat"], primary_ds["lon"])
            state_frame = S0_CONTEXT["conus_states_gdf"].copy()
            state_frame["state_numeric"] = state_frame[S0_CONTEXT["state_fips_col"]].astype(int)
            state_codes = rasterize_match(
                state_frame,
                primary_ds["lat"].values,
                primary_ds["lon"].values,
                value_column="state_numeric",
            )
            summaries = summarize_key_numbers(primary_ds, area, state_codes=state_codes)
            summaries["overall"]["status"] = "computed"

            def load_grid_exact_retain_mask(source_path, variable_name, stage_name):
                if source_path is None:
                    return None
                source_path = Path(source_path)
                if not source_path.exists():
                    raise FileNotFoundError(f"{stage_name} retain-mask source does not exist: {source_path}")
                with xr.open_dataset(source_path) as mask_ds:
                    if variable_name not in mask_ds:
                        raise KeyError(f"{stage_name} retain-mask variable is missing: {variable_name}")
                    retain_mask = mask_ds[variable_name].squeeze(drop=True).load()
                _, retain_mask = xr.align(primary_ds["current_abandonment"], retain_mask, join="exact")
                values = np.unique(retain_mask.values[np.isfinite(retain_mask.values)])
                if not np.isin(values, [0, 1]).all():
                    raise ValueError(f"{stage_name} retain mask must contain only 0/1, got {values.tolist()}")
                return retain_mask.fillna(0).astype(np.uint8) == 1

            pv_retain_mask = load_grid_exact_retain_mask(
                PV_DEDUP_RETAIN_MASK_SOURCE,
                PV_DEDUP_RETAIN_MASK_VARIABLE,
                "pv_dedup",
            )
            feature_retain_mask = load_grid_exact_retain_mask(
                FEATURE_COMPLETE_RETAIN_MASK_SOURCE,
                FEATURE_COMPLETE_RETAIN_MASK_VARIABLE,
                "feature_complete",
            )
            current_mask = primary_ds["current_abandonment"].fillna(0).astype(np.uint8) == 1
            downstream_rows = []
            pv_selected = None
            if pv_retain_mask is None:
                downstream_rows.append({"metric": "pv_deduplicated", "pixels": pd.NA, "area_ha": np.nan, "status": "not_configured"})
            else:
                pv_selected = current_mask & pv_retain_mask
                downstream_rows.append(
                    {
                        "metric": "pv_deduplicated",
                        "pixels": int(pv_selected.sum().compute().item()),
                        "area_ha": float(area.where(pv_selected, 0).sum().compute().item() / 10_000.0),
                        "status": "computed",
                    }
                )
            if feature_retain_mask is None or pv_selected is None:
                downstream_rows.append({"metric": "complete_features", "pixels": pd.NA, "area_ha": np.nan, "status": "not_configured"})
            else:
                complete_selected = pv_selected & feature_retain_mask
                downstream_rows.append(
                    {
                        "metric": "complete_features",
                        "pixels": int(complete_selected.sum().compute().item()),
                        "area_ha": float(area.where(complete_selected, 0).sum().compute().item() / 10_000.0),
                        "status": "computed",
                    }
                )
            summaries["overall"] = pd.concat(
                [summaries["overall"], pd.DataFrame(downstream_rows)],
                ignore_index=True,
            )
            state_lookup = state_frame[["state_numeric", S0_CONTEXT["state_code_col"], "NAME"]].rename(
                columns={"state_numeric": "state_code", S0_CONTEXT["state_code_col"]: "state_abbreviation", "NAME": "state_name"}
            )
            summaries["by_state"] = summaries["by_state"].merge(state_lookup, on="state_code", how="left")

            year_area_columns = {
                "detected": "detected_area_ha",
                "cutoff_qualified": "cutoff_qualified_area_ha",
                "current_abandonment": "current_area_ha",
                "recultivated": "recultivated_area_ha",
            }
            state_area_columns = year_area_columns
            reconciliation = {"state_area_difference_ha": {}, "year_area_difference_ha": {}, "passed": True}
            for metric, area_column in year_area_columns.items():
                national_area = float(summaries["overall"].loc[summaries["overall"]["metric"] == metric, "area_ha"].iloc[0])
                state_total = float(summaries["by_state"][state_area_columns[metric]].sum()) if area_column in summaries["by_state"] else 0.0
                year_total = float(summaries["by_year"][area_column].sum()) if area_column in summaries["by_year"] else 0.0
                state_difference = state_total - national_area
                year_difference = year_total - national_area
                tolerance_ha = max(1e-6, abs(national_area) * 1e-9)
                reconciliation["state_area_difference_ha"][metric] = state_difference
                reconciliation["year_area_difference_ha"][metric] = year_difference
                reconciliation["passed"] = bool(
                    reconciliation["passed"]
                    and abs(state_difference) <= tolerance_ha
                    and abs(year_difference) <= tolerance_ha
                )
            S0_CONTEXT["area_m2"] = area
            S0_CONTEXT["state_codes"] = state_codes
            S0_CONTEXT["key_number_summaries"] = summaries
            S0_CONTEXT["key_number_reconciliation"] = reconciliation
            S0_CONTEXT["downstream_stage_complete"] = bool(
                pv_retain_mask is not None and feature_retain_mask is not None
            )
            for result_row in summaries["overall"].itertuples(index=False):
                if result_row.status == "computed":
                    print(
                        f"[RESULT] metric={result_row.metric} pixels={result_row.pixels} "
                        f"area_ha={result_row.area_ha:.2f} area_mha={result_row.area_ha / 1_000_000.0:.6f} status=computed"
                    )
                else:
                    print(f"[RESULT] metric={result_row.metric} pixels=not_evaluated area_ha=not_evaluated status={result_row.status}")
            print(f"[RESULT] state_rows={len(summaries['by_state'])} year_rows={len(summaries['by_year'])}")
            print(f"[QA] key_number_reconciliation={reconciliation}")
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] key-number summary completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-key-numbers",
        ),
        code(
            """
            stage_started = time.time()
            robustness_rows = []
            legacy_mean_round = (
                S0_CONTEXT["landcover_300m_filtered"]
                .coarsen(lat=MODE_AGG_FACTOR, lon=MODE_AGG_FACTOR, boundary="trim")
                .mean()
                .round()
                .astype(np.uint8)
            )
            scenario_defs = [
                {"name": "native_300m", "branch": "300m_native", "min_years": MIN_ABANDONMENT_YEARS, "cutoff": REQUIRED_THROUGH_YEAR, "excluded": {6, 7}, "detector": "stable_crop"},
                {"name": "mode_1km", "branch": "1km_mode", "min_years": MIN_ABANDONMENT_YEARS, "cutoff": REQUIRED_THROUGH_YEAR, "excluded": {6, 7}, "detector": "stable_crop"},
                {"name": "legacy_mean_round", "branch": "legacy_mean_round", "min_years": MIN_ABANDONMENT_YEARS, "cutoff": REQUIRED_THROUGH_YEAR, "excluded": {6, 7}, "detector": "stable_crop"},
                {"name": "min_7", "branch": PRIMARY_GRID, "min_years": 7, "cutoff": REQUIRED_THROUGH_YEAR, "excluded": {6, 7}, "detector": "stable_crop"},
                {"name": "min_10", "branch": PRIMARY_GRID, "min_years": 10, "cutoff": REQUIRED_THROUGH_YEAR, "excluded": {6, 7}, "detector": "stable_crop"},
                {"name": "cutoff_2018", "branch": PRIMARY_GRID, "min_years": MIN_ABANDONMENT_YEARS, "cutoff": 2018, "excluded": {6, 7}, "detector": "stable_crop"},
                {"name": "cutoff_2022", "branch": PRIMARY_GRID, "min_years": MIN_ABANDONMENT_YEARS, "cutoff": 2022, "excluded": {6, 7}, "detector": "stable_crop"},
                {"name": "settlement_only", "branch": PRIMARY_GRID, "min_years": MIN_ABANDONMENT_YEARS, "cutoff": REQUIRED_THROUGH_YEAR, "excluded": {7}, "detector": "stable_crop"},
                {"name": "strict_post_classes", "branch": PRIMARY_GRID, "min_years": MIN_ABANDONMENT_YEARS, "cutoff": REQUIRED_THROUGH_YEAR, "excluded": {6, 7, 8, 9}, "detector": "stable_crop"},
                {"name": "xie_window_proxy", "branch": PRIMARY_GRID, "min_years": MIN_ABANDONMENT_YEARS, "cutoff": REQUIRED_THROUGH_YEAR, "excluded": {7}, "detector": "xie_window"},
            ]
            scenario_bar = tqdm(scenario_defs, disable=not SHOW_PROGRESS, desc="robustness")
            for scenario in scenario_bar:
                branch_da = legacy_mean_round if scenario["branch"] == "legacy_mean_round" else S0_CONTEXT["grid_branches"][scenario["branch"]]
                scenario_ds = detect_abandonment(
                    branch_da,
                    schema="simplified9",
                    built_up_codes=tuple(sorted({7} & scenario["excluded"])),
                    wetland_codes=tuple(sorted(scenario["excluded"] - {7})),
                    baseline_years=BASELINE_CROP_YEARS,
                    min_abandonment_years=scenario["min_years"],
                    required_through_year=scenario["cutoff"],
                    analysis_end_year=max(ANALYSIS_END_YEAR, scenario["cutoff"]),
                    recultivation_years=RECULTIVATION_YEARS,
                    detector=scenario["detector"],
                    pre_window=5,
                    post_window=5,
                    max_noncrop_pre_years=1 if scenario["detector"] == "xie_window" else 0,
                    max_crop_post_years=1 if scenario["detector"] == "xie_window" else 0,
                    time_dim="time",
                )
                baseline_year_index = pd.to_datetime(branch_da["time"].values).year
                baseline_mask = (baseline_year_index >= BASELINE_CROP_YEARS[0]) & (baseline_year_index <= BASELINE_CROP_YEARS[1])
                scenario_ds["eligible_cropland"] = (
                    branch_da.isel(time=np.where(baseline_mask)[0]).fillna(0).astype(np.uint8) == 1
                ).all("time").astype(np.uint8)
                scenario_area = cell_area_m2(scenario_ds["lat"], scenario_ds["lon"])
                summary = summarize_key_numbers(scenario_ds, scenario_area)["overall"]
                qualifies_row = summary.loc[summary["metric"] == "cutoff_qualified"].iloc[0]
                robustness_rows.append(
                    {
                        "scenario": scenario["name"],
                        "grid": scenario["branch"],
                        "min_years": scenario["min_years"],
                        "cutoff_year": scenario["cutoff"],
                        "qualifies_pixel_count": int(qualifies_row["pixels"]),
                        "qualifies_area_ha": float(qualifies_row["area_ha"]),
                    }
                )
                if SHOW_PROGRESS:
                    scenario_bar.set_postfix(
                        done=scenario["name"], skipped=0, failed=0,
                        valid_pixels=int(qualifies_row["pixels"]),
                        area_ha=f"{qualifies_row['area_ha']:.2f}",
                    )
            S0_CONTEXT["robustness_df"] = pd.DataFrame(robustness_rows)
            print(f"[RESULT] robustness_scenarios={len(S0_CONTEXT['robustness_df'])}")
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] robustness evaluation completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-robustness",
        ),
        code(
            """
            stage_started = time.time()
            poisson_results = None
            if sm is None:
                print("[QA] statsmodels unavailable; skipping Poisson/Negative Binomial robustness")
            else:
                counties = gpd.read_file(COUNTY_SHAPEFILE).to_crs("EPSG:4326")
                primary_ds = S0_CONTEXT["primary_materialized"]
                county_codes = rasterize_match(
                    counties.assign(county_numeric=np.arange(1, len(counties) + 1)),
                    primary_ds["lat"].values,
                    primary_ds["lon"].values,
                    value_column="county_numeric",
                )
                event_years = primary_ds["abandonment_year"].where(primary_ds["qualifies_at_cutoff"] == 1)
                rows = []
                for year in sorted(int(v) for v in np.unique(event_years.values[np.isfinite(event_years.values)])):
                    year_mask = event_years == year
                    for county_code in sorted(int(v) for v in np.unique(county_codes.values[np.isfinite(county_codes.values)]) if int(v) != 0):
                        county_mask = county_codes == county_code
                        response = int((year_mask & county_mask).sum().compute().item())
                        exposure = int(primary_ds["eligible_cropland"].where(county_mask, 0).sum().compute().item())
                        if exposure > 0:
                            rows.append({"county_code": county_code, "year": year, "events": response, "eligible_pixels": exposure})
                county_year = pd.DataFrame(rows)
                if len(county_year) >= 10:
                    county_year["intercept"] = 1.0
                    poisson_model = sm.GLM(
                        county_year["events"],
                        county_year[["intercept"]],
                        family=sm.families.Poisson(),
                        offset=np.log(county_year["eligible_pixels"]),
                    ).fit()
                    dispersion = float(((poisson_model.resid_pearson ** 2).sum()) / poisson_model.df_resid)
                    poisson_results = {"pearson_dispersion": dispersion, "n_rows": len(county_year)}
                    if dispersion > 1.5 and NegativeBinomial is not None:
                        nb_model = NegativeBinomial(
                            county_year["events"],
                            county_year[["intercept"]],
                            offset=np.log(county_year["eligible_pixels"]),
                        ).fit(disp=False)
                        poisson_results["negative_binomial_llf"] = float(nb_model.llf)
                    print(f"[RESULT] county_year_rows={len(county_year)} pearson_dispersion={dispersion:.4f}")
                else:
                    print("[QA] county-year table too small for stable Poisson estimation; skipping fit")
            S0_CONTEXT["poisson_results"] = poisson_results
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] Poisson robustness completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-poisson",
        ),
        markdown(
            """
            ### （5）GLBRC/Xie外部验证与验收发布

            在共同可检测年份内对齐GLBRC事件年份栅格，按数据源隔离保存比较结果；仅在QA门槛通过后发布仓库manifest、汇总CSV和报告。
            """,
            "s0-title-external",
        ),
        code(
            """
            stage_started = time.time()
            external_bar = tqdm(total=1, disable=not SHOW_PROGRESS, desc="external sources")
            if not XIE_ARCHIVE_LOCAL_PATH.exists() and not ALLOW_NETWORK_DOWNLOADS:
                raise FileNotFoundError(
                    f"Missing Xie archive and downloads are disabled: {XIE_ARCHIVE_LOCAL_PATH}"
                )

            xie_path = fetch_reference_raster(
                archive_source=XIE_DIRECT_DOWNLOAD_URL if ALLOW_NETWORK_DOWNLOADS else None,
                archive_path=XIE_ARCHIVE_LOCAL_PATH,
                archive_member_name=XIE_ARCHIVE_MEMBER,
                extracted_raster_path=XIE_TIFF_LOCAL_PATH,
                expected_archive_sha256=XIE_ARCHIVE_SHA256,
                expected_archive_size=XIE_ARCHIVE_SIZE_BYTES,
                expected_raster_sha256=XIE_TIFF_SHA256,
                expected_raster_size=XIE_TIFF_SIZE_BYTES,
            )
            xie_meta = validate_reference_raster(
                xie_path,
                expected_sha256=XIE_TIFF_SHA256,
                expected_size=XIE_TIFF_SIZE_BYTES,
                expected_crs="EPSG:4326",
                expected_count=1,
                expected_dtype="int16",
                expected_nodata=0,
                expected_year_min=XIE_VALID_YEAR_MIN,
                expected_year_max=XIE_VALID_YEAR_MAX,
            )
            xie_year = open_reference_abandonment_year(xie_path, nodata=0)
            xie_readme = XIE_TIFF_LOCAL_PATH.with_name("readme.txt")
            readme_range = None
            if xie_readme.exists():
                readme_text = xie_readme.read_text(encoding="utf-8", errors="ignore")
                matched = re.search(r"Pixel values in the dataset range from (\\d{4}) to (\\d{4})", readme_text)
                if matched:
                    readme_range = f"{matched.group(1)}-{matched.group(2)}"

            print(f"[PATH] xie_source_page={XIE_SOURCE_PAGE_URL}")
            print(f"[PATH] xie_direct_url={XIE_DIRECT_DOWNLOAD_URL}")
            print(f"[PATH] xie_archive={XIE_ARCHIVE_LOCAL_PATH}")
            print(f"[PATH] xie_local_tiff={xie_path}")
            print(f"[INPUT] xie_meta={xie_meta}")
            print(f"[INPUT] xie_readme_year_range={readme_range}")
            print(f"[INPUT] geotiff_grid_contract={json.dumps(SOURCE_GRID_CONTRACTS['xie_2024_30m'], ensure_ascii=False)}")

            primary_ds = S0_CONTEXT.get("primary_materialized")
            if primary_ds is None:
                print("[QA] primary detection unavailable; skipping Xie alignment summary")
                if SHOW_PROGRESS:
                    external_bar.set_postfix(done=0, skipped=1, failed=0, valid_pixels=0, area_ha="0.00")
            else:
                comparison_years = (
                    max(BASELINE_CROP_YEARS[1] + 1, XIE_VALID_YEAR_MIN),
                    min(ANALYSIS_END_YEAR, XIE_VALID_YEAR_MAX),
                )
                if comparison_years[0] > comparison_years[1]:
                    raise ValueError(f"No common detectable interval for Xie comparison: {comparison_years}")
                xie_aligned = align_reference_abandonment(
                    xie_year,
                    primary_ds["abandonment_year"],
                    method="mode",
                )
                xie_california_lccs_path = None
                if EXECUTION_MODE == "california":
                    xie_california_lccs_path = D_XIE_SUBDIRS["intermediate"] / "xie_2024_30m_ca_lccs.nc"
                    xie_california_lccs_tmp = D_XIE_SUBDIRS["tmp"] / "xie_2024_30m_ca_lccs.tmp.nc"
                    xie_california_lccs_tmp.unlink(missing_ok=True)
                    xie_california_ds = xie_aligned.fillna(0).astype(np.int16).rename("abandonment_year").to_dataset()
                    xie_california_ds.attrs.update(
                        {
                            "source_key": "xie_2024_30m",
                            "source_path": str(XIE_TIFF_LOCAL_PATH),
                            "source_semantics": "single_band_abandonment_event_year_not_annual_lccs",
                            "aoi": "california_approximate_bbox",
                            "aoi_bounds_epsg4326": json.dumps(CALIFORNIA_BOUNDS, sort_keys=True),
                            "alignment_target": f"{INPUT_SOURCE_KEY}:{PRIMARY_GRID}",
                            "alignment_method": "categorical_mode",
                            "comparison_years": f"{comparison_years[0]}-{comparison_years[1]}",
                            "parameter_hash": PARAMETER_HASH,
                        }
                    )
                    try:
                        xie_california_ds.to_netcdf(
                            xie_california_lccs_tmp,
                            encoding={
                                "abandonment_year": {
                                    "dtype": "int16",
                                    "zlib": True,
                                    "complevel": 4,
                                    "shuffle": True,
                                    "_FillValue": np.int16(0),
                                }
                            },
                        )
                        os.replace(xie_california_lccs_tmp, xie_california_lccs_path)
                    finally:
                        xie_california_lccs_tmp.unlink(missing_ok=True)
                    print(f"[PATH] xie_california_lccs={xie_california_lccs_path}")
                comparison = compare_reference_abandonment(
                    primary_ds,
                    xie_aligned,
                    S0_CONTEXT["area_m2"],
                    state_codes=S0_CONTEXT["state_codes"],
                    comparison_years=comparison_years,
                )
                S0_CONTEXT["reference_abandonment_year"] = xie_aligned
                S0_CONTEXT["external_check_df"] = comparison["overall"]
                S0_CONTEXT["external_comparison_years"] = comparison_years
                comparison_paths = {
                    "overall": D_XIE_SUBDIRS["diagnostics"] / "xie_external_comparison_overall.csv",
                    "by_state": D_XIE_SUBDIRS["diagnostics"] / "xie_external_comparison_by_state.csv",
                    "by_year": D_XIE_SUBDIRS["diagnostics"] / "xie_external_comparison_by_year.csv",
                }
                for comparison_name, comparison_path in comparison_paths.items():
                    comparison_tmp = D_XIE_SUBDIRS["tmp"] / f"{comparison_path.name}.tmp"
                    try:
                        comparison[comparison_name].to_csv(comparison_tmp, index=False)
                        os.replace(comparison_tmp, comparison_path)
                    finally:
                        comparison_tmp.unlink(missing_ok=True)
                S0_CONTEXT["external_check_paths"] = comparison_paths
                S0_CONTEXT["xie_california_lccs_path"] = xie_california_lccs_path
                reference_valid_pixels = int(xie_aligned.notnull().sum().item())
                reference_area_ha = float(S0_CONTEXT["area_m2"].where(xie_aligned.notnull(), 0).sum().item() / 10_000.0)
                print(
                    f"[RESULT] external_rows={len(comparison['overall'])} "
                    f"state_rows={len(comparison['by_state'])} comparison_years={comparison_years[0]}-{comparison_years[1]}"
                )
                if SHOW_PROGRESS:
                    external_bar.set_postfix(
                        done=1, skipped=0, failed=0,
                        valid_pixels=reference_valid_pixels,
                        area_ha=f"{reference_area_ha:.2f}",
                    )
                if EXECUTION_MODE == "california":
                    california_summary_paths = {
                        "overall": D_PRIMARY_SUBDIRS["diagnostics"] / "california_abandonment_key_numbers.csv",
                        "by_state": D_PRIMARY_SUBDIRS["diagnostics"] / "california_abandonment_by_state.csv",
                        "by_year": D_PRIMARY_SUBDIRS["diagnostics"] / "california_abandonment_by_year.csv",
                    }
                    for summary_name, summary_path in california_summary_paths.items():
                        summary_tmp = D_PRIMARY_SUBDIRS["tmp"] / f"{summary_path.name}.tmp"
                        try:
                            S0_CONTEXT["key_number_summaries"][summary_name].to_csv(summary_tmp, index=False)
                            os.replace(summary_tmp, summary_path)
                        finally:
                            summary_tmp.unlink(missing_ok=True)

                    california_validation = {
                        "run_id": RUN_ID,
                        "parameter_hash": PARAMETER_HASH,
                        "execution_mode": EXECUTION_MODE,
                        "aoi": "california_approximate_bbox",
                        "aoi_bounds_epsg4326": CALIFORNIA_BOUNDS,
                        "target_year": ANALYSIS_END_YEAR,
                        "comparison_years": list(comparison_years),
                        "source_files": {
                            INPUT_SOURCE_KEY: {
                                "path": str(S0_CONTEXT["california_lccs_path"]),
                                "size_bytes": S0_CONTEXT["california_lccs_path"].stat().st_size,
                                "sha256": sha256_file(S0_CONTEXT["california_lccs_path"]),
                                "semantics": "annual_simplified9_lccs_1992_2022",
                            },
                            f"{INPUT_SOURCE_KEY}_1km_mode": {
                                "path": str(S0_CONTEXT["california_1km_lccs_path"]),
                                "size_bytes": S0_CONTEXT["california_1km_lccs_path"].stat().st_size,
                                "sha256": sha256_file(S0_CONTEXT["california_1km_lccs_path"]),
                                "semantics": "five_year_temporal_majority_then_3x3_categorical_mode_lccs",
                            },
                            "xie_2024_30m": {
                                "path": str(xie_california_lccs_path),
                                "size_bytes": xie_california_lccs_path.stat().st_size,
                                "sha256": sha256_file(xie_california_lccs_path),
                                "semantics": "aligned_single_band_abandonment_event_year",
                            },
                        },
                        "outputs": {
                            "merged_detection": str(S0_CONTEXT["merged_path"]),
                            "execution_log": str(S0_CONTEXT.get("execution_log_path", "")),
                            "summaries": {key: str(path) for key, path in california_summary_paths.items()},
                            "external_comparisons": {key: str(path) for key, path in comparison_paths.items()},
                        },
                        "qa": {
                            "roundtrip": S0_CONTEXT["roundtrip"],
                            "key_number_reconciliation": S0_CONTEXT["key_number_reconciliation"],
                            "chunk_stats": S0_CONTEXT["chunk_stats"],
                        },
                    }
                    california_validation_path = D_PRIMARY_SUBDIRS["diagnostics"] / "california_validation_chain.json"
                    california_validation_tmp = D_PRIMARY_SUBDIRS["tmp"] / "california_validation_chain.json.tmp"
                    try:
                        california_validation_tmp.write_text(
                            json.dumps(california_validation, indent=2, ensure_ascii=False, default=json_default),
                            encoding="utf-8",
                        )
                        os.replace(california_validation_tmp, california_validation_path)
                    finally:
                        california_validation_tmp.unlink(missing_ok=True)

                    run_manifest_path = D_ACTIVE_RUN / "run_manifest.json"
                    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
                    run_manifest["california_validation_chain"] = california_validation
                    run_manifest_tmp = D_PRIMARY_SUBDIRS["tmp"] / "run_manifest.california.json.tmp"
                    try:
                        run_manifest_tmp.write_text(
                            json.dumps(run_manifest, indent=2, ensure_ascii=False, default=json_default),
                            encoding="utf-8",
                        )
                        os.replace(run_manifest_tmp, run_manifest_path)
                    finally:
                        run_manifest_tmp.unlink(missing_ok=True)
                    S0_CONTEXT["california_validation_path"] = california_validation_path
                    print(f"[QA] california_validation_chain={california_validation_path}")
            external_bar.update(1)
            external_bar.close()
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] external benchmark check completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-external-check",
        ),
        code(
            """
            stage_started = time.time()
            if EXECUTION_MODE != "full":
                raise RuntimeError("Smoke runs are diagnostic-only and cannot publish scientific summaries.")
            qa_result = S0_CONTEXT.get("roundtrip") or {}
            chunk_stats = S0_CONTEXT.get("chunk_stats", {})
            chunk_paths = S0_CONTEXT.get("chunk_paths", [])
            chunk_complete = bool(
                chunk_stats.get("failed", 1) == 0
                and chunk_stats.get("total", -1) == len(chunk_paths)
                and all(Path(path).exists() for path in chunk_paths)
            )
            reconciliation = S0_CONTEXT.get("key_number_reconciliation") or {}
            external_paths = S0_CONTEXT.get("external_check_paths") or {}
            external_complete = bool(external_paths and all(Path(path).exists() for path in external_paths.values()))
            stale_tmp_files = [str(path) for path in D_ACTIVE_RUN.rglob("*.tmp*") if path.is_file()]
            qa_gate = {
                "roundtrip_passed": bool(qa_result.get("passed", False)),
                "chunk_complete": chunk_complete,
                "key_number_reconciliation_passed": bool(reconciliation.get("passed", False)),
                "downstream_stage_complete": bool(S0_CONTEXT.get("downstream_stage_complete", False)),
                "external_comparison_complete": external_complete,
                "no_stale_tmp_files": not stale_tmp_files,
                "stale_tmp_files": stale_tmp_files,
            }
            qa_gate["passed"] = all(
                qa_gate[key]
                for key in (
                    "roundtrip_passed",
                    "chunk_complete",
                    "key_number_reconciliation_passed",
                    "external_comparison_complete",
                    "no_stale_tmp_files",
                )
            )
            if not qa_gate["passed"]:
                raise RuntimeError(f"Refusing to publish because the acceptance gate failed: {qa_gate}")
            manifest = {
                "run_id": RUN_ID,
                "parameter_hash": PARAMETER_HASH,
                "parameter_snapshot": PARAMETER_SNAPSHOT,
                "input_source": str(INPUT_SOURCES[INPUT_SOURCE_KEY]),
                "d_run_root": str(D_ACTIVE_RUN),
                "source_runs": {key: str(path) for key, path in D_SOURCE_RUN_ROOTS.items()},
                "source_grid_contracts": SOURCE_GRID_CONTRACTS,
                "external_comparison_years": list(S0_CONTEXT.get("external_comparison_years", ())),
                "input_metadata": S0_CONTEXT.get("input_metadata", {}),
                "chunk_stats": chunk_stats,
                "qa": {
                    "gate": qa_gate,
                    "roundtrip": qa_result,
                    "key_number_reconciliation": reconciliation,
                },
                "xie_reference": {
                    "source_page": XIE_SOURCE_PAGE_URL,
                    "download_url": XIE_DIRECT_DOWNLOAD_URL,
                    "archive_path": str(XIE_ARCHIVE_LOCAL_PATH),
                    "archive_sha256": XIE_ARCHIVE_SHA256,
                    "raster_path": str(XIE_TIFF_LOCAL_PATH),
                    "raster_sha256": XIE_TIFF_SHA256,
                    "diagnostic_paths": {key: str(path) for key, path in external_paths.items()},
                    "comparison_summary": json.loads(
                        S0_CONTEXT.get("external_check_df", pd.DataFrame()).to_json(orient="records")
                    ),
                },
            }
            summaries = {
                "abandonment_key_numbers": S0_CONTEXT.get("key_number_summaries", {}).get("overall", pd.DataFrame()),
                "abandonment_by_state": S0_CONTEXT.get("key_number_summaries", {}).get("by_state", pd.DataFrame()),
                "abandonment_by_year": S0_CONTEXT.get("key_number_summaries", {}).get("by_year", pd.DataFrame()),
                "abandonment_robustness": S0_CONTEXT.get("robustness_df", pd.DataFrame()),
            }
            report_text = "\\n".join(
                [
                    "# U.S. cropland abandonment validation report",
                    "",
                    f"- Run ID: {RUN_ID}",
                    f"- Parameter hash: {PARAMETER_HASH}",
                    f"- Input source: {INPUT_SOURCES[INPUT_SOURCE_KEY]}",
                    f"- Source runs: {json.dumps({key: str(path) for key, path in D_SOURCE_RUN_ROOTS.items()}, ensure_ascii=False)}",
                    f"- Source grid contracts: {json.dumps(SOURCE_GRID_CONTRACTS, ensure_ascii=False)}",
                    f"- External comparison years: {list(S0_CONTEXT.get('external_comparison_years', ()))}",
                    f"- Legacy comparison value: 4.703357 Mha",
                    f"- Xie source page: {XIE_SOURCE_PAGE_URL}",
                    f"- Xie direct URL: {XIE_DIRECT_DOWNLOAD_URL}",
                    f"- Acceptance gate: {json.dumps(qa_gate, ensure_ascii=False)}",
                    f"- Round-trip QA: {json.dumps(qa_result, ensure_ascii=False)}",
                    f"- Key-number reconciliation: {json.dumps(reconciliation, ensure_ascii=False)}",
                    f"- Downstream PV/feature stages complete: {S0_CONTEXT.get('downstream_stage_complete', False)}",
                ]
            )
            d_manifest_path = D_ACTIVE_RUN / "run_manifest.json"
            d_manifest_tmp = D_PRIMARY_SUBDIRS["tmp"] / "run_manifest.final.json.tmp"
            try:
                d_manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                os.replace(d_manifest_tmp, d_manifest_path)
            finally:
                d_manifest_tmp.unlink(missing_ok=True)
            published = publish_validation_results(
                run_id=RUN_ID,
                repo_result_root=REPO_RESULT_ROOT,
                manifest=manifest,
                summaries=summaries,
                report_text=report_text,
            )
            repo_manifest = json.loads(published["manifest"].read_text(encoding="utf-8"))
            d_manifest = json.loads(d_manifest_path.read_text(encoding="utf-8"))
            if repo_manifest != d_manifest:
                raise RuntimeError("Repository and D-drive manifests differ after publication.")
            S0_CONTEXT["published"] = published
            print(f"[PUBLISH] manifest={published['manifest']}")
            print(f"[PUBLISH] key_numbers={published['abandonment_key_numbers']}")
            print(f"[PUBLISH] report={published['report']}")
            print(f"[QA] manifest_identity=True d_manifest={d_manifest_path}")
            if PRINT_STAGE_SUMMARY:
                print(f"[STAGE] publish completed in {time.time() - stage_started:.2f}s")
            """,
            "s0-abandon-publish",
        ),
        markdown(
            """
            ## 4、Mode修正后的全球旧版等价撂荒重建

            本节只替换错误的1 km分类均值重采样，保持旧版候选mask、正则事件和两年复耕定义，以便将新旧结果差异归因于 `3×3 categorical mode`。严格S0规则仍由上一节独立审计，不进入本节训练数据。

            | 阶段 | 主要函数 | 输出 |
            |---|---|---|
            | 写盘门槛 | `run_mode_readiness()` | 正确1 km输入验收快照 |
            | 候选mask与检测 | `run_mode_mask()`、`run_mode_detection()` | `final_mask_1km_new.nc`、500×500 chunks |
            | Unit QA | `run_mode_chunk_qa()` | 预期键、坐标、dtype、数值抽样报告 |
            | Landcover贴合 | `run_mode_merge()` | 含31年 `landcover` 的合并chunks |
            | 一键运行 | `run_mode_end_to_end()` | 全阶段manifest与Embedding交接信息 |
            """,
            "mode-chain-overview",
        ),
        markdown(
            """
            ### （1）正确1 km文件的持续探针与参数

            每60秒检查文件大小和mtime；连续10分钟稳定后必须通过解除锁定、xarray重开、维度、坐标、类别样本和attrs验收。旧版compression level 9与新版level 4的字节比只记录，不作为硬门槛。
            """,
            "mode-chain-monitor-title",
        ),
        code(
            r'''
            import importlib.util
            import json
            import os
            import tempfile
            from pathlib import Path

            import numpy as np
            import xarray as xr
            from tqdm.auto import tqdm

            MODE_RECLASS_PATH = Path(r"D:\xarray\reclass_lccs_1km.nc")
            HISTORY_RECLASS_PATH = Path(r"D:\xarray\history_data\reclass_lccs_1km.nc")
            MODE_MASK_PATH = Path(r"D:\xarray\final_mask_1km_new.nc")
            MODE_ABANDON_DIR = Path(r"D:\xarray\abandon_2")
            MODE_MERGED_DIR = Path(r"D:\xarray\merged_chunk_2")
            MODE_CHUNK_SIZE = 500
            MODE_MONITOR_POLL_SECONDS = 60
            MODE_MONITOR_STABLE_SECONDS = 600
            MODE_MONITOR_TIMEOUT_SECONDS = 24 * 60 * 60
            MODE_RANDOM_SEED = 20260818

            module_path = Path("function/cropland_abandonment.py").resolve()
            module_spec = importlib.util.spec_from_file_location("legacy_mode_pipeline", module_path)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(module_path)
            mode_pipeline = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(mode_pipeline)

            monitor_mode_reclass_file = mode_pipeline.monitor_mode_reclass_file
            build_legacy_candidate_mask = mode_pipeline.build_legacy_candidate_mask
            candidate_chunk_keys = mode_pipeline.candidate_chunk_keys
            write_legacy_equivalent_chunks = mode_pipeline.write_legacy_equivalent_chunks
            validate_legacy_chunk_set = mode_pipeline.validate_legacy_chunk_set
            merge_landcover_into_legacy_chunks = mode_pipeline.merge_landcover_into_legacy_chunks

            MODE_CONTEXT = {}
            print(f"[CONFIG] corrected_1km={MODE_RECLASS_PATH}")
            print(f"[PATH] abandon_chunks={MODE_ABANDON_DIR}")
            print(f"[PATH] merged_chunks={MODE_MERGED_DIR}")
            ''',
            "mode-chain-parameters",
        ),
        code(
            """
            def run_mode_readiness():
                progress = tqdm(total=MODE_MONITOR_STABLE_SECONDS, desc="1 km write stability", unit="s")

                def update_probe(probe):
                    stable = int(probe.get("stable_seconds", 0))
                    progress.n = min(stable, MODE_MONITOR_STABLE_SECONDS)
                    progress.set_postfix(
                        bytes=probe.get("size_bytes", 0),
                        stable_s=stable,
                        ready=False,
                    )
                    progress.refresh()

                try:
                    readiness = monitor_mode_reclass_file(
                        MODE_RECLASS_PATH,
                        poll_seconds=MODE_MONITOR_POLL_SECONDS,
                        stable_seconds=MODE_MONITOR_STABLE_SECONDS,
                        timeout_seconds=MODE_MONITOR_TIMEOUT_SECONDS,
                        progress_callback=update_probe,
                    )
                finally:
                    progress.close()
                old_size = HISTORY_RECLASS_PATH.stat().st_size if HISTORY_RECLASS_PATH.exists() else None
                readiness["history_size_bytes"] = old_size
                readiness["size_ratio_diagnostic"] = (
                    readiness["size_bytes"] / old_size if old_size else None
                )
                readiness["size_comparable"] = False
                readiness["size_comparison_note"] = "new compression level 4; history compression level 9"
                MODE_ABANDON_DIR.mkdir(parents=True, exist_ok=True)
                MODE_MERGED_DIR.mkdir(parents=True, exist_ok=True)
                MODE_CONTEXT["readiness"] = readiness
                print(f"[QA] corrected_1km_ready=True size_bytes={readiness['size_bytes']}")
                return readiness
            """,
            "mode-chain-monitor",
        ),
        markdown(
            """
            ### （2）旧版等价候选Mask与500×500检测Chunk

            候选条件和正则事件规则保持旧版；仅使用正确mode分类输入。Chunk键由新mask推导，允许与旧版974块不同。
            """,
            "mode-chain-detection-title",
        ),
        code(
            """
            def run_mode_mask():
                if not MODE_CONTEXT.get("readiness", {}).get("passed"):
                    raise RuntimeError("Corrected 1 km input has not passed readiness QA")
                with xr.open_dataset(
                    MODE_RECLASS_PATH,
                    chunks={"time": -1, "lat": MODE_CHUNK_SIZE, "lon": MODE_CHUNK_SIZE},
                ) as source:
                    mask = build_legacy_candidate_mask(source["lccs_class"])
                    mask_dataset = mask.to_dataset()
                    mask_dataset.attrs.update(
                        {
                            "source_path": str(MODE_RECLASS_PATH),
                            "source_size_bytes": MODE_CONTEXT["readiness"]["size_bytes"],
                            "detector_contract": "legacy_equivalent_mode_resample_only",
                        }
                    )
                    temporary = MODE_MASK_PATH.with_suffix(".tmp.nc")
                    temporary.unlink(missing_ok=True)
                    try:
                        mask_dataset.to_netcdf(
                            temporary,
                            encoding={
                                "final_mask": {
                                    "dtype": "uint8",
                                    "zlib": True,
                                    "complevel": 4,
                                    "shuffle": True,
                                    "_FillValue": None,
                                    "chunksizes": (MODE_CHUNK_SIZE, MODE_CHUNK_SIZE),
                                }
                            },
                        )
                        os.replace(temporary, MODE_MASK_PATH)
                    finally:
                        temporary.unlink(missing_ok=True)
                with xr.open_dataset(MODE_MASK_PATH, chunks={"lat": MODE_CHUNK_SIZE, "lon": MODE_CHUNK_SIZE}) as mask_source:
                    keys = candidate_chunk_keys(mask_source["final_mask"], chunk_size=MODE_CHUNK_SIZE)
                MODE_CONTEXT["expected_keys"] = keys
                MODE_CONTEXT["mask_path"] = MODE_MASK_PATH
                print(f"[RESULT] candidate_chunk_keys={len(keys)} mask={MODE_MASK_PATH}")
                return keys
            """,
            "mode-chain-mask",
        ),
        code(
            """
            def run_mode_detection():
                keys = MODE_CONTEXT.get("expected_keys")
                if keys is None:
                    raise RuntimeError("Candidate mask stage has not completed")
                progress = tqdm(total=len(keys), desc="legacy-equivalent chunks", unit="chunk")

                def update_chunk(status):
                    progress.n = status["generated"] + status["skipped"]
                    progress.set_postfix(
                        generated=status["generated"],
                        skipped=status["skipped"],
                        failed=status["failed"],
                    )
                    progress.refresh()

                try:
                    manifest = write_legacy_equivalent_chunks(
                        MODE_RECLASS_PATH,
                        MODE_MASK_PATH,
                        MODE_ABANDON_DIR,
                        chunk_size=MODE_CHUNK_SIZE,
                        expected_keys=keys,
                        resume=True,
                        progress_callback=update_chunk,
                    )
                finally:
                    progress.close()
                MODE_CONTEXT["detection_manifest"] = manifest
                print(f"[CHUNK] detection={manifest['stats']}")
                return manifest
            """,
            "mode-chain-detect-chunks",
        ),
        markdown(
            """
            ### （3）Unit Check：完整性、编码与抽样逐值复算

            检查预期键集合、坐标、边界块、变量、年份/持续时间、0/1编码和临时文件；固定随机种子抽样候选像元并从源时间序列重算。
            """,
            "mode-chain-qa-title",
        ),
        code(
            """
            def run_mode_chunk_qa():
                report = validate_legacy_chunk_set(
                    MODE_RECLASS_PATH,
                    MODE_MASK_PATH,
                    MODE_ABANDON_DIR,
                    chunk_size=MODE_CHUNK_SIZE,
                    sample_pixels=2000,
                    random_seed=MODE_RANDOM_SEED,
                )
                MODE_CONTEXT["chunk_qa"] = report
                print(f"[QA] legacy_chunk_validation={report}")
                if not report["passed"]:
                    raise RuntimeError("Legacy-equivalent chunk validation failed")
                return report
            """,
            "mode-chain-qa",
        ),
        markdown(
            """
            ### （4）精确贴合31年Landcover并建立新Merged Chunk目录

            `lccs_class` 重命名为 `landcover`；同一网格必须精确选择并通过 `xr.align(join="exact")`，不使用nearest插值掩盖偏移。
            """,
            "mode-chain-merge-title",
        ),
        code(
            """
            def run_mode_merge():
                if not MODE_CONTEXT.get("chunk_qa", {}).get("passed"):
                    raise RuntimeError("Chunk QA must pass before landcover merge")
                progress = tqdm(
                    total=MODE_CONTEXT["chunk_qa"]["actual_count"],
                    desc="merge exact landcover",
                    unit="chunk",
                )

                def update_merge(status):
                    progress.n = status["done"]
                    progress.set_postfix(total=status["total"], failed=0)
                    progress.refresh()

                try:
                    report = merge_landcover_into_legacy_chunks(
                        MODE_RECLASS_PATH,
                        MODE_ABANDON_DIR,
                        MODE_MERGED_DIR,
                        progress_callback=update_merge,
                    )
                finally:
                    progress.close()
                MODE_CONTEXT["merge_qa"] = report
                if not report["passed"]:
                    raise RuntimeError("Merged chunk validation failed")
                print(f"[QA] merged_chunk_validation={report}")
                return report
            """,
            "mode-chain-merge",
        ),
        markdown(
            """
            ### （5）一键端到端运行与Embedding交接

            运行下方Cell将严格按“监控→mask→检测→unit QA→精确合并”执行。全部PASS后，再运行 `2.2 process_csv_for_embedding.ipynb` 末尾的Mode重建块。
            """,
            "mode-chain-handoff-title",
        ),
        code(
            """
            def run_mode_end_to_end():
                readiness = run_mode_readiness()
                keys = run_mode_mask()
                detection = run_mode_detection()
                chunk_qa = run_mode_chunk_qa()
                merge_qa = run_mode_merge()
                summary = {
                    "readiness": readiness,
                    "candidate_chunk_count": len(keys),
                    "detection": detection["stats"],
                    "chunk_qa_passed": chunk_qa["passed"],
                    "merge_qa_passed": merge_qa["passed"],
                    "embedding_notebook": "2.2 process_csv_for_embedding.ipynb",
                }
                manifest_path = MODE_ABANDON_DIR / "end_to_end_manifest.json"
                temporary = MODE_ABANDON_DIR / "end_to_end_manifest.tmp"
                temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                os.replace(temporary, manifest_path)
                MODE_CONTEXT["end_to_end"] = summary
                print(f"[RESULT] mode_chain_complete=True manifest={manifest_path}")
                print(f"[PATH] next=2.2 process_csv_for_embedding.ipynb")
                return summary

            mode_chain_result = run_mode_end_to_end()
            """,
            "mode-chain-run-all",
        ),
        markdown(
            """
            ## 5、300m/30m多源运行入口

            本节只在Notebook尾部暴露多尺度入口，不在此处展开300 m原生分支、30 m外部基准分支或跨尺度对照的大计算。实际执行由 `run_multiscale_abandonment_end_to_end()` 统一接管。
            """,
            "multiscale-overview",
        ),
        markdown(
            """
            ### （1）多源参数与入口契约

            这里固定声明300 m原生源、30 m外部源、输出根和执行模式，并保持调用面稳定，便于后续独立工具接管。
            """,
            "multiscale-parameters-title",
        ),
        code(
            """
            from pathlib import Path
            from tqdm.auto import tqdm

            from tools.run_multiscale_abandonment import (
                CANONICAL_RUN_ID,
                D_RUN_ROOT,
                run_multiscale_abandonment_end_to_end,
            )

            MULTISCALE_PRIMARY_300M_SOURCE = Path(r"D:\\xarray\\reclass_lccs_300m_esa_cci.nc")
            MULTISCALE_LCMAP_30M_SOURCE = Path(r"D:\\xarray\\reclass_lccs_30m_lcmap_c13.nc")
            MULTISCALE_XIE_30M_SOURCE = Path(r"D:\\xarray\\project_temdata\\abanMaps_5yr_v4__abanDef2_noRec_noUr\\abanMaps_5yr_v4__abanDef2_noRec_noUr.tif")
            MULTISCALE_ACTIVE_RUN = D_RUN_ROOT / CANONICAL_RUN_ID
            MULTISCALE_EXECUTION_MODE = "full"

            MULTISCALE_CONTEXT = {}
            print(f"[CONFIG] primary_300m_source={MULTISCALE_PRIMARY_300M_SOURCE}")
            print(f"[PATH] lcmap_30m_source={MULTISCALE_LCMAP_30M_SOURCE}")
            print(f"[PATH] xie_30m_source={MULTISCALE_XIE_30M_SOURCE}")
            print(f"[PATH] active_run={MULTISCALE_ACTIVE_RUN}")
            """,
            "multiscale-parameters",
        ),
        markdown(
            """
            ### （2）一键300m/30m多源运行

            运行下方Cell会先打包Notebook层配置，再调用 `run_multiscale_abandonment_end_to_end()`。进度条只显示阶段，不把像元级计算留在Notebook。
            """,
            "multiscale-run-title",
        ),
        code(
            """
            multiscale_abandonment_result = run_multiscale_abandonment_end_to_end(
                execution_mode=MULTISCALE_EXECUTION_MODE
            )
            MULTISCALE_CONTEXT["result"] = multiscale_abandonment_result
            print(f"[RESULT] multiscale_run_complete={MULTISCALE_ACTIVE_RUN}")
            """,
            "multiscale-run-all",
        ),
        markdown(
            """
            ## 6、公共本地撂荒入口

            最后一个 active cell 只保留六个公有参数、一次 `run_abandonment_detection(...)` 调用，以及紧凑的 receipt 观察，避免把局部实现细节再写回 Notebook。
            """,
            "public-abandonment-overview",
        ),
        markdown(
            """
            ### （1）公有 API 调用与紧凑 receipt 检查

            这一格只做最小入口验证：参数、调用、返回摘要。
            """,
            "public-abandonment-run-title",
        ),
        code(
            """
            from pathlib import Path

            from function.cropland_abandonment import run_abandonment_detection

            PUBLIC_TARGET_NC = Path(r"D:\\xarray\\reclass_lccs_1km.nc")
            PUBLIC_WINDOW_YEAR = 5
            PUBLIC_START_YEAR = 1992
            PUBLIC_CURRENT_END_YEAR = 2020
            PUBLIC_INIT_CROPLAND = 2
            PUBLIC_EXTEND_VALIDATION = True

            public_abandonment_receipt = run_abandonment_detection(
                target_nc=PUBLIC_TARGET_NC,
                window_year=PUBLIC_WINDOW_YEAR,
                start_year=PUBLIC_START_YEAR,
                current_end_year=PUBLIC_CURRENT_END_YEAR,
                init_cropland=PUBLIC_INIT_CROPLAND,
                extend_validation=PUBLIC_EXTEND_VALIDATION,
            )
            print(
                f"[RESULT] status={public_abandonment_receipt['status']} "
                f"feature={public_abandonment_receipt['feature']} "
                f"accepted={public_abandonment_receipt['accepted']}"
            )
            print(f"[PATH] abandon_chunks={public_abandonment_receipt['abandonment_chunk_root']}")
            print(f"[PATH] merged_chunks={public_abandonment_receipt['merged_chunk_root']}")
            print(f"[PATH] required_csv={public_abandonment_receipt['required_csv_path']}")
            print(f"[PATH] acceptance={public_abandonment_receipt['required_acceptance_path']}")
            """,
            "public-abandonment-run",
        ),
    ]


def _preserve_or_restore_legacy(cell: nbformat.NotebookNode, head_cell: nbformat.NotebookNode | None) -> None:
    original_source = cell.get("source", "")
    if not str(original_source).startswith("# LEGACY_DISABLED:"):
        cell["source"] = comment_legacy_source_text(str(original_source))
    if (not cell.get("outputs")) and head_cell is not None:
        cell["outputs"] = copy.deepcopy(head_cell.get("outputs", []))
    if cell.get("execution_count") is None and head_cell is not None and head_cell.get("execution_count") is not None:
        cell["execution_count"] = head_cell.get("execution_count")


def _replace_source_keep_state(existing: nbformat.NotebookNode, desired: nbformat.NotebookNode) -> None:
    existing["cell_type"] = desired["cell_type"]
    existing["source"] = desired["source"]
    existing["metadata"] = copy.deepcopy(existing.get("metadata", {}))


def transform_notebook(
    notebook_path: Path = NOTEBOOK_PATH,
    head_reference: nbformat.NotebookNode | None = None,
) -> nbformat.NotebookNode:
    head_reference = head_reference or load_head_notebook(NOTEBOOK_PATH)
    try:
        nb = nbformat.read(str(notebook_path), as_version=4)
    except Exception:
        if head_reference is None:
            raise
        nb = copy.deepcopy(head_reference)
    _normalize_stream_outputs(nb)
    nb.nbformat_minor = max(getattr(nb, "nbformat_minor", 0), 5)
    desired_new_cells = build_new_cells()

    managed_ids = set(NEW_CELL_ORDER)
    existing_managed = {
        cell.get("id"): cell
        for cell in nb.cells
        if cell.get("id") in managed_ids
    }
    original_cells = [cell for cell in nb.cells if cell.get("id") not in managed_ids]

    legacy_id_to_index = {cell_id: index for index, cell_id in LEGACY_CODE_INDICES.items()}
    head_cell_count = len(head_reference.cells) if head_reference is not None else None
    for index, cell in enumerate(original_cells):
        existing_id = cell.get("id")
        if not existing_id:
            cell["id"] = stable_id(index, cell)
        if existing_id == MODE_RESAMPLE_CELL_ID or (
            not existing_id and index == MODE_RESAMPLE_INDEX
        ):
            cell["id"] = MODE_RESAMPLE_CELL_ID
            _replace_source_keep_state(cell, build_mode_resample_cell())
            cell["outputs"] = []
            cell["execution_count"] = None
        elif existing_id in legacy_id_to_index or (
            not existing_id and head_cell_count == len(original_cells) and index in LEGACY_CODE_INDICES
        ):
            head_index = legacy_id_to_index.get(existing_id, index)
            cell["id"] = LEGACY_CODE_INDICES[head_index]
            head_cell = head_reference.cells[head_index] if head_reference is not None else None
            _preserve_or_restore_legacy(cell, head_cell)
    proc_cell = next(
        (cell for cell in original_cells if cell.get("id") == "proc-abandon-section"),
        original_cells[PROC_INDEX],
    )
    proc_cell["id"] = "proc-abandon-section"

    if head_reference is not None:
        head_index = 0
        for cell in original_cells:
            if cell.get("id") in PRESERVED_EXTRA_CELL_IDS:
                continue
            if head_index >= len(head_reference.cells):
                raise ValueError("Current notebook has unmatched non-managed cells after the HEAD baseline")
            head_cell = head_reference.cells[head_index]
            if cell.get("id") != MODE_RESAMPLE_CELL_ID:
                if cell.get("cell_type") == "code":
                    cell["outputs"] = copy.deepcopy(head_cell.get("outputs", []))
                    cell["execution_count"] = head_cell.get("execution_count")
                else:
                    cell.pop("outputs", None)
                    cell.pop("execution_count", None)
            head_index += 1
        if head_index != len(head_reference.cells):
            raise ValueError("Current notebook is missing cells from the HEAD baseline")

    managed_cells = []
    for desired_cell in desired_new_cells:
        cell_id = desired_cell["id"]
        if cell_id in existing_managed:
            managed_cell = existing_managed[cell_id]
            _replace_source_keep_state(managed_cell, desired_cell)
        else:
            managed_cell = desired_cell
        managed_cells.append(managed_cell)

    nb.cells = original_cells + managed_cells

    nbformat.validate(nb)
    temporary_path = notebook_path.with_suffix(".tmp.ipynb")
    try:
        nbformat.write(nb, str(temporary_path))
        temporary_path.replace(notebook_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return nb


def main() -> None:
    transform_notebook(NOTEBOOK_PATH, load_head_notebook(NOTEBOOK_PATH))


if __name__ == "__main__":
    main()
