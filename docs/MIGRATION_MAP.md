# Migration Map: Notebook → Global Pipeline

## Overview

This document maps the original Jupyter Notebook workflow to the new modularized Global Pipeline architecture.

---

## Stage Mapping Table

| Original Notebook | Pipeline Stage | Module | Description |
|-------------------|----------------|--------|-------------|
| `0.0 PV_dataset.ipynb` | `stage0_ingest.py` | `landuse.data` | PV站点数据质量控制 |
| `2.1 process_csv_for_aligning.ipynb` | `stage1_align.py` | `landuse.indicators.align` | 空间网格对齐到1/120° |
| `2.2 process_csv_for_embedding.ipynb` | `stage2_embed.py` | `landuse.indicators.features` | 15维环境特征提取 |
| `2.3 process_csv_for_prediction.ipynb` | `stage3_predprep.py` | `landuse.data` | 预测数据集准备 |
| `3.0 pre-training.ipynb` | `stage4_env_train.py` | `landuse.env_model` | GMM + Transformer-ResNet训练 |
| `Process.ipynb` | `stage5_env_post.py` | `landuse.env_model` | 训练结果后处理 |
| `4.1 Emission_reduction_potential.ipynb` | `stage6_carbon.py` | `landuse.carbon` | 光伏减排 vs LNCS碳汇 |
| `5.1 Economical_feasibility.ipynb` | `stage7_econ.py` | `landuse.econ` | NPV计算（多情景） |
| `6.4 3E_synergy_index.ipynb` | `stage8_synergy.py` | `landuse.synergy` | WCCD协同指数 |
| `6.5-6.9 Figure*.ipynb` | `stage9_figures.py` | `landuse.visualization` | 主要图表生成 |
| `7.0-7.1 Analysis_*.ipynb` | Analysis modules | `landuse.analysis` | 国家/州层面分析 |
| `8.0 Multi-objective.ipynb` | Analysis extension | `landuse.optimization` | Pareto多目标优化 |
| `9.0 Energy_demand_adjust.ipynb` | **NOT MIGRATED** | N/A | **美国特定，全球不适用** |

---

## Detailed Migration

### Stage 0: Data Ingestion

**Original**: `0.0 PV_dataset.ipynb`

**New Implementation**:
- **Script**: `pipelines/global/stage0_ingest.py`
- **Modules Used**: `landuse.data.manifest`, `landuse.io.gcs`
- **Key Changes**:
  - ESA-CCI land cover time series → NetCDF processing
  - Abandonment detection (5-year moving window)
  - Output directly to GCS or local storage
  - Manifest tracking for data lineage

**Cloud Integration**:
```python
# Local mode
python pipelines/global/stage0_ingest.py --config configs/global.yaml

# Cloud mode
python pipelines/global/stage0_ingest.py --config configs/cloud.yaml
```

---

### Stage 1: Spatial Alignment

**Original**: `2.1 process_csv_for_aligning.ipynb`

**New Implementation**:
- **Script**: `pipelines/global/stage1_align.py`
- **Modules Used**: `landuse.indicators.align`
- **Key Changes**:
  - `align_datasets()` function for batch alignment
  - `align_to_grid()` for single dataset
  - Supports bilinear/cubic/nearest interpolation
  - Tile-based processing for large datasets

**Migration Example**:
```python
# Old (Notebook)
ds_aligned = ds.interp(lon=new_lon, lat=new_lat, method='linear')

# New (Pipeline)
from landuse.indicators import align_datasets

aligned = align_datasets(
    datasets={'abandonment': ds, 'features': features},
    target_resolution=0.00833333,
    method='bilinear'
)
```

---

### Stage 2-3: Feature Extraction

**Original**: `2.2 process_csv_for_embedding.ipynb`, `2.3 process_csv_for_prediction.ipynb`

**New Implementation**:
- **Scripts**: `stage2_embed.py`, `stage3_predprep.py`
- **Modules Used**: `landuse.indicators.features.FeatureExtractor`
- **Key Changes**:
  - 15 features organized by category (physical/climate/socioeconomic)
  - Standardized normalization and missing value handling
  - Supports CSV and NetCDF input
  - Distance raster calculations (roads, grids, towns)

**Feature Extraction**:
```python
from landuse.indicators import FeatureExtractor

extractor = FeatureExtractor(config)
features = extractor.extract(datasets, mask=abandonment_mask)
feature_array = extractor.to_array(features, flatten=True)
```

---

### Stage 4-5: Environmental Suitability Modeling

**Original**: `3.0 pre-training.ipynb`, `Process.ipynb`

**New Implementation**:
- **Script**: `stage4_env_train.py`, `stage5_env_post.py`
- **Modules Used**: 
  - `landuse.env_model.GMMTrainer`
  - `landuse.env_model.TransformerResNetClassifier`
  - `landuse.env_model.NegativeSampler`

**Key Changes**:
- GMM training with BIC criterion
- Negative sampling from low-density regions (5th percentile)
- Transformer-ResNet hybrid architecture
- Model checkpointing and versioning

**Training Pipeline**:
```python
# GMM training
gmm_trainer = GMMTrainer(config)
gmm_pipeline = gmm_trainer.train(X_positive, search_components=True)
gmm_trainer.calibrate(X_calib)
gmm_trainer.save("models/gmm_pipeline.pkl")

# Negative sampling
sampler = NegativeSampler(config)
X_negative = sampler.sample(candidates, log_density, n_positive)

# Classifier training
classifier = TransformerResNetClassifier(config)
classifier.build()
classifier.train(X_train, y_train, X_val, y_val)
classifier.save("models/transformer_resnet.h5")
```

---

### Stage 6: Carbon Emission Reduction

**Original**: `4.1 Emission_reduction_potential.ipynb`

**New Implementation**:
- **Script**: `stage6_carbon.py`
- **Modules Used**: `landuse.carbon`
- **Key Changes**:
  - PV generation calculation with temperature correction
  - LNCS carbon sequestration (3 strategies)
  - Spatial allocation using K-d tree + IDW
  - Net emission reduction = PV - LNCS

**Carbon Calculation**:
```python
from landuse.carbon import calculate_pv_emission_reduction, calculate_lncs_carbon

pv_carbon = calculate_pv_emission_reduction(
    solar_radiation, temperature, config
)

lncs_carbon = calculate_lncs_carbon(
    land_use_history, biomass, soc, config
)

net_emission = pv_carbon - sum(lncs_carbon.values())
```

---

### Stage 7: Economic Feasibility

**Original**: `5.1 Economical_feasibility.ipynb`

**New Implementation**:
- **Script**: `stage7_econ.py`
- **Modules Used**: `landuse.econ.NPVCalculator`
- **Key Changes**:
  - NPV calculation across 11 AR6 scenarios
  - Discounted cash flow analysis (5% discount rate)
  - LNCS opportunity cost integration
  - Scenario-specific price trajectories

**NPV Calculation**:
```python
from landuse.econ import NPVCalculator

calculator = NPVCalculator(config)
npv_by_scenario = calculator.calculate(
    pv_generation,
    pv_capacity,
    lncs_opportunity_cost,
    scenario_prices
)
```

---

### Stage 8: 3E-Synergy Index

**Original**: `6.4 3E_synergy_index.ipynb`

**New Implementation**:
- **Script**: `stage8_synergy.py`
- **Modules Used**: `landuse.synergy.WCCDCalculator`, `landuse.synergy.PriorityRanker`
- **Key Changes**:
  - WCCD with adaptive pixel-level weights
  - SLSQP optimization
  - Priority ranking and cumulative benefit curves
  - Efficiency/equity/robustness attention kernels

**Synergy Calculation**:
```python
from landuse.synergy import WCCDCalculator, PriorityRanker

wccd_calc = WCCDCalculator(config)
synergy, weights = wccd_calc.calculate(environment, emission, economic)

ranker = PriorityRanker(config)
priority = ranker.rank(synergy, environment, emission, economic)
cumulative = ranker.calculate_cumulative_benefits(
    priority, environment, emission, economic
)
```

---

### Stage 9: Visualization

**Original**: `6.5-6.9 Figure*.ipynb`

**New Implementation**:
- **Script**: `stage9_figures.py`
- **Modules Used**: `landuse.visualization`
- **Key Changes**:
  - Automated figure generation pipeline
  - Configurable DPI, format (PDF/PNG)
  - Albers Equal Area projection for US maps
  - Publication-ready styling

**Figure Generation**:
- Figure 1: Environmental suitability spatial distribution
- Figure 2: Policy scenario matrix + Priority map
- Figure 3: Carbon reduction comparison (PV vs LNCS)
- Figure 4: Cumulative benefit curves

---

### Multi-Objective Optimization (Optional Analysis)

**Original**: `8.0 Multi-objective.ipynb`

**New Implementation**:
- **Module**: `landuse.optimization`
- **Classes**: `ParetoOptimizer`, `EfficiencyKernel`
- **Key Changes**:
  - Pareto frontier analysis for 3E dimensions
  - Efficiency kernel functions (decreasing/uniform/increasing)
  - Integration with pymoo (optional dependency)
  - Heuristic fallback when pymoo unavailable

**Optimization Framework**:
```python
from landuse.optimization import ParetoOptimizer

optimizer = ParetoOptimizer(config)
pareto_solutions = optimizer.optimize(
    environment, emission, economic, areas,
    objectives=["environment", "emission", "economic"]
)
```

**Objectives**:
- **Environment**: Environmental suitability (predicted_prob)
- **Emission**: Emission mitigation (net_benefit)
- **Economic**: Economic feasibility (avg_npv)

**Output**:
- Pareto frontier rankings
- Multi-objective trade-off curves
- Cross-objective cumulative benefits

---

### ⚠️ Excluded: Energy Demand Adjustment

**Original**: `9.0 Energy_demand_adjust.ipynb`

**Status**: **NOT MIGRATED TO GLOBAL PIPELINE**

**Reason**: 
This notebook is **specific to the United States** and not applicable to global-scale analysis.

**US-Specific Dependencies**:
1. **NREL Energy Data**: US-only electricity demand projections
2. **State-Level Analysis**: 51 US states (including DC)
3. **Electrification Scenarios**: US policy scenarios
   - HIGH ELECTRIFICATION
   - MEDIUM ELECTRIFICATION
   - REFERENCE ELECTRIFICATION
   - LOW ELECTRICITY GROWTH
   - ELECTRIFICATION TECHNICAL POTENTIAL
4. **2050 Demand Targets**: Based on US Department of Energy projections

**Data Files (US-Only)**:
- `data/US_data/US_electricity/NREL/energy.csv.gzip`
- `data/cb_2018_us_state_500k.shp` (US state boundaries)
- `data/US_data/cb_2018_us_nation_5m.shp` (US national boundary)

**Global Alternative**:
For global-scale energy demand analysis, use:
- IEA World Energy Outlook scenarios
- Country-level energy demand projections
- Regional aggregation instead of US states

**Implementation Note**:
If US-specific analysis is needed, keep the original notebook as a separate analysis script, but **do not integrate it into the global pipeline**.

---

## Function/Class Migration Reference

### Data Processing

| Old Function (Notebook) | New Location |
|------------------------|--------------|
| `align_raster()` | `landuse.indicators.align.align_to_grid()` |
| `extract_features_from_csv()` | `landuse.indicators.features.FeatureExtractor.extract()` |
| `calculate_distance_to_grid()` | `landuse.indicators.align.create_distance_raster()` |

### Modeling

| Old Function (Notebook) | New Location |
|------------------------|--------------|
| `select_and_train_gmm()` | `landuse.env_model.GMMTrainer.train()` |
| `build_transformer_resnet_model()` | `landuse.env_model.build_transformer_resnet()` |
| `generate_negative_samples()` | `landuse.env_model.NegativeSampler.sample()` |

### Analysis

| Old Function (Notebook) | New Location |
|------------------------|--------------|
| `calculate_pv_emission()` | `landuse.carbon.calculate_pv_emission_reduction()` |
| `calculate_npv()` | `landuse.econ.NPVCalculator.calculate()` |
| `calculate_3e_synergy()` | `landuse.synergy.WCCDCalculator.calculate()` |

---

## Data Flow

```
Stage 0: Ingest
    ↓
  abandonment_mask.nc
    ↓
Stage 1: Align
    ↓
  *_aligned.nc (15 features + abandonment)
    ↓
Stage 2-3: Feature Extraction
    ↓
  pv_features.csv, candidate_features.csv
    ↓
Stage 4-5: Train Environmental Model
    ↓
  gmm_pipeline.pkl, transformer_resnet.h5
    ↓
  env_probability.nc (suitability map)
    ↓
Stage 6-7: Carbon & Economics
    ↓
  net_emission.nc, npv_scenarios.nc
    ↓
Stage 8: 3E-Synergy
    ↓
  3e_synergy.nc, priority_ranks.nc
    ↓
Stage 9: Visualization
    ↓
  Figure1-4.pdf/png
```

---

## Breaking Changes

### 1. Data Format
- **Old**: Mixed CSV + NetCDF + in-memory arrays
- **New**: Standardized NetCDF for spatial data, CSV for tabular data
- **Action**: Update data loading scripts to use `xarray.open_dataset()`

### 2. Model Persistence
- **Old**: Manual pickle save/load
- **New**: Managed through `ModelSaver` class with versioning
- **Action**: Use `.save()` and `.load()` methods

### 3. Configuration
- **Old**: Hardcoded parameters in notebooks
- **New**: Centralized `configs/global.yaml`
- **Action**: Move all parameters to config file

### 4. Paths
- **Old**: Absolute local paths
- **New**: Relative paths or GCS URIs managed by `DataCatalog`
- **Action**: Use `catalog.get_path()` for all file access

---

## Testing Migration

### Unit Tests
Each module has corresponding unit tests in `tests/`:
```
tests/
├── test_indicators.py    # Alignment and features
├── test_env_model.py     # GMM and Transformer-ResNet
├── test_carbon.py        # PV and LNCS calculations
├── test_econ.py          # NPV calculations
└── test_synergy.py       # WCCD and priority ranking
```

### Integration Tests
Pipeline stages can be tested with small datasets:
```bash
# Test single stage
python pipelines/global/stage0_ingest.py --config configs/test.yaml

# Test full pipeline
bash scripts/run_test_pipeline.sh
```

---

## Performance Considerations

### Memory Optimization
- **Old**: Load entire dataset into memory
- **New**: 
  - Tile-based processing with `TileManager`
  - Lazy loading with `xarray.open_mfdataset()`
  - Dask for parallel computation

### Compute Optimization
- **GMM Training**: ~10-30 minutes (unchanged)
- **Transformer-ResNet**: ~2-4 hours on GPU (unchanged)
- **Prediction**: Now supports batch processing across tiles

### Storage Optimization
- **NetCDF Compression**: `zlib=True, complevel=4`
- **GCS Lifecycle**: Automatic archival of intermediate results

---

## Rollback Strategy

If issues arise, original notebooks remain unchanged in `master` branch:
```bash
# Switch back to notebook workflow
git checkout master
cd /path/to/Landuse_Zhong_clean
jupyter notebook
```

All pipeline outputs are versioned in manifest, allowing comparison:
```python
from landuse.data import DataManifest

manifest = DataManifest("manifest.json")
manifest.load()

# Compare outputs
old_result = xr.open_dataset("old_output.nc")
new_result = xr.open_dataset(manifest.get_artifact("stage8_synergy", "3e_synergy")["path"])
```

---

## Contact

For migration issues:
- Check `docs/AGENT_RUNBOOK.md` for detailed instructions
- Review `docs/README.md` for troubleshooting
- Consult original notebooks in `master` branch for reference
