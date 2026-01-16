# Data Dependencies Analysis

## Overview

This document analyzes the data dependencies of the Global Pipeline and clarifies which components are global-applicable vs. region-specific.

---

## ✅ Global Pipeline Components

### Included Notebooks & Their Dependencies

| Notebook | Data Sources | Geographic Scope | Status |
|----------|--------------|------------------|--------|
| `0.0 PV_dataset.ipynb` | ESA-CCI Land Cover | Global | ✅ Included |
| `2.1-2.3 process_csv_*.ipynb` | Multi-source features | Global | ✅ Included |
| `3.0 pre-training.ipynb` | PV sites + Features | Global | ✅ Included |
| `4.1 Emission_reduction_potential.ipynb` | CHELSA Climate + LNCS | Global | ✅ Included |
| `5.1 Economical_feasibility.ipynb` | IPCC AR6 Scenarios | Global | ✅ Included |
| `6.4 3E_synergy_index.ipynb` | 3E Dimensions | Global | ✅ Included |
| `6.5-6.9 Figure*.ipynb` | Analysis Results | Global | ✅ Included |
| `7.0-7.1 Analysis_*.ipynb` | Multi-scale Analysis | Adaptable | ✅ Included |
| `8.0 Multi-objective.ipynb` | 3E Optimization | Global | ✅ Included |

---

## ❌ Excluded Components

### 9.0 Energy_demand_adjust.ipynb

**Status**: **NOT MIGRATED - US-SPECIFIC**

#### Hard Dependencies on US Data

1. **NREL Electricity Projections**
   - File: `data/US_data/US_electricity/NREL/energy.csv.gzip`
   - Source: US National Renewable Energy Laboratory
   - Scope: 51 US states (including District of Columbia)
   - Years: Projections to 2050
   - Sectors: Residential, Commercial, Industrial, Transportation

2. **US Geospatial Boundaries**
   - `data/cb_2018_us_state_500k.shp` - US State boundaries
   - `data/US_data/cb_2018_us_nation_5m.shp` - US National boundary
   - Source: US Census Bureau

3. **US Electrification Scenarios**
   - HIGH ELECTRIFICATION - MODERATE TECHNOLOGY ADVANCEMENT
   - MEDIUM ELECTRIFICATION - MODERATE TECHNOLOGY ADVANCEMENT
   - REFERENCE ELECTRIFICATION - MODERATE TECHNOLOGY ADVANCEMENT
   - LOW ELECTRICITY GROWTH - MODERATE TECHNOLOGY ADVANCEMENT
   - ELECTRIFICATION TECHNICAL POTENTIAL - MODERATE TECHNOLOGY ADVANCEMENT

#### Code Snippets Showing US-Specificity

```python
# Hard-coded US states
energy_df['STATE'].unique()
# Output: ['ALABAMA', 'ALASKA', 'ARIZONA', ..., 'WYOMING']  # 51 states

# US-specific filtering
filtered_df = energy_df[
    (energy_df['STATE'] == 'CALIFORNIA') &
    (energy_df['YEAR'] == 2050) &
    (energy_df['SECTOR'] == 'RESIDENTIAL') &
    (energy_df['SCENARIO'] == scenario) &
    (energy_df['FINAL_ENERGY'] == 'ELECTRICITY')
]
```

#### Why Not Applicable to Global Pipeline

1. **Geographic Constraint**: Data only covers 51 US states, no other countries
2. **Policy Scenarios**: Based on US Department of Energy projections
3. **Energy System**: US-specific electricity grid structure
4. **Target Year**: 2050 targets aligned with US policy goals
5. **Sectoral Breakdown**: Specific to US energy consumption patterns

---

## 🌍 Global Data Sources

### Core Data Used in Global Pipeline

| Data Type | Source | Coverage | Resolution | Used In |
|-----------|--------|----------|------------|---------|
| **Land Cover** | ESA-CCI | Global | 300m | Stage 0 |
| **Climate** | CHELSA / WorldClim | Global | 1km | Stage 6 |
| **Solar Radiation** | CMSAF / PVGIS | Global | 0.05° | Stage 6 |
| **DEM** | SRTM / ASTER | Global | 90m/30m | Stage 2 |
| **Population** | WorldPop / GPW | Global | 1km | Stage 2 |
| **GDP** | Kummu et al. | Global | 5' | Stage 2 |
| **Roads** | OpenStreetMap | Global | Vector | Stage 2 |
| **Biomass** | GlobBiomass | Global | 100m | Stage 6 |
| **SOC** | SoilGrids | Global | 250m | Stage 6 |
| **Economic Scenarios** | IPCC AR6 | Global | Country | Stage 7 |

---

## 🔄 Global Alternative for Energy Demand

If energy demand analysis is needed for global pipeline, use:

### Recommended Global Data Sources

1. **IEA World Energy Outlook**
   - Coverage: Global, 200+ countries
   - Scenarios: Stated Policies, Sustainable Development, Net Zero
   - Sectors: Power, Industry, Transport, Buildings
   - URL: https://www.iea.org/weo

2. **IIASA Energy Scenarios**
   - SSP scenarios (Shared Socioeconomic Pathways)
   - Integrated with IPCC AR6
   - Global coverage with regional detail

3. **BP Energy Outlook**
   - Global energy demand projections
   - Multiple transition scenarios

### Implementation Strategy

```python
# Instead of US-specific:
# energy_df = pd.read_csv('data/US_data/US_electricity/NREL/energy.csv.gzip')

# Use global data:
from landuse.data import load_global_energy_scenarios

energy_scenarios = load_global_energy_scenarios(
    source="IEA",
    countries=["USA", "CHN", "IND", "EU", ...],  # All countries
    scenarios=["Stated Policies", "Net Zero"],
    target_year=2050
)
```

---

## 📊 Data Dependency Graph

```
Global Pipeline Data Flow:
==========================

ESA-CCI Land Cover (Global)
    ↓
[Stage 0: Abandonment Detection]
    ↓
Abandonment Mask (Global)
    ↓
15 Global Features:
- Climate (CHELSA)           ✅ Global
- Topography (SRTM)          ✅ Global
- Socioeconomic (GPW/OSM)    ✅ Global
    ↓
[Stage 1-3: Alignment & Features]
    ↓
[Stage 4-5: ML Models]
    ↓
[Stage 6: Carbon]
- Solar Radiation (CMSAF)    ✅ Global
- Biomass/SOC (Global)       ✅ Global
    ↓
[Stage 7: Economics]
- IPCC AR6 Scenarios         ✅ Global
    ↓
[Stage 8: 3E-Synergy]
    ↓
[Stage 9: Visualization]
    ↓
[Optional: Multi-objective]  ✅ Global

❌ US Energy Demand (NREL)   ❌ US-Only
```

---

## 🔧 Configuration for Regional Analysis

### Global Mode (Default)

```yaml
# configs/global.yaml
regional_exclusions:
  us_specific:
    enabled: false  # Disable US-specific features
```

### US-Specific Mode (Optional)

```yaml
# configs/us_specific.yaml
regional_exclusions:
  us_specific:
    enabled: true
    energy_scenarios:
      - "HIGH ELECTRIFICATION"
      - "REFERENCE ELECTRIFICATION"
    data_paths:
      nrel_energy: "data/US_data/US_electricity/NREL/energy.csv.gzip"
      state_boundaries: "data/cb_2018_us_state_500k.shp"
```

---

## 📝 Summary

### Global Pipeline Includes

✅ **All stages (0-9)** - Use global data sources
✅ **Multi-objective optimization (8.0)** - Applicable globally
✅ **3E-Synergy framework** - Country-agnostic methodology

### Global Pipeline Excludes

❌ **US Energy Demand Adjustment (9.0)** - Use only for US-specific studies
❌ **NREL data dependencies** - Replace with IEA/IIASA for global
❌ **US state-level analysis** - Adapt to country/regional level

### Migration Checklist

When adapting for specific regions:

- [ ] Replace US shapefiles with target region boundaries
- [ ] Use regional energy scenarios (IEA/local sources)
- [ ] Adjust policy scenarios to local context
- [ ] Update economic parameters (discount rates, costs)
- [ ] Verify all data sources cover target region
- [ ] Test with sample data from target region

---

## 📧 Contact

For questions about data dependencies or regional adaptations, see:
- `docs/MIGRATION_MAP.md` for component mapping
- `docs/AGENT_RUNBOOK.md` for execution guidelines
- Original notebooks in `master` branch for reference

**Last Updated**: 2025-01-16  
**Version**: 1.0.0
