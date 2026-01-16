# Global Pipeline for PV Deployment Suitability Analysis

> **Refactored from**: Notebook-based workflow  
> **Status**: Active Development  
> **Version**: 1.0.0

This is the **Global Cloud-First Pipeline** version of the Landuse project, designed for scalable, reproducible analysis of photovoltaic deployment on abandoned cropland.

---

## 🎯 Quick Start

### Local Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure pipeline
cp configs/global.yaml configs/local.yaml
# Edit local.yaml with your data paths

# 3. Run single stage
python pipelines/global/stage0_ingest.py --config configs/local.yaml

# 4. Run full pipeline
bash scripts/run_pipeline.sh configs/local.yaml
```

### Cloud Execution

```bash
# 1. Build and push Docker image
cd cloud/docker
docker build -t gcr.io/YOUR_PROJECT/landuse-pipeline:latest .
docker push gcr.io/YOUR_PROJECT/landuse-pipeline:latest

# 2. Configure cloud settings
cp configs/cloud.yaml configs/my_cloud.yaml
# Edit my_cloud.yaml with GCS bucket and project

# 3. Submit job
python cloud/submit_job.py \
    --stage stage4_env_train \
    --config configs/my_cloud.yaml \
    --gpu
```

---

## 📁 Project Structure

```
├── src/landuse/              # Core library
│   ├── io/                   # I/O abstractions (GCS, local)
│   ├── data/                 # Manifest, tiling, catalog
│   ├── indicators/           # Alignment & feature extraction
│   ├── env_model/            # GMM + Transformer-ResNet
│   ├── carbon/               # Carbon emission calculations
│   ├── econ/                 # Economic NPV analysis
│   └── synergy/              # 3E-Synergy index (WCCD)
│
├── pipelines/global/         # Stage execution scripts
│   ├── stage0_ingest.py      # Abandonment detection
│   ├── stage1_align.py       # Spatial alignment
│   ├── stage4_env_train.py   # ML model training
│   ├── stage8_synergy.py     # 3E-Synergy calculation
│   └── ...
│
├── configs/                  # Configuration files
│   ├── global.yaml           # Main configuration
│   └── cloud.yaml            # Cloud-specific settings
│
├── cloud/                    # Cloud deployment
│   ├── docker/Dockerfile     # Container definition
│   ├── submit_job.py         # Job submission
│   └── README_cloud.md       # Cloud guide
│
└── docs/                     # Documentation
    ├── MIGRATION_MAP.md      # Notebook → Pipeline mapping
    ├── AGENT_RUNBOOK.md      # Execution guide
    └── README.md             # Documentation index
```

---

## 🔄 Pipeline Stages

| Stage | Script | Input | Output | Description |
|-------|--------|-------|--------|-------------|
| 0 | `stage0_ingest.py` | ESA-CCI land cover | `abandonment_mask.nc` | Detect abandoned cropland |
| 1 | `stage1_align.py` | Multiple datasets | `*_aligned.nc` | Align to 1/120° grid |
| 2 | `stage2_embed.py` | Aligned data | `features.csv` | Extract 15D features |
| 3 | `stage3_predprep.py` | Features | `candidates.csv` | Prepare prediction data |
| 4 | `stage4_env_train.py` | PV + candidates | `gmm_pipeline.pkl`, `transformer_resnet.h5` | Train GMM + Transformer-ResNet |
| 5 | `stage5_env_post.py` | Trained models | `env_probability.nc` | Predict suitability |
| 6 | `stage6_carbon.py` | Suitability + climate | `net_emission.nc` | Calculate carbon reduction |
| 7 | `stage7_econ.py` | Suitability + prices | `npv_scenarios.nc` | Compute NPV across scenarios |
| 8 | `stage8_synergy.py` | Environment + Emission + Economic | `3e_synergy.nc`, `priority_ranks.nc` | Calculate 3E-Synergy & rank |
| 9 | `stage9_figures.py` | All results | `Figure1-4.pdf` | Generate publication figures |

---

## 🔧 Configuration

### Key Parameters

Edit `configs/global.yaml`:

```yaml
# Run mode
run:
  mode: local  # or 'cloud'

# Data resolution
data:
  resolution: 0.00833333  # 1/120° (~1km)

# Model hyperparameters
model:
  gmm:
    n_components: 23
  transformer_resnet:
    num_heads: 4
    dropout_rate: 0.3
    epochs: 100

# Economic scenarios
economics:
  scenarios:
    - {name: "P1", description: "No coordination"}
    - {name: "P2a", description: "Immediate action"}
```

---

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific module
pytest tests/test_indicators.py

# With coverage
pytest --cov=src/landuse tests/
```

### Integration Tests

```bash
# Test single stage with small data
python pipelines/global/stage4_env_train.py --config configs/test.yaml

# Test full pipeline
bash scripts/test_pipeline.sh
```

---

## 📊 Data Flow

```
ESA-CCI → [Stage 0] → Abandonment Mask
                         ↓
        [Stage 1] → Aligned Features (15D)
                         ↓
        [Stage 2-3] → PV Features + Candidates
                         ↓
        [Stage 4] → GMM (23 components) + Transformer-ResNet
                         ↓
        [Stage 5] → Environmental Suitability Map
                         ↓
        [Stage 6] → Net Emission Reduction
        [Stage 7] → Economic NPV (11 scenarios)
                         ↓
        [Stage 8] → 3E-Synergy Index → Priority Ranking
                         ↓
        [Stage 9] → Publication Figures
```

---

## 🌐 Cloud Deployment

### Prerequisites

1. GCP account with billing enabled
2. Docker installed locally
3. `gcloud` CLI configured

### Quick Deploy

```bash
# 1. Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build image
cd cloud/docker
docker build -t gcr.io/YOUR_PROJECT_ID/landuse-pipeline:latest .
docker push gcr.io/YOUR_PROJECT_ID/landuse-pipeline:latest

# 3. Submit job
python cloud/submit_job.py \
    --stage stage4_env_train \
    --config configs/cloud.yaml \
    --machine-type n1-highmem-8 \
    --gpu --gpu-type nvidia-tesla-t4
```

See `cloud/README_cloud.md` for detailed instructions.

---

## 📖 Documentation

- **Migration Guide**: `docs/MIGRATION_MAP.md` - Maps original notebooks to pipeline stages
- **Agent Runbook**: `docs/AGENT_RUNBOOK.md` - Execution instructions for Cursor Agent
- **Cloud Guide**: `cloud/README_cloud.md` - Cloud deployment instructions
- **API Reference**: See docstrings in `src/landuse/`

---

## 🔬 Research Context

This pipeline implements the methodology from:

> **"Policy-informed priority for effectively releasing photovoltaic potential from abandoned cropland in the United States"**  
> Zhong, P., Yue, W., Chen, Y., Wang, T., & Meng, S. (2026)

### Key Findings

- Identified **4.7 Mha** abandoned cropland suitable for PV
- Average environmental suitability: **84.0 ± 0.2%**
- Potential emission reduction: **62.83 ± 17.05 Gt CO₂** (2020-2050)
- 3E-Synergy improves policy efficiency by **38.4%**

---

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repo-url>
cd landuse-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in editable mode
pip install -e .

# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/
```

### Code Style

```bash
# Format code
black src/ pipelines/

# Lint
flake8 src/ pipelines/

# Type check
mypy src/
```

---

## 📝 License

See `LICENSE` file

---

## 📧 Contact

- **Main Author**: Pengyu Zhong
- **Institution**: Zhejiang University / University of Notre Dame
- **Issues**: Open an issue on GitHub
- **Documentation**: See `docs/` directory

---

## 🔄 Version History

- **v1.0.0** (2025-01-16): Initial Global Pipeline release
  - Refactored from notebook workflow
  - Added GCS support
  - Implemented modular architecture
  - Cloud deployment ready

---

## 🙏 Acknowledgments

- ESA-CCI for land cover data
- IPCC AR6 for policy scenarios
- Google Cloud Platform for compute infrastructure
- TensorFlow team for deep learning framework

---

**Status**: This pipeline is actively maintained and continuously improved. For the original notebook-based workflow, see the `master` branch.
