## Agent Runbook: Global Pipeline Execution

### Purpose

This document provides step-by-step instructions for Cursor Agent to execute and maintain the Global Pipeline. It defines task boundaries, execution order, and commit conventions.

---

## 📋 Pre-Flight Checklist

Before executing any pipeline stage, verify:

1. ✅ You are in `Landuse_Global_Pipeline_Worktree` directory (not `master`)
2. ✅ Current branch is `Landuse_Global_Pipeline`
3. ✅ `configs/global.yaml` is properly configured
4. ✅ Python environment is activated (if using conda/venv)
5. ✅ Required dependencies are installed

```bash
# Verify location
pwd  # Should show: .../Landuse_Global_Pipeline_Worktree

# Verify branch
git branch  # Should show: * Landuse_Global_Pipeline

# Verify config
ls configs/global.yaml  # Should exist
```

---

## 🚫 Prohibited Actions

**Agent MUST NOT**:

1. ❌ Modify `master` branch or files in `Landuse_Zhong_clean` directory
2. ❌ Modify original Notebook files (*.ipynb in root)
3. ❌ Run long-duration training without explicit user approval
4. ❌ Commit large data files (>100MB) - use `.gitignore`
5. ❌ Push to remote without user confirmation
6. ❌ Execute `git` commands directly (read-only access)

---

## ✅ Allowed Actions

**Agent CAN**:

1. ✅ Create/modify files in `src/landuse/`, `pipelines/`, `configs/`, `docs/`
2. ✅ Create/modify test files in `tests/`
3. ✅ Read original Notebooks for reference
4. ✅ Execute pipeline stages with test data
5. ✅ Generate documentation and diagrams
6. ✅ Propose code improvements and refactoring

---

## 📂 Directory Structure Reference

```
Landuse_Global_Pipeline_Worktree/
├── src/landuse/              # Core library code
│   ├── io/                   # I/O abstractions (GCS, local)
│   ├── data/                 # Manifest, tiling, catalog
│   ├── indicators/           # Alignment & features
│   ├── env_model/            # GMM + Transformer-ResNet
│   ├── carbon/               # PV & LNCS emission
│   ├── econ/                 # NPV calculations
│   └── synergy/              # 3E-Synergy index
│
├── pipelines/global/         # Stage execution scripts
│   ├── stage0_ingest.py      # Data ingestion
│   ├── stage1_align.py       # Spatial alignment
│   ├── stage2_embed.py       # Feature embedding
│   ├── stage3_predprep.py    # Prediction prep
│   ├── stage4_env_train.py   # GMM + Model training
│   ├── stage5_env_post.py    # Post-processing
│   ├── stage6_carbon.py      # Carbon reduction
│   ├── stage7_econ.py        # Economics
│   ├── stage8_synergy.py     # 3E-Synergy
│   └── stage9_figures.py     # Visualization
│
├── configs/                  # Configuration files
│   ├── global.yaml           # Main config
│   ├── cloud.yaml            # Cloud-specific config
│   └── test.yaml             # Testing config
│
├── cloud/                    # Cloud deployment
│   ├── docker/
│   │   └── Dockerfile        # Container definition
│   ├── submit_job.py         # Job submission script
│   └── README_cloud.md       # Cloud instructions
│
├── docs/                     # Documentation
│   ├── MIGRATION_MAP.md      # Notebook → Pipeline mapping
│   ├── AGENT_RUNBOOK.md      # This file
│   └── REQUIREMENTS.md       # Technical requirements
│
└── tests/                    # Unit & integration tests
```

---

## 🔄 Execution Workflow

### Sequential Stage Execution

Stages must be executed in order (each depends on previous outputs):

```
0. Ingest → 1. Align → 2. Embed → 3. PredPrep 
    → 4. Train → 5. Post → 6. Carbon → 7. Econ 
    → 8. Synergy → 9. Figures
```

### Task Breakdown

**Each pipeline stage task should**:

1. ✅ Be self-contained (one stage = one commit)
2. ✅ Include error handling and logging
3. ✅ Register outputs in manifest
4. ✅ Support both local and cloud modes
5. ✅ Be executable with test data

---

## 📝 Task Templates

### Template 1: Create New Pipeline Stage

**Context**: User asks to create `stageX_name.py`

**Steps**:

1. **Read Reference**:
   ```python
   # Read corresponding notebook
   read(f"{X.Y} original_notebook.ipynb")
   ```

2. **Create Stage Script**:
   - Copy template from existing stage
   - Extract core logic from notebook
   - Add GCS support via `DataCatalog`
   - Add manifest registration

3. **Create Module** (if needed):
   - Add functions to appropriate `src/landuse/` module
   - Follow existing module structure
   - Add type hints and docstrings

4. **Test Execution**:
   ```bash
   python pipelines/global/stageX_name.py --config configs/test.yaml
   ```

5. **Document**:
   - Update `MIGRATION_MAP.md` with new mapping
   - Add stage description to this runbook

---

### Template 2: Refactor Notebook Logic

**Context**: User asks to move notebook code to module

**Steps**:

1. **Identify Functions**:
   - List all functions/classes in notebook
   - Categorize by module (indicators/carbon/econ/etc.)

2. **Extract and Refactor**:
   - Remove hardcoded paths → use `DataCatalog`
   - Remove global variables → use config
   - Add function signatures and docstrings

3. **Update Imports**:
   - Update stage scripts to use new module functions
   - Remove duplicated code

4. **Validate**:
   - Run unit tests
   - Run integration test with small dataset

---

### Template 3: Add Cloud Support

**Context**: User asks to enable GCS for a stage

**Steps**:

1. **Update Stage Script**:
   ```python
   # Add at top of stage script
   from landuse.io import GCSManager
   from landuse.data import DataCatalog
   
   catalog = DataCatalog(config)
   
   # Replace local paths
   # Old: "data/features.nc"
   # New: catalog.get_path("features", "features.nc")
   ```

2. **Update Config**:
   ```yaml
   # configs/cloud.yaml
   run:
     mode: cloud
   gcs:
     bucket: your-bucket
     prefix: landuse/global
   ```

3. **Test Locally First**:
   ```bash
   # Test with local mode
   python stage.py --config configs/global.yaml
   
   # Then test with cloud mode (if GCS configured)
   python stage.py --config configs/cloud.yaml
   ```

---

## 🧪 Testing Guidelines

### Unit Tests

**When to Write**:
- New functions added to `src/landuse/`
- Complex logic extracted from notebooks

**How to Test**:
```bash
# Run specific test
pytest tests/test_indicators.py::test_align_datasets

# Run all tests in module
pytest tests/test_indicators.py

# Run all tests
pytest tests/
```

**Template**:
```python
# tests/test_new_module.py
import pytest
from landuse.new_module import new_function

def test_new_function():
    # Arrange
    input_data = ...
    expected = ...
    
    # Act
    result = new_function(input_data)
    
    # Assert
    assert result == expected
```

---

### Integration Tests

**Purpose**: Test full stage execution with small dataset

**How to Test**:
```bash
# Create test config with small data
cp configs/global.yaml configs/test.yaml

# Edit test.yaml to use small subset
# Then run stage
python pipelines/global/stage4_env_train.py --config configs/test.yaml
```

---

## 📊 Progress Tracking

Use TODO tool to track multi-stage tasks:

**Example**:
```python
# When user requests: "Implement Stage 2-5"

todo_write([
    {"id": "1", "content": "Create stage2_embed.py", "status": "in_progress"},
    {"id": "2", "content": "Create stage3_predprep.py", "status": "pending"},
    {"id": "3", "content": "Create stage4_env_train.py", "status": "pending"},
    {"id": "4", "content": "Create stage5_env_post.py", "status": "pending"},
    {"id": "5", "content": "Test stages 2-5 end-to-end", "status": "pending"},
], merge=False)

# Update as you complete each stage
todo_write([
    {"id": "1", "status": "completed"},
    {"id": "2", "status": "in_progress"},
], merge=True)
```

---

## 🔍 Debugging Workflow

### Issue: Stage Fails with "File Not Found"

**Check**:
1. Manifest contains required artifact:
   ```python
   from landuse.data import DataManifest
   manifest = DataManifest("manifest.json")
   manifest.load()
   manifest.get_artifact("stage_name", "artifact_name")
   ```

2. DataCatalog path resolution:
   ```python
   from landuse.data import DataCatalog
   catalog = DataCatalog(config)
   path = catalog.get_path("dataset", "file.nc")
   print(path)  # Verify path is correct
   ```

3. Previous stage completed successfully:
   ```bash
   # Check logs
   tail -n 50 logs/stage_name.log
   ```

---

### Issue: GCS Access Denied

**Check**:
1. GCP credentials:
   ```bash
   # Verify credentials
   gcloud auth list
   
   # Set application default
   gcloud auth application-default login
   ```

2. Bucket permissions:
   ```bash
   gsutil ls gs://your-bucket/
   ```

3. Config file:
   ```yaml
   gcs:
     bucket: your-bucket  # No gs:// prefix!
     project: your-project-id
   ```

---

### Issue: Model Training OOM

**Solutions**:
1. Reduce batch size in config:
   ```yaml
   model:
     transformer_resnet:
       batch_size: 256  # Reduce from 512
   ```

2. Enable tiling:
   ```yaml
   tiling:
     enabled: true
     tile_size: 5  # degrees
   ```

3. Use cloud GPU:
   ```bash
   python cloud/submit_job.py --stage stage4_env_train --machine-type n1-highmem-8 --gpu
   ```

---

## 📦 Dependency Management

### Adding New Dependencies

**Process**:
1. Add to appropriate requirements file:
   - Core: `requirements.txt`
   - Dev/test: `requirements-dev.txt`
   - Cloud: `cloud/docker/requirements.txt`

2. Update conda environment (if used):
   ```bash
   conda env update -f environment.yml
   ```

3. Document in `docs/REQUIREMENTS.md`

**Example**:
```txt
# requirements.txt
numpy>=1.21.0
xarray>=2022.3.0
dask[complete]>=2022.1.0
google-cloud-storage>=2.0.0
```

---

## 🚀 Cloud Deployment

### Submitting Cloud Jobs

**Prerequisites**:
1. Docker image built and pushed to GCR
2. GCS bucket configured
3. Config file updated for cloud mode

**Workflow**:
```bash
# 1. Build and push Docker image
cd cloud/docker
docker build -t gcr.io/your-project/landuse-pipeline:latest .
docker push gcr.io/your-project/landuse-pipeline:latest

# 2. Submit job for specific stage
python cloud/submit_job.py \
    --stage stage4_env_train \
    --config configs/cloud.yaml \
    --machine-type n1-standard-8 \
    --gpu \
    --gpu-type nvidia-tesla-t4

# 3. Monitor job
python cloud/monitor_job.py --job-id JOB_ID

# 4. Retrieve results
gsutil cp gs://your-bucket/landuse/global/results/* ./results/
```

**Batch Processing**:
```bash
# Submit multiple tiles in parallel
python cloud/submit_jobs.py \
    --stage stage4_env_train \
    --config configs/cloud.yaml \
    --tiles tile_000_000,tile_000_001,tile_001_000 \
    --workers 10
```

---

## 📋 Checklist: Before Requesting User Review

Before asking user to review/merge work:

- [ ] All TODO items completed
- [ ] Code follows project structure
- [ ] Docstrings added to all public functions
- [ ] Type hints added where appropriate
- [ ] No hardcoded paths (use `DataCatalog`)
- [ ] Config parameters moved to `global.yaml`
- [ ] Stage script tested with `test.yaml`
- [ ] Manifest registration added
- [ ] GCS support implemented
- [ ] Error handling and logging added
- [ ] `MIGRATION_MAP.md` updated
- [ ] No large files committed (check `.gitignore`)
- [ ] No sensitive data in code

---

## 📚 Reference Links

- **Original Notebooks**: See `master` branch
- **Migration Map**: `docs/MIGRATION_MAP.md`
- **Cloud Setup**: `cloud/README_cloud.md`
- **Requirements**: `docs/REQUIREMENTS.md`
- **GCP Documentation**: https://cloud.google.com/python/docs/reference

---

## 🆘 Escalation

If you encounter:
1. **Ambiguous requirements** → Ask user for clarification
2. **Missing original code** → Request access to missing notebooks
3. **Complex refactoring** → Propose plan before implementing
4. **Breaking changes needed** → Discuss with user first
5. **Performance issues** → Profile and report findings

---

## 📌 Version Info

- **Pipeline Version**: 1.0.0
- **Last Updated**: 2025-01-16
- **Maintainer**: Pengyu Zhong
- **Status**: Active Development
