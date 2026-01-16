# Cloud Deployment Guide

## Overview

This guide covers deploying the Global Pipeline to Google Cloud Platform (GCP) for large-scale processing.

---

## Prerequisites

1. **GCP Account**:
   - Active GCP project
   - Billing enabled
   - APIs enabled: Vertex AI, Cloud Storage, Container Registry

2. **Local Setup**:
   ```bash
   # Install gcloud CLI
   # See: https://cloud.google.com/sdk/docs/install
   
   # Authenticate
   gcloud auth login
   gcloud auth application-default login
   
   # Set project
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Docker**:
   ```bash
   # Install Docker
   # See: https://docs.docker.com/get-docker/
   
   # Verify installation
   docker --version
   ```

---

## Quick Start

### 1. Configure GCS Bucket

```bash
# Create bucket
gsutil mb -l us-central1 gs://your-landuse-bucket

# Set lifecycle policy (optional)
gsutil lifecycle set cloud/gcs_lifecycle.json gs://your-landuse-bucket
```

### 2. Update Configuration

Edit `configs/cloud.yaml`:

```yaml
run:
  mode: cloud

gcs:
  bucket: your-landuse-bucket
  project: your-project-id
  prefix: landuse/global
```

### 3. Build and Push Docker Image

```bash
cd cloud/docker

# Build image
docker build -t gcr.io/YOUR_PROJECT_ID/landuse-pipeline:latest .

# Push to GCR
docker push gcr.io/YOUR_PROJECT_ID/landuse-pipeline:latest

# Update config with image path
# In configs/cloud.yaml:
#   cloud_job:
#     docker:
#       image: "gcr.io/YOUR_PROJECT_ID/landuse-pipeline:latest"
```

### 4. Submit Jobs

```bash
# Single stage
python cloud/submit_job.py \
    --stage stage4_env_train \
    --config configs/cloud.yaml \
    --machine-type n1-standard-8 \
    --gpu \
    --gpu-type nvidia-tesla-t4

# Batch processing (multiple tiles)
python cloud/submit_job.py \
    --stage stage5_env_post \
    --config configs/cloud.yaml \
    --tiles tile_000_000,tile_000_001,tile_001_000 \
    --max-workers 10
```

---

## Architecture

### Storage Structure

```
gs://your-landuse-bucket/
└── landuse/global/
    ├── data/
    │   ├── raw/              # Input data
    │   ├── aligned/          # Stage 1 output
    │   ├── features/         # Stage 2-3 output
    │   └── abandonment/      # Stage 0 output
    ├── models/
    │   ├── gmm_pipeline.pkl
    │   └── transformer_resnet.h5
    ├── results/
    │   ├── 3e_synergy.nc
    │   ├── priority_ranks.nc
    │   └── cumulative_benefits.json
    └── logs/
        └── stage_*.log
```

### Compute Options

| Backend | Use Case | Cost | Setup Complexity |
|---------|----------|------|------------------|
| **Cloud Run** | Single-stage jobs, no GPU | Low | Easy |
| **Vertex AI** | Multi-stage, GPU training | Medium | Medium |
| **Compute Engine** | Custom setup, long-running | Variable | Complex |

---

## Stage-Specific Recommendations

### Stage 0-3: Data Processing

- **Machine**: `n1-standard-4` to `n1-standard-8`
- **Memory**: 16-32 GB
- **Storage**: Read from GCS, write back
- **Parallelization**: Tile-based

```bash
python cloud/submit_job.py \
    --stage stage1_align \
    --machine-type n1-standard-8 \
    --tiles tile_000_000,...
```

### Stage 4: GMM + Model Training

- **Machine**: `n1-highmem-8`
- **GPU**: `nvidia-tesla-t4` (1x)
- **Memory**: 52 GB
- **Duration**: 2-4 hours

```bash
python cloud/submit_job.py \
    --stage stage4_env_train \
    --machine-type n1-highmem-8 \
    --gpu \
    --gpu-type nvidia-tesla-t4
```

### Stage 5: Prediction

- **Machine**: `n1-standard-4`
- **GPU**: Optional (faster with GPU)
- **Parallelization**: Tile-based

```bash
# CPU-only batch prediction
python cloud/submit_job.py \
    --stage stage5_env_post \
    --machine-type n1-standard-4 \
    --tiles $(python -c "from landuse.data import TileManager; \
                          tm = TileManager(); \
                          print(','.join([t.tile_id for t in tm.tiles]))")
```

### Stage 6-9: Analysis

- **Machine**: `n1-standard-4`
- **Memory**: 16 GB
- **No GPU needed**

---

## Cost Estimation

### Typical Run

| Stage | Machine Type | Duration | Cost (USD) |
|-------|--------------|----------|------------|
| 0 | n1-standard-4 | 1h | $0.20 |
| 1 | n1-standard-8 | 2h | $0.80 |
| 2-3 | n1-standard-4 | 1h | $0.20 |
| 4 | n1-highmem-8 + T4 GPU | 4h | $4.00 |
| 5 | n1-standard-4 (x10 tiles) | 2h | $2.00 |
| 6-9 | n1-standard-4 | 1h | $0.20 |
| **Total** | | | **~$7.40** |

*Note: Costs vary by region and actual usage.*

### Storage Costs

- **GCS Standard**: $0.020 per GB/month
- **Estimated data**: ~500 GB → $10/month

---

## Monitoring

### Check Job Status

```bash
# List jobs
gcloud ai custom-jobs list --region us-central1

# Describe specific job
gcloud ai custom-jobs describe JOB_ID --region us-central1

# View logs
gcloud ai custom-jobs stream-logs JOB_ID --region us-central1
```

### Dashboard

View jobs in GCP Console:
- Navigate to: **Vertex AI → Training → Custom Jobs**
- Monitor: CPU/GPU utilization, memory usage, logs

---

## Troubleshooting

### Issue: Image Pull Error

**Symptom**: Job fails with "Failed to pull image"

**Solution**:
```bash
# Verify image exists
gcloud container images list --repository=gcr.io/YOUR_PROJECT_ID

# Check permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member=serviceAccount:SERVICE_ACCOUNT@PROJECT.iam.gserviceaccount.com \
    --role=roles/storage.objectViewer
```

### Issue: OOM (Out of Memory)

**Symptom**: Job crashes with memory error

**Solutions**:
1. Increase machine memory:
   ```bash
   --machine-type n1-highmem-16  # 104 GB RAM
   ```

2. Enable tiling in config:
   ```yaml
   tiling:
     enabled: true
     tile_size: 5  # smaller tiles
   ```

3. Reduce batch size:
   ```yaml
   model:
     transformer_resnet:
       batch_size: 256  # from 512
   ```

### Issue: GCS Permission Denied

**Symptom**: "Access denied" when reading/writing GCS

**Solution**:
```bash
# Grant Storage Admin role to Compute Engine service account
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
    --role=roles/storage.admin
```

---

## Best Practices

### 1. Use Preemptible Instances (Optional)

Save ~70% on compute costs:

```bash
python cloud/submit_job.py \
    --stage stage5_env_post \
    --preemptible \
    --max-retries 3
```

**Trade-off**: Jobs may be interrupted (but automatically retried)

### 2. Organize with Prefixes

```yaml
gcs:
  prefix: landuse/global/experiment_20250116
```

### 3. Enable Logging

```python
# In pipeline scripts
import logging
from google.cloud import logging as cloud_logging

client = cloud_logging.Client()
client.setup_logging()

logger = logging.getLogger(__name__)
logger.info("Job started")  # Appears in GCP Console
```

### 4. Tag Resources

```bash
gcloud ai custom-jobs create \
    --labels=project=landuse,stage=env_train,version=v1
```

---

## Cleanup

### Delete Completed Jobs

```bash
# List jobs
gcloud ai custom-jobs list --region us-central1 --filter="state:JOB_STATE_SUCCEEDED"

# Delete job
gcloud ai custom-jobs delete JOB_ID --region us-central1
```

### Archive GCS Data

```bash
# Move to Archive storage class
gsutil rewrite -s ARCHIVE gs://your-bucket/landuse/global/experiment_*/
```

---

## Advanced: Custom Vertex AI Pipeline

For complex multi-stage workflows, use Vertex AI Pipelines:

```python
# pipelines/vertex_pipeline.py
from kfp.v2 import dsl

@dsl.pipeline(name="landuse-global-pipeline")
def landuse_pipeline(
    bucket: str,
    project: str
):
    # Stage 0
    ingest_op = dsl.ContainerOp(
        name="stage0-ingest",
        image="gcr.io/.../landuse-pipeline:latest",
        arguments=["pipelines/global/stage0_ingest.py", "--config", "configs/cloud.yaml"]
    )
    
    # Stage 1 (depends on Stage 0)
    align_op = dsl.ContainerOp(
        name="stage1-align",
        image="gcr.io/.../landuse-pipeline:latest",
        arguments=["pipelines/global/stage1_align.py", "--config", "configs/cloud.yaml"]
    ).after(ingest_op)
    
    # ... more stages
```

Submit pipeline:
```bash
python cloud/submit_pipeline.py --pipeline pipelines/vertex_pipeline.py
```

---

## Support

- **GCP Issues**: https://cloud.google.com/support
- **Pipeline Issues**: See `docs/AGENT_RUNBOOK.md`
- **Documentation**: https://cloud.google.com/vertex-ai/docs
