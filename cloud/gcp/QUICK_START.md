# 快速开始指南

## 一键执行流程

### 1. 准备数据（本地）
```bash
cd Supplymentary  # 或从项目根目录
python cloud/gcp/prepare_data.py --bucket pv_cropland
```

### 2. 构建并推送镜像

**方式A：使用Cloud Build（推荐，无需Docker）**
```bash
# 从项目根目录运行
# 首次使用需要启用API
gcloud services enable cloudbuild.googleapis.com --project=trial-285112

# 构建并推送镜像
gcloud builds submit --config=cloud/gcp/cloudbuild.yaml --project=trial-285112
```

**方式B：本地构建（需要Docker Desktop）**
```bash
# 安装Docker Desktop后运行
gcloud auth configure-docker
docker build -t gcr.io/trial-285112/sensitivity-analysis:latest -f cloud/gcp/Dockerfile .
docker push gcr.io/trial-285112/sensitivity-analysis:latest
```

### 3. 提交任务（本地）
```powershell
# Windows PowerShell（从项目根目录运行）
python cloud/gcp/submit_jobs.py `
  --project_id trial-285112 `
  --region us-central1 `
  --bucket pv_cropland `
  --image gcr.io/trial-285112/sensitivity-analysis:latest

# 或者单行（所有平台）
python cloud/gcp/submit_jobs.py --project_id trial-285112 --region us-central1 --bucket pv_cropland --image gcr.io/trial-285112/sensitivity-analysis:latest
```

### 4. 监控任务
```bash
gcloud ai custom-jobs list --region=us-central1 --project=trial-285112
```

### 5. 聚合结果（任务完成后）
```powershell
# Windows PowerShell（从项目根目录运行）
python cloud/gcp/aggregate_results.py `
    --bucket pv_cropland `
    --output_dir Supplymentary/ML_sensitivity

# 或者单行
python cloud/gcp/aggregate_results.py --bucket pv_cropland --output_dir Supplymentary/ML_sensitivity
```

### 6. 下载模型文件（可选）
```powershell
# Windows PowerShell（下载所有18个实验的模型）
python cloud/gcp/download_models.py `
    --bucket pv_cropland `
    --output_dir Supplymentary/ML_sensitivity/models

# 或者单行
python cloud/gcp/download_models.py --bucket pv_cropland --output_dir Supplymentary/ML_sensitivity/models
```

## 检查清单

- [ ] GCP项目已设置（trial-285112）
- [ ] GCS bucket已创建（pv_cropland）
- [ ] API已启用（AI Platform, Storage）
- [ ] GPU配额充足（18个GPU）
- [ ] 本地数据文件已准备好
- [ ] Docker已安装并配置
- [ ] gcloud CLI已安装并登录

## 预计时间

- 数据上传: 10-30分钟（取决于网络速度）
- 镜像构建: 5-10分钟
- 镜像推送: 5-10分钟
- 任务执行: 2-4小时/任务（18个并行）
- 结果聚合: 1-2分钟

## 常见问题

**Q: 如何检查任务状态？**  
A: 使用 `gcloud ai custom-jobs list` 或访问GCP控制台

**Q: 任务失败怎么办？**  
A: 查看日志 `gcloud ai custom-jobs stream-logs JOB_ID`，检查错误信息

**Q: 如何重新运行失败的实验？**  
A: 修改 `submit_jobs.py`，只提交失败的实验ID，或手动运行单个实验

