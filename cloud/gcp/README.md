# Google Cloud AI Platform 敏感性分析部署指南

## 概述

本目录包含将敏感性分析任务部署到Google Cloud AI Platform的完整方案。通过并行执行18个实验，大幅缩短敏感性分析时间。

## 项目信息

- **GCP项目ID**: `trial-285112`
- **GCS存储桶**: `gs://pv_cropland/`
- **账户**: `zpy1023840872@gmail.com`
- **数据位置**: 本地 `data/` 目录（与 `Supplymentary/` 同级）

## 目录结构

```
cloud/gcp/
├── Dockerfile                  # 容器镜像定义
├── requirements.txt            # Python依赖
├── sensitivity_utils.py        # 工具函数（正交表、参数映射等）
├── prepare_data.py            # 数据准备和上传脚本
├── run_experiment.py          # 单个实验执行脚本
├── submit_jobs.py             # 任务提交脚本
├── aggregate_results.py        # 结果聚合脚本
└── README.md                  # 本文档
```

## 前置要求

### 1. 安装Google Cloud SDK

```bash
# 安装gcloud CLI
# https://cloud.google.com/sdk/docs/install

# 登录并设置项目
gcloud auth login
gcloud config set project trial-285112
```

### 2. 启用必要的API

```bash
# 启用AI Platform API
gcloud services enable aiplatform.googleapis.com

# 启用Storage API
gcloud services enable storage-component.googleapis.com
```

### 3. 安装Python依赖

```bash
pip install google-cloud-aiplatform google-cloud-storage
```

### 4. 检查GPU配额

确保GCP项目有足够的GPU配额（18个任务需要18个GPU）：

```bash
gcloud compute project-info describe --project=trial-285112
```

查看输出中的 `GPUS_ALL_REGIONS` 或 `NVIDIA_T4_GPUS` 配额。

**重要：如果GPU配额为0，需要先申请配额才能运行任务。**

#### 申请GPU配额步骤：

1. **访问GCP控制台配额页面**：
   ```
   https://console.cloud.google.com/iam-admin/quotas?project=trial-285112
   ```

2. **筛选GPU配额**：
   - 在搜索框输入：`GPU` 或 `NVIDIA`
   - 选择区域（如 `us-central1`）
   - 查找 `NVIDIA T4 GPUs` 或 `GPUs (all regions)`

3. **申请增加配额**：
   - 点击配额项 → "EDIT QUOTAS"
   - 输入新限制值（建议至少20，留有余量）
   - 填写申请理由："需要运行18个并行的机器学习训练任务进行敏感性分析"
   - 提交申请

4. **等待审批**：
   - 配额申请通常需要几小时到几天时间
   - 审批通过后会收到邮件通知

**注意**：Trial账户可能无法申请GPU配额，需要升级到付费账户。

## 使用流程

### 步骤1: 准备数据

从项目根目录或 `Supplymentary/` 目录运行：

```bash
python cloud/gcp/prepare_data.py --bucket pv_cropland
```

此脚本会：
- 自动定位 `data/` 目录
- 加载和预处理数据（df_embedding_fill, df_abandon_filtered）
- 上传到GCS：
  - `data/df_positive.pkl`
  - `data/df_prediction_pool.pkl`
  - `data/features.json`
  - `data/gmm_model.pkl` (如果存在)
  - `data/US_data/cb_2018_us_nation_5m.*` (shapefile文件)

### 步骤2: 构建并推送Docker镜像

有两种方式：

#### 方式A：使用Google Cloud Build（推荐，无需本地Docker）

从项目根目录运行：

```bash
# 启用Cloud Build API（首次使用）
gcloud services enable cloudbuild.googleapis.com --project=trial-285112

# 提交构建任务到Cloud Build（使用配置文件）
gcloud builds submit --config=cloud/gcp/cloudbuild.yaml --project=trial-285112

# 或者使用简化的tag方式（会自动创建临时配置）
gcloud builds submit --tag gcr.io/trial-285112/sensitivity-analysis:latest --project=trial-285112
```

此命令会：
- 自动上传项目文件到Cloud Build
- 在GCP上构建镜像
- 自动推送到GCR（Google Container Registry）

**优点**：无需本地安装Docker，构建速度更快

#### 方式B：本地构建（需要安装Docker Desktop）

1. **安装Docker Desktop for Windows**：
   - 下载：https://www.docker.com/products/docker-desktop/
   - 安装后重启计算机
   - 确保Docker Desktop正在运行

2. **配置Docker以使用GCR**：
   ```bash
   gcloud auth configure-docker
   ```

3. **构建并推送镜像**：
   ```bash
   docker build -t gcr.io/trial-285112/sensitivity-analysis:latest -f cloud/gcp/Dockerfile .
   docker push gcr.io/trial-285112/sensitivity-analysis:latest
   ```

### 步骤3: 提交任务

从项目根目录运行：

**Linux/Mac (bash)**:
```bash
python cloud/gcp/submit_jobs.py \
  --project_id trial-285112 \
  --region us-central1 \
  --bucket pv_cropland \
  --image gcr.io/trial-285112/sensitivity-analysis:latest
```

**Windows PowerShell**:
```powershell
python cloud/gcp/submit_jobs.py `
  --project_id trial-285112 `
  --region us-central1 `
  --bucket pv_cropland `
  --image gcr.io/trial-285112/sensitivity-analysis:latest
```

**或者单行（所有平台）**:
```bash
python cloud/gcp/submit_jobs.py --project_id trial-285112 --region us-central1 --bucket pv_cropland --image gcr.io/trial-285112/sensitivity-analysis:latest
```

此脚本会：
- 生成18个参数组合（使用L18(3^6)正交表）
- 上传配置文件到GCS
- 提交18个并行任务到GCP AI Platform
- 每个任务配置：
  - 机器类型: `n1-standard-8` (8 vCPU, 30GB RAM)
  - GPU: `NVIDIA_TESLA_T4` x 1
  - 区域: `us-central1` (可修改)

任务ID会保存到 `cloud/gcp/submitted_jobs.json`

### 步骤4: 监控任务

```bash
# 查看所有任务状态
gcloud ai custom-jobs list --region=us-central1 --project=trial-285112

# 查看特定任务日志
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

或在GCP控制台查看：
https://console.cloud.google.com/ai/platform/custom-jobs

### 步骤5: 聚合结果

等待所有任务完成后，从项目根目录运行：

**Linux/Mac (bash)**:
```bash
python cloud/gcp/aggregate_results.py \
  --bucket pv_cropland \
  --output_dir Supplymentary/ML_sensitivity
```

**Windows PowerShell**:
```powershell
python cloud/gcp/aggregate_results.py `
  --bucket pv_cropland `
  --output_dir Supplymentary/ML_sensitivity
```

**或者单行**:
```bash
python cloud/gcp/aggregate_results.py --bucket pv_cropland --output_dir Supplymentary/ML_sensitivity
```

此脚本会：
- 从GCS下载所有18个实验的结果
- 创建6个DataFrame（overfitting, accuracy, precision, recall, f1, mean_prob）
- 生成摘要JSON（包括最佳参数组合）
- 保存到 `Supplymentary/ML_sensitivity/`

### 步骤6: 下载模型文件（可选）

如果需要下载所有实验的模型文件到本地：

**Linux/Mac (bash)**:
```bash
# 下载所有18个实验的模型
python cloud/gcp/download_models.py \
  --bucket pv_cropland \
  --output_dir Supplymentary/ML_sensitivity/models

# 只下载特定实验的模型
python cloud/gcp/download_models.py \
  --bucket pv_cropland \
  --exp_ids 1 2 3 \
  --output_dir Supplymentary/ML_sensitivity/models
```

**Windows PowerShell**:
```powershell
# 下载所有18个实验的模型
python cloud/gcp/download_models.py `
  --bucket pv_cropland `
  --output_dir Supplymentary/ML_sensitivity/models

# 只下载特定实验的模型
python cloud/gcp/download_models.py `
  --bucket pv_cropland `
  --exp_ids 1 2 3 `
  --output_dir Supplymentary/ML_sensitivity/models
```

**只列出GCS中的模型（所有平台）**:
```bash
python cloud/gcp/download_models.py --bucket pv_cropland --list_only
```

**模型保存位置**：
- GCS: `gs://pv_cropland/results/{E1-E18}/models/`
- 本地（下载后）: `Supplymentary/ML_sensitivity/models/{E1-E18}/`

**每个实验的模型包含**：
- `{exp_id}_transformer_generation.pkl` - 主模型文件（包含所有路径信息）
- `{exp_id}_transformer_generation_gmm.pkl` - GMM模型
- `{exp_id}_transformer_generation_dl.h5` - 深度学习模型（TensorFlow格式）
- `{exp_id}_transformer_generation_preprocessor.pkl` - 预处理器
- `{exp_id}_transformer_generation_test_data.npz` - 测试数据
- `{exp_id}_transformer_generation_config.json` - 配置文件

## 输出文件

### 结果文件（在 `Supplymentary/ML_sensitivity/` 目录）

- `sensitivity_results_all.csv` - 所有实验的完整结果
- `df_overfitting.csv` - 过拟合分数
- `df_accuracy.csv` - 准确率
- `df_precision.csv` - 精确率
- `df_recall.csv` - 召回率
- `df_f1.csv` - F1分数
- `df_mean_prob.csv` - 平均概率
- `sensitivity_summary.json` - 结果摘要（最佳参数组合）
- `orthogonal_config.json` - 正交表配置信息

### 模型文件（在GCS和本地）

**GCS位置**: `gs://pv_cropland/results/{E1-E18}/models/`

**本地位置**（下载后）: `Supplymentary/ML_sensitivity/models/{E1-E18}/`

**每个实验的模型文件**（共6个文件）：
1. `{exp_id}_transformer_generation.pkl` - 主模型文件（包含所有组件路径）
2. `{exp_id}_transformer_generation_gmm.pkl` - GMM环境相似度模型
3. `{exp_id}_transformer_generation_dl.h5` - Transformer深度学习模型
4. `{exp_id}_transformer_generation_preprocessor.pkl` - 数据预处理器
5. `{exp_id}_transformer_generation_test_data.npz` - 测试数据集
6. `{exp_id}_transformer_generation_config.json` - 训练配置和元数据

**模型加载方式**：
```python
from function.model_saving import load_complete_model_pipeline

# 加载模型
model_data = load_complete_model_pipeline(
    'Supplymentary/ML_sensitivity/models/E1/E1_transformer_generation.pkl'
)

# 使用模型
gmm_pipeline = model_data['gmm_pipeline']
dl_model = model_data['dl_model']
preprocessor = model_data['preprocessor']
```

## 成本估算

- **机器类型**: `n1-standard-8` + `NVIDIA_TESLA_T4`
- **每个任务预计运行时间**: 2-4小时
- **18个任务总计算时间**: 36-72小时（并行执行）
- **预估成本**: 约 $200-400（取决于实际运行时间和区域）

## 故障排除

### 1. 数据上传失败

**问题**: `prepare_data.py` 上传失败

**解决方案**:
- 检查GCS bucket权限
- 确认bucket名称正确
- 检查本地数据文件是否存在

### 2. 任务提交失败

**问题**: `submit_jobs.py` 提交失败

**解决方案**:
- 检查GPU配额是否足够
- 确认镜像已成功推送到GCR
- 检查区域是否支持GPU实例

### 3. 任务运行失败

**问题**: 任务在GCP上运行失败

**解决方案**:
- 查看任务日志：`gcloud ai custom-jobs stream-logs JOB_ID`
- 检查数据文件是否已正确上传
- 确认容器镜像包含所有依赖

### 4. 结果聚合失败

**问题**: `aggregate_results.py` 无法下载结果

**解决方案**:
- 确认所有任务已完成
- 检查GCS中是否存在结果文件
- 验证bucket权限

## 注意事项

1. **路径处理**: 脚本会自动查找项目根目录，确保从正确的位置运行
2. **数据大小**: 数据文件较大，上传可能需要一些时间
3. **GPU配额**: 确保有足够的GPU配额，否则任务会排队等待
4. **区域选择**: 建议选择有GPU可用性的区域（如 us-central1, us-east1）
5. **成本控制**: 任务完成后及时删除不需要的资源

## 高级配置

### 修改训练参数

编辑 `submit_jobs.py` 中的 `config` 字典：

```python
config = {
    'epochs': 80,              # 训练轮数
    'batch_size': 256,         # 批次大小
    'learning_curve_epochs': 20,  # 学习曲线分析轮数
    # ... 其他参数
}
```

### 修改机器类型

在 `submit_jobs.py` 中修改：

```python
machine_type='n1-standard-8',  # 可改为其他类型
accelerator_type='NVIDIA_TESLA_T4',  # 可改为其他GPU类型
```

### 选择不同区域

```bash
python cloud/gcp/submit_jobs.py \
  --region us-east1 \  # 或其他区域
  ...
```

## 联系支持

如遇到问题，请检查：
1. GCP控制台的任务日志
2. 本地脚本的错误输出
3. GCS bucket中的文件结构

## 更新日志

- 2024-11-24: 初始版本，支持18个并行实验

