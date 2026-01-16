
* 你已经在 **同一仓库中创建了 `Landuse_Global_Pipeline` 分支**
* 当前工作分支仍是 `master`
* 目标是：**不乱码、不污染 master，让 Cursor agent 在“新目录 + 新分支”中完成 global + cloud-first 重构**


---

````md
# Global Pipeline Refactor & Cursor Agent Execution Guide

## 0. 文档目的

本文档用于**固化 Landuse 项目从 Notebook 串行流程 → Global Cloud-First Pipeline 的代码重构规范**，并明确：

- Git 分支与目录的正确使用方式
- Cursor agent 在新目录 / 新分支中的执行边界
- Notebook 到 pipeline stage 的迁移规则
- Google Cloud (GCS + Cloud Job) 的最小接入方案

**核心目标只有一个：**
> 在不破坏现有论文与本地流程（master）的前提下，构建一个可扩展、可云端运行的 Global Pipeline。

---

## 1. 当前 Git 状态（已确认）

```bash
C:\Dev\Landuse_Zhong_clean> git branch
  Landuse_Global_Pipeline
* master
````

### 各分支角色定义

| 分支                        | 角色          | 规则              |
| ------------------------- | ----------- | --------------- |
| `master`                  | 论文/历史主线     | 冻结；不允许 agent 重构 |
| `Landuse_Global_Pipeline` | Global 重构分支 | 只用于 pipeline 升级 |

⚠️ **重要说明**
`git branch` ≠ 新目录。
若直接在 `master` 目录里 `checkout Landuse_Global_Pipeline` 并让 agent 工作，**极易产生状态错乱（尤其是 Notebook）**。

---

## 2. 强制执行的 Git + 目录策略（必须遵守）

### 2.1 原则（不可破）

> **一个物理目录 = 一个逻辑分支 = 一个 Cursor agent**

---

### 2.2 正确做法（立刻执行）

在 `master` 目录中执行：

```bash
cd C:\Dev\Landuse_Zhong_clean
git status   # 必须干净

git worktree add ..\Landuse_Global_Pipeline_Worktree Landuse_Global_Pipeline
```

结果应为：

```text
C:\Dev\
├── Landuse_Zhong_clean                (master, 冻结)
└── Landuse_Global_Pipeline_Worktree   (Landuse_Global_Pipeline, 开发)
```

✅ **此后规则**

* ❌ 不在 `Landuse_Zhong_clean` 里运行 Cursor agent
* ✅ 用 Cursor 打开 `Landuse_Global_Pipeline_Worktree`
* ✅ agent 的所有修改仅发生在该目录

---

## 3. 原始 Notebook 流程（逻辑基线）

原始项目采用 **串行 Notebook 执行逻辑**：

1. 数据准备
   `0.0 → 2.1 → 2.2 → 2.3`

2. 环境适宜性建模
   `3.0 (+ Process.ipynb)`

3. 减排与经济评估
   `4.1 → 5.1`

4. 3E 协同分析
   `6.4`

5. 可视化
   `6.5 – 6.9`

6. 多尺度分析
   `7.0 / 7.1 / 8.0 / 9.0`

⚠️ 该逻辑 **仅作为方法论参考**，不再作为 global pipeline 的执行方式。

---

## 4. Global Pipeline 的 Stage 化重构方案

### 4.1 Notebook → Pipeline Stage 映射

| 原 Notebook                               | Pipeline Stage   |
| ---------------------------------------- | ---------------- |
| `0.0 PV_dataset.ipynb`                   | stage0_ingest    |
| `2.1 process_csv_for_aligning.ipynb`     | stage1_align     |
| `2.2 process_csv_for_embedding.ipynb`    | stage2_embed     |
| `2.3 process_csv_for_prediction.ipynb`   | stage3_predprep  |
| `3.0 pre-training.ipynb`                 | stage4_env_train |
| `Process.ipynb`                          | stage5_env_post  |
| `4.1 Emission_reduction_potential.ipynb` | stage6_carbon    |
| `5.1 Economical_feasibility.ipynb`       | stage7_econ      |
| `6.4 3E_synergy_index.ipynb`             | stage8_synergy   |
| `6.5–6.9 Figure*.ipynb`                  | stage9_figures   |
| `7.x / 8.0 / 9.0`                        | analysis modules |

---

## 5. Global Pipeline 目标目录结构（必须生成）

在 `Landuse_Global_Pipeline_Worktree` 中建立：

```text
Landuse_Global_Pipeline_Worktree/
├── src/landuse/
│   ├── io/                # local / gcs I/O abstraction
│   ├── data/              # manifest, tiling, catalog
│   ├── indicators/        # align & feature extraction
│   ├── env_model/         # GMM + Transformer-ResNet
│   ├── carbon/            # LNCS emission
│   ├── econ/              # NPV & scenarios
│   └── synergy/           # 3E index (WCCD)
│
├── pipelines/global/
│   ├── stage0_ingest.py
│   ├── stage1_align.py
│   ├── stage2_embed.py
│   ├── stage3_predprep.py
│   ├── stage4_env_train.py
│   ├── stage5_env_post.py
│   ├── stage6_carbon.py
│   ├── stage7_econ.py
│   ├── stage8_synergy.py
│   └── stage9_figures.py
│
├── configs/
│   └── global.yaml
│
├── cloud/
│   ├── docker/Dockerfile
│   ├── submit_job.py
│   └── README_cloud.md
│
├── notebooks/
│   └── sandbox/           # 仅用于调试
│
├── docs/
│   ├── REQUIREMENTS.md
│   ├── MIGRATION_MAP.md
│   └── AGENT_RUNBOOK.md
│
└── pyproject.toml / requirements.txt
```

---

## 6. Google Cloud 接入规范（最小可用）

### 6.1 设计原则

* **所有大规模数据与中间结果直接写入 GCS**
* 本地仅保存：

  * 配置文件
  * 小样例
  * 日志与 manifest

### 6.2 必须实现的模块

**`src/landuse/io/gcs.py`**

* `upload(local_path, gcs_path)`
* `download(gcs_path, local_path)`
* `open_gcs(gcs_path)`
* 支持 csv / parquet / tif

### 6.3 配置入口

**`configs/global.yaml`**

```yaml
run:
  mode: cloud    # local | cloud

gcs:
  bucket: your-bucket
  prefix: landuse/global

tiling:
  enabled: true
  tile_size: 5deg
```

---

## 7. Cursor Agent 执行规范（必须遵守）

### 7.1 Agent 允许做的事

* 创建目录结构
* 抽取 Notebook 逻辑为模块函数
* 生成 pipeline stage 脚本
* 添加 GCS I/O
* 生成 Docker / cloud job 脚本

### 7.2 Agent 禁止做的事

* ❌ 修改 `master`
* ❌ 修改原始 Notebook（仅可读取）
* ❌ 提交大数据文件
* ❌ 长时间运行真实 global 任务

---

## 8. 推荐的 Agent 任务拆分（顺序执行）

1. **任务 1：结构与迁移映射**

   * 生成 `MIGRATION_MAP.md`
   * 建立空 pipeline skeleton

2. **任务 2：Stage 1–3（对齐与特征）**

   * 支持 GCS 输出
   * 单 tile 可运行

3. **任务 3：Stage 4（GMM + Transformer）**

   * 抽离训练逻辑
   * 提供 cloud job 入口

4. **任务 4：Stage 6–8（碳 / 经济 / 3E）**

每个任务 **一个 commit**，commit message 标注 stage。

---

## 9. 合并策略（最后一步）

```bash
cd C:\Dev\Landuse_Zhong_clean
git merge Landuse_Global_Pipeline
```

合并前必须满足：

* pipeline 可运行（至少单 tile）
* 不依赖 Notebook 顺序执行
* 所有大数据路径为 `gs://`

---

## 10. 终极原则（务必牢记）

> **Notebook 是方法说明，不是工程执行单元**
> **Global pipeline = stage + config + cloud job**

```

---

### 如果你愿意的下一步
我可以**直接再给你生成**：

- `AGENT_RUNBOOK.md`（你复制给 Cursor agent 的标准指令集）
- `MIGRATION_MAP.md`（按你现有 notebook 精确到 cell 的迁移表）
- `configs/global.yaml` 初始模板（与你的指标体系一一对应）

你只需要说一句：  
👉 **“下一步生成哪一个文件”**
```
