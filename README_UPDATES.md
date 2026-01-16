# Global Pipeline 更新说明

## 📋 最新更新 (2025-01-16)

### ✅ 新增功能

#### 1. 多目标优化模块 (`landuse.optimization`)

已将 `8.0 Multi-objective.ipynb` 的功能集成到 Global Pipeline：

**新增文件**：
- `src/landuse/optimization/__init__.py`
- `src/landuse/optimization/pareto.py` (421 行)
- `src/landuse/optimization/ranking.py` (117 行)

**功能**：
- **Pareto 前沿分析**: 3E 维度的多目标优化
- **效率核函数**: 递减/均匀/递增/指数/幂核
- **pymoo 集成**: 支持 NSGA-II 算法（可选依赖）
- **启发式后备**: 当 pymoo 不可用时的简化实现

**使用示例**：
```python
from landuse.optimization import ParetoOptimizer

optimizer = ParetoOptimizer(config)
pareto_solutions = optimizer.optimize(
    environment, emission, economic, areas,
    objectives=["environment", "emission", "economic"]
)
```

**配置**：
```yaml
# configs/global.yaml
optimization:
  enabled: true
  algorithm: "nsga2"
  population_size: 100
  n_generations: 200
  kernel:
    type: "decreasing"
```

---

#### 2. 数据依赖分析与排除

**排除美国特定功能** (`9.0 Energy_demand_adjust.ipynb`):

**原因**：
- 仅使用 NREL 美国电力数据
- 按 51 个美国州分析
- 美国特定的电气化情景
- 不适用于全球尺度

**数据依赖**：
```
❌ data/US_data/US_electricity/NREL/energy.csv.gzip
❌ data/cb_2018_us_state_500k.shp (美国州边界)
❌ US Department of Energy 2050 目标
```

**全球替代方案**：
- 使用 IEA World Energy Outlook（全球覆盖）
- 按国家/地区聚合（而非美国州）
- 国际能源情景（SSP/IEA scenarios）

**配置标记**：
```yaml
# configs/global.yaml
regional_exclusions:
  us_specific:
    enabled: false  # 全球 pipeline 中禁用
    note: "9.0 Energy_demand_adjust is US-specific"
```

---

### 📚 文档更新

#### 新增文档

1. **`docs/DATA_DEPENDENCIES.md`** - 数据依赖详细分析
   - 全球数据源列表
   - 美国特定数据识别
   - 全球替代方案建议
   - 数据流向图

#### 更新文档

1. **`docs/MIGRATION_MAP.md`**
   - 添加 8.0 Multi-objective 映射
   - 标注 9.0 为"NOT MIGRATED"
   - 详细说明排除原因

2. **`configs/global.yaml`**
   - 添加 `optimization` 配置节
   - 添加 `regional_exclusions` 说明
   - 明确标记美国特定功能

3. **`PROJECT_REFACTOR_SUMMARY.md`**
   - 更新模块统计（32 → 36 个文件）
   - 添加数据依赖关系表
   - 更新代码总量（~8,500 → ~9,000 行）

---

## 📊 Notebook 迁移状态更新

| Notebook | 状态 | 目标模块 | 说明 |
|----------|------|---------|------|
| `0.0-7.1` | ✅ 已迁移 | Stage 0-9 | 核心 pipeline |
| `8.0 Multi-objective.ipynb` | ✅ 已集成 | `landuse.optimization` | 多目标优化 |
| `9.0 Energy_demand_adjust.ipynb` | ⚠️ 已排除 | N/A | 美国特定，不迁移 |

---

## 🔧 使用指南

### 1. 运行多目标优化（可选）

```bash
# 在配置中启用优化
vim configs/global.yaml
# 设置: optimization.enabled = true

# 运行优化（作为 Stage 8 的扩展）
python pipelines/global/stage8_synergy.py --config configs/global.yaml --optimize
```

### 2. 检查数据依赖

```bash
# 查看数据依赖文档
cat docs/DATA_DEPENDENCIES.md

# 验证数据源
python scripts/verify_data_sources.py --config configs/global.yaml
```

### 3. 区域特定配置

```bash
# 全球模式（默认）
python pipelines/global/stage0_ingest.py --config configs/global.yaml

# 美国特定分析（如需要）
python pipelines/global/stage0_ingest.py --config configs/us_specific.yaml
```

---

## ⚠️ 重要说明

### 依赖项

如果要使用多目标优化的完整功能，需要安装可选依赖：

```bash
# 安装 pymoo (可选)
pip install pymoo

# 如果不安装，将使用简化的启发式算法
```

### 美国特定功能

**不要在全球 pipeline 中使用以下内容**：
- ❌ `9.0 Energy_demand_adjust.ipynb`
- ❌ NREL 能源数据
- ❌ US 州级边界
- ❌ US Department of Energy 情景

**如果需要美国特定分析**：
- 保持原 notebook 作为独立脚本
- 使用 `configs/us_specific.yaml`
- 不要集成到全球 pipeline

---

## 📝 下一步计划

### 已完成
- ✅ 多目标优化模块
- ✅ 数据依赖分析
- ✅ 美国特定功能排除
- ✅ 文档更新

### 待完善
1. **测试多目标优化模块**
   - 使用小样本数据测试
   - 验证 Pareto 前沿计算
   - 测试 pymoo 集成

2. **全球能源需求模块（可选）**
   - 集成 IEA 数据
   - 按国家聚合
   - 替代 NREL 功能

3. **区域适配指南**
   - 欧洲特定配置
   - 亚洲特定配置
   - 非洲特定配置

---

## 📧 问题反馈

如有疑问，请查阅：
- `docs/DATA_DEPENDENCIES.md` - 数据依赖详情
- `docs/MIGRATION_MAP.md` - Notebook 映射
- `docs/AGENT_RUNBOOK.md` - 执行手册

---

**更新日期**: 2025-01-16  
**版本**: 1.0.1  
**主要变更**: 添加多目标优化，排除美国特定功能
