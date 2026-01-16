# 快速入门指南 (Quick Start Guide)

本指南帮助你在30分钟内完成环境配置并运行第一个示例。

---

## 🚀 15分钟快速演示

### Step 0: 环境准备 (5分钟)

```bash
# 1. 克隆或进入项目目录
cd c:\Dev\Landuse_Zhong_clean

# 2. 创建conda环境
conda env create -f geo.yml
conda activate geo

# 3. 验证安装
python -c "import xarray; import geopandas; import tensorflow; print('✅ All packages installed')"
```

### Step 1: 数据准备演示 (5分钟)

```bash
# 启动Jupyter
jupyter notebook

# 打开以下notebook进行快速测试:
# 0.0 PV_dataset.ipynb
```

**在notebook中运行**:
```python
import pandas as pd
import geopandas as gpd

# 检查数据结构
# 这一步仅检查数据格式,不执行完整流程
print("✅ 数据加载测试完成")
```

### Step 2: 快速训练示例 (5分钟)

```python
# 在 3.0 pre-training.ipynb 中运行小样本测试

from function import *

# 使用500个样本进行快速测试
df_positive_sample = df_positive.sample(500, random_state=42)
df_prediction_sample = df_prediction.sample(1000, random_state=42)

# 运行快速训练
results = run_correct_training_pipeline(
    df_positive_sample, 
    df_prediction_sample,
    features_no_coords,
    epochs=10,  # 快速测试仅10个epoch
    plot_learning_curve=True
)

print("✅ 模型训练测试完成")
```

---

## 📝 完整流程运行指南

### 前置条件检查清单

- [ ] **硬件**: 至少16GB RAM, 建议32GB
- [ ] **存储**: 至少100GB可用空间
- [ ] **Python**: 3.8-3.10 (推荐3.9)
- [ ] **数据**: 已下载必需的数据集

### 数据目录结构建议

```
D:\xarray\                          # 主数据目录 (可自定义)
├── merged_chunk_2\                 # 废弃农田数据
│   └── *.nc
├── aligned2\
│   ├── Feature_all\                # 15个环境特征
│   │   └── *.nc
│   ├── economic_cost\              # AR6经济数据
│   │   └── national_growth_rate\
│   │       ├── AR6_Scenarios_*.csv
│   └── carbon\                     # 碳汇数据
│       └── *.nc
└── output\                         # 输出结果
    └── models\
```

---

## 🎯 阶段性运行指南

### 阶段一: 数据预处理 (预计4-6小时)

#### 1.1 废弃农田识别
```bash
jupyter notebook "0.0 PV_dataset.ipynb"
```

**关键配置**:
```python
# 修改数据路径
PATHS = {
    'abandonment': r"你的路径\merged_chunk_2\*.nc",
    'csv': "对齐后的数据.csv"
}

# 5年移动窗口参数
WINDOW_SIZE = 5
MIN_DURATION = 5  # 最小废弃年限
```

**预期输出**:
- ✅ 废弃农田CSV (~4.7M行)
- ✅ 空间分布地图

#### 1.2 数据对齐与特征提取
```bash
# 依次运行
jupyter notebook "2.1 process_csv_for_aligning.ipynb"
jupyter notebook "2.2 process_csv_for_embedding.ipynb"
jupyter notebook "2.3 process_csv_for_prediction.ipynb"
```

**关键步骤**:
```python
# 在 2.2 中提取15维特征
features_to_extract = [
    # 物理地理
    'DEM', 'Slope', 'land_cover', 'gdmp',
    # 气候
    'tas', 'wind', 'rsds',
    # 社会经济
    'Population', 'GDPpc', 'GDPtot',
    'GURdist', 'Powerdist',
    'PrimaryRoad', 'SecondaryRoad', 'TertiaryRoad'
]
```

**验证检查**:
```python
# 检查特征矩阵
print(f"特征矩阵形状: {df.shape}")  # 应该是 (N, 15+其他列)
print(f"缺失值统计:\n{df.isnull().sum()}")  # 应该没有或很少缺失值
```

---

### 阶段二: 环境适宜性建模 (预计6-10小时)

#### 2.1 GMM训练与负样本生成
```bash
jupyter notebook "3.0 pre-training.ipynb"
```

**核心代码**:
```python
from function import run_correct_training_pipeline

# 完整训练配置
results = run_correct_training_pipeline(
    df_positive=df_pv,                    # 光伏正样本
    df_prediction_pool=df_abandoned,      # 废弃农田
    features_no_coords=features_15,       # 15个特征
    
    # 负样本策略
    negative_strategy='selection',        # 'selection'或'generation'
    negative_ratio=1.0,                   # 正负比例1:1
    sampling_strategy='pit_based',        # 基于PIT的分层采样
    difficulty_levels=3,                  # 3个难度级别
    
    # 训练参数
    model_type='transformer',             # 'transformer', 'mlp', 或 'rf'
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    
    # 模型架构
    transformer_config={
        'd_model': 64,
        'num_heads': 4,
        'num_layers': 2
    },
    resnet_layers=[128, 128, 64],
    
    # 诊断
    plot_learning_curve=True,
    run_shap=True,                        # 特征重要性分析
    
    random_state=42
)
```

**中间检查**:
```python
# 检查GMM结果
print(f"GMM组件数: {results['gmm_pipeline'].named_steps['gmm'].n_components}")
print(f"BIC值: {results['gmm_pipeline'].named_steps['gmm'].bic(X)}")

# 检查负样本质量
neg_scores = results['negative_samples']['gmm_score']
print(f"负样本得分范围: [{neg_scores.min():.3f}, {neg_scores.max():.3f}]")
```

**预期输出**:
- ✅ `gmm_model_23c_fixed.pkl` (~50MB)
- ✅ 训练历史曲线图
- ✅ SHAP特征重要性图
- ✅ F1 Score > 0.85

#### 2.2 模型预测
```python
# 获取预测结果
predictions = results['prediction_results']
print(f"预测概率均值: {predictions['predicted_prob'].mean():.3f}")
print(f"高适宜性(>0.7)比例: {(predictions['predicted_prob']>0.7).mean():.1%}")

# 保存预测结果
predictions.to_csv('output/prediction_probability.csv', index=False)
```

---

### 阶段三: 碳减排潜力 (预计3-5小时)

```bash
jupyter notebook "4.1 Emission_reduction_potential.ipynb"
```

**关键参数**:
```python
# 光伏参数
PV_PARAMS = {
    'efficiency': 0.17,           # kW/m²
    'system_loss': 0.8,           # 系统效率
    'lifetime': 30,               # 年
    'annual_hours': 8760          # 小时/年
}

# LNCS策略权重(从历史数据学习)
lncs_weights = calculate_lncs_probability(
    df_abandoned, 
    strategy='knn_idw',
    k_neighbors=10
)
```

**输出验证**:
```python
# 检查减排结果
pv_mitigation = df['pv_carbon_total'].sum() / 1e9  # Gt CO2
lncs_mitigation = df['lncs_carbon_total'].sum() / 1e9
net_mitigation = pv_mitigation - lncs_mitigation

print(f"光伏总减排: {pv_mitigation:.2f} Gt CO₂")
print(f"LNCS总碳汇: {lncs_mitigation:.2f} Gt CO₂")
print(f"净减排: {net_mitigation:.2f} Gt CO₂")

# 应该接近: 光伏~62.83, LNCS~3.91, 净~58.92
```

---

### 阶段四: 经济可行性 (预计2-4小时)

```bash
jupyter notebook "5.1 Economical_feasibility.ipynb"
```

**AR6情景加载**:
```python
# 读取AR6数据
df_ar6 = pd.read_csv('AR6_Scenarios_Database_R10_regions_v1.1.csv')

# 筛选美国数据
df_us = df_ar6[df_ar6['Region'] == 'R10NORTH_AM']

# 提取关键变量
scenarios = ['P1a', 'P1b', 'P2a', 'P2b', 'P2c', 'P3a', 'P3b', 'P3c']
years = [2020, 2030, 2040, 2050]

# 提取电价、投资成本、运营成本
electricity_prices = extract_prices(df_us, scenarios, years)
investment_costs = extract_costs(df_us, scenarios, years, 'Capital')
operation_costs = extract_costs(df_us, scenarios, years, 'O&M')
```

**NPV计算**:
```python
def calculate_npv(row, scenario, discount_rate=0.05):
    """计算单个像元的NPV"""
    revenue = row['generation'] * electricity_prices[scenario]
    costs = investment_costs[scenario] + operation_costs[scenario]
    lncs_cost = row['lncs_opportunity_cost']
    
    npv = sum([
        (revenue[t] - costs[t]) / ((1 + discount_rate) ** t)
        for t in range(30)
    ]) - lncs_cost
    
    return npv

# 应用到所有像元
for scenario in scenarios:
    df[f'npv_{scenario}'] = df.apply(
        lambda row: calculate_npv(row, scenario), 
        axis=1
    )

# 计算均值
df['avg_npv'] = df[[f'npv_{s}' for s in scenarios]].mean(axis=1)
```

---

### 阶段五: 3E协同分析 (预计2-3小时)

#### 5.1 3E-Synergy指数
```bash
jupyter notebook "6.4 3E_synergy_index.ipynb"
```

**归一化处理**:
```python
from sklearn.preprocessing import MinMaxScaler

# 三个维度
e1_env = df['predicted_prob']              # 环境 (已在0-1)
e2_emission = df['net_carbon_mitigation']  # 减排 (需归一化)
e3_economic = df['avg_npv']                # 经济 (需归一化)

# 归一化
scaler = MinMaxScaler()
df['E1'] = e1_env
df['E2'] = scaler.fit_transform(e2_emission.values.reshape(-1, 1))
df['E3'] = scaler.fit_transform(e3_economic.values.reshape(-1, 1))
```

**WCCD计算**:
```python
from scipy.optimize import minimize

def calculate_3e_synergy(row):
    """为单个像元计算3E-synergy"""
    U = [row['E1'], row['E2'], row['E3']]
    
    def objective(w):
        """优化目标: 最大化CCD"""
        C = np.prod([u**w[i] for i, u in enumerate(U)]) / (np.mean(U) ** 3)
        T = sum(w[i] * U[i] for i in range(3))
        return -(C * T) ** 0.5  # 负号因为minimize
    
    # 约束: Σw=1, w≥0
    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}
    bounds = [(0, 1) for _ in range(3)]
    
    result = minimize(
        objective, 
        x0=[1/3, 1/3, 1/3],
        method='SLSQP',
        constraints=constraints,
        bounds=bounds
    )
    
    return -result.fun  # 返回最大CCD值

# 应用到所有像元
df['3e_synergy'] = df.apply(calculate_3e_synergy, axis=1)
```

#### 5.2 优先级排序
```bash
jupyter notebook "6.5 Figure2_priority_total.ipynb"
```

**对比分析**:
```python
# 四种策略
strategies = {
    'env_optimal': df.sort_values('E1', ascending=False),
    'emission_optimal': df.sort_values('E2', ascending=False),
    'economic_optimal': df.sort_values('E3', ascending=False),
    '3e_synergy': df.sort_values('3e_synergy', ascending=False)
}

# 累积性能
def cumulative_performance(df_sorted, target_area=0.1):
    """计算累积性能"""
    n = int(len(df_sorted) * target_area)
    top_n = df_sorted.head(n)
    
    return {
        'env': top_n['E1'].mean(),
        'emission': top_n['E2'].sum(),
        'economic': top_n['E3'].sum()
    }

# 对比
for name, df_sorted in strategies.items():
    perf = cumulative_performance(df_sorted, 0.1)
    print(f"{name}: E1={perf['env']:.3f}, E2={perf['emission']:.1f}Gt, E3=${perf['economic']:.0f}B")
```

---

### 阶段六: 多尺度分析 (预计2-3小时)

#### 6.1 探索性数据分析(可选)
```bash
jupyter notebook "6.1 EDA_data.ipynb"
```

#### 6.2 国家层面分析
```bash
jupyter notebook "7.0 Analysis_Nation_level.ipynb"
```

#### 6.3 州层面分析
```bash
jupyter notebook "7.1 Analysis_State_level.ipynb"
```

#### 6.4 多目标优化与能源需求
```bash
jupyter notebook "8.0 Multi-objective.ipynb"
jupyter notebook "9.0 Energy_demand_adjust.ipynb"
```

#### 6.5 附录分析
```bash
jupyter notebook "7.3 Appendix_figure.ipynb"
```

**州级汇总**:
```python
# 空间连接
gdf = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs='EPSG:4326'
)

# 与州边界叠加
us_states = gpd.read_file('data/us_states.shp')
gdf_with_state = gpd.sjoin(gdf, us_states, how='left')

# 州级汇总
state_summary = gdf_with_state.groupby('STATE_NAME').agg({
    'abandoned_area': 'sum',           # kha
    'pv_capacity': 'sum',              # GW
    'E1': 'mean',                      # 环境
    'net_carbon_mitigation': 'sum',    # Gt CO2
    'avg_npv': 'mean',                 # k USD/ha
    '3e_synergy': 'mean',              # 协同指数
    'energy_demand': 'first'           # TWh
}).reset_index()

# 能源需求满足度
state_summary['demand_met'] = (
    state_summary['pv_capacity'] * 8760 * 0.2 / 1000  # TWh
) / state_summary['energy_demand']

print(f"可100%满足需求的州数: {(state_summary['demand_met'] >= 1).sum()}")
```

---

---

## 🎨 可视化快速生成

**注意**: 可视化模块需要先完成前面的核心计算步骤

### 主图快速生成
```bash
# Figure 1: 环境适宜性空间分布
jupyter notebook "6.6 Figure1_Enviromental_plot.ipynb"

# Figure 2: 政策情景矩阵与优先级
jupyter notebook "6.7 Figure2_Policy_matrix.ipynb"
jupyter notebook "6.5 Figure2_priority_total.ipynb"

# Figure 3: 光伏vs LNCS碳减排对比
jupyter notebook "6.8 Figure3_Carbon_LNCS.ipynb"

# Figure 4: 累积收益曲线
jupyter notebook "6.9 Figure4_Cumulative_pirority.ipynb"
```

**执行顺序建议**:
1. 先完成阶段1-5的核心计算
2. 确保生成了所有中间结果文件
3. 再运行可视化notebook生成图表

### 基础绘图模板
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 空间分布图
fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(
    column='3e_synergy',
    cmap='RdYlGn',
    legend=True,
    ax=ax,
    vmin=0, vmax=1
)
us_states.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
ax.set_title('3E-Synergy Index Spatial Distribution')
plt.savefig('figure/3e_synergy_map.pdf', dpi=300, bbox_inches='tight')

# 累积曲线
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (dim, label) in enumerate([('E1', 'Environment'), 
                                    ('E2', 'Emission'), 
                                    ('E3', 'Economic')]):
    for name, df_sorted in strategies.items():
        cumsum = df_sorted[dim].cumsum() / df_sorted[dim].sum()
        axes[i].plot(
            np.linspace(0, 1, len(cumsum)),
            cumsum,
            label=name
        )
    axes[i].set_title(label)
    axes[i].legend()
plt.savefig('figure/cumulative_curves.pdf', dpi=300, bbox_inches='tight')
```

---

## 💾 结果保存建议

```python
# 1. 预测结果
df_results = df[[
    'lon', 'lat', 'STATE_NAME',
    'predicted_prob', 'net_carbon_mitigation', 'avg_npv',
    'E1', 'E2', 'E3', '3e_synergy'
]]
df_results.to_csv('output/final_results.csv', index=False)

# 2. 州级汇总
state_summary.to_csv('output/state_summary.csv', index=False)

# 3. 模型文件
results['model'].save('output/models/transformer_model.h5')
joblib.dump(results['gmm_pipeline'], 'output/models/gmm_pipeline.pkl')

# 4. 栅格输出 (可选)
from rasterio.transform import from_origin

# 转为栅格
raster = df_results.pivot(index='lat', columns='lon', values='3e_synergy')
with rasterio.open(
    'output/3e_synergy.tif', 'w',
    driver='GTiff',
    height=raster.shape[0],
    width=raster.shape[1],
    count=1,
    dtype=raster.values.dtype,
    crs='EPSG:4326',
    transform=from_origin(raster.columns.min(), raster.index.max(), 0.00833, 0.00833)
) as dst:
    dst.write(raster.values, 1)
```

---

## 🐛 常见错误速查

### 错误1: "ModuleNotFoundError: No module named 'function'"
```bash
# 解决: 确保在项目根目录运行
cd c:\Dev\Landuse_Zhong_clean
python -c "import function; print(function.__file__)"
```

### 错误2: "KeyError: 'predicted_prob'"
```bash
# 解决: 检查是否完成阶段二训练
# 预测结果应该包含'predicted_prob'列
```

### 错误3: "MemoryError"
```python
# 解决: 分批处理
chunk_size = 10000
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    process(chunk)
```

### 错误4: "ValueError: operands could not be broadcast"
```python
# 解决: 检查数组形状
print(f"数组形状: {arr.shape}")
arr = arr.reshape(-1, 1)  # 转为列向量
```

---

## 📚 进阶学习资源

1. **论文原文**: `【2020-Policy-informed priority...】.md`
2. **方法论文档**: `docs/REGRESSION_ANALYSIS_COMPREHENSIVE_GUIDE.md`
3. **GPU环境**: `docs/HOW_TO_USE_BAYES_GPU_KERNEL.md`
4. **模型保存**: `docs/SAVE_MODEL_PARAMS.md`

---

## 📞 获取帮助

- **文档**: 查看 `README.md` 和 `PIPELINE_VISUALIZATION.md`
- **代码注释**: 所有function模块都有详细docstring
- **Issues**: 项目issue tracker (如有)

---

**祝你研究顺利! 🎉**
