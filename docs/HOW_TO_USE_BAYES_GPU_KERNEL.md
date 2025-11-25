# 🚀 如何在Cursor中使用bayes-gpu kernel（GPU加速）

## ✅ 环境配置完成总结

### bayes-gpu环境状态
- ✅ **GPU可用**: CudaDevice(id=0)
- ✅ **Python**: 3.10.19
- ✅ **numpy**: 2.2.6（支持JAX GPU）
- ✅ **pandas**: 2.3.3
- ✅ **geopandas**: 1.1.1（支持numpy 2.x）
- ✅ **JAX**: 0.4.38（GPU加速）
- ✅ **PyMC**: 5.25.1
- ✅ **NumPyro**: 0.19.0
- ✅ **Shapefile读取**: 正常
- ✅ **Jupyter kernel**: 已注册

### geo环境状态
- ✅ **numpy**: 1.26.4
- ✅ **pandas**: 2.2.2
- ✅ **geopandas**: 1.1.0
- ✅ **Shapefile读取**: 正常
- ❌ **GPU**: 不支持

---

## 📋 在Cursor中切换kernel的步骤

### Step 1: 重新加载Cursor窗口

**方法1：使用命令面板**
1. 按 `Ctrl+Shift+P`（Windows）或 `Cmd+Shift+P`（Mac）
2. 输入 `Reload Window`
3. 选择 `Developer: Reload Window`

**方法2：重启Cursor**
- 关闭Cursor
- 重新打开Cursor

### Step 2: 打开notebook并选择kernel

1. **打开文件**：`7.1 Analysis_State_level.ipynb`

2. **点击右上角的kernel选择器**：
   - 看到类似 `Python 3.11.x (geo)` 的按钮
   - 点击它

3. **在弹出的列表中选择**：
   ```
   Python (bayes-gpu GPU)
   ```

4. **等待kernel启动**（首次可能需要10-20秒）

5. **验证kernel已切换**：
   - 右上角应显示 `Python (bayes-gpu GPU)`

---

## 🎯 运行notebook的完整流程

### 方案A：一站式GPU加速（推荐）

**使用bayes-gpu kernel运行整个notebook**

1. **选择kernel**: `Python (bayes-gpu GPU)`

2. **运行Cell 1**（验证环境）：
   ```python
   # 添加新cell在notebook开头
   import jax
   import numpy
   import geopandas
   print(f"JAX devices: {jax.devices()}")
   print(f"numpy: {numpy.__version__}")
   print(f"geopandas: {geopandas.__version__}")
   ```
   
   **预期输出**：
   ```
   JAX devices: [CudaDevice(id=0)]
   numpy: 2.2.6
   geopandas: 1.1.1
   ```

3. **运行Cell 1-64**（数据加载、State-level、Pixel-level分析）
   - 正常运行，无需修改

4. **运行Cell 65**（State-between贝叶斯分析，自动使用GPU）
   - 确认代码是 `use_bayesian=True`
   - GPU将自动检测并使用
   - **预计耗时：5-15分钟**（5个州）

5. **查看GPU使用情况**（可选）：
   - 打开新的PowerShell窗口
   - 运行 `nvidia-smi`
   - 应该看到GPU使用率上升（30-80%）

---

### 方案B：双环境协作（最稳定）

**geo环境做前期分析，bayes-gpu环境做GPU分析**

#### Part 1: 在geo环境运行Cell 1-64

1. **选择kernel**: `Python 3.11.x (geo)`
2. **运行Cell 1-64**：数据加载、State-level、Pixel-level分析
3. **保存中间结果**（已自动保存到CSV）

#### Part 2: 切换到bayes-gpu运行Cell 65

1. **点击右上角kernel选择器**
2. **选择**: `Python (bayes-gpu GPU)`
3. **等待kernel切换完成**
4. **运行Cell 65**：State-between贝叶斯分析（GPU加速）
5. **完成后查看结果**

---

## 🔍 验证GPU是否正在使用

### 方法1：在notebook中检查

在运行分析前，添加一个cell：

```python
import jax
print(f"可用设备: {jax.devices()}")
print(f"GPU数量: {len(jax.devices('gpu'))}")

# 测试GPU计算
import jax.numpy as jnp
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
print(f"GPU测试通过: {y.shape}")
```

**预期输出**：
```
可用设备: [CudaDevice(id=0)]
GPU数量: 1
GPU测试通过: (1000, 1000)
```

### 方法2：监控GPU使用率

**在PowerShell中运行**（在分析运行期间）：

```powershell
# 持续监控GPU（每2秒刷新）
while($true) { nvidia-smi; sleep 2; cls }
```

**GPU使用的标志**：
- GPU-Util列显示 30-90%
- Memory-Usage增加（显存使用）
- 进程列表中有python进程

---

## 📊 性能对比预期

### 5个目标州（~15k pixels）

| 模型 | geo环境(CPU) | bayes-gpu环境(GPU) | 加速比 |
|------|-------------|-------------------|--------|
| State-between贝叶斯 | 30-90分钟 | **5-15分钟** | **6-10x** |
| MixedLM | 5-10分钟 | N/A | N/A |

### 全部48州（~65k pixels）

| 模型 | geo环境(CPU) | bayes-gpu环境(GPU) | 加速比 |
|------|-------------|-------------------|--------|
| State-between贝叶斯 | 3-6小时 | **20-40分钟** | **9-15x** |

---

## ⚠️ 常见问题

### 问题1：Cursor中看不到bayes-gpu kernel

**原因**：Cursor还没有刷新kernel列表

**解决方案**：
```
Ctrl+Shift+P → Developer: Reload Window
```

### 问题2：选择kernel后启动失败

**症状**：显示"Kernel启动失败"或"连接超时"

**解决方案1**：在WSL中手动启动Jupyter
```bash
source ~/bayes-gpu/bin/activate
cd /mnt/c/Dev/Landuse_Zhong_clean
jupyter notebook
```

**解决方案2**：检查kernel配置
```bash
cat ~/.local/share/jupyter/kernels/bayes-gpu/kernel.json
```

### 问题3：GPU没有被使用（GPU-Util=0%）

**可能原因**：
1. 代码中`use_bayesian=False`（检查Cell 65）
2. JAX没有检测到GPU（在notebook中运行`jax.devices()`）
3. 数据量太小（GPU开销大于收益）

**解决方案**：
- 确认`use_bayesian=True`
- 在notebook cell中运行：
  ```python
  import jax
  print(jax.devices())  # 应显示CudaDevice
  ```

### 问题4：显存不足（OOM错误）

**症状**：报错 "XLA allocation failed"

**解决方案**：在notebook开头添加：
```python
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # 限制70%显存
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```

---

## 🎉 快速开始检查清单

在运行分析前，确认：

- [ ] Cursor已重新加载（Reload Window）
- [ ] 打开了 `7.1 Analysis_State_level.ipynb`
- [ ] 右上角显示 `Python (bayes-gpu GPU)`
- [ ] 运行验证cell确认GPU可用（`jax.devices()`）
- [ ] Cell 65的`use_bayesian=True`
- [ ] 准备好监控GPU（`nvidia-smi`）

**一切就绪！开始您的GPU加速分析吧！** 🚀

---

## 📂 相关文件

- `ENVIRONMENT_FIXED.md` - 环境修复详情
- `REGRESSION_ANALYSIS_COMPREHENSIVE_GUIDE.md` - 完整分析指南
- `gpu-bayesian-regression.plan.md` - GPU配置计划（可归档）


