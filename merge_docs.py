#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并REGRESSION_ANALYSIS_COMPREHENSIVE_GUIDE.md和GPU配置文档
"""

# 读取原始文档
with open('REGRESSION_ANALYSIS_COMPREHENSIVE_GUIDE.md', 'r', encoding='utf-8') as f:
    original_content = f.read()

# 新增的"零章"：环境配置与准备
chapter_zero = """# 回归分析综合指南（含GPU加速）

---

## 零、环境配置与准备

### 0.1 基础环境要求

#### 必需软件与包

**Python环境（必需）：**
- Python 3.10+ 或 3.11
- 包管理器：conda 或 pip

**核心依赖包：**
```bash
# 数据处理与分析
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# 统计建模
statsmodels>=0.13.0
scikit-learn>=1.0.0

# 可视化
matplotlib>=3.4.0
seaborn>=0.11.0

# 地理空间分析
geopandas>=0.10.0
shapely>=1.8.0

