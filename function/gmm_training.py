# -*- coding: utf-8 -*-
"""
整合版本的GMM+深度学习训练管道
- 统一的GMM训练和深度学习流程
- 解决数据泄露：预处理器在Pipeline中
- 完整的负样本采样和模型评估
- 简化的API设计

Author: you + ChatGPT 
"""
from __future__ import annotations

import os
import warnings
from typing import List, Tuple, Dict, Any, Optional, Sequence

from sklearn.pipeline import Pipeline as SkPipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    PowerTransformer,
    FunctionTransformer,
    OneHotEncoder,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, mean_squared_error, mean_absolute_error,
    roc_auc_score
)
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple


SkPipeline = Pipeline

# 深度学习库
try:
    import tensorflow as tf
    # 使用 tf.keras 而不是 from tensorflow import keras（避免递归错误）
    # TensorFlow 2.15 兼容方式
    keras = tf.keras
    layers = keras.layers  # 使用 keras.layers 而不是 tensorflow.keras.layers
    TENSORFLOW_AVAILABLE = True
    print("[OK] TensorFlow available")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[WARN] TensorFlow not available")
except RecursionError as e:
    # 捕获递归错误（TensorFlow 2.15 的已知问题）
    TENSORFLOW_AVAILABLE = False
    print(f"[WARN] TensorFlow import recursion error: {e}")
    print("[INFO] This may be a TensorFlow 2.15 compatibility issue")
except Exception as e:
    # 捕获所有异常，不仅仅是 ImportError
    TENSORFLOW_AVAILABLE = False
    print(f"[WARN] TensorFlow not available: {type(e).__name__}: {e}")

# scikeras 延迟导入（在 TensorFlow 成功导入后，避免 keras.api 兼容性问题）
SCIKERAS_AVAILABLE = False
KerasClassifier = None

def _ensure_scikeras():
    """确保 scikeras 已导入，处理 keras.api 兼容性问题"""
    global SCIKERAS_AVAILABLE, KerasClassifier
    if SCIKERAS_AVAILABLE and KerasClassifier is not None:
        return True
    
    if not TENSORFLOW_AVAILABLE:
        return False
    
    try:
        # 在 TensorFlow 已导入的情况下，延迟导入 scikeras
        from scikeras.wrappers import KerasClassifier as _KerasClassifier
        KerasClassifier = _KerasClassifier
        SCIKERAS_AVAILABLE = True
        print("[OK] scikeras available")
        return True
    except ModuleNotFoundError as e:
        if 'keras.api' in str(e):
            print(f"[WARN] scikeras version incompatible with TensorFlow 2.11: {e}")
            print("[INFO] Try upgrading scikeras: pip install --upgrade scikeras>=0.12.0")
        SCIKERAS_AVAILABLE = False
        return False
    except ImportError:
        SCIKERAS_AVAILABLE = False
        print("[WARN] scikeras not available (pip install scikeras)")
        return False
    except Exception as e:
        SCIKERAS_AVAILABLE = False
        print(f"[WARN] scikeras import failed: {type(e).__name__}: {e}")
        return False

# 如果 TensorFlow 已可用，尝试导入 scikeras
if TENSORFLOW_AVAILABLE:
    _ensure_scikeras()

# SHAP（可选）
try:
    import shap
    SHAP_AVAILABLE = True
    print("[OK] SHAP available")
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not available")
except Exception as e:
    # 捕获所有异常，不仅仅是 ImportError
    SHAP_AVAILABLE = False
    print(f"[WARN] SHAP not available: {type(e).__name__}: {e}")

# 可视化设置
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ------------------------------
# 修复：组合预处理器（解决数据泄露）
# ------------------------------


class CombinedPreprocessor(BaseEstimator, TransformerMixin):
    """
    组合预处理器：数值 + 类别（One-Hot），确保特征维度一致性
    """
    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.numeric_preprocessor = None
        self.categorical_preprocessor = None
        self.feature_names_out_ = None
        
        # ✅ 关键修复：预定义landcover的所有可能类别
        self.known_landcover_categories = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 固定1-9类别
        
        # 数值特征预处理器
        if self.numeric_features:
            self.numeric_preprocessor = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
        
        # ✅ 类别特征预处理器 - 固定类别
        if self.categorical_features:
            self.categorical_preprocessor = OneHotEncoder(
                categories=[self.known_landcover_categories],  # 固定类别
                sparse_output=False, 
                handle_unknown='ignore',  # 忽略未知类别
                drop=None  # 保留所有类别
            )

    def fit(self, X, y=None):
        """拟合预处理器"""
        # 确保输入是DataFrame并且列存在
        if not isinstance(X, pd.DataFrame):
            raise ValueError("输入必须是pandas DataFrame")
        
        # 检查特征列是否存在
        missing_features = [f for f in (self.numeric_features + self.categorical_features) if f not in X.columns]
        if missing_features:
            raise ValueError(f"以下特征列在输入数据中不存在: {missing_features}")
        
        # 拟合数值特征预处理器
        if self.numeric_features:
            self.numeric_preprocessor.fit(X[self.numeric_features])
        
        # 拟合类别特征预处理器
        if self.categorical_features:
            # ✅ 确保landcover列是整数类型
            X_cat = X[self.categorical_features].copy()
            for col in self.categorical_features:
                if col == 'landcover':
                    X_cat[col] = X_cat[col].astype(int)
            
            # ✅ 关键修复：重新初始化OneHotEncoder并强制设置categories
            self.categorical_preprocessor = OneHotEncoder(
                categories=[self.known_landcover_categories],  # 重新设置固定类别
                sparse_output=False, 
                handle_unknown='ignore',
                drop=None
            )
            self.categorical_preprocessor.fit(X_cat)
            
        # ✅ 生成固定的特征名称
        self._generate_feature_names()
        return self

    def transform(self, X):
        """转换数据"""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("输入必须是pandas DataFrame")
        
        # 检查是否已拟合
        if ((self.numeric_features and self.numeric_preprocessor is None) or 
            (self.categorical_features and self.categorical_preprocessor is None)):
            raise ValueError("必须先调用fit方法")
        
        results = []
        
        # 转换数值特征
        if self.numeric_features:
            X_num_transformed = self.numeric_preprocessor.transform(X[self.numeric_features])
            results.append(X_num_transformed)
        
        # 转换类别特征
        if self.categorical_features:
            X_cat = X[self.categorical_features].copy()
            for col in self.categorical_features:
                if col == 'landcover':
                    X_cat[col] = X_cat[col].astype(int)
            
            X_cat_transformed = self.categorical_preprocessor.transform(X_cat)
            results.append(X_cat_transformed)
        
        # 合并结果
        if results:
            return np.hstack(results)
        else:
            return np.array([]).reshape(X.shape[0], 0)

    def _generate_feature_names(self):
        """生成特征名称"""
        feature_names = []
        
        # 数值特征名称
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
        
        # ✅ 类别特征名称 - 固定生成
        if self.categorical_features:
            for col in self.categorical_features:
                cat_names = self.categorical_preprocessor.get_feature_names_out([col])
                feature_names.extend(cat_names)
        
        self.feature_names_out_ = np.array(feature_names)

    def get_feature_names_out(self, input_features=None):
        """获取输出特征名称"""
        if self.feature_names_out_ is None:
            raise ValueError("必须先调用fit方法")
        return self.feature_names_out_

    def get_params(self, deep=True):
        """获取参数"""
        return {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }

    def set_params(self, **params):
        """设置参数"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


#---------------------------
# Top-level, picklable transformers (no lambdas!)
# ------------------------------
class ReplaceInfWithNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X[~np.isfinite(X)] = np.nan
        return X

class SafeLog1p(BaseEstimator, TransformerMixin):
    def __init__(self, lower_bound: float = -1 + 1e-6):
        self.lower_bound = lower_bound
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X = np.where(X <= self.lower_bound, self.lower_bound, X)
        return np.log1p(X)

# ------------------------------
# Utilities
# ------------------------------
def get_adaptive_n_quantiles(n_samples: int) -> int:
    """Adaptive n_quantiles for QuantileTransformer: 10..1000 and ≤ n_samples."""
    return int(max(10, min(1000, n_samples)))

# ------------------------------
# Light-weight quality check helpers
# ------------------------------
def _cov_condition_number(X: np.ndarray) -> Dict[str, float]:
    X = np.asarray(X, dtype=float)
    C = np.cov(X, rowvar=False)
    C += np.eye(C.shape[0]) * 1e-12
    try:
        w = np.linalg.eigvalsh(C)
        w = np.clip(w, 0.0, None)
        w_min = float(np.min(w))
        w_max = float(np.max(w))
        cond = float(w_max / (w_min + 1e-18))
        return {"min_eig": w_min, "max_eig": w_max, "condition_number": cond, "eigen_ratio": cond}
    except np.linalg.LinAlgError:
        return {"min_eig": np.nan, "max_eig": np.nan, "condition_number": np.inf, "eigen_ratio": np.inf}

def comprehensive_data_quality_check(
    X: np.ndarray, feature_names: List[str] | None = None, verbose: bool = True
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    X = np.asarray(X)
    n, d = X.shape
    var = np.nanvar(X, axis=0)
    zero_thr = 1e-12
    low_thr = 1e-4

    report: Dict[str, Any] = {
        "shape": (n, d),
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum()),
        "zero_count": int((X == 0).sum()),
        "variance_range": (float(np.nanmin(var)), float(np.nanmax(var))),
        "zero_variance_count": int(np.sum(var <= zero_thr)),
        "low_variance_count": int(np.sum(var <= low_thr)),
    }
    cov_info = _cov_condition_number(X)
    report["covariance_analysis"] = cov_info

    if verbose:
        print("数据形状:", report["shape"])
        print("NaN值数量:", report["nan_count"])
        print("Inf值数量:", report["inf_count"])
        print("零值数量:", report["zero_count"])
        print("方差范围:", "[%.2e, %.2e]" % report["variance_range"])
        print("零方差特征数:", report["zero_variance_count"], "；低方差特征数:", report["low_variance_count"])
        print("协方差矩阵条件数: %.2e" % cov_info["condition_number"])
        print("最小/最大特征值: %.2e / %.2e" % (cov_info["min_eig"], cov_info["max_eig"]))

    recs: List[Dict[str, Any]] = []
    return report, recs



def select_and_train_gmm(df_pos: pd.DataFrame, bandwidths=None, use_bic=False):

    """
    修复版本的GMM训练函数（升级版）：
    - 解决数据泄露（预处理器进Pipeline）；
    - 搜索 n_init 与 reg_covar；
    - 打印单组合排行榜；
    - 分类型绘图并带误差条。
    """
    print(f"输入数据形状: {df_pos.shape}")

    if "landcover" in df_pos.columns:
        landcover_values = df_pos["landcover"].value_counts().sort_index()
        print("\n检查 landcover 分布:")
        print(f"唯一值: {sorted(df_pos['landcover'].unique())}")
        print(f"分布: {dict(landcover_values)}")

    LOG = ['GDPpc', 'GDPtot', 'Population', 'Powerdist']
    DEM_SLOPE = ['DEM', 'Slope']
    DIST = ['GURdist', 'PrimaryRoad', 'SecondaryRoad', 'TertiaryRoad']
    NORMAL = ['tas', 'gdmp', 'rsds', 'wind']
    CAT = ['landcover']

    all_numeric_features = LOG + DEM_SLOPE + DIST + NORMAL
    available_numeric = [f for f in all_numeric_features if f in df_pos.columns]
    available_categorical = [f for f in CAT if f in df_pos.columns]
    if not available_numeric and not available_categorical:
        raise ValueError("没有找到任何可用的特征列")

    # 预处理器
    print("\n 创建组合预处理器...")
    combined_preprocessor = CombinedPreprocessor(available_numeric, available_categorical)

    # 预处理探查
    print("测试预处理器...")
    test_preprocessor = CombinedPreprocessor(available_numeric, available_categorical)
    X_test = test_preprocessor.fit_transform(df_pos)
    print(f"预处理后特征形状: {X_test.shape}")

    # 质量检查
    quality_report, _ = comprehensive_data_quality_check(
        X_test, feature_names=[f"f{i}" for i in range(X_test.shape[1])], verbose=True
    )
    # print("\n数据质量概要：")
    # print(f"NaN: {quality_report['nan_count']}  |  Inf: {quality_report['inf_count']}")
    # print("方差范围: [%.2e, %.2e]" % quality_report["variance_range"])

    cond = quality_report["covariance_analysis"]["condition_number"]
    if cond > 1e12:
        # print("⚠️ 数值较不稳定，使用保守参数")
        gmm_params = dict(n_components=1, covariance_type="diag", reg_covar=1e-3, random_state=0)
    else:
        print("✅ 数值稳定，使用标准参数")
        gmm_params = dict(n_components=1, covariance_type="full", reg_covar=1e-6, random_state=0)

    # 完整Pipeline
    print("\n构建完整Pipeline（包含预处理器）...")
    full_pipe = Pipeline([
        ("preprocessor", combined_preprocessor),
        ("gmm", GaussianMixture(**gmm_params)),
    ])

    # 参数网格
    if bandwidths is not None and np.size(bandwidths) > 0:
        comps = sorted({int(max(1, round(float(b)))) for b in np.ravel(bandwidths)})
    else:
        n_samples = len(df_pos)
        max_k = min(35, n_samples // 50)  
        comps = list(range(15, max_k + 1, 2))
    
    cov_types = ["diag", "full"]
    n_init_list = [20]
    reg_list = [1e-7, 1e-6]
    
    param_grid = {
        "gmm__n_components": comps,
        "gmm__covariance_type": cov_types,
        "gmm__n_init": n_init_list,
        "gmm__reg_covar": reg_list,
    }
    
    if use_bic:
        def gmm_bic_scorer(estimator, X, y=None):
            Xt = estimator[:-1].transform(X)
            return -estimator[-1].bic(Xt)
        scoring = gmm_bic_scorer
        print("使用BIC作为评分标准")
    else:
        scoring = None  # 使用 Pipeline.score -> GMM 的平均对数似然
        print("使用对数似然作为评分标准")

    
    grid = GridSearchCV(
        estimator=full_pipe,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,  # ✅ 可选的评分标准
        n_jobs=-1,
        refit=True,
        verbose=1,
        error_score="raise"
    )

    # 进度条（按网格规模粗略估计）
    total_iters = len(comps) * len(cov_types) * len(n_init_list) * len(reg_list)
    print("\n开始训练...")
    with tqdm(total=total_iters, desc="GMM训练") as pbar:
        grid.fit(df_pos)
        pbar.update(total_iters)

    # 结果
    best_params = grid.best_params_
    best_score  = grid.best_score_
    best_pipe   = grid.best_estimator_

    print("\n" + "=" * 60)
    print("训练完成！最佳参数:")
    print("=" * 60)
    print(f"n_components   : {best_params['gmm__n_components']}")
    print(f"covariance_type: {best_params['gmm__covariance_type']}")
    print(f"n_init         : {best_params['gmm__n_init']}")
    print(f"reg_covar      : {best_params['gmm__reg_covar']:.1e}")
    print(f"最佳CV均值对数似然: {best_score:.6f}")

    # --- 打印排行榜（单组合粒度） ---
    res = pd.DataFrame(grid.cv_results_)
    cols = [
        "param_gmm__n_components",
        "param_gmm__covariance_type",
        "param_gmm__n_init",
        "param_gmm__reg_covar",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    leaderboard = res[cols].sort_values("mean_test_score", ascending=False)
    print("\nTop-15 单组合排行榜（越高越好）:")
    print(leaderboard.head(15).to_string(index=False))

    # --- 两条曲线 + 误差条（每个 cov、K 选取该组最佳组合的 std） ---
    try:
        plot_cv_by_covariance_with_errorbars(grid.cv_results_, best_params)
    except Exception as e:
        print(f"⚠️ 分类型误差条绘图失败: {e}")

    # 经典的（跨类型平均）的总览图（可选）
    try:
        plot_loglik_vs_components(grid.cv_results_, best_params)
    except Exception as e:
        print(f"⚠️ 总览图绘制失败: {e}")

    # 边界提醒
    if (best_params["gmm__n_components"] == max(comps)
        and best_params["gmm__covariance_type"] == "full"):
        print("\n💡 提示：最佳模型在 K 上触到上界，后续可扩大 K 或继续细化 reg_covar 网格。")

    # 保存Pipeline
    model_filename = f"gmm_model_{best_params['gmm__n_components']}c_fixed.pkl"
    try:
        joblib.dump(best_pipe, model_filename)
        print(f"\n✅ 完整Pipeline已保存到: {model_filename}")
    except Exception as e:
        print(f"⚠️ 模型保存失败: {e}")

    return best_pipe


# ------------------------------
# 修复：评分API（提升便捷性）
# ------------------------------
def score_env(
    gmm_pipeline: Pipeline,
    df_query: pd.DataFrame,
    method: str = 'sigmoid',
    sigmoid_alpha: float = 1.0,
    reference_stats: Dict | None = None,
    return_logdens: bool = False,
):
    """
    一致性评分函数（支持固定标定）：
    - 优先使用 gmm_pipeline.calibration_（若存在），否则用 reference_stats，
      再否则退回当前批次自适应。
    - 支持 method ∈ {'sigmoid','minmax','zscore'}。
    - 可返回 logdens 以便后续分析。

    reference_stats 可包含的键：
      - 对 sigmoid/zscore：'mu' 或 'mean'，以及可选 'std'
      - 对 minmax：'min','max'
    """
    # 1) 预处理 + GMM打分（log域更稳定）
    Xp = gmm_pipeline.named_steps['preprocessor'].transform(df_query)
    gmm: GaussianMixture = gmm_pipeline.named_steps['gmm']
    logdens = gmm.score_samples(Xp)
    dens = np.exp(logdens)

    # 2) 选择标定参数来源：pipeline.calibration_ > reference_stats > 当前批次
    calib = getattr(gmm_pipeline, "calibration_", None)
    ref = reference_stats or {}
    # 统一取值
    def pick(keys, default=None):
        for k in keys:
            if calib and k in calib:
                return calib[k]
            if k in ref:
                return ref[k]
        return default

    mu  = pick(['mu', 'mean'], float(logdens.mean()))
    std = pick(['std'], float(logdens.std()))
    vmin = pick(['min'], float(logdens.min()))
    vmax = pick(['max'], float(logdens.max()))

    # 3) 计算分数（带数值保护）
    if method == 'sigmoid':
        # 数值裁剪，避免 exp 溢出
        x = np.clip(sigmoid_alpha * (logdens - mu), -50.0, 50.0)
        scores = 1.0 / (1.0 + np.exp(-x))
    elif method == 'minmax':
        rng = max(vmax - vmin, 1e-12)
        scores = (logdens - vmin) / rng
        # 防止轻微越界
        scores = np.clip(scores, 0.0, 1.0)
    elif method == 'zscore':
        s = std if std and std > 1e-12 else 1.0
        scores = (logdens - mu) / s
    else:
        raise ValueError(f"Unknown method: {method}")

    # 4) 输出统计（用当前批次的，用于日志/回写）
    stats_out = {
        'mean': float(logdens.mean()),
        'mu': float(mu),
        'std': float(std),
        'min': float(logdens.min()),
        'max': float(logdens.max()),
    }

    if return_logdens:
        return dens, scores, stats_out, logdens
    else:
        return dens, scores, stats_out





def split_pos_for_calibration(df_pos: pd.DataFrame, calib_frac: float = 0.2, random_state: int = 42):
    """把正样本拆成训练(1-calib_frac)与标定(calib_frac)。返回 df_train, df_calib。"""
    idx = np.arange(len(df_pos))
    rs = np.random.RandomState(random_state)
    rs.shuffle(idx)
    cut = int(len(idx) * (1 - calib_frac))
    return df_pos.iloc[idx[:cut]].copy(), df_pos.iloc[idx[cut:]].copy()



def attach_env_calibration(gmm_pipeline: Pipeline, df_calib: pd.DataFrame, robust: bool = True):
    """
    在独立的标定集上估计 log-density 的统计量并挂到 pipeline.calibration_。
    robust=True 用 median/MAD，重尾更稳。
    """
    Xp = gmm_pipeline.named_steps['preprocessor'].transform(df_calib)
    gmm: GaussianMixture = gmm_pipeline.named_steps['gmm']
    logp = gmm.score_samples(Xp)
    if robust:
        med = float(np.median(logp))
        mad = float(np.median(np.abs(logp - med)) + 1e-12)
        std = 1.4826 * mad  # 把 MAD 转成近似标准差
        mu = med
    else:
        mu = float(np.mean(logp))
        std = float(np.std(logp) + 1e-12)
    gmm_pipeline.calibration_ = {
        "mu": mu, "std": std,
        "min": float(np.min(logp)),
        "max": float(np.max(logp))
    }
    return gmm_pipeline



def logdensity_reference_stats(gmm_pipeline: Pipeline, df_ref: pd.DataFrame, qs=(0.01, 0.05, 0.10)):
    Xp = gmm_pipeline.named_steps['preprocessor'].transform(df_ref)
    gmm: GaussianMixture = gmm_pipeline.named_steps['gmm']
    logp = gmm.score_samples(Xp)
    stats = {
        "mu": float(np.mean(logp)),
        "std": float(np.std(logp) + 1e-12),
        "quantiles": {f"Q{int(q*100)}": float(np.quantile(logp, q)) for q in qs}
    }
    return stats, logp

def assess_similarity_by_logdensity(gmm_pipeline: Pipeline,
                                    df_ref_pos: pd.DataFrame,
                                    df_query: pd.DataFrame,
                                    q_cut: float = 0.05):
    """
    返回：每个查询样本是否“相似”（logp >= 参考集 q_cut 分位阈值）、其 z-score 与 logp。
    """
    ref_stats, ref_logp = logdensity_reference_stats(gmm_pipeline, df_ref_pos, qs=(q_cut,))
    Xq = gmm_pipeline.named_steps['preprocessor'].transform(df_query)
    gmm: GaussianMixture = gmm_pipeline.named_steps['gmm']
    logp_q = gmm.score_samples(Xq)
    z = (logp_q - ref_stats["mu"]) / ref_stats["std"]
    thr = list(ref_stats["quantiles"].values())[0]
    is_similar = logp_q >= thr
    return {
        "threshold": thr,
        "ref_mu": ref_stats["mu"], "ref_std": ref_stats["std"],
        "similar_mask": is_similar,
        "logp_query": logp_q,
        "z_query": z
    }



def plot_loglik_vs_components(grid_results: Dict[str, Any], best_params: Dict[str, Any]) -> None:
    """绘制对数似然vs组件数（风格仿 Figure5）"""
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12

    df = pd.DataFrame(grid_results)
    if "param_gmm__n_components" not in df.columns:
        warnings.warn("Grid results do not contain 'param_gmm__n_components'. Skipping plot.")
        return
    series = df.groupby("param_gmm__n_components")["mean_test_score"].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(series.index, series.values, marker="o", color='#1F78B4', linewidth=2)
    plt.axvline(x=best_params["gmm__n_components"], linestyle="--", color='#E31A1C', linewidth=2, label=f"Best: {best_params['gmm__n_components']}")
    plt.xlabel("n_components")
    plt.ylabel("CV mean log-likelihood (higher is better)")
    plt.title("Log-likelihood vs n_components (CV)")
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()



def plot_cv_by_covariance_with_errorbars(cv_results: Dict[str, Any], best_params: Dict[str, Any]) -> None:
    """按协方差类型分组绘制CV结果（风格仿 Figure5）"""
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12

    df = pd.DataFrame(cv_results)
    need_cols = [
        "param_gmm__n_components",
        "param_gmm__covariance_type",
        "param_gmm__n_init",
        "param_gmm__reg_covar",
        "mean_test_score",
        "std_test_score",
    ]
    for c in need_cols:
        if c not in df.columns:
            warnings.warn(f"cv_results_ 缺少列: {c}，跳过分类型误差图")
            return
    idx = df.groupby(["param_gmm__covariance_type", "param_gmm__n_components"])["mean_test_score"].idxmax()
    best_per_k_cov = df.loc[idx].sort_values(["param_gmm__covariance_type", "param_gmm__n_components"])
    fig, ax = plt.subplots(figsize=(9, 5))
    color_map = {
        'full': '#1F78B4',
        'tied': '#33A02C',
        'diag': '#E31A1C',
        'spherical': '#FDBF6F'
    }
    for cov, sub in best_per_k_cov.groupby("param_gmm__covariance_type"):
        color = color_map.get(cov, None)
        ax.errorbar(
            sub["param_gmm__n_components"],
            sub["mean_test_score"],
            yerr=sub["std_test_score"],
            marker="o",
            capsize=3,
            label=f"{cov}",
            color=color,
            linewidth=2
        )
    ax.axvline(best_params["gmm__n_components"], linestyle="--", color='#E31A1C', linewidth=2, label=f"Best K={best_params['gmm__n_components']}")
    ax.set_xlabel("n_components (K)")
    ax.set_ylabel("CV mean log-likelihood (↑)")
    ax.set_title("CV score by K (split by covariance_type) with error bars")
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def _ecdf(values: np.ndarray):
    v = np.sort(values)
    y = np.linspace(0, 1, len(v), endpoint=False)
    return v, y

def _hist_bins_clip(x: np.ndarray, pct_lo=0.5, pct_hi=99.5, max_bins=60):
    lo, hi = np.percentile(x, [pct_lo, pct_hi])
    xs = np.clip(x, lo, hi)
    return xs, max_bins, (lo, hi)

def _get_component_covs(gmm: GaussianMixture):
    """返回每个组件的协方差矩阵列表（full/diag/tied 统一为 full 矩阵）"""
    cov_type = gmm.covariance_type
    n_comp = gmm.n_components
    covs = []
    if cov_type == "full":
        covs = [gmm.covariances_[k] for k in range(n_comp)]
    elif cov_type == "diag":
        for k in range(n_comp):
            covs.append(np.diag(gmm.covariances_[k]))
    elif cov_type == "tied":
        covs = [gmm.covariances_ for _ in range(n_comp)]
    else:
        # 兜底：按 diag 处理
        for k in range(n_comp):
            covs.append(np.diag(gmm.covariances_[k]))
    return covs

def _mahalanobis2_per_sample(Xp: np.ndarray, gmm: GaussianMixture):
    """对每个样本：选责任度最大的组件，计算该组件下的马氏距离平方（全维）"""
    resp = gmm.predict_proba(Xp)           # (n, K)
    assign = resp.argmax(axis=1)           # (n,)
    means = gmm.means_                     # (K, d)
    covs = _get_component_covs(gmm)        # list of (d,d)

    md2 = np.empty(len(Xp), dtype=float)
    eps = 1e-9
    for i in range(len(Xp)):
        k = assign[i]
        delta = Xp[i] - means[k]
        C = covs[k] + np.eye(covs[k].shape[0]) * eps
        # 用solve比显式逆更稳
        md2[i] = float(delta @ np.linalg.solve(C, delta))
    return md2, assign




def visualize_similarity_diagnostics(gmm_pipeline: Pipeline,
                                     df_ref_pos: pd.DataFrame,
                                     df_query: pd.DataFrame,
                                     q_cut: float = 0.05,
                                     max_points_pca: int = 8000,
                                     random_state: int = 42):
    """
    增强版相似性诊断函数 - 包含详细的统计分析
    新增功能：
    1. 5个相似性层级的详细统计
    2. PIT均匀性测试和分布偏重分析
    3. 马氏距离的多个分位数统计
    4. 综合相似性评分系统
    5. 实际应用意义的解释
    6. PIT分布的上下四分位数和均值稳健性估计（attach_env）
    """
    from sklearn.decomposition import PCA

    pre = gmm_pipeline.named_steps['preprocessor']
    gmm: GaussianMixture = gmm_pipeline.named_steps['gmm']

    # ----- 1) 计算 log-density -----
    Xr = pre.transform(df_ref_pos)
    Xq = pre.transform(df_query)
    logp_ref = gmm.score_samples(Xr)
    logp_q = gmm.score_samples(Xq)

    # 参考分布的阈值（默认5%）
    thr = float(np.quantile(logp_ref, q_cut))
    mu, std = float(np.mean(logp_ref)), float(np.std(logp_ref) + 1e-12)
    z_q = (logp_q - mu) / std

    # ----- 2) 画图 -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # (a) log-density 重叠直方图
    ref_clip, bins_ref, _ = _hist_bins_clip(logp_ref)
    q_clip,   bins_q,   _ = _hist_bins_clip(logp_q)
    bins = max(bins_ref, bins_q)
    ax1.hist(ref_clip, bins=bins, alpha=0.6, label=f"Ref pos (n={len(logp_ref)})")
    ax1.hist(q_clip,   bins=bins, alpha=0.6, label=f"Query (n={len(logp_q)})")
    ax1.axvline(thr, color="red", linestyle="--", label=f"{int(q_cut*100)}% threshold")
    ax1.set_title("Log-density overlap (clipped to [0.5%, 99.5%])")
    ax1.set_xlabel("log p(x)  (transformed feature space)")
    ax1.set_ylabel("count")
    ax1.legend(frameon=False)

    # (b) PIT/分位直方图：q 的 logp 在 ref 的 ECDF 中的百分位
    v_ref, y_ref = _ecdf(logp_ref)
    ranks = np.searchsorted(v_ref, logp_q, side="left")
    pit = ranks / max(1, len(v_ref))  # ∈[0,1)
    ax2.hist(pit, bins=20, range=(0, 1), alpha=0.85)
    ax2.set_title("PIT of query w.r.t ref (Uniform≈well-matched; Left-heavy≈OOD)")
    ax2.set_xlabel("percentile")
    ax2.set_ylabel("count")

    # (c) PCA-2D + GMM 椭圆（95% 等概率轮廓）
    rs = np.random.RandomState(random_state)
    idx_r = np.arange(len(Xr))
    idx_q = np.arange(len(Xq))
    if len(idx_r) > max_points_pca:
        idx_r = rs.choice(idx_r, size=max_points_pca, replace=False)
    if len(idx_q) > max_points_pca:
        idx_q = rs.choice(idx_q, size=max_points_pca, replace=False)
    Xr_s = Xr[idx_r]
    Xq_s = Xq[idx_q]

    pca = PCA(n_components=2, random_state=random_state)
    Zr = pca.fit_transform(Xr_s)
    Zq = pca.transform(Xq_s)

    ax3.scatter(Zr[:,0], Zr[:,1], s=6, alpha=0.25, label="Ref pos (PCA2)")
    ax3.scatter(Zq[:,0], Zq[:,1], s=8, alpha=0.5, label="Query (PCA2)")

    # 画 GMM 组件在 PCA-2D 下的95%椭圆
    comps = pca.components_[:2]
    means = gmm.means_
    covs = _get_component_covs(gmm)
    chi2_95 = 5.991
    for k in range(gmm.n_components):
        m_full = means[k][None, :]
        m_2d = pca.transform(m_full)[0]
        C = covs[k]
        C2 = comps @ C @ comps.T
        w, V = np.linalg.eigh(C2)
        w = np.maximum(w, 1e-12)
        width, height = 2*np.sqrt(chi2_95*w)
        angle = np.degrees(np.arctan2(V[1,0], V[0,0]))
        from matplotlib.patches import Ellipse
        ell = Ellipse(xy=m_2d, width=width, height=height, angle=angle,
                      edgecolor='k', facecolor='none', lw=1.5, alpha=0.8)
        ax3.add_patch(ell)

    ax3.set_title("PCA-2D with GMM 95% ellipses (for intuition only)")
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2")
    ax3.legend(frameon=False)

    # (d) 马氏距离（全维、按责任度最近簇）
    md2_ref, _ = _mahalanobis2_per_sample(Xr, gmm)
    md2_q,   _ = _mahalanobis2_per_sample(Xq, gmm)
    thr_md2 = float(np.quantile(md2_ref, 0.95))
    xr_clip, _, _ = _hist_bins_clip(md2_ref)
    xq_clip, _, _ = _hist_bins_clip(md2_q)
    ax4.hist(xr_clip, bins=60, alpha=0.6, label="Ref pos")
    ax4.hist(xq_clip, bins=60, alpha=0.6, label="Query")
    ax4.axvline(thr_md2, color='red', linestyle='--', label="Ref 95% MD²")
    ax4.set_title("Mahalanobis distance² by assigned component (full-dim)")
    ax4.set_xlabel("MD²"); ax4.set_ylabel("count")
    ax4.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    # =====================================================
    # 计算详细的相似性统计指标
    # =====================================================

    # 1. 基于 log-density 的相似性分析
    # 5%分位数以下（极度相似）
    below_5pct = np.sum(logp_q >= thr)
    pct_below_5 = 100 * below_5pct / len(logp_q)

    # 不同相似性层次的划分
    ref_quantiles = np.percentile(logp_ref, [5, 25, 50, 75, 95])
    q5, q25, q50, q75, q95 = ref_quantiles

    # 查询样本在不同层次的分布
    extremely_similar = np.sum(logp_q >= q95)  # 前5%
    highly_similar = np.sum((logp_q >= q75) & (logp_q < q95))  # 75%-95%
    moderately_similar = np.sum((logp_q >= q25) & (logp_q < q75))  # 25%-75%
    poorly_similar = np.sum((logp_q >= q5) & (logp_q < q25))  # 5%-25%
    outliers = np.sum(logp_q < q5)  # 后5%

    similarity_levels = {
        'extremely_similar': (extremely_similar, 100 * extremely_similar / len(logp_q)),
        'highly_similar': (highly_similar, 100 * highly_similar / len(logp_q)),
        'moderately_similar': (moderately_similar, 100 * moderately_similar / len(logp_q)),
        'poorly_similar': (poorly_similar, 100 * poorly_similar / len(logp_q)),
        'outliers': (outliers, 100 * outliers / len(logp_q))
    }

    # 2. PIT 分析（概率积分变换）
    pit_uniform_test = np.abs(np.histogram(pit, bins=10, range=(0,1))[0] - len(pit)/10).mean()
    pit_left_heavy = np.sum(pit < 0.2) / len(pit)  # 左偏重（OOD指标）
    pit_right_heavy = np.sum(pit > 0.8) / len(pit)  # 右偏重

    # ==== PIT分布的上下四分位数、均值及其稳健性估计 ====
    def attach_env(arr):
        arr = np.asarray(arr)
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        n = len(arr)
        sem = std / np.sqrt(n)
        # 四分位数的标准误估计（IQR/1.349/sqrt(n) 近似，适用于大样本）
        iqr = q75 - q25
        q25_sem = iqr / 1.349 / np.sqrt(n)
        q75_sem = iqr / 1.349 / np.sqrt(n)
        q25_ci95 = (q25 - 1.96*q25_sem, q25 + 1.96*q25_sem)
        q75_ci95 = (q75 - 1.96*q75_sem, q75 + 1.96*q75_sem)
        median_sem = 1.253 * sem  # 正态近似下中位数标准误
        median_ci95 = (q50 - 1.96*median_sem, q50 + 1.96*median_sem)
        return {
            "q25": q25,
            "q25_sem": q25_sem,
            "q25_ci95": q25_ci95,
            "median": q50,
            "median_sem": median_sem,
            "median_ci95": median_ci95,
            "q75": q75,
            "q75_sem": q75_sem,
            "q75_ci95": q75_ci95,
            "mean": mean,
            "mean_sem": sem,
            "mean_ci95": (mean - 1.96*sem, mean + 1.96*sem)
        }
    pit_env = attach_env(pit)

    # 3. 马氏距离分析
    md2_ref_quantiles = np.percentile(md2_ref, [50, 90, 95, 99])
    md2_q50, md2_q90, md2_q95, md2_q99 = md2_ref_quantiles

    md2_within_median = np.sum(md2_q <= md2_q50) / len(md2_q)
    md2_within_90pct = np.sum(md2_q <= md2_q90) / len(md2_q)
    md2_within_95pct = np.sum(md2_q <= md2_q95) / len(md2_q)
    md2_outliers = np.sum(md2_q > md2_q99) / len(md2_q)

    # 4. 综合相似性评分（0-100分）
    from scipy import stats
    logp_score = 100 * (1 - stats.percentileofscore(logp_ref, np.median(logp_q), kind='weak') / 100)
    pit_score = 100 * (1 - pit_uniform_test / (len(pit)/10))  # 越接近均匀分布越好
    md2_score = 100 * md2_within_90pct  # 90%以内的比例

    overall_similarity = (logp_score * 0.4 + pit_score * 0.3 + md2_score * 0.3)

    # =====================================================
    # 打印详细统计报告
    # =====================================================
    print("\n" + "="*80)
    print("                    相似性诊断详细报告")
    print("="*80)

    print(f"\n📊 数据概览:")
    print(f"  参考正样本: {len(logp_ref):,} 个")
    print(f"  查询样本: {len(logp_q):,} 个")

    print(f"\n🎯 基于Log-Density的相似性分层:")
    for level, (count, pct) in similarity_levels.items():
        level_names = {
            'extremely_similar': '极度相似 (>95%分位)',
            'highly_similar': '高度相似 (75%-95%)',
            'moderately_similar': '中等相似 (25%-75%)',
            'poorly_similar': '低度相似 (5%-25%)',
            'outliers': '异常值 (<5%分位)'
        }
        emoji = "🔥" if level == 'extremely_similar' else "✨" if level == 'highly_similar' else "📈" if level == 'moderately_similar' else "⚠️" if level == 'poorly_similar' else "❌"
        print(f"  {emoji} {level_names[level]}: {count:,} 个 ({pct:.1f}%)")

    print(f"\n🎲 概率积分变换(PIT)分析:")
    print(f"  均匀性偏离度: {pit_uniform_test:.3f} (越小越好, <{len(pit)/20:.1f}为良好)")
    print(f"  左偏重比例: {pit_left_heavy:.1%} (高值表示查询样本偏离参考分布)")
    print(f"  右偏重比例: {pit_right_heavy:.1%}")
    print(f"  PIT分布上下四分位数: Q1={pit_env['q25']:.3f} ± {pit_env['q25_sem']:.3f} (95%CI: {pit_env['q25_ci95'][0]:.3f} ~ {pit_env['q25_ci95'][1]:.3f})")
    print(f"                      Q2/中位数={pit_env['median']:.3f} ± {pit_env['median_sem']:.3f} (95%CI: {pit_env['median_ci95'][0]:.3f} ~ {pit_env['median_ci95'][1]:.3f})")
    print(f"                      Q3={pit_env['q75']:.3f} ± {pit_env['q75_sem']:.3f} (95%CI: {pit_env['q75_ci95'][0]:.3f} ~ {pit_env['q75_ci95'][1]:.3f})")
    print(f"  PIT均值: {pit_env['mean']:.3f} ± {pit_env['mean_sem']:.3f} (95%CI: {pit_env['mean_ci95'][0]:.3f} ~ {pit_env['mean_ci95'][1]:.3f})")

    print(f"\n📏 马氏距离统计:")
    print(f"  中位数以内: {md2_within_median:.1%}")
    print(f"  90%分位以内: {md2_within_90pct:.1%}")
    print(f"  95%分位以内: {md2_within_95pct:.1%}")
    print(f"  异常值(>99%): {md2_outliers:.1%}")

    print(f"\n🏆 综合相似性评分:")
    print(f"  Log-density得分: {logp_score:.1f}/100")
    print(f"  PIT均匀性得分: {pit_score:.1f}/100") 
    print(f"  马氏距离得分: {md2_score:.1f}/100")
    print(f"  ⭐ 总体相似性: {overall_similarity:.1f}/100")

    # 解释评分意义
    if overall_similarity >= 80:
        interpretation = "🟢 优秀 - 查询样本与参考分布高度一致"
    elif overall_similarity >= 60:
        interpretation = "🟡 良好 - 查询样本与参考分布较为一致"
    elif overall_similarity >= 40:
        interpretation = "🟠 一般 - 查询样本与参考分布存在一定差异"
    else:
        interpretation = "🔴 较差 - 查询样本明显偏离参考分布"

    print(f"  📝 解释: {interpretation}")

    print(f"\n💡 实际应用意义:")
    if pct_below_5 >= 30:
        print(f"  ✅ {pct_below_5:.1f}%的撂荒地具有与已建光伏高度相似的环境特征")
    elif pct_below_5 >= 15:
        print(f"  ⚠️ {pct_below_5:.1f}%的撂荒地具有相似特征，建议进一步筛选")
    else:
        print(f"  ❌ 仅{pct_below_5:.1f}%的撂荒地相似，可能需要调整选址策略")

    extreme_count = similarity_levels['extremely_similar'][0]
    high_count = similarity_levels['highly_similar'][0]
    print(f"  🎯 建议优先开发: {extreme_count:,}个极度相似 + {high_count:,}个高度相似地点")

    print("="*80)

    # 返回详细的统计数据
    return {
        # 原有数据
        "logp_ref_mean": mu, "logp_ref_std": std,
        "logp_threshold_q": q_cut, "logp_threshold_value": thr,
        "z_query": z_q,
        "pit_histogram": np.histogram(pit, bins=20, range=(0,1))[0],
        "md2_ref_q95": thr_md2,

        # 新增详细统计
        "similarity_levels": similarity_levels,
        "logp_quantiles": ref_quantiles,
        "pit_uniformity_deviation": pit_uniform_test,
        "pit_left_heavy_ratio": pit_left_heavy,
        "pit_right_heavy_ratio": pit_right_heavy,
        "pit_env": pit_env,
        "md2_quantiles": md2_ref_quantiles,
        "md2_within_ratios": {
            "median": md2_within_median,
            "90pct": md2_within_90pct, 
            "95pct": md2_within_95pct,
            "outliers": md2_outliers
        },
        "scores": {
            "logp_score": logp_score,
            "pit_score": pit_score,
            "md2_score": md2_score,
            "overall_similarity": overall_similarity,
            "interpretation": interpretation
        }
    }

