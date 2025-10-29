# 文件: function/negative_sampling.py (完整版)

# -*- coding: utf-8 -*-
"""
统一负样本生成接口

策略类型：
1. 选择式 (Selection-based): 从现有样本池中筛选
2. 生成式 (Generation-based): 使用GMM生成新样本
3. 混合式 (Hybrid): 结合两种策略

Author: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Literal
from abc import ABC, abstractmethod

# 必要的导入（根据实际情况调整）
# 这些函数如果在新文件中，需要确保导入路径正确
try:
    from function.gmm_training import score_env, attach_env_calibration, visualize_sampling_results_pit
except ImportError:
    # 如果在notebook中，可能需要从全局命名空间导入
    pass


# ==========================================
# 策略基类
# ==========================================

class NegativeSamplingStrategy(ABC):
    """负样本策略抽象基类"""
    
    @abstractmethod
    def generate(
        self, 
        df_positive: pd.DataFrame,
        df_prediction_pool: Optional[pd.DataFrame] = None,
        features: List[str] = None,
        negative_ratio: float = 0.3,
        random_state: int = 42
    ) -> tuple:
        """
        生成负样本
        
        Returns:
        --------
        (df_negative_samples, df_remaining, df_combined)
        """
        pass


# ==========================================
# 策略实现（包装现有函数）
# ==========================================

class SelectionBasedStrategy(NegativeSamplingStrategy):
    """选择式策略：包装现有的generate_negative_samples_from_abandon"""
    
    def __init__(
        self,
        gmm_pipeline,
        sampling_strategy: Literal["simple", "mixed", "hard"] = "mixed",
        difficulty_levels: int = 3
    ):
        self.gmm_pipeline = gmm_pipeline
        self.sampling_strategy = sampling_strategy
        self.difficulty_levels = difficulty_levels
    
    def generate(
        self,
        df_positive,
        df_prediction_pool,
        features,
        negative_ratio=0.3,
        random_state=42
    ):
        """调用现有的generate_negative_samples_from_abandon"""
        print(f"\n{'='*60}")
        print(f"选择式负样本策略")
        print(f"  策略: {self.sampling_strategy}")
        print(f"  难度层级: {self.difficulty_levels}")
        print(f"{'='*60}")
        
        # 直接调用现有函数
        return generate_negative_samples_from_abandon(
            df_positive, 
            df_prediction_pool, 
            features, 
            self.gmm_pipeline,
            negative_ratio=negative_ratio,
            random_state=random_state,
            sampling_strategy=self.sampling_strategy,
            difficulty_levels=self.difficulty_levels
        )


class GenerationBasedStrategy(NegativeSamplingStrategy):
    """生成式策略：纯粹的GMM生成式采样"""
    
    def __init__(self, gmm_pipeline, augmentation_ratio=0.3):
        self.gmm_pipeline = gmm_pipeline
        self.augmentation_ratio = augmentation_ratio
    

    def generate(self, df_positive, df_prediction_pool=None, features=None,
                negative_ratio=0.3, random_state=42):
        """纯粹生成式：从GMM采样生成负样本"""
        print(f"\n{'='*60}")
        print(f"生成式负样本策略（纯粹GMM采样）")
        print(f"{'='*60}")
        
        np.random.seed(random_state)
        
        gmm = self.gmm_pipeline.named_steps['gmm']
        preprocessor = self.gmm_pipeline.named_steps['preprocessor']
        
        # 计算需要生成的负样本数
        n_pos = len(df_positive)
        n_neg = int(n_pos * negative_ratio)
        
        print(f"\n生成配置:")
        print(f"  正样本数: {n_pos}")
        print(f"  目标负样本数: {n_neg}")
        
        # ✅ 动态采样策略：逐步增加采样量，保持阈值严格
        X_pos_processed = preprocessor.transform(df_positive[features])
        ref_logp = np.mean(gmm.score_samples(X_pos_processed))
        ref_std = np.std(gmm.score_samples(X_pos_processed))
        threshold = ref_logp - ref_std  # 保持严格阈值
        
        print(f"\n筛选标准:")
        print(f"  正样本平均log概率: {ref_logp:.3f}")
        print(f"  阈值 (均值 - 1std): {threshold:.3f}")
        
        max_attempts = 8  # 最多尝试8次
        sampling_multiplier = 2  # 初始采样倍数
        candidate_samples = None
        total_sampled = 0
        
        for attempt in range(max_attempts):
            sample_count = n_neg * sampling_multiplier
            print(f"\n尝试 {attempt + 1}/{max_attempts}: 采样 {sample_count} 个...")
            
            # 从GMM采样
            generated_samples_highdim, _ = gmm.sample(sample_count)
            total_sampled += sample_count
            
            # 筛选低概率样本
            generated_logps = gmm.score_samples(generated_samples_highdim)
            low_density_mask = generated_logps < threshold
            n_valid = low_density_mask.sum()
            
            print(f"  实际采样: {sample_count} 个")
            print(f"  符合阈值: {n_valid} 个 ({n_valid/sample_count:.1%})")
            
            if candidate_samples is None:
                candidate_samples = generated_samples_highdim[low_density_mask].copy()
            else:
                # 合并新采样的有效样本
                new_candidates = generated_samples_highdim[low_density_mask]
                candidate_samples = np.vstack([candidate_samples, new_candidates])
            
            n_available = len(candidate_samples)
            print(f"  累计有效样本: {n_available} 个")
            
            if n_available >= n_neg:
                # 有足够的样本，截取目标数量
                candidate_samples = candidate_samples[:n_neg]
                print(f"  ✅ 已获得足够样本 ({len(candidate_samples)} 个)")
                break
            elif attempt < max_attempts - 1:
                # 继续下一次尝试，增加采样倍数
                sampling_multiplier = int(sampling_multiplier * 1.5)  # 增加50%
                print(f"  ⏳ 继续增加采样量 (下次: {n_neg * sampling_multiplier} 个)...")
            else:
                print(f"  ⚠️ 已达到最大尝试次数，使用所有可用样本 ({n_available} 个)")
        
        # 统计信息
        print(f"\n采样统计:")
        print(f"  总采样次数: {total_sampled} 个")
        print(f"  有效样本率: {len(candidate_samples)/total_sampled:.2%}")
        print(f"  最终负样本数: {len(candidate_samples)} 个")
        
        # ✅ 逆变换：最近邻映射（优化版 - 批量处理）
        from scipy.spatial import cKDTree
        tree = cKDTree(X_pos_processed)
        
        print(f"\n逆变换映射到原始特征空间（批量处理）...")
        
        # 批量查找最近邻
        distances, nearest_indices = tree.query(candidate_samples, k=1)
        
        if nearest_indices.ndim > 1:
            nearest_indices = nearest_indices.ravel()

        # 批量获取基础特征（添加 int() 转换确保索引是整数）
        base_features_list = [df_positive[features].iloc[int(idx)].copy() for idx in nearest_indices]
        df_base = pd.DataFrame(base_features_list).reset_index(drop=True)
        
        # 向量化添加微扰
        numeric_cols = df_positive[features].select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df_base.columns:
                std_val = df_positive[col].std()
                perturbations = np.random.normal(0, 0.15 * std_val, size=len(df_base))
                df_base[col] += perturbations
        
        # 批量计算log-density
        df_base_features = df_base[features].copy()
        X_generated_processed = preprocessor.transform(df_base_features)
        generated_logps = gmm.score_samples(X_generated_processed)

        # 添加元数据
        df_generated = df_base.copy()
        df_generated['gmm_logp'] = generated_logps
        df_generated['sample_type'] = 'generated'
        df_generated['label'] = 0

        print(f"逆变换完成: {len(df_generated)} 个负样本")
        print(f"平均log-density: {df_generated['gmm_logp'].mean():.3f}")

        # 训练数据
        df_combined = pd.concat([
            df_positive[features].assign(label=1, sample_type='positive'),
            df_generated[features + ['gmm_logp', 'sample_type', 'label']]
        ], ignore_index=True)

        print(f"生成完成: {len(df_generated)} 个负样本")
        print(f"平均log-density: {df_generated['gmm_logp'].mean():.3f}")

        # 生成式策略返回处理
        if df_prediction_pool is not None and len(df_prediction_pool) > 0:
            print(f"保留整个预测池用于预测: {len(df_prediction_pool)} 个样本")
            df_remaining = df_prediction_pool.copy()

            # 可选：添加GMM评分到remaining（与选择式策略保持一致的数据格式）
            X_remaining = preprocessor.transform(df_remaining[features])
            remaining_logp = gmm.score_samples(X_remaining)
            df_remaining['gmm_logp'] = remaining_logp
            df_remaining['gmm_score'] = None  # 后续可以计算sigmoid分数

        else:
            print("⚠️ 未提供预测池，返回空DataFrame")
            df_remaining = pd.DataFrame()

        return df_generated, df_remaining, df_combined

class HybridStrategy(NegativeSamplingStrategy):
    """混合策略：选择式 + 生成式"""
    
    def __init__(
        self,
        selection_strategy: SelectionBasedStrategy,
        generation_strategy: GenerationBasedStrategy,
        selection_weight: float = 0.7
    ):
        self.selection_strategy = selection_strategy
        self.generation_strategy = generation_strategy
        self.selection_weight = selection_weight
    
    def generate(
        self,
        df_positive,
        df_prediction_pool,
        features,
        negative_ratio=0.3,
        random_state=42
    ):
        """混合策略生成"""
        print(f"\n{'='*60}")
        print(f"混合负样本策略")
        print(f"  选择式权重: {self.selection_weight:.1%}")
        print(f"  生成式权重: {1-self.selection_weight:.1%}")
        print(f"{'='*60}")
        
        # 分配比例
        n_pos = len(df_positive)
        n_total_neg = int(n_pos * negative_ratio)
        n_selection = int(n_total_neg * self.selection_weight)
        n_generation = n_total_neg - n_selection
        
        # 1. 选择式
        if n_selection > 0 and df_prediction_pool is not None and len(df_prediction_pool) > 0:
            df_neg_sel, df_remaining, _ = self.selection_strategy.generate(
                df_positive, df_prediction_pool, features,
                negative_ratio=n_selection / n_pos,
                random_state=random_state
            )
        else:
            df_neg_sel = pd.DataFrame()
            df_remaining = df_prediction_pool if df_prediction_pool is not None else pd.DataFrame()
        
        # 2. 生成式（基于选择式的负样本）
        if n_generation > 0 and len(df_neg_sel) > 0:
            # 使用augment_negative_samples_with_generation增强
            df_neg_gen = augment_negative_samples_with_generation(
                df_neg_sel.sample(min(len(df_neg_sel), n_generation // 2)),
                df_positive,
                features,
                self.selection_strategy.gmm_pipeline,
                augmentation_ratio=1.0,
                random_state=random_state + 1
            )
            df_neg_gen['sample_type'] = 'generation'
        else:
            df_neg_gen = pd.DataFrame()
        
        # 合并所有负样本
        if len(df_neg_sel) > 0 and len(df_neg_gen) > 0:
            all_neg = pd.concat([df_neg_sel, df_neg_gen], ignore_index=True)
        elif len(df_neg_sel) > 0:
            all_neg = df_neg_sel
        elif len(df_neg_gen) > 0:
            all_neg = df_neg_gen
        else:
            raise ValueError("No negative samples generated")
        
        # 确保不超过目标数量
        if len(all_neg) > n_total_neg:
            all_neg = all_neg.sample(n=n_total_neg, random_state=random_state)
        
        # 标记来源
        n_from_selection = min(len(df_neg_sel), n_selection)
        n_from_generation = len(all_neg) - n_from_selection
        
        print(f"\n混合结果:")
        print(f"  选择式: {n_from_selection} 个")
        print(f"  生成式: {n_from_generation} 个")
        print(f"  总计: {len(all_neg)} 个")
        
        # 训练数据
        df_combined = pd.concat([
            df_positive[features].assign(label=1, sample_type='positive'),
            all_neg
        ], ignore_index=True)
        
        return all_neg, df_remaining, df_combined


# ==========================================
# 统一接口
# ==========================================

def generate_negative_samples_unified(
    strategy_type: Literal["selection", "generation", "hybrid"],
    df_positive: pd.DataFrame,
    df_prediction_pool: Optional[pd.DataFrame] = None,
    features: List[str] = None,
    gmm_pipeline=None,
    **kwargs
) -> tuple:
    """
    统一的负样本生成接口
    
    Parameters:
    -----------
    strategy_type: str, 策略类型
        - "selection": 选择式（需要df_prediction_pool）
        - "generation": 生成式（不需要df_prediction_pool）
        - "hybrid": 混合式
    df_positive: DataFrame, 正样本
    df_prediction_pool: DataFrame, 预测样本池（选择式和混合式需要）
    features: List, 特征列表
    gmm_pipeline: Pipeline, GMM模型
    **kwargs: 策略特定参数
        - sampling_strategy: "simple"/"mixed"/"hard"
        - difficulty_levels: int
        - augmentation_ratio: float
        - selection_weight: float
        - negative_ratio: float
        - random_state: int
    
    Returns:
    --------
    (df_negative_samples, df_remaining, df_combined_training)
    """
    
    print(f"\n{'='*80}")
    print(f"统一负样本生成接口 - 策略: {strategy_type}")
    print(f"{'='*80}")
    
    # 创建策略对象
    if strategy_type == "selection":
        strategy = SelectionBasedStrategy(
            gmm_pipeline=gmm_pipeline,
            sampling_strategy=kwargs.get('sampling_strategy', 'mixed'),
            difficulty_levels=kwargs.get('difficulty_levels', 3)
        )
    
    elif strategy_type == "generation":
        strategy = GenerationBasedStrategy(
            gmm_pipeline=gmm_pipeline,
            augmentation_ratio=kwargs.get('augmentation_ratio', 0.3)
        )
    
    elif strategy_type == "hybrid":
        selection_strategy = SelectionBasedStrategy(
            gmm_pipeline=gmm_pipeline,
            sampling_strategy=kwargs.get('sampling_strategy', 'mixed'),
            difficulty_levels=kwargs.get('difficulty_levels', 3)
        )
        generation_strategy = GenerationBasedStrategy(
            gmm_pipeline=gmm_pipeline,
            augmentation_ratio=kwargs.get('augmentation_ratio', 0.3)
        )
        strategy = HybridStrategy(
            selection_strategy,
            generation_strategy,
            selection_weight=kwargs.get('selection_weight', 0.7)
        )
    
    else:
        raise ValueError(f"未知策略类型: {strategy_type}")
    
    # 执行生成
    return strategy.generate(
        df_positive=df_positive,
        df_prediction_pool=df_prediction_pool,
        features=features,
        negative_ratio=kwargs.get('negative_ratio', 0.3),
        random_state=kwargs.get('random_state', 42)
    )




# 选择式负样本策略
def generate_negative_samples_from_abandon(df_positive, df_prediction_pool, features_no_coords, 
                                           gmm_pipeline, negative_ratio=0.1, random_state=42,
                                           sampling_strategy="mixed", difficulty_levels=3):
    """
    基于已训练GMM + 参考分位校准的负样本采样（PIT 分层）
    
    改进点：
    - 使用已训练的GMM，不重新训练
    - 基于参考集（正样本）计算log-density分位阈值
    - 使用PIT (percentile-in-training) 进行分层采样
    - 采样仅基于log-density/PIT，sigmoid分数仅用于可视化
    - 训练数据不包含任何GMM打分列，避免泄露
    
    Parameters:
    -----------
    df_positive : 正样本数据（作为参考集）
    df_prediction_pool : 预测样本池（从中抽取负样本）
    features_no_coords : 特征列表
    gmm_pipeline : 已训练好的GMM管道
    negative_ratio : 负样本比例
    random_state : 随机种子
    sampling_strategy : str, 采样策略 {"simple", "mixed", "hard"}
    difficulty_levels : int, 难度分层数（默认3层）
    
    Returns:
    --------
    df_negative_samples : 被选中的负样本
    df_remaining_prediction : 剩余的预测样本
    df_combined_training : 组合的训练数据（正样本+负样本）
    """
    print("=" * 60)
    print("基于已训练GMM + 参考分位校准的负样本采样（PIT 分层）")
    print("=" * 60)
    
    try:
        np.random.seed(random_state)
        rng = np.random.RandomState(random_state)
        
        # 0) Validate GMM pipeline
        if (gmm_pipeline is None or 
            not hasattr(gmm_pipeline, 'named_steps') or
            'preprocessor' not in gmm_pipeline.named_steps or 
            'gmm' not in gmm_pipeline.named_steps):
            raise ValueError("Invalid gmm_pipeline: must contain both 'preprocessor' and 'gmm' steps")

        pre = gmm_pipeline.named_steps['preprocessor']
        gmm = gmm_pipeline.named_steps['gmm']
        
        print(f"Using trained GMM model: {gmm.n_components} components, {gmm.covariance_type} covariance")
        
        # 1) Compute log-density for reference (positive) and prediction pool
        print("Computing log-density for reference set and prediction pool...")
        X_ref = pre.transform(df_positive[features_no_coords])
        logp_ref = gmm.score_samples(X_ref)
        
        X_pool = pre.transform(df_prediction_pool[features_no_coords])
        logp_pool = gmm.score_samples(X_pool)
        
        print(f"Reference set log-density: mean={np.mean(logp_ref):.3f}, std={np.std(logp_ref):.3f}")
        print(f"Prediction pool log-density: mean={np.mean(logp_pool):.3f}, std={np.std(logp_pool):.3f}")
        
        # 2) Compute PIT (Percentile-In-Training)
        # PIT(x) = P_ref(logp <= logp(x)), using sorted reference logp as empirical distribution
        print("Building reference empirical distribution and computing PIT...")
        ref_sorted = np.sort(logp_ref)
        
        def compute_pit(logp_values):
            """Compute PIT for given log-density values"""
            indices = np.searchsorted(ref_sorted, logp_values, side='right')
            return indices.astype(float) / len(ref_sorted)
        
        pit_pool = compute_pit(logp_pool)
        
        # 3) (Optional) Attach robust calibration parameters to pipeline
        try:
            gmm_pipeline = attach_env_calibration(
                gmm_pipeline, df_positive[features_no_coords], robust=True
            )
            print("✅ Robust calibration parameters attached to pipeline")
        except Exception as e:
            print(f"Calibration attachment failed (does not affect sampling): {e}")
        
        # 4) Compute sigmoid scores and density (for visualization/audit only, not for sampling)
        print("Computing sigmoid scores (for visualization only)...")
        dens_pool = np.exp(logp_pool)
        
        # Use unified scoring function to compute sigmoid scores
        _, env_scores_pool, stats_pool = score_env(
            gmm_pipeline,
            df_prediction_pool[features_no_coords],
            method='sigmoid', 
            sigmoid_alpha=0.2,
            return_logdens=False
        )
        
        # 5) Add all scoring info to prediction pool
        df_pool_scored = df_prediction_pool.copy()
        df_pool_scored['original_id'] = np.arange(len(df_pool_scored))
        df_pool_scored['gmm_logp'] = logp_pool
        df_pool_scored['gmm_pit'] = pit_pool
        df_pool_scored['gmm_density'] = dens_pool
        df_pool_scored['gmm_score'] = env_scores_pool  # for visualization only
        
        # 6) Diagnostic statistics
        ref_quantiles = np.percentile(logp_ref, [5, 20, 40, 60, 80, 95])
        pit_quantiles = np.percentile(pit_pool, [5, 10, 25, 50, 75, 90])
        
        print(f"Reference log-density quantiles: Q5={ref_quantiles[0]:.3f}, Q20={ref_quantiles[1]:.3f}, Q40={ref_quantiles[2]:.3f}")
        print(f"Prediction pool PIT quantiles: Q5={pit_quantiles[0]:.3f}, Q10={pit_quantiles[1]:.3f}, Q25={pit_quantiles[2]:.3f}, Q50={pit_quantiles[3]:.3f}, Q75={pit_quantiles[4]:.3f}")

        # 7) Define PIT bin boundaries and strategy weights
        # 以5%和25%分位数为主要负样本采样源头，5%最重要，25%次之，保留diffculty的动态判别接口
        if difficulty_levels == 3:
            # [0,5%)=最容易, [5%,25%)=次容易, [25%,60%)=难
            pit_bins = [0.0, 0.05, 0.25, 0.60]
        elif difficulty_levels == 2:
            # [0,5%)=最容易, [5%,30%)=难
            pit_bins = [0.0, 0.05, 0.30]
        else:
            # 动态分层，前5%和25%分位数为主，剩下均分
            # 先用5%和25%分位数，再均分剩余区间
            pit_bins = [0.0, 0.05, 0.25]
            if difficulty_levels > 2:
                # 剩余区间 [0.25, 0.6] 均分
                extra_bins = list(np.linspace(0.25, 0.6, difficulty_levels - 2 + 1))[1:]  
                pit_bins += extra_bins
        pit_bins = np.asarray(pit_bins, dtype=float)

        # Strategy weights definition
        # 主要权重分配给5%和25%分位数
        if difficulty_levels == 3:
            strategy_weights = {
                "simple": [1.0, 0.0, 0.0],      # 5%最多，25%次之
                "mixed":  [0.6, 0.3, 0.1],
                "hard":   [0.5, 0.4, 0.1]
            }
        elif difficulty_levels == 2:
            strategy_weights = {
                "simple": [0.8, 0.2],
                "mixed":  [0.6, 0.4],
                "hard":   [0.4, 0.6]
            }
        else:
            # 动态分配，前两个bin权重高，剩下均分
            def dynamic_weights(levels):
                w = np.ones(levels)
                w[0] = 0.5  # 5%分位数
                w[1] = 0.3  # 25%分位数
                if levels > 2:
                    w[2:] = 0.2 / (levels - 2)
                return w / w.sum()
            strategy_weights = {
                "simple": dynamic_weights(difficulty_levels),
                "mixed":  dynamic_weights(difficulty_levels),
                "hard":   dynamic_weights(difficulty_levels)
            }

        if sampling_strategy not in strategy_weights:
            print(f"⚠️ Unknown sampling strategy: {sampling_strategy}, using default: mixed")
            sampling_strategy = "mixed"

        base_weights = strategy_weights[sampling_strategy]
        layer_weights = np.asarray(base_weights[:difficulty_levels], dtype=float)
        layer_weights = layer_weights / layer_weights.sum()  # normalize
        
        # 8) Calculate required number of negative samples
        n_pos = len(df_positive)
        n_neg_total = int(max(1, round(n_pos * negative_ratio)))
        
        print(f"\nSampling configuration:")
        print(f"  Number of positive samples: {n_pos}")
        print(f"  Target negative samples: {n_neg_total} (ratio={negative_ratio:.2f})")
        print(f"  PIT bin boundaries: {pit_bins}")
        print(f"  Layer weights: {np.round(layer_weights, 3)}")
        print(f"  Sampling strategy: {sampling_strategy}")
        
        # 9) Perform PIT-based stratified sampling
        print("\nPerforming PIT stratified sampling...")
        layer_samples = []
        layer_info = []
        layer_names = ["Easy", "Medium", "Hard", "Very Hard", "Extremely Hard"]
        
        for i in range(difficulty_levels):
            lo, hi = pit_bins[i], pit_bins[i+1]
            layer_name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
            
            # Define PIT range for current layer
            if i == difficulty_levels - 1:  # last layer uses closed interval
                mask = (df_pool_scored['gmm_pit'] >= lo) & (df_pool_scored['gmm_pit'] <= hi)
            else:
                mask = (df_pool_scored['gmm_pit'] >= lo) & (df_pool_scored['gmm_pit'] < hi)
            
            candidates = df_pool_scored[mask]
            need = int(round(n_neg_total * layer_weights[i]))
            
            print(f"  {layer_name} layer: PIT range [{lo:.3f},{hi:.3f}], candidates {len(candidates):,}, needed {need:,}")
            
            if len(candidates) == 0:
                print(f"    ⚠️ No candidates in {layer_name} layer, skipping")
                continue
            
            # Random sampling within this layer
            if need >= len(candidates):
                print(f"    ⚠️ Requested more than available in {layer_name} layer, using all")
                sampled = candidates
            else:
                sampled = candidates.sample(n=need, random_state=random_state + i)
            
            layer_samples.append(sampled)
            layer_info.append({
                'layer_name': layer_name,
                'pit_range': (float(lo), float(hi)),
                'n_candidates': int(len(candidates)),
                'n_sampled': int(len(sampled)),
                'weight': layer_weights[i],
                'mean_logp': float(sampled['gmm_logp'].mean()),
                'std_logp': float(sampled['gmm_logp'].std()),
                'mean_pit': float(sampled['gmm_pit'].mean()),
                'mean_score': float(sampled['gmm_score'].mean())  # for display only
            })
        
        # 10) Concatenate all sampled layers
        if not layer_samples:
            raise ValueError("No negative samples could be drawn from any layer. Please check bin boundaries or pool size.")
        
        df_negative_samples = pd.concat(layer_samples, ignore_index=True)
        
        # 11) Build remaining prediction pool
        picked_ids = set(df_negative_samples['original_id'])
        remaining_mask = ~df_pool_scored['original_id'].isin(picked_ids)
        df_remaining_prediction = df_pool_scored[remaining_mask].copy()
        
        # Data integrity check
        total_expected = len(df_prediction_pool)
        total_actual = len(df_negative_samples) + len(df_remaining_prediction)
        if total_actual != total_expected:
            print(f"❌ Data inconsistency: expected {total_expected}, got {total_actual}")
            raise ValueError("Data integrity check failed")
        
        print(f"\nSampling results:")
        print(f"  Total negative samples drawn: {len(df_negative_samples):,}")
        print(f"  Remaining prediction samples: {len(df_remaining_prediction):,}")
        print(f"  ✅ Data integrity check passed")
        
        # 12) Build training data (without any GMM scoring columns to avoid leakage)
        print("\nBuilding training dataset (no GMM scoring columns, to avoid leakage)...")
        
        df_pos_train = df_positive[features_no_coords].copy()
        df_pos_train['label'] = 1
        df_pos_train['sample_type'] = 'positive'
        
        df_neg_train = df_negative_samples[features_no_coords].copy()
        df_neg_train['label'] = 0
        df_neg_train['sample_type'] = 'negative_sample'
        
        df_combined_training = pd.concat([df_pos_train, df_neg_train], ignore_index=True)
        
        print(f"Training set built:")
        print(f"  Positives: {len(df_pos_train):,} | Negatives: {len(df_neg_train):,}")
        print(f"  Total: {len(df_combined_training):,} | Positive ratio: {df_combined_training['label'].mean():.3f}")
        
        # 13) Detailed stratified statistics
        print(f"\n📊 PIT stratified sampling statistics:")
        print("-" * 60)
        for info in layer_info:
            print(f"{info['layer_name']} layer:")
            print(f"  PIT range: [{info['pit_range'][0]:.3f}, {info['pit_range'][1]:.3f}]")
            print(f"  Candidates/Sampled: {info['n_candidates']:,} / {info['n_sampled']:,}")
            print(f"  Mean log-density: {info['mean_logp']:.3f} ± {info['std_logp']:.3f}")
            print(f"  Mean PIT: {info['mean_pit']:.3f}")
            print(f"  Mean sigmoid score: {info['mean_score']:.3f} (for reference only)")
        
        # 14) Quick quality validation
        print(f"\nValidating sampling quality...")
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline as SkPipeline
            
            # Prepare validation data (use training data, no GMM scores)
            X_pos = df_pos_train[features_no_coords]
            X_neg = df_neg_train[features_no_coords]
            X_combined = pd.concat([X_pos, X_neg], ignore_index=True)
            y_combined = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
            
            # Simple validation pipeline
            val_pipeline = SkPipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=random_state, max_iter=1000))
            ])
            
            cv_scores = cross_val_score(val_pipeline, X_combined, y_combined, 
                                      cv=3, scoring='f1', n_jobs=-1)
            
            quality_score = np.mean(cv_scores)
            print(f"  Sampling quality (3-fold CV F1): {quality_score:.3f} ± {np.std(cv_scores):.3f}")
            
            if quality_score > 0.8:
                print("  ✅ Good sampling quality")
            elif quality_score > 0.6:
                print("  ⚠️ Moderate sampling quality, consider adjusting strategy")
            else:
                print("  ❌ Poor sampling quality, consider resampling")
                
        except Exception as e:
            print(f"  ⚠️ Quality validation failed: {e}")
        
        # 15) Visualize sampling results
        try:
            visualize_sampling_results_pit(
                logp_pool, pit_pool, env_scores_pool, 
                df_negative_samples, pit_bins, layer_info
            )
        except Exception as e:
            print(f"ℹ️ Sampling visualization failed (ignored): {e}")
        
        return df_negative_samples, df_remaining_prediction, df_combined_training
        
    except Exception as e:
        print(f"❌ Error in generate_negative_samples_from_abandon: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None



# 生成式负样本
def augment_negative_samples_with_generation(
    df_negative_samples, df_positive, features_no_coords, gmm_pipeline,
    augmentation_ratio=0.3, random_state=42):
    """
    使用GMM生成式采样增强负样本
    
    核心逻辑：
    1. GMM已经学到了正样本的分布
    2. 从低概率区域随机采样生成新样本
    3. 使用最近邻+微扰将高维样本映射回原始特征空间
    
    Parameters:
    -----------
    df_negative_samples : pd.DataFrame, 已有负样本（从选择策略获得）
    df_positive : pd.DataFrame, 正样本（用作参考）
    features_no_coords : List[str], 特征列表
    gmm_pipeline : Pipeline, 已训练的GMM
    augmentation_ratio : float, 增广比例
    random_state : int, 随机种子
    """
    print("\n" + "=" * 60)
    print("GMM生成式负样本增广")
    print("=" * 60)
    
    np.random.seed(random_state)
    
    gmm = gmm_pipeline.named_steps['gmm']
    preprocessor = gmm_pipeline.named_steps['preprocessor']
    
    # 1) 计算参考log-density（正样本的密度）
    X_pos_processed = preprocessor.transform(df_positive[features_no_coords])
    ref_logp = np.mean(gmm.score_samples(X_pos_processed))
    ref_std = np.std(gmm.score_samples(X_pos_processed))
    
    # 2) 从GMM采样（生成高维样本）
    n_generate = int(len(df_negative_samples) * augmentation_ratio)
    generated_samples_highdim, _ = gmm.sample(n_generate * 3)  # 多生成一些用于筛选
    
    # 3) 筛选低概率样本（远离正样本分布）
    generated_logps = gmm.score_samples(generated_samples_highdim)
    threshold = ref_logp - ref_std  # 低于平均密度1个标准差
    
    low_density_mask = generated_logps < threshold
    candidate_samples = generated_samples_highdim[low_density_mask][:n_generate]
    
    print(f"\n生成过程:")
    print(f"  参考log-density: {ref_logp:.3f} ± {ref_std:.3f}")
    print(f"  筛选阈值: {threshold:.3f}")
    print(f"  生成候选: {len(candidate_samples)}")
    
    if len(candidate_samples) < n_generate:
        print(f"⚠️ 生成样本不足，使用所有 {len(candidate_samples)} 个")
        n_generate = len(candidate_samples)
        candidate_samples = generated_samples_highdim[low_density_mask][:n_generate]
    
    # 4) 逆变换：将高维样本映射回原始特征空间
    from scipy.spatial import cKDTree
    
    # 构建KD树
    tree = cKDTree(X_pos_processed)
    df_augmented = []
    
    for i, sample_highdim in enumerate(candidate_samples[:n_generate]):
        # 找最近邻
        _, nearest_idx = tree.query(sample_highdim.reshape(1, -1), k=1)
        
        # 使用最近邻特征，添加控制性扰动
        base_features = df_positive[features_no_coords].iloc[nearest_idx[0]].copy()
        
        # 添加定向扰动（向低密度区域偏移）
        perturbation_scale = 0.15
        for col in base_features.index:
            if col in df_positive[features_no_coords].select_dtypes(include=[np.number]).columns:
                std_val = df_positive[col].std()
                # 根据特征重要性调整扰动方向
                perturbation = np.random.normal(0, perturbation_scale * std_val)
                base_features[col] += perturbation
        
        # 验证生成样本的质量
        sample_df = pd.DataFrame([base_features])[features_no_coords]
        sample_processed = preprocessor.transform(sample_df)
        sample_logp = gmm.score_samples(sample_processed)[0]
        
        base_features['gmm_logp'] = sample_logp
        base_features['generated'] = True  # 标记为生成样本
        
        df_augmented.append(base_features)
    
    df_augmented = pd.DataFrame(df_augmented)
    
    # 5) 验证增广质量
    X_aug_processed = preprocessor.transform(df_augmented[features_no_coords])
    aug_logps = gmm.score_samples(X_aug_processed)
    
    print(f"\n增广样本质量:")
    print(f"  增广数量: {len(df_augmented)}")
    print(f"  平均log-density: {np.mean(aug_logps):.3f} (应低于参考: {ref_logp:.3f})")
    print(f"  密度比: {np.exp(np.mean(aug_logps) - ref_logp):.3f}")
    
    return df_augmented