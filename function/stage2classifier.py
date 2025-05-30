# stage2_classifier.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_stage2(
    df_stage2,
    cv: bool = False
):
    """
    训练一个随机森林分类器：
      - 特征: ['env_score','abandonment_year','abandonment_year_miss',
                 'abandonment_duration','abandonment_duration_miss']
      - 标签: 'label'
    如果 cv=True，可替换为 GridSearchCV / StratifiedKFold 内部验证。
    """

    # 这里相当于对于模型根据新的特征进行微调
    feats = [
        'env_score',
        'abandonment_year','abandonment_year_miss',
        'abandonment_duration','abandonment_duration_miss'
    ]
    X = df_stage2[feats].values
    y = df_stage2['label'].values

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X, y)
    return clf


def predict_stage2(
    clf,
    df_stage2,
    unlabeled_idx: np.ndarray
) -> np.ndarray:
    """
    对 df_stage2 中与 unlabeled_idx 对应的行（label==0 部分）进行预测，
    返回概率数组，长度等于 len(unlabeled_idx)。
    """
    feats = [
        'env_score',
        'abandonment_year','abandonment_year_miss',
        'abandonment_duration','abandonment_duration_miss'
    ]
    # 只取那些 unlabeled_idx 对应的行（注意：df_stage2 的行索引跟 unlabeled_idx 映射一致）
    Xq = df_stage2.loc[unlabeled_idx, feats].values
    probs = clf.predict_proba(Xq)[:, 1]
    return probs
