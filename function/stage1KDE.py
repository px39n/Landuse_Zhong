# # stage1_kde.py
# import numpy as np
# import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
# from function.data_utils import NUMERIC_FEATURES, CAT_COLS, ZERO_COLS, ABANDON_COLS, NONE_ABANDON_COLS
# from function import *


# def select_and_train_kde(
#     df_pos: pd.DataFrame,
#     bandwidths: list,
#     cv: int = 5,
#     n_jobs: int = 1,
#     verbose: int = 1
# ) -> Pipeline:
#     """
#     使用 Pipeline(ColumnTransformer→KDE) + GridSearchCV，
#     在每个 fold 内先 fit(StandardScaler、OneHotEncoder)，再 fit(KernelDensity)。
#     返回最终全量训练好的 Pipeline。
#     """
#     preproc = ColumnTransformer([
#         ('num', StandardScaler(), NUMERIC_FEATURES),
#         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
#     ])

#     pipe = Pipeline([
#         ('preproc', preproc),
#         ('kde',     KernelDensity(kernel='gaussian'))
#     ])

#     grid = GridSearchCV(
#         estimator=pipe,
#         param_grid={'kde__bandwidth': bandwidths},
#         cv=cv,
#         scoring='neg_log_loss',
#         n_jobs=n_jobs,
#         verbose=verbose
#     )
#     # df_pos 中必须只包含 NUMERIC_FEATURES + CAT_COLS 列
#     grid.fit(df_pos)

#     best_bw = grid.best_params_['kde__bandwidth']
#     print(f"[stage1_kde] Selected bandwidth = {best_bw}")
#     return grid.best_estimator_


# def score_env(
#     kde_pipeline: Pipeline,
#     df_query: pd.DataFrame
# ) -> np.ndarray:
#     """
#     用训练好的 KDE Pipeline 对 df_query 计算相似度分数并线性归一到 [0,1]。
#     """
#     # pipeline.score_samples 会自动调用 preproc.transform
#     log_dens = kde_pipeline.score_samples(df_query)
#     dens     = np.exp(log_dens)
#     return (dens - dens.min()) / (dens.max() - dens.min())
