import math

import pandas as pd


class log(float):
    """Log wrapper for math.log to get better documentation"""

    def __new__(cls, log_val, *args, **kwargs):
        return super(log, cls).__new__(cls, math.log(log_val))

    def __repr__(cls):
        return f'Log({round(math.exp(cls), 10)})'


def reshape_covariate_dfs(spatial_df: pd.DataFrame, temporal_df: pd.DataFrame):
    """
    Funciton used to transform covariates coming from two dataframes one for spatial and
    one for temporal into a single dataframe with the right shape for the BKTR Regressor.
    This is useful when the temporal covariates do not vary trough space and the spatial
    covariates do not vary trough time (Like in the BIXI example).
    """
    from pyBKTR.bktr import BKTRRegressor

    spa_index_name = BKTRRegressor.spatial_index_name
    temp_index_name = BKTRRegressor.temporal_index_name
    if spatial_df.index.name != spa_index_name:
        raise ValueError(f'Index name of spatial_df must be {spa_index_name}')
    if temporal_df.index.name != temp_index_name:
        raise ValueError(f'Index name of temporal_df must be {temp_index_name}')

    spatial_df_cp = spatial_df.copy()
    temporal_df_cp = temporal_df.copy()
    spatial_df_cp.reset_index(inplace=True)
    temporal_df_cp.reset_index(inplace=True)
    covariates_df = pd.merge(spatial_df_cp, temporal_df_cp, how='cross')
    covariates_df.set_index([spa_index_name, temp_index_name], inplace=True)
    covariates_df.sort_index(inplace=True, ascending=False)
    return covariates_df
