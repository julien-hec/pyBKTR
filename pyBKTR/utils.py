import math
from typing import Any, Literal

import pandas as pd


class log(float):
    """Log wrapper for math.log to get better documentation"""

    def __new__(cls, log_val, *args, **kwargs):
        return super(log, cls).__new__(cls, math.log(log_val))

    def __repr__(cls):
        return f'Log({round(math.exp(cls), 10)})'


def get_label_index_or_raise(
    label: Any, label_list: list[Any], label_type: Literal['spatial', 'temporal', 'feature']
) -> int:
    """Get the index of a label in a list of labels. If the label is not in the list,
    raise an error.

    Args:
        label (Any): The label for which we want to get the index
        label_list (List[Any]): The list of labels
        label_type (Literal['spatial', 'temporal', 'feature']): The label type

    Returns:
        int: The index of the label in the list
    """
    try:
        return label_list.index(label)
    except ValueError:
        raise ValueError(f'Label `{label}` does not exist in {label_type} labels.')


def get_label_indexes(
    labels: list[Any],
    available_labels: list[Any],
    label_type: Literal['spatial', 'temporal', 'feature'],
) -> list[int]:
    """return the indexes of a given set of labels that can be found in a list of available labels

    Args:
        labels (list[Any]): The labels for which we want to get the indexes
        available_labels (list[Any]): The list of available labels
        label_type (Literal[&#39;spatial&#39;, &#39;temporal&#39;, &#39;feature&#39;]): Type
            of label for which we want to get indexes

    Raises:
        ValueError: Error if the list of labels is empty

    Returns:
        list[int]: The indexes of the labels in the list of available labels
    """
    if len(labels) == 0:
        raise ValueError(f'No {label_type} labels provided.')
    return [get_label_index_or_raise(lab, available_labels, label_type) for lab in labels]


def reshape_covariate_dfs(spatial_df: pd.DataFrame, temporal_df: pd.DataFrame):
    """
    Function used to transform covariates coming from two dataframes one for spatial and
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
