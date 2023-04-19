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


def reshape_covariate_dfs(
    spatial_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    y_df: pd.DataFrame,
    y_column_name: str = 'y',
) -> pd.DataFrame:
    """
    Function used to transform covariates coming from two dataframes one for spatial and
    one for temporal into a single dataframe with the right shape for the BKTR Regressor.
    This is useful when the temporal covariates do not vary trough space and the spatial
    covariates do not vary trough time (Like in the BIXI example). The function also adds
    a column for the target variable at the beginning of the dataframe.

    Args:
        spatial_df (pd.DataFrame): Spatial covariates dataframe with an index named
          location and a shape of (n_locations, n_spatial_covariates)
        temporal_df (pd.DataFrame): Temporal covariates dataframe with an index named
            time and a shape of (n_times, n_temporal_covariates)
        y_df (pd.DataFrame): The dataframe containing the target variable. It should have
            a shape of (n_locations, n_times). The columns and index names of this dataframe
            should be correspond to the one of the spatial_df and temporal_df.

    Raises:
        ValueError: If the index names of spatial_df and y_df are not `location`
        ValueError: If the index name of temporal_df is not `time`
        ValueError: If the index values of spatial_df and y_df are not the same
        ValueError: If the temporal_df index values and y_df columns names are not the same

    Returns:
        pd.DataFrame: The reshaped covariates dataframe with a shape of
            (n_locations * n_times, 1 + n_spatial_covariates + n_temporal_covariates).
            The first column is the target variable and the other columns are the
            covariates. The index is a MultiIndex with the cartesian product of all
            location and time as levels.
    """
    spa_index_name = 'location'
    temp_index_name = 'time'
    if spatial_df.index.name != spa_index_name or y_df.index.name != spa_index_name:
        raise ValueError(f'Index names of spatial_df and y_df must be {spa_index_name}')
    if temporal_df.index.name != temp_index_name:
        raise ValueError(f'Index name of temporal_df must be {temp_index_name}')

    spatial_df_cp = spatial_df.copy()
    temporal_df_cp = temporal_df.copy()
    y_df_cp = y_df.copy()

    # Sort indexes and columns
    for df in [spatial_df_cp, temporal_df_cp, y_df_cp]:
        df.sort_index(inplace=True)
    y_df_cp.sort_index(axis=1, inplace=True)

    # Compare indexes values
    if spatial_df_cp.index.to_list() != y_df_cp.index.to_list():
        raise ValueError('Index values of spatial_df and y_df must be the same')
    if temporal_df_cp.index.to_list() != y_df_cp.columns.to_list():
        raise ValueError('temporal_df index values and y_df columns names must be the same')

    # Create cartesian product for covariates
    spatial_df_cp.reset_index(inplace=True)
    temporal_df_cp.reset_index(inplace=True)
    data_df = pd.merge(spatial_df_cp, temporal_df_cp, how='cross')
    data_df.set_index([spa_index_name, temp_index_name], inplace=True)
    data_df.sort_index(inplace=True)
    data_df.insert(0, y_column_name, y_df_cp.to_numpy().flatten())
    return data_df
