from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import torch

from pyBKTR.distances import DIST_TYPE
from pyBKTR.tensor_ops import TSR

if TYPE_CHECKING:
    from pyBKTR.kernels import Kernel


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


def get_dim_labels(dim_prefix: str, max_value: int) -> str:
    """Utility function to get the dimension labels for a
        given dimension prefix and max value.
        TODO: Do not export

    Args:
        dim_prefix (str): The prefix of the dimension labels
        max_value (int): The maximum value of the dimension labels

    Returns:
        str: The dimension labels
    """
    max_digits = len(str(max_value))
    int_format = f'{{:0{max_digits}d}}'
    return [f'{dim_prefix}_{int_format.format(i)}' for i in range(max_value)]


def simulate_spatiotemporal_data(
    nb_spatial_locations: int,
    nb_time_points: int,
    nb_spatial_dimensions: int,
    spatial_scale: float,
    time_scale: float,
    spatial_covariates_means: list[float],
    temporal_covariates_means: list[float],
    spatial_kernel: Kernel,
    temporal_kernel: Kernel,
    noise_variance_scale: float,
    jitter_value: float | None = 1e-3,
) -> dict[str, pd.DataFrame]:
    """Simulate spatiotemporal data using kernel covariances.

    Args:
        nb_spatial_locations (int): Number of spatial locations
        nb_time_points (int): Number of time points
        nb_spatial_dimensions (int): Number of spatial dimensions
        spatial_scale (float): Spatial scale
        time_scale (float): Time scale
        spatial_covariates_means (list[float]): Spatial covariates means
        temporal_covariates_means (list[float]): Temporal covariates means
        spatial_kernel (Kernel): Spatial kernel
        temporal_kernel (Kernel): Temporal kernel
        noise_variance_scale (float): Noise variance scale
        jitter_value (float, optional): Jitter value added to the completed covariance
            matrix to help with numerical stability. Defaults to 1e-4.

    Returns:
        dict[str, pd.DataFrame]: A dictionnary containing 4 dataframes:
            - `data_df` contains the response variable and the covariates
            - `spatial_locations_df` contains the spatial locations and their coordinates
            - `temporal_points_df` contains the time points and their coordinates
            - `beta_df` contains the true beta coefficients
    """
    spa_pos = TSR.rand([nb_spatial_locations, nb_spatial_dimensions]) * spatial_scale
    temp_pos = TSR.tensor(
        [time_scale / (nb_time_points - 1) * x for x in range(0, nb_time_points)]
    )
    temp_pos = temp_pos.reshape([nb_time_points, 1])

    # Dimension labels
    s_dims = get_dim_labels('s_dim', nb_spatial_dimensions)
    s_locs = get_dim_labels('s_loc', nb_spatial_locations)
    t_points = get_dim_labels('t_point', nb_time_points)
    s_covs = get_dim_labels('s_cov', len(spatial_covariates_means))
    t_covs = get_dim_labels('t_cov', len(temporal_covariates_means))

    spa_pos_df = pd.DataFrame(spa_pos, columns=s_dims, index=s_locs)
    temp_pos_df = pd.DataFrame(temp_pos, columns=['time'], index=t_points)

    spa_means = TSR.tensor(spatial_covariates_means)
    nb_spa_covariates = len(spa_means)
    spa_covariates = TSR.randn([nb_spatial_locations, nb_spa_covariates])
    spa_covariates = spa_covariates + spa_means

    temp_means = TSR.tensor(temporal_covariates_means)
    nb_temp_covariates = len(temp_means)
    temp_covariates = TSR.randn([nb_time_points, nb_temp_covariates])
    temp_covariates = temp_covariates + temp_means
    inter_covariates = TSR.ones([nb_spatial_locations, nb_time_points, 1])
    covs = torch.concat(
        [
            inter_covariates,
            spa_covariates.unsqueeze(1).expand(
                nb_spatial_locations, nb_time_points, nb_spa_covariates
            ),
            temp_covariates.unsqueeze(0).expand(
                nb_spatial_locations, nb_time_points, nb_temp_covariates
            ),
        ],
        dim=2,
    )
    nb_covs = 1 + nb_spa_covariates + nb_temp_covariates

    covs_covariance = torch.distributions.Wishart(
        df=nb_covs,
        covariance_matrix=TSR.eye(nb_covs),
    ).sample()

    spatial_kernel.distance_type = DIST_TYPE.EUCLIDEAN
    spatial_kernel.set_positions(spa_pos_df)
    spatial_covariance = spatial_kernel.kernel_gen()
    temporal_kernel.distance_type = DIST_TYPE.EUCLIDEAN
    temporal_kernel.set_positions(temp_pos_df)
    temporal_covariance = temporal_kernel.kernel_gen()

    beta_covariance = TSR.kronecker_prod(
        TSR.kronecker_prod(spatial_covariance, temporal_covariance),
        covs_covariance,
    )
    # add a jitter to the covariance matrix to make it positive definite
    if jitter_value:
        beta_covariance += TSR.eye(beta_covariance.shape[0]) * jitter_value
    beta_means = TSR.zeros(nb_spatial_locations * nb_time_points * nb_covs)
    beta_vec: torch.Tensor = torch.distributions.MultivariateNormal(
        beta_means, beta_covariance
    ).sample()
    beta_values = beta_vec.reshape([nb_spatial_locations, nb_time_points, nb_covs])
    y_val = torch.einsum('ijk,ijk->ij', covs, beta_values)
    err = TSR.randn([nb_spatial_locations, nb_time_points]) * (noise_variance_scale**0.5)
    y_val += err
    y_val = y_val.reshape([nb_spatial_locations * nb_time_points, 1])
    # We remove the intercept from the covariates
    covs = covs.reshape([nb_spatial_locations * nb_time_points, nb_covs])[:, 1:]
    spa_temp_df_index = pd.MultiIndex.from_product(
        [spa_pos_df.index, temp_pos_df.index],
        names=['location', 'time'],
    )
    data_df = pd.DataFrame(
        torch.concat([y_val, covs], dim=1),
        columns=['y'] + s_covs + t_covs,
        index=spa_temp_df_index,
    )
    beta_df = pd.DataFrame(
        beta_values.reshape([nb_spatial_locations * nb_time_points, nb_covs]),
        columns=['Intercept'] + s_covs + t_covs,
        index=spa_temp_df_index,
    )
    return {
        'data_df': data_df,
        'spatial_locations_df': spa_pos_df,
        'temporal_points_df': temp_pos_df,
        'beta_df': beta_df,
    }
