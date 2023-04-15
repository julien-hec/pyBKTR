from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

import pandas as pd
import torch

from pyBKTR.tensor_ops import TSR
from pyBKTR.utils import get_label_indexes


class ResultLogger:
    __slots__ = [
        'y',
        'omega',
        'covariates',
        'nb_burn_in_iter',
        'nb_sampling_iter',
        'logged_params_map',
        'beta_estimates',
        'y_estimates',
        'total_elapsed_time',
        'spatial_labels',
        'temporal_labels',
        'feature_labels',
        'spatial_decomp_per_iter',
        'temporal_decomp_per_iter',
        'covs_decomp_per_iter',
        'sum_beta_est',
        'sum_y_est',
        'beta_estimates_df',
        'y_estimates_df',
        'beta_covariates_summary_df',
        'last_time_stamp',
        'export_path',
        'export_suffix',
        'error_metrics',
        'total_sq_error',
    ]

    # Metrics used to create beta summaries
    moment_metrics = ['Mean', 'Var']
    quantile_metrics = [
        'Min',
        '2.5th Percentile',
        '1st Quartile',
        'Median',
        '3rd Quartile',
        '97.5th Percentile',
        'Max',
    ]
    quantile_values = [0, 0.025, 0.25, 0.5, 0.75, 0.975, 1]

    def __init__(
        self,
        y: torch.Tensor,
        omega: torch.Tensor,
        covariates: torch.Tensor,
        nb_burn_in_iter: int,
        nb_sampling_iter: int,
        rank_decomp: int,
        spatial_labels: list,
        temporal_labels: list,
        feature_labels: list,
        results_export_dir: str | None = None,
        results_export_suffix: str | None = None,
    ):
        # Create a tensor dictionary holding scalar data gathered through all iterations
        self.logged_params_map = defaultdict(list)

        # Set export dir
        if results_export_dir is None and results_export_suffix is not None:
            raise ValueError(
                'Cannot set a suffix for the export file if no export directory is provided.'
            )
        elif results_export_dir is None:
            self.export_path = None
        else:
            export_path = Path(results_export_dir)
            if not export_path.is_dir():
                raise ValueError(f'Path {export_path} does not exists.')
            self.export_path = export_path
        self.export_suffix = results_export_suffix

        # Create tensors that accumulate values needed for estimates
        self.spatial_labels = spatial_labels
        self.temporal_labels = temporal_labels
        self.feature_labels = feature_labels
        self.spatial_decomp_per_iter = TSR.zeros(
            (len(spatial_labels), rank_decomp, nb_sampling_iter)
        )
        self.temporal_decomp_per_iter = TSR.zeros(
            (len(temporal_labels), rank_decomp, nb_sampling_iter)
        )
        self.covs_decomp_per_iter = TSR.zeros(
            (len(self.feature_labels), rank_decomp, nb_sampling_iter)
        )
        self.sum_beta_est = TSR.zeros(covariates.shape)
        self.sum_y_est = TSR.zeros(y.shape)
        self.total_elapsed_time = 0

        self.y = y
        self.omega = omega
        self.covariates = covariates
        self.nb_burn_in_iter = nb_burn_in_iter
        self.nb_sampling_iter = nb_sampling_iter

        # Set initial timer value to calculate iterations' processing time
        self.last_time_stamp = time()

    def collect_iter_samples(self, iter: int, iter_logged_params: dict[str, float]):
        """
        Collect current iteration values inside the historical data tensor list.
        To be noted that errors have already been calculated before tau sampling.
        """
        elapsed_time_dict = self._get_elapsed_time_dict()

        if iter > self.nb_burn_in_iter:
            self.sum_beta_est += self.beta_estimates
            self.sum_y_est += self.y_estimates

        total_logged_params = {
            **{'iter': iter},
            **{'is_burn_in': 1 if iter <= self.nb_burn_in_iter else 0},
            **iter_logged_params,
            **self.error_metrics,
            **elapsed_time_dict,
        }

        for k, v in total_logged_params.items():
            self.logged_params_map[k].append(v)

        self._print_iter_result(iter, {**elapsed_time_dict, **self.error_metrics})

    def _get_elapsed_time_dict(self):
        iter_elapsed_time = time() - self.last_time_stamp
        self.total_elapsed_time += iter_elapsed_time
        self.last_time_stamp = time()
        return {'Elapsed Time': iter_elapsed_time}

    def _set_error_metrics(self) -> dict[str, float]:
        """Calculate error metrics of interest (MAE, RMSE, Total Error)

        Returns:
            dict[str, float]: A dictionary containing the values of the error metrics
        """
        nb_observ = self.omega.sum()
        err_matrix = (self.y_estimates - self.y) * self.omega
        total_sq_error = err_matrix.norm() ** 2
        mae = err_matrix.abs().sum() / nb_observ
        rmse = (total_sq_error / nb_observ).sqrt()
        self.total_sq_error = float(total_sq_error)
        self.error_metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
        }
        return self.error_metrics

    def set_y_and_beta_estimates(self, decomp_tensors_map: dict[str, torch.Tensor], iter: int):
        """Calculate the estimated y and beta tensors

        Args:
            decomposition_tensors_map (dict[str, torch.Tensor]): A dictionnary that
                contains all the spatial, temporal and covariates decomposition
        """
        if iter > self.nb_burn_in_iter:
            iter_indx = iter - self.nb_burn_in_iter - 1
            self.spatial_decomp_per_iter[:, :, iter_indx] = decomp_tensors_map['spatial_decomp']
            self.temporal_decomp_per_iter[:, :, iter_indx] = decomp_tensors_map['temporal_decomp']
            self.covs_decomp_per_iter[:, :, iter_indx] = decomp_tensors_map['covs_decomp']

        self.beta_estimates = torch.einsum(
            'im,jm,km->ijk',
            [
                decomp_tensors_map['spatial_decomp'],
                decomp_tensors_map['temporal_decomp'],
                decomp_tensors_map['covs_decomp'],
            ],
        )
        self.y_estimates = torch.einsum('ijk,ijk->ij', self.covariates, self.beta_estimates)

    def _print_iter_result(self, iter: int, result_dict: dict[str, float]):
        formated_results = [f'{k.replace("_", " ")} is {v:7.4f}' for k, v in result_dict.items()]
        print(f'** Result for iter {iter:<5} : {" || ".join(formated_results)} **')

    def _get_file_name(self, export_type_name: str):
        time_now = datetime.now()
        name_components = [export_type_name, time_now.strftime('%Y%m%d_%H%M')]
        if self.export_suffix:
            name_components.append(self.export_suffix)
        file_name = '_'.join(name_components)
        return self.export_path.joinpath(f'{file_name}.csv')

    def log_final_iter_results(self):
        self.beta_estimates = self.sum_beta_est / self.nb_sampling_iter
        self.y_estimates = self.sum_y_est / self.nb_sampling_iter
        beta_covariates_summary = self._create_beta_values_summary(
            self.beta_estimates.reshape(-1, len(self.feature_labels)).cpu(), dim=0
        )
        self.beta_covariates_summary_df = pd.DataFrame(
            beta_covariates_summary.t(),
            index=self.feature_labels,
            columns=self.moment_metrics + self.quantile_metrics,
        )
        self.y_estimates_df = pd.DataFrame(
            self.y_estimates.flatten().cpu(),
            columns=['y_estimate'],
            index=pd.MultiIndex.from_product([self.spatial_labels, self.temporal_labels]),
        )
        self.beta_estimates_df = pd.DataFrame(
            self.beta_estimates.reshape([-1, len(self.feature_labels)]).cpu(),
            columns=self.feature_labels,
            index=pd.MultiIndex.from_product([self.spatial_labels, self.temporal_labels]),
        )
        error_metrics = self._set_error_metrics()
        iters_summary_dict = {'Elapsed Time': self.total_elapsed_time} | error_metrics
        self._print_iter_result('TOTAL', iters_summary_dict)
        iter_results_df = pd.DataFrame.from_dict(self.logged_params_map)
        if self.export_path is not None:
            iter_results_df.to_csv(self._get_file_name('iter_results'), index=False)

    @classmethod
    def _create_beta_values_summary(
        cls, beta_values: torch.Tensor, dim: int = None
    ) -> torch.Tensor:
        """Create a summary for a given tensor of beta values across a given dimension
        for the metrics set in the class.

        Args:
            beta_values (torch.Tensor): Beta values to summarize
            dim (int, optional): Dimension of the tensor we want to summaryize. If None,
                we want to summarize the whole tensor and flatten it. Defaults to None.

        Returns:
            torch.Tensor: A tensor with summaries for the given beta values
        """
        all_metrics = cls.moment_metrics + cls.quantile_metrics
        summary_shape = [len(all_metrics)]
        if dim is not None:
            beta_val_shape = list(beta_values.shape)
            summary_shape = summary_shape + beta_val_shape[:dim] + beta_val_shape[dim + 1 :]
        beta_summaries = TSR.zeros(summary_shape)
        # Dimension for moment calculations are a bit different than for quantile
        moment_dim = dim if dim is not None else []
        beta_summaries[0] = beta_values.mean(dim=moment_dim)
        beta_summaries[1] = beta_values.var(dim=moment_dim)
        beta_summaries[len(cls.moment_metrics) :] = torch.quantile(
            beta_values, TSR.tensor(cls.quantile_values), dim=dim
        )
        return beta_summaries

    def get_beta_summary_df(
        self,
        spatial_labels: list[Any] = None,
        temporal_labels: list[Any] = None,
        feature_labels: list[Any] = None,
    ) -> pd.DataFrame:
        """Get a summary of the beta values. If no labels are given, then the summary is for all
            the betas. If labels are given, then the summary is for the given labels.

        Args:
            spatial_labels (list[Any], optional): The spatial labels to get the summary for.
                Defaults to None.
            temporal_labels (list[Any], optional): The temporal labels to get the summary for.
                Defaults to None.
            feature_labels (list[Any], optional): The feature labels to get the summary for.
                Defaults to None.

        Returns:
            pd.DataFrame: A dataframe with the summary for the given labels.
        """
        spatial_labs = spatial_labels if spatial_labels is not None else self.spatial_labels
        temporal_labs = temporal_labels if temporal_labels is not None else self.temporal_labels
        feature_labs = feature_labels if feature_labels is not None else self.feature_labels
        iteration_betas = self.get_iteration_betas_tensor(
            spatial_labs, temporal_labs, feature_labs
        )
        beta_summary = self._create_beta_values_summary(iteration_betas, dim=1).t()
        return pd.DataFrame(
            beta_summary.cpu(),
            columns=self.moment_metrics + self.quantile_metrics,
            index=pd.MultiIndex.from_product([spatial_labs, temporal_labs, feature_labs]),
        )

    def get_iteration_betas_tensor(
        self, spatial_labels: list[Any], temporal_labels: list[Any], feature_labels: list[Any]
    ) -> torch.Tensor:
        spatial_indexes = get_label_indexes(spatial_labels, self.spatial_labels, 'spatial')
        temporal_indexes = get_label_indexes(temporal_labels, self.temporal_labels, 'temporal')
        feature_indexes = get_label_indexes(feature_labels, self.feature_labels, 'feature')
        betas_per_iterations = torch.einsum(
            'sri,tri,cri->stci',
            [
                self.spatial_decomp_per_iter[spatial_indexes, :, :],
                self.temporal_decomp_per_iter[temporal_indexes, :, :],
                self.covs_decomp_per_iter[feature_indexes, :, :],
            ],
        )
        return betas_per_iterations.reshape([-1, self.nb_sampling_iter])
