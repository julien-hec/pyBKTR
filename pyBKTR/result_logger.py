import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Literal

import pandas as pd
import torch
from formulaic import Formula

from pyBKTR.kernels import Kernel, KernelComposed
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
        'formula',
        'rank_decomp',
        'spatial_labels',
        'temporal_labels',
        'feature_labels',
        'spatial_kernel',
        'temporal_kernel',
        'hparam_labels',
        'hparam_per_iter',
        'spatial_decomp_per_iter',
        'temporal_decomp_per_iter',
        'covs_decomp_per_iter',
        'sum_beta_est',
        'sum_y_est',
        'beta_estimates_df',
        'y_estimates_df',
        'beta_covariates_summary_df',
        'hparam_per_iter_df',
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
        'p2.5',
        'Q1',
        'Median',
        'Q3',
        'p97.5',
        'Max',
    ]
    quantile_values = [0, 0.025, 0.25, 0.5, 0.75, 0.975, 1]

    # Summary parameters
    LINE_NCHAR = 79
    TAB_STR = '  '
    LINE_SEPARATOR = LINE_NCHAR * '='
    DF_DISTRIB_STR_PARAMS = {
        'float_format': '{:,.3f}'.format,
        'col_space': 7,
        'line_width': LINE_NCHAR,
        'max_colwidth': 20,
        'formatters': {'__index__': lambda x: f'{x:20}'},
    }
    DISTRIB_COLS = ['p2.5', 'Q1', 'Median', 'Mean', 'Q3', 'p97.5']

    def __init__(
        self,
        y: torch.Tensor,
        omega: torch.Tensor,
        covariates: torch.Tensor,
        nb_burn_in_iter: int,
        nb_sampling_iter: int,
        rank_decomp: int,
        formula: Formula,
        spatial_labels: list,
        temporal_labels: list,
        feature_labels: list,
        spatial_kernel: Kernel,
        temporal_kernel: Kernel,
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
        self.hparam_labels = [
            'Tau',
            *[f'Spatial - {p.full_name}' for p in spatial_kernel.parameters if not p.is_fixed],
            *[f'Temporal - {p.full_name}' for p in temporal_kernel.parameters if not p.is_fixed],
        ]
        self.spatial_decomp_per_iter = TSR.zeros(
            (len(spatial_labels), rank_decomp, nb_sampling_iter)
        )
        self.temporal_decomp_per_iter = TSR.zeros(
            (len(temporal_labels), rank_decomp, nb_sampling_iter)
        )
        self.covs_decomp_per_iter = TSR.zeros(
            (len(self.feature_labels), rank_decomp, nb_sampling_iter)
        )
        self.hparam_per_iter = TSR.zeros((len(self.hparam_labels), nb_sampling_iter))
        self.sum_beta_est = TSR.zeros(covariates.shape)
        self.sum_y_est = TSR.zeros(y.shape)
        self.total_elapsed_time = 0

        self.y = y
        self.omega = omega
        self.covariates = covariates
        self.formula = formula
        self.rank_decomp = rank_decomp
        self.spatial_kernel = spatial_kernel
        self.temporal_kernel = temporal_kernel
        self.nb_burn_in_iter = nb_burn_in_iter
        self.nb_sampling_iter = nb_sampling_iter

        # Set initial timer value to calculate iterations' processing time
        self.last_time_stamp = time()

    def collect_iter_samples(self, iter: int, tau_value: float):
        """
        Collect current iteration values inside the historical data tensor list.
        To be noted that errors have already been calculated before tau sampling.
        """
        elapsed_time_dict = self._get_elapsed_time_dict()

        if iter > self.nb_burn_in_iter:
            self.sum_beta_est += self.beta_estimates
            self.sum_y_est += self.y_estimates

            # Collect Hyperparameters
            s_iter = iter - self.nb_burn_in_iter - 1
            s_params = [p for p in self.spatial_kernel.parameters if not p.is_fixed]
            t_params = [p for p in self.temporal_kernel.parameters if not p.is_fixed]
            self.hparam_per_iter[0, s_iter] = tau_value
            self.hparam_per_iter[1 : len(s_params) + 1, s_iter] = TSR.tensor(
                [p.value for p in s_params]
            )
            self.hparam_per_iter[1 + len(s_params) :, s_iter] = TSR.tensor(
                [p.value for p in t_params]
            )

        total_logged_params = {
            **{'iter': iter},
            **{'is_burn_in': 1 if iter <= self.nb_burn_in_iter else 0},
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
        beta_covariates_summary = self._create_distrib_values_summary(
            self.beta_estimates.reshape(-1, len(self.feature_labels)).cpu(), dim=0
        )
        self.beta_covariates_summary_df = pd.DataFrame(
            beta_covariates_summary.t(),
            index=self.feature_labels,
            columns=self.moment_metrics + self.quantile_metrics,
        )
        self.y_estimates_df = pd.DataFrame(
            self.y_estimates.flatten().cpu(),
            columns=[str(self.formula.lhs)],
            index=pd.MultiIndex.from_product([self.spatial_labels, self.temporal_labels]),
        )
        self.beta_estimates_df = pd.DataFrame(
            self.beta_estimates.reshape([-1, len(self.feature_labels)]).cpu(),
            columns=self.feature_labels,
            index=pd.MultiIndex.from_product([self.spatial_labels, self.temporal_labels]),
        )
        self.hparam_per_iter_df = pd.DataFrame(
            self.hparam_per_iter.t().cpu(),
            columns=self.hparam_labels,
        )
        error_metrics = self._set_error_metrics()
        iters_summary_dict = {'Elapsed Time': self.total_elapsed_time} | error_metrics
        self._print_iter_result('TOTAL', iters_summary_dict)
        iter_results_df = pd.DataFrame.from_dict(self.logged_params_map)
        if self.export_path is not None:
            iter_results_df.to_csv(self._get_file_name('iter_results'), index=False)

    @classmethod
    def _create_distrib_values_summary(cls, values: torch.Tensor, dim: int = None) -> torch.Tensor:
        """Create a summary for a given tensor of beta values across a given dimension
        for the metrics set in the class.

        Args:
            values (torch.Tensor): Values to summarize
            dim (int, optional): Dimension of the tensor we want to summaryize. If None,
                we want to summarize the whole tensor and flatten it. Defaults to None.

        Returns:
            torch.Tensor: A tensor with summaries for the given beta values
        """
        all_metrics = cls.moment_metrics + cls.quantile_metrics
        summary_shape = [len(all_metrics)]
        if dim is not None:
            beta_val_shape = list(values.shape)
            summary_shape = summary_shape + beta_val_shape[:dim] + beta_val_shape[dim + 1 :]
        beta_summaries = TSR.zeros(summary_shape)
        # Dimension for moment calculations are a bit different than for quantile
        moment_dim = dim if dim is not None else []
        beta_summaries[0] = values.mean(dim=moment_dim)
        beta_summaries[1] = values.var(dim=moment_dim)
        beta_summaries[len(cls.moment_metrics) :] = torch.quantile(
            values, TSR.tensor(cls.quantile_values), dim=dim
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
        beta_summary = self._create_distrib_values_summary(iteration_betas, dim=1).t()
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

    def summary(self) -> str:
        """Print a summary of the BKTR regressor.

        Returns:
            str: A string representation of the BKTR regressor after MCMC sampling.
        """
        title_format = f'^{self.LINE_NCHAR}'
        summary_str = [
            '',
            self.LINE_SEPARATOR,
            '',
            f'{"BKTR Regressor Summary":{title_format}}',
            '',
            self.LINE_SEPARATOR,
            self._get_formula_str(),
            '',
            f'Burn-in iterations: {self.nb_burn_in_iter}',
            f'Sampling iterations: {self.nb_sampling_iter}',
            f'Rank decomposition: {self.rank_decomp}',
            f'Nb Spatial Locations: {len(self.spatial_labels)}',
            f'Nb Temporal Points: {len(self.temporal_labels)}',
            f'Nb Covariates: {len(self.feature_labels)}',
            self.LINE_SEPARATOR,
            'In Sample Errors:',
            f'{self.TAB_STR}RMSE: {self.error_metrics["RMSE"]:.3f}',
            f'{self.TAB_STR}MAE: {self.error_metrics["MAE"]:.3f}',
            f'Computation time: {self.total_elapsed_time:.2f}s.',
            self.LINE_SEPARATOR,
            'Kernels',
            '',
            '-- Spatial Kernel --',
            self._kernel_summary(self.spatial_kernel, 'spatial'),
            '',
            '-- Temporal Kernel --',
            self._kernel_summary(self.temporal_kernel, 'temporal'),
            self.LINE_SEPARATOR,
            self._beta_summary(),
            self.LINE_SEPARATOR,
            '',
        ]
        return '\n'.join(summary_str)

    def _get_formula_str(self) -> str:
        formula_str = f'Formula: {self.formula.lhs} ~ {self.formula.rhs}'
        f_wrap = textwrap.wrap(formula_str, width=self.LINE_NCHAR, subsequent_indent=self.TAB_STR)
        return '\n'.join(f_wrap)

    def _kernel_summary(
        self, kernel: Kernel, kernel_type: Literal['spatial', 'temporal'], indent_count=0
    ) -> str:
        """Get a string representation of a given kernel. Since the kernel can be composed, this
            function needs to be recursive.


        Args:
            kernel (Kernel): The kernel to get the summary for.
            kernel_type (Literal[&#39;spatial&#39;, &#39;temporal&#39;]): The type of kernel.
            indent_count (int, optional): Indentation level (related to the depth of composition).
                Defaults to 0.

        Returns:
            str: A string representation of the kernel. Containing the name of the kernel,
                the estimated parameters distribution and the fixed parameters.
        """
        params = kernel.parameters
        if kernel.__class__ == KernelComposed:
            new_ind_nb = indent_count + 1
            kernel_elems = [
                f'Composed Kernel ({str(kernel.composition_operation.value).capitalize()})',
                f'{self._kernel_summary(kernel.left_kernel, kernel_type, new_ind_nb)}',
                f'{self.TAB_STR}{"+" if kernel.composition_operation == "add" else "*"}',
                f'{self._kernel_summary(kernel.right_kernel, kernel_type, new_ind_nb)}',
            ]
        else:
            fixed_params = [p for p in params if p.is_fixed]
            sampled_params = [p for p in params if not p.is_fixed]
            sampled_par_indexes = [
                self.hparam_labels.index(f'{kernel_type.capitalize()} - {p.full_name}')
                for p in sampled_params
            ]
            sampled_par_tsr = self.hparam_per_iter[sampled_par_indexes, :]
            sampled_par_summary = self._create_distrib_values_summary(sampled_par_tsr, dim=1)
            sampled_par_df = pd.DataFrame(
                sampled_par_summary.cpu().t(),
                columns=self.moment_metrics + self.quantile_metrics,
                index=[p.name for p in sampled_params],
            )[self.DISTRIB_COLS]
            sampled_params_str = sampled_par_df.to_string(**self.DF_DISTRIB_STR_PARAMS)
            constant_params_strs = [
                f'{p.name:20}   Fixed Value: {p.value:.3f}' for p in fixed_params
            ]
            kernel_elems = [
                kernel._name,
                'Parameter(s):',
                sampled_params_str,
                *constant_params_strs,
            ]
        kernel_str = '\n'.join(kernel_elems)
        return textwrap.indent(kernel_str, self.TAB_STR * indent_count)

    def _beta_summary(self) -> str:
        """Get a string representation of the beta estimates aggregated per covariates.
            (This shows the distribution of the beta hats per covariates)

        Returns:
            str: A string representation of the beta estimates.
        """

        distrib_df = self.beta_covariates_summary_df[self.DISTRIB_COLS].copy()
        beta_distrib_str = distrib_df.to_string(**self.DF_DISTRIB_STR_PARAMS)
        beta_summary_str = [
            'Beta Estimates Summary (Aggregated Per Covariates)',
            '',
            beta_distrib_str,
        ]
        return '\n'.join(beta_summary_str)
