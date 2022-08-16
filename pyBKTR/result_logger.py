from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch

from pyBKTR.tensor_ops import TSR


class ResultLogger:
    __slots__ = [
        'y',
        'omega',
        'covariates',
        'nb_iter',
        'nb_burn_in_iter',
        'export_dir',
        'results_export_dir',
        'sampled_beta_indexes',
        'sampled_y_indexes',
        'logged_params_map',
        'beta_estimates',
        'y_estimates',
        'sum_beta_est',
        'sum_y_est',
        'last_time_stamp',
        'export_path',
        'error_metrics',
        'tsr',
        'seed',
    ]

    def __init__(
        self,
        y: torch.Tensor,
        omega: torch.Tensor,
        covariates: torch.Tensor,
        nb_iter: int,
        nb_burn_in_iter: int,
        tensor_instance: TSR,
        results_export_dir: str | None = None,
        sampled_beta_indexes: list[int] = [],
        sampled_y_indexes: list[int] = [],
        seed: int | None = None,
    ):
        # Create a tensor dictionary holding scalar data gathered through all iterations
        self.logged_params_map = defaultdict(list)

        # Set export dir
        if results_export_dir is None:
            self.export_dir = None
        else:
            export_path = Path(results_export_dir)
            if not export_path.is_dir():
                raise ValueError(f'Path {export_path} does not exists.')
            self.export_path = export_path

        self.seed = seed

        # Create tensor that accumulate values needed for mean of evaluation
        self.tsr = tensor_instance
        self.sum_beta_est = self.tsr.zeros(covariates.shape)
        self.sum_y_est = self.tsr.zeros(y.shape)

        self.y = y
        self.omega = omega
        self.covariates = covariates
        self.nb_iter = nb_iter
        self.nb_burn_in_iter = nb_burn_in_iter

        # Indexes that need to be logged in the result output
        self.sampled_beta_indexes = sampled_beta_indexes
        self.sampled_y_indexes = sampled_y_indexes

        # Set initial timer value to calculate iterations' processing time
        self.last_time_stamp = time()

    def collect_iter_samples(self, iter: int, iter_logged_params: dict[str, float]):
        """
        Collect current iteration values inside the historical data tensor list.
        To be noted that errors have already been calculated before tau sampling.
        """
        elapsed_time_dict = self._get_elapsed_time_dict()
        y_beta_sampled_values = self._get_y_and_beta_sampled_values(
            self.y_estimates, self.beta_estimates
        )

        if iter > self.nb_burn_in_iter:
            self.sum_beta_est += self.beta_estimates
            self.sum_y_est += self.y_estimates

        total_logged_params = {
            **{'iter': iter},
            **{'is_burn_in': 1 if iter <= self.nb_burn_in_iter else 0},
            **iter_logged_params,
            **self.error_metrics,
            **y_beta_sampled_values,
            **elapsed_time_dict,
        }

        for k, v in total_logged_params.items():
            self.logged_params_map[k].append(v)

        self._print_iter_result(iter, {**elapsed_time_dict, **self.error_metrics})

    def _get_elapsed_time_dict(self):
        iter_elapsed_time = time() - self.last_time_stamp
        self.last_time_stamp = time()
        return {'elapsed_time': iter_elapsed_time}

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
        self.error_metrics = {
            'total_sq_error': float(total_sq_error),
            'mae': float(mae),
            'rmse': float(rmse),
        }
        return self.error_metrics

    def set_y_and_beta_estimates(self, decomp_tensors_map: dict[str, torch.Tensor]):
        """Calculate the estimated y and beta tensors

        Args:
            decomposition_tensors_map (dict[str, torch.Tensor]): A dictionnary that
                contains all the spatial, temporal and covariates decomposition
        """

        # Calculate Coefficient Estimation
        self.beta_estimates = torch.einsum(
            'im,jm,km->ijk',
            [
                decomp_tensors_map['spatial_decomp'],
                decomp_tensors_map['temporal_decomp'],
                decomp_tensors_map['covs_decomp'],
            ],
        )
        self.y_estimates = torch.einsum('ijk,ijk->ij', self.covariates, self.beta_estimates)

    def _get_y_and_beta_sampled_values(
        self, y_estimates: torch.Tensor, beta_estimates: torch.Tensor
    ) -> dict[str, float]:
        y_beta_dict = {}
        flat_y_est, flat_beta_est = y_estimates.cpu().flatten(), beta_estimates.cpu().flatten()

        for idx in self.sampled_y_indexes:
            y_beta_dict[f'y_{idx}'] = float(flat_y_est[idx])
        for idx in self.sampled_beta_indexes:
            y_beta_dict[f'beta_{idx}'] = float(flat_beta_est[idx])

        return y_beta_dict

    def _print_iter_result(self, iter: int, result_dict: dict[str, float]):
        formated_results = [f'{k.replace("_", " ")} is {v:.4f}' for k, v in result_dict.items()]
        print(f'** Result for iter {iter:<4} : {" || ".join(formated_results)} **')

    def _get_avg_estimates(self) -> dict[str, float | torch.Tensor]:
        """Calculate the final dictionary of values returned by the MCMC sampling

        The final values include the y estimation, the average estimated betas and the errors

        Returns:
            dict [str, float | torch.Tensor]: A dictionary of the MCMC's values of interest
        """
        nb_after_burn_in_iter = self.nb_iter - self.nb_burn_in_iter
        self.beta_estimates = self.sum_beta_est / nb_after_burn_in_iter
        self.y_estimates = self.sum_y_est / nb_after_burn_in_iter
        error_metrics = self._set_error_metrics()
        self._print_iter_result(-1, error_metrics)
        return {
            'y_est': self.y_estimates,
            'beta_est': self.beta_estimates,
            **error_metrics,
        }

    def _get_file_name(self, file_prefix: str):
        time_now = datetime.now()
        file_name = f'{file_prefix}_{time_now:%Y%m%d_%H%M}'
        if self.seed is not None:
            file_name = f'{file_name}__s{self.seed}'
        return self.export_path.joinpath(f'{file_name}.csv')

    def log_iter_results(self):
        iter_results_df = pd.DataFrame.from_dict(self.logged_params_map)
        avg_estimates = self._get_avg_estimates()
        if self.export_path is None:
            print(iter_results_df)
        else:
            iter_results_df.to_csv(self._get_file_name('iter_results'), index=False)
            y_est = avg_estimates['y_est'].cpu().flatten()
            beta_est = avg_estimates['beta_est'].cpu().flatten()
            np.savetxt(self._get_file_name('y_estimates'), y_est, delimiter=',')
            np.savetxt(self._get_file_name('beta_estimates'), beta_est, delimiter=',')
        return avg_estimates
