from typing import Union

import numpy as np
import torch

from pyBKTR.bktr_config import BKTRConfig
from pyBKTR.kernel_generators import (
    KernelGenerator,
    SpatialKernelGenerator,
    TemporalKernelGenerator,
)
from pyBKTR.likelihood_evaluator import MarginalLikelihoodEvaluator
from pyBKTR.samplers import (
    KernelParamSampler,
    PrecisionMatrixSampler,
    TauSampler,
    get_cov_decomp_chol,
    sample_norm_multivariate,
)


class BKTRRegressor:
    """Class encapsulating the BKTR regression steps

    A BKTRRegressor holds all the key elements to accomplish the MCMC sampling
    algorithm (**Algorithm 1** of the paper).
    """

    __slots__ = [
        'config',
        'spatial_distance_tensor',
        'y',
        'omega',
        'covariates',
        'covariates_dim',
        'logged_params_tensor',
        'tau',
        'spatial_decomp',
        'temporal_decomp',
        'covs_decomp',
        'y_estimate',
        'total_sq_error',
        'mae',
        'rmse',
        'temporal_kernel_generator',
        'spatial_kernel_generator',
        'spatial_length_sampler',
        'decay_scale_sampler',
        'periodic_length_sampler',
        'tau_sampler',
        'precision_matrix_sampler',
        'spatial_ll_evaluator',
        'temporal_ll_evaluator',
        'avg_y_est',
        'sum_beta_est',
    ]
    config: BKTRConfig
    spatial_distance_tensor: torch.Tensor
    y: torch.Tensor
    omega: torch.Tensor
    covariates: torch.Tensor
    covariates_dim: dict[str, int]
    logged_params_tensor: dict[str, torch.Tensor]
    tau: float
    # Covariate decompositions (change during iter)
    spatial_decomp: torch.Tensor  # U
    temporal_decomp: torch.Tensor  # V
    covs_decomp: torch.Tensor  # C or W
    # Y estimates and errors (change during iter)
    y_estimate: torch.Tensor
    total_sq_error: float
    mae: float
    rmse: float
    # Kernel Generators
    temporal_kernel_generator: KernelGenerator
    spatial_kernel_generator: KernelGenerator
    # Samplers
    spatial_length_sampler: KernelParamSampler
    decay_scale_sampler: KernelParamSampler
    periodic_length_sampler: KernelParamSampler
    tau_sampler: TauSampler
    precision_matrix_sampler: PrecisionMatrixSampler
    # Likelihood evaluators
    spatial_ll_evaluator: MarginalLikelihoodEvaluator
    temporal_ll_evaluator: MarginalLikelihoodEvaluator
    # used to collect mcmc samples
    avg_y_est: torch.Tensor
    sum_beta_est: torch.Tensor

    def __init__(
        self,
        bktr_config: BKTRConfig,
        temporal_covariate_matrix: np.ndarray,
        spatial_covariate_matrix: np.ndarray,
        spatial_distance_matrix: np.ndarray,
        y: np.ndarray,
        omega: np.ndarray,
    ):
        """Create a new *BKTRRegressor* object.

        Args:
            bktr_config (BKTRConfig): Configuration used for the BKTR regression
            temporal_covariate_matrix (np.ndarray):  Temporal Covariates
            spatial_covariate_matrix (np.ndarray): Spatial Covariates
            spatial_distance_matrix (np.ndarray): Distance between spatial entities
            y (np.ndarray): Response variable that we are trying to predict
            omega (np.ndarray): Mask showing if a y observation is missing or not
        """
        self.config = bktr_config
        # Set tensor backend according to config
        # torch.set_device_type(self.config.torch_device)
        # torch.set_tensor_type(self.config.torch_dtype)
        # Assignation
        self.spatial_distance_tensor = torch.tensor(spatial_distance_matrix)
        self.y = torch.tensor(y)
        self.omega = torch.tensor(omega)
        self.tau = 1 / torch.tensor(self.config.sigma_r)
        self._reshape_covariates(
            torch.tensor(spatial_covariate_matrix), torch.tensor(temporal_covariate_matrix)
        )
        self._initialize_params()

    def mcmc_sampling(self) -> dict[str, Union[float, torch.Tensor]]:
        """Launch the MCMC sampling process for a predefined number of iterations

        1. Sample the spatial length scale hyperparameter
        2. Sample the decay time scale hyperparameter
        3. Sample the periodic length scale hyperparameter
        4. Sample the precision matrix from a wishart distribution
        5. Sample a new spatial covariate decomposition
        6. Sample a new covariate decomposition
        7. Sample a new temporal covariate decomposition
        8. Calculate respective errors for the iterations
        9. Sample a new tau value
        10. Collect all the important data for the iteration

        Returns:
            dict [str, Union[float, torch.Tensor]]: A dictionary with the MCMC sampling's results
        """
        for i in range(1, self.config.max_iter + 1):
            print(f'*** Running iter {i} ***')
            self._sample_kernel_hparam()
            self._sample_precision_wish()
            self._sample_spatial_decomp()
            self._sample_covariate_decomp()
            self._sample_temporal_decomp()
            self._set_y_estimate_and_errors(i)
            self._sample_precision_tau()
            self._collect_iter_samples(i)
        return self._calculate_avg_estimates()

    def _reshape_covariates(
        self, spatial_covariate_tensor: torch.Tensor, temporal_covariate_tensor: torch.Tensor
    ):
        """Reshape the covariate tensors into one single tensor and set it as a property

        Args:
            spatial_covariate_tensor (torch.Tensor): Temporal Covariates
            temporal_covariate_tensor (torch.Tensor): Spatial Covariates
        """
        nb_spaces, nb_spatial_covariates = spatial_covariate_tensor.shape
        nb_times, nb_temporal_covariates = temporal_covariate_tensor.shape
        self.covariates_dim = {
            'nb_spaces': nb_spaces,  # S
            'nb_times': nb_times,  # T
            'nb_spatial_covariates': nb_spatial_covariates,
            'nb_temporal_covariates': nb_temporal_covariates,
            'nb_covariates': 1 + nb_spatial_covariates + nb_temporal_covariates,  # P
        }

        intersect_covs = torch.ones([nb_spaces, nb_times, 1])
        spatial_covs = spatial_covariate_tensor.unsqueeze(1).expand(
            [nb_spaces, nb_times, nb_spatial_covariates]
        )
        time_covs = temporal_covariate_tensor.unsqueeze(0).expand(
            [nb_spaces, nb_times, nb_temporal_covariates]
        )

        self.covariates = torch.dstack([intersect_covs, spatial_covs, time_covs])

    def _init_covariate_decomp(self):
        """Initialize CP decomposed covariate tensors with normally distributed random values"""
        rank_decomp = self.config.rank_decomp
        covs_dim = self.covariates_dim

        self.spatial_decomp = torch.randn(
            [covs_dim['nb_spaces'], rank_decomp], dtype=torch.float64
        )
        self.temporal_decomp = torch.randn(
            [covs_dim['nb_times'], rank_decomp], dtype=torch.float64
        )
        self.covs_decomp = torch.randn(
            [covs_dim['nb_covariates'], rank_decomp], dtype=torch.float64
        )

    @staticmethod
    def _calculate_error_metrics(
        y_estimate: torch.Tensor, y: torch.Tensor, omega: torch.Tensor
    ) -> dict[str, float]:
        """Calculate error metrics of interest (MAE, RMSE, Total Error)

        Args:
            y_estimate (torch.Tensor): Estimation of the response variable
            y (torch.Tensor): Response variable that we are trying to predict
            omega (torch.Tensor): Mask showing if a y observation is missing or not

        Returns:
            dict[str, float]: A dictionary containing the values of the error metrics
        """
        nb_observ = omega.sum()
        err_matrix = (y_estimate - y) * omega
        total_sq_error = err_matrix.norm() ** 2
        mae = err_matrix.abs().sum() / nb_observ
        rmse = (total_sq_error / nb_observ).sqrt()
        return {'total_sq_error': total_sq_error, 'mae': mae, 'rmse': rmse}

    def _set_y_estimate_and_errors(self, iter):
        """Calculate the estimated y and set the iteration errors appropriately

        Args:
            iter (int): Current iteration index
        """
        # Calculate Coefficient Estimation
        coefficient_estimate = torch.einsum(
            'im,jm,km->ijk', [self.spatial_decomp, self.temporal_decomp, self.covs_decomp]
        )
        self.y_estimate = torch.einsum('ijk,ijk->ij', self.covariates, coefficient_estimate)

        # Iteration errors
        err_metrics = self._calculate_error_metrics(self.y_estimate, self.y, self.omega)
        self.total_sq_error = err_metrics['total_sq_error']

        # Average errors
        burn_in_iter = self.config.burn_in_iter
        if iter == 0:
            avg_y_est = self.y_estimate
        else:
            lbound_indx = max(0, iter - burn_in_iter)
            sum_y_est = (
                self.logged_params_tensor['y_estimate'][lbound_indx:].sum(0) + self.y_estimate
            )
            avg_y_est = sum_y_est / (iter - lbound_indx)
        avg_err_metrics = self._calculate_error_metrics(avg_y_est, self.y, self.omega)
        self.mae = avg_err_metrics['mae']
        self.rmse = avg_err_metrics['rmse']
        print(f'Iter Error: MAE is {self.mae.cpu():.4f} || RMSE is {self.rmse.cpu():.4f}')

        # Collect sample for mcmc results
        self.avg_y_est = avg_y_est
        if iter == burn_in_iter + 1:
            self.sum_beta_est = coefficient_estimate
        elif iter > burn_in_iter + 1:
            self.sum_beta_est = self.sum_beta_est + coefficient_estimate

    def _create_iter_estim_tensors(self) -> dict[str, torch.Tensor]:
        """Create the tensor dictionary holding all the data gathered through all iterations

        Returns:
            dict[str, torch.Tensor]: Tensor Dictionary of historical data
        """
        self.logged_params_tensor = {
            k: torch.zeros(
                [self.config.max_iter]
                + (list(v.shape) if torch.is_tensor(v) and list(v.shape) else [1])
            )
            for k, v in self._get_logged_params_dict().items()
        }

    def _create_kernel_generators(self):
        """Create and set the kernel generators for the spatial and temporal kernels"""
        self.spatial_kernel_generator = SpatialKernelGenerator(
            self.spatial_distance_tensor,
            self.config.spatial_smoothness_factor,
            self.config.kernel_variance,
        )
        self.temporal_kernel_generator = TemporalKernelGenerator(
            'periodic_se',
            self.covariates_dim['nb_times'],
            self.config.temporal_period_length,
            self.config.kernel_variance,
        )

    def _create_likelihood_evaluators(self):
        """Create and set the evaluators for the spatial and the temporal likelihoods"""
        rank_decomp = self.config.rank_decomp

        self.spatial_ll_evaluator = MarginalLikelihoodEvaluator(
            rank_decomp,
            self.covariates_dim['nb_covariates'],
            self.covariates,
            self.omega,
            self.y,
            is_transposed=False,
        )
        self.temporal_ll_evaluator = MarginalLikelihoodEvaluator(
            rank_decomp,
            self.covariates_dim['nb_covariates'],
            self.covariates,
            self.omega,
            self.y,
            is_transposed=True,
        )

    def _create_hparam_samplers(self):
        """Create hyperparameter samplers

        Create and set the hyperparameter samplers for the
        spatial length scale, decay time scale, periodic length scale,
        tau and the precision matrix
        """
        self.spatial_length_sampler = KernelParamSampler(
            config=self.config.spatial_length_config,
            kernel_generator=self.spatial_kernel_generator,
            marginal_ll_eval_fn=self._calc_spatial_marginal_ll,
            kernel_hparam_name='spatial_length_scale',
        )
        self.decay_scale_sampler = KernelParamSampler(
            config=self.config.decay_scale_config,
            kernel_generator=self.temporal_kernel_generator,
            marginal_ll_eval_fn=self._calc_temporal_marginal_ll,
            kernel_hparam_name='decay_time_scale',
        )
        self.periodic_length_sampler = KernelParamSampler(
            config=self.config.periodic_scale_config,
            kernel_generator=self.temporal_kernel_generator,
            marginal_ll_eval_fn=self._calc_temporal_marginal_ll,
            kernel_hparam_name='periodic_length_scale',
        )

        self.tau_sampler = TauSampler(self.config.a_0, self.config.b_0, self.omega.sum())

        self.precision_matrix_sampler = PrecisionMatrixSampler(
            self.covariates_dim['nb_covariates'], self.config.rank_decomp
        )

    def _calc_spatial_marginal_ll(self):
        """Calculate the spatial marginal likelihood"""
        return self.spatial_ll_evaluator.calc_likelihood(
            self.spatial_kernel_generator.kernel, self.temporal_decomp, self.covs_decomp, self.tau
        )

    def _calc_temporal_marginal_ll(self):
        """Calculate the temporal marginal likelihood"""
        return self.temporal_ll_evaluator.calc_likelihood(
            self.temporal_kernel_generator.kernel, self.spatial_decomp, self.covs_decomp, self.tau
        )

    def _sample_kernel_hparam(self):
        """Sample new kernel hyperparameters"""
        self.spatial_length_sampler.sample()
        self.decay_scale_sampler.sample()
        self.periodic_length_sampler.sample()

    def _sample_precision_wish(self):
        """Sample the precision matrix from a Wishart distribution"""
        self.precision_matrix_sampler.sample(self.covs_decomp)

    def _sample_decomp_norm(
        self, initial_decomp: torch.Tensor, chol_l: torch.Tensor, uu: torch.Tensor
    ):
        """Sample a new covariate decomposition from a mulivariate normal distribution

        Args:
            initial_decomp (torch.Tensor): Decomposition of the previous iteration
            chol_l (torch.Tensor): The cholesky decomposition of the #TODO
            uu (torch.Tensor): _description_ #TODO

        Returns:
            torch.Tensor: A tensor containing the newly sampled covariate decomposition
        """
        precision_mat = chol_l.t()
        mean_vec = self.tau * torch.linalg.solve(precision_mat, uu)
        return sample_norm_multivariate(mean_vec, precision_mat).reshape_as(initial_decomp.t()).t()

    def _sample_spatial_decomp(self):
        """Sample a new spatial covariate decomposition"""
        ll_eval = self.spatial_ll_evaluator
        self.spatial_decomp = self._sample_decomp_norm(
            self.spatial_decomp, ll_eval.chol_lu, ll_eval.uu
        )

    def _sample_covariate_decomp(self):
        """Sample a new covariate decomposition"""
        chol_res = get_cov_decomp_chol(
            self.spatial_decomp,
            self.temporal_decomp,
            self.covariates,
            self.config.rank_decomp,
            self.omega,
            self.tau,
            self.y,
            self.precision_matrix_sampler.wish_precision_tensor,
        )
        self.covs_decomp = self._sample_decomp_norm(
            self.covs_decomp, chol_res['chol_lc'], chol_res['cc']
        )

    def _sample_temporal_decomp(self):
        """Sample a new temporal covariate decomposition

        Need to recalculate uu and chol_u since covariate decomp changed
        """
        self._calc_temporal_marginal_ll()
        ll_eval = self.temporal_ll_evaluator
        self.temporal_decomp = self._sample_decomp_norm(
            self.temporal_decomp, ll_eval.chol_lu, ll_eval.uu
        )

    def _sample_precision_tau(self):
        """Sample a new tau"""
        self.tau = self.tau_sampler.sample(self.total_sq_error)

    def _get_logged_params_dict(self):
        """Get a list of current iteration values needed for historical data"""
        return {
            'spatial_decomp': self.spatial_decomp,
            'temporal_decomp': self.temporal_decomp,
            'covs_decomp': self.covs_decomp,
            'tau': self.tau,
            'y_estimate': self.y_estimate,
            'mae': self.mae,
            'rmse': self.rmse,
            'spatial_length': self.spatial_length_sampler.theta_value,
            'decay_scale': self.decay_scale_sampler.theta_value,
            'periodic_length': self.periodic_length_sampler.theta_value,
        }

    def _collect_iter_samples(self, iter):
        """Collect current iteration values inside the historical data tensor list"""
        logged_params = self._get_logged_params_dict()
        for k, v in logged_params.items():
            self.logged_params_tensor[k][iter - 1] = v

    def _calculate_avg_estimates(self) -> dict[str, Union[float, torch.Tensor]]:
        """Calculate the final dictionary of values returned by the MCMC sampling

        The final values include the y estimation, the average estimated betas and the errors

        Returns:
            dict [str, Union[float, torch.Tensor]]: A dictionary of the MCMC's values of interest
        """
        return {
            'y_est': self.avg_y_est,
            'beta_est': self.sum_beta_est / (self.config.max_iter - self.config.burn_in_iter + 1),
            'mae': self.mae,
            'rmse': self.rmse,
        }

    def _initialize_params(self):
        """Initialize all parameters that are needed before we start the MCMC sampling"""
        self._init_covariate_decomp()
        self._set_y_estimate_and_errors(0)
        self._create_kernel_generators()
        self._create_likelihood_evaluators()
        self._create_hparam_samplers()
        self._create_iter_estim_tensors()
