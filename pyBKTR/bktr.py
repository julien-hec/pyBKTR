import numpy as np
import torch

from pyBKTR.bktr_config import BKTRConfig
from pyBKTR.kernel_generators import (
    KernelGenerator,
    SpatialKernelGenerator,
    TemporalKernelGenerator,
)
from pyBKTR.likelihood_evaluator import MarginalLikelihoodEvaluator
from pyBKTR.result_logger import ResultLogger
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
        'result_logger',
        'temporal_kernel_generator',
        'spatial_kernel_generator',
        'spatial_length_sampler',
        'decay_scale_sampler',
        'periodic_length_sampler',
        'tau_sampler',
        'precision_matrix_sampler',
        'spatial_ll_evaluator',
        'temporal_ll_evaluator',
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
    # Kernel Generators
    temporal_kernel_generator: KernelGenerator
    spatial_kernel_generator: KernelGenerator
    # Result Logger
    result_logger: ResultLogger
    # Samplers
    spatial_length_sampler: KernelParamSampler
    decay_scale_sampler: KernelParamSampler
    periodic_length_sampler: KernelParamSampler
    tau_sampler: TauSampler
    precision_matrix_sampler: PrecisionMatrixSampler
    # Likelihood evaluators
    spatial_ll_evaluator: MarginalLikelihoodEvaluator
    temporal_ll_evaluator: MarginalLikelihoodEvaluator

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
        # torch.set_default_dtype(self.config.torch_dtype)
        if self.config.torch_seed is not None:
            torch.manual_seed(self.config.torch_seed)
        # Assignation
        self.spatial_distance_tensor = torch.tensor(spatial_distance_matrix)
        self.y = torch.tensor(y)
        self.omega = torch.tensor(omega)
        self.tau = 1 / torch.tensor(self.config.sigma_r)
        self._reshape_covariates(
            torch.tensor(spatial_covariate_matrix), torch.tensor(temporal_covariate_matrix)
        )
        self._initialize_params()

    def mcmc_sampling(self) -> dict[str, float | torch.Tensor]:
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
            dict [str, float | torch.Tensor]: A dictionary with the MCMC sampling's results
        """
        for i in range(1, self.config.max_iter + 1):
            self._sample_kernel_hparam()
            self._sample_precision_wish()
            self._sample_spatial_decomp()
            self._sample_covariate_decomp()
            self._sample_temporal_decomp()
            self._set_errors_and_sample_precision_tau()
            self._collect_iter_values(i)
        return self._log_iter_results()

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

    def _create_result_logger(self):
        self.result_logger = ResultLogger(
            y=self.y,
            omega=self.omega,
            covariates=self.covariates,
            nb_iter=self.config.max_iter,
            nb_burn_in_iter=self.config.burn_in_iter,
            sampled_beta_indexes=self.config.sampled_beta_indexes,
            sampled_y_indexes=self.config.sampled_y_indexes,
            results_export_dir=self.config.results_export_dir,
        )

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
        mean_vec = (
            self.tau
            * torch.linalg.solve_triangular(precision_mat, uu.unsqueeze(1), upper=True).squeeze()
        )
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

    def _set_errors_and_sample_precision_tau(self):
        """Sample a new tau and set errors"""
        self.result_logger.set_y_and_beta_estimates(self._decomposition_tensors)
        error_metrics = self.result_logger._set_error_metrics()
        self.tau = self.tau_sampler.sample(error_metrics['total_sq_error'])

    @property
    def _logged_scalar_params(self):
        """Get a dict of current iteration values needed for historical data"""
        return {
            'tau': float(self.tau),
            'spatial_length': self.spatial_length_sampler.theta_value,
            'decay_scale': self.decay_scale_sampler.theta_value,
            'periodic_length': self.periodic_length_sampler.theta_value,
        }

    @property
    def _decomposition_tensors(self):
        """Get a dict of current iteration decomposition needed to calculate estimated betas"""
        return {
            'spatial_decomp': self.spatial_decomp,
            'temporal_decomp': self.temporal_decomp,
            'covs_decomp': self.covs_decomp,
        }

    def _collect_iter_values(self, iter: int):
        """Collect current iteration values inside the historical data tensor list"""
        self.result_logger.collect_iter_samples(iter, self._logged_scalar_params)

    def _log_iter_results(self):
        return self.result_logger.log_iter_results()

    def _initialize_params(self):
        """Initialize all parameters that are needed before we start the MCMC sampling"""
        self._init_covariate_decomp()
        self._create_result_logger()
        self._create_kernel_generators()
        self._create_likelihood_evaluators()
        self._create_hparam_samplers()
