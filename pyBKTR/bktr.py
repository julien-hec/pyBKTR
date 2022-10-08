import numpy as np
import torch

from pyBKTR.kernels import Kernel, KernelMatern, KernelSE
from pyBKTR.likelihood_evaluator import MarginalLikelihoodEvaluator
from pyBKTR.result_logger import ResultLogger
from pyBKTR.samplers import (
    KernelParamSampler,
    PrecisionMatrixSampler,
    TauSampler,
    get_cov_decomp_chol,
    sample_norm_multivariate,
)
from pyBKTR.tensor_ops import TSR


class BKTRRegressor:
    """Class encapsulating the BKTR regression steps

    A BKTRRegressor holds all the key elements to accomplish the MCMC sampling
    algorithm (**Algorithm 1** of the paper).
    """

    __slots__ = [
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
        'temporal_kernel',
        'spatial_kernel',
        'spatial_params_sampler',
        'temporal_params_sampler',
        'tau_sampler',
        'precision_matrix_sampler',
        'spatial_ll_evaluator',
        'temporal_ll_evaluator',
        'rank_decomp',
        'burn_in_iter',
        'sampling_iter',
        'max_iter',
        'a_0',
        'b_0',
        'sampled_beta_indexes',
        'sampled_y_indexes',
        'results_export_dir',
    ]
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
    # Kernels
    temporal_kernel: Kernel
    spatial_kernel: Kernel
    # Result Logger
    result_logger: ResultLogger
    # Samplers
    spatial_params_sampler: KernelParamSampler
    temporal_params_sampler: KernelParamSampler
    tau_sampler: TauSampler
    precision_matrix_sampler: PrecisionMatrixSampler
    # Likelihood evaluators
    spatial_ll_evaluator: MarginalLikelihoodEvaluator
    temporal_ll_evaluator: MarginalLikelihoodEvaluator
    # Params
    rank_decomp: int
    burn_in_iter: int
    sampling_iter: int
    max_iter: int
    a_0: float
    b_0: float
    # Export Params
    sampled_beta_indexes: list[int]
    sampled_y_indexes: list[int]
    results_export_dir: str | None

    def __init__(
        self,
        temporal_covariate_matrix: np.ndarray,
        spatial_covariate_matrix: np.ndarray,
        y: np.ndarray,
        omega: np.ndarray,
        rank_decomp: int,
        burn_in_iter: int,
        sampling_iter: int,
        spatial_kernel: Kernel = KernelMatern(smoothness_factor=3),
        spatial_kernel_x: None | torch.Tensor = None,
        spatial_kernel_dist: None | torch.Tensor = None,
        temporal_kernel: Kernel = KernelSE(),
        temporal_kernel_x: None | torch.Tensor = None,
        temporal_kernel_dist: None | torch.Tensor = None,
        sigma_r: float = 1e-2,
        a_0: float = 1e-6,
        b_0: float = 1e-6,
        sampled_beta_indexes: list[int] = [],
        sampled_y_indexes: list[int] = [],
        results_export_dir: str | None = None,
    ):
        """Create a new *BKTRRegressor* object.

        Args:
            temporal_covariate_matrix (np.ndarray):  Temporal Covariates
            spatial_covariate_matrix (np.ndarray): Spatial Covariates
            y (np.ndarray): Response variable that we are trying to predict
            omega (np.ndarray): Mask showing if a y observation is missing or not
            rank_decomp (int): Rank of the CP decomposition (Paper -- :math:`R`)
            burn_in_iter (int): Number of iteration before sampling (Paper -- :math:`K_1`)
            sampling_iter (int): Number of sampling iterations
            spatial_kernel (Kernel, optional): Spatial kernel Used.
                Defaults to KernelMatern(smoothness_factor=3).
            spatial_kernel_x (None | torch.Tensor, optional): Spatial kernel input tensor
                used to calculate covariate distance. Defaults to None.
            spatial_kernel_dist (None | torch.Tensor, optional): Spatial kernel covariate
                distance. Can be used instead of `spatial_kernel_x` if distance is already
                calculated. Defaults to None.
            temporal_kernel (Kernel, optional): Temporal kernel used.
                Defaults to KernelSE().
            temporal_kernel_x (None | torch.Tensor, optional): Temporal kernel input tensor
                used to calculate covariate distance. Defaults to None.
            temporal_kernel_dist (None | torch.Tensor, optional): Temporal kernel covariate
                distance. Can be used instead of `temporal_kernel_x` if distance is already
                calculated. Defaults to None.
            sigma_r (float, optional): Variance of the white noise process TODO
                (Paper -- :math:`\\tau^{-1}`). Defaults to 1e-2.
            a_0 (float, optional): Initial value for the shape (:math:`\\alpha`) in the gamma
                function generating tau. Defaults to 1e-6.
            b_0 (float, optional): Initial value for the rate (:math:`\\beta`) in the gamma
                function generating tau. Defaults to 1e-6.
            sampled_beta_indexes (list[int], optional): Indexes of beta estimates that need
                to be sampled through iterations. Defaults to [].
            sampled_y_indexes (list[int], optional): Indexes of y estimates that need
                to be sampled through iterations. Defaults to [].
            results_export_dir (str | None, optional): Path of the folder where the csv file
                will be exported (if None it is printed). Defaults to None.

        Raises:
            ValueError: If none or both `spatial_kernel_x` and `spatial_kernel_dist` are provided
            ValueError: If none or both `temporal_kernel_x` and `temporal_kernel_dist` are provided
        """
        # Param assignation
        self.rank_decomp = rank_decomp
        self.burn_in_iter = burn_in_iter
        self.sampling_iter = sampling_iter
        self.max_iter = self.burn_in_iter + self.sampling_iter
        self.a_0 = a_0
        self.b_0 = b_0
        self.sampled_beta_indexes = sampled_beta_indexes
        self.sampled_y_indexes = sampled_y_indexes
        self.results_export_dir = results_export_dir
        # Tensor assignation
        self.y = TSR.tensor(y)
        self.omega = TSR.tensor(omega)
        self.tau = 1 / TSR.tensor(sigma_r)
        self._reshape_covariates(
            TSR.tensor(spatial_covariate_matrix), TSR.tensor(temporal_covariate_matrix)
        )
        # Kernel assignation
        self.spatial_kernel = spatial_kernel
        self.temporal_kernel = temporal_kernel
        if (spatial_kernel_x is None) == (spatial_kernel_dist is None):
            raise ValueError('Either `spatial_kernel_x` or `spatial_kernel_dist` must be provided')
        if (temporal_kernel_x is None) == (temporal_kernel_dist is None):
            raise ValueError(
                'Either `temporal_kernel_x` or `temporal_kernel_dist` must be provided'
            )
        self.spatial_kernel.set_distance_matrix(spatial_kernel_x, spatial_kernel_dist)
        self.temporal_kernel.set_distance_matrix(temporal_kernel_x, temporal_kernel_dist)

    def mcmc_sampling(self) -> dict[str, float | torch.Tensor]:
        """Launch the MCMC sampling process for a predefined number of iterations

        1. Sample spatial kernel hyperparameters
        2. Sample temporal kernel hyperparameters
        3. Sample the precision matrix from a wishart distribution
        4. Sample a new spatial covariate decomposition
        5. Sample a new covariate decomposition
        6. Sample a new temporal covariate decomposition
        7. Calculate respective errors for the iterations
        8. Sample a new tau value
        9. Collect all the important data for the iteration

        Returns:
            dict [str, float | torch.Tensor]: A dictionary with the MCMC sampling's results
        """
        self._initialize_params()
        for i in range(1, self.max_iter + 1):
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

        intersect_covs = TSR.ones([nb_spaces, nb_times, 1])
        spatial_covs = spatial_covariate_tensor.unsqueeze(1).expand(
            [nb_spaces, nb_times, nb_spatial_covariates]
        )
        time_covs = temporal_covariate_tensor.unsqueeze(0).expand(
            [nb_spaces, nb_times, nb_temporal_covariates]
        )

        self.covariates = torch.dstack([intersect_covs, spatial_covs, time_covs])

    def _init_covariate_decomp(self):
        """Initialize CP decomposed covariate tensors with normally distributed random values"""
        rank_decomp = self.rank_decomp
        covs_dim = self.covariates_dim

        self.spatial_decomp = TSR.randn([covs_dim['nb_spaces'], rank_decomp])
        self.temporal_decomp = TSR.randn([covs_dim['nb_times'], rank_decomp])
        self.covs_decomp = TSR.randn([covs_dim['nb_covariates'], rank_decomp])

    def _create_result_logger(self):
        self.result_logger = ResultLogger(
            y=self.y,
            omega=self.omega,
            covariates=self.covariates,
            nb_iter=self.max_iter,
            nb_burn_in_iter=self.burn_in_iter,
            sampled_beta_indexes=self.sampled_beta_indexes,
            sampled_y_indexes=self.sampled_y_indexes,
            results_export_dir=self.results_export_dir,
        )

    def _create_likelihood_evaluators(self):
        """Create and set the evaluators for the spatial and the temporal likelihoods"""
        self.spatial_ll_evaluator = MarginalLikelihoodEvaluator(
            self.rank_decomp,
            self.covariates_dim['nb_covariates'],
            self.covariates,
            self.omega,
            self.y,
            is_transposed=False,
        )
        self.temporal_ll_evaluator = MarginalLikelihoodEvaluator(
            self.rank_decomp,
            self.covariates_dim['nb_covariates'],
            self.covariates,
            self.omega,
            self.y,
            is_transposed=True,
        )

    def _create_hparam_samplers(self):
        """Create hyperparameter samplers

        Create and set the hyperparameter samplers for the spatial/temporal kernel parameters,
        tau and the precision matrix
        """
        self.spatial_params_sampler = KernelParamSampler(
            kernel=self.spatial_kernel,
            marginal_ll_eval_fn=self._calc_spatial_marginal_ll,
        )
        self.temporal_params_sampler = KernelParamSampler(
            kernel=self.temporal_kernel,
            marginal_ll_eval_fn=self._calc_temporal_marginal_ll,
        )

        self.tau_sampler = TauSampler(self.a_0, self.b_0, self.omega.sum())

        self.precision_matrix_sampler = PrecisionMatrixSampler(
            self.covariates_dim['nb_covariates'], self.rank_decomp
        )

    def _calc_spatial_marginal_ll(self):
        """Calculate the spatial marginal likelihood"""
        return self.spatial_ll_evaluator.calc_likelihood(
            self.spatial_kernel.kernel, self.temporal_decomp, self.covs_decomp, self.tau
        )

    def _calc_temporal_marginal_ll(self):
        """Calculate the temporal marginal likelihood"""
        return self.temporal_ll_evaluator.calc_likelihood(
            self.temporal_kernel.kernel, self.spatial_decomp, self.covs_decomp, self.tau
        )

    def _sample_kernel_hparam(self):
        """Sample new kernel hyperparameters"""
        self.spatial_params_sampler.sample()
        self.temporal_params_sampler.sample()

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
            self.rank_decomp,
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
        temporal_params = {p.full_name: p.value for p in self.temporal_kernel.parameters}
        spatial_params = {p.full_name: p.value for p in self.spatial_kernel.parameters}
        return {'tau': float(self.tau), **temporal_params, **spatial_params}

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
        self._create_likelihood_evaluators()
        self._create_hparam_samplers()
