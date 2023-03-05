from typing import Literal

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
        omega: np.ndarray,  # TODO omega could be determined from y missing values (np.NaN) #12
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
            spatial_covariate_matrix (np.ndarray | torch.Tensor): Spatial Covariates. A
                two dimension matrix (nb spatial points x nb spatial covariates).
            temporal_covariate_matrix (np.ndarray | torch.Tensor):  Temporal Covariates. A
                two dimension matrix (nb temporal points x nb temporal covariates).
            y (np.ndarray | torch.Tensor): Response variable (`Y`). Variable that we want to
                predict. A two dimensions matrix (nb spatial points x nb spatial points).
            omega (np.ndarray | torch.Tensor): Mask showing if a y observation is missing or
                not. A two dimensions matrix (nb spatial points x nb temporal points).
            rank_decomp (int): Rank of the CP decomposition (Paper -- :math:`R`)
            burn_in_iter (int): Number of iteration before sampling (Paper -- :math:`K_1`).
            sampling_iter (int): Number of sampling iterations (Paper -- :math:`K_2`).
            spatial_kernel (Kernel, optional): Spatial kernel Used.
                Defaults to KernelMatern(smoothness_factor=3).
            spatial_kernel_x (None | np.ndarray | torch.Tensor, optional): Spatial kernel input
                tensor used to calculate covariate distance. Vector of length equal to nb
                spatial points. Defaults to None.
            spatial_kernel_dist (None | np.ndarray | torch.Tensor, optional): Spatial kernel
                covariate distance. A two dimensions matrix (nb spatial points x nb spatial
                points).  Should be used instead of `spatial_kernel_x` if distance was already
                calculated. Defaults to None.
            temporal_kernel (Kernel, optional): Temporal kernel used.
                Defaults to KernelSE().
            temporal_kernel_x (None | torch.Tensor, optional): Temporal kernel input tensor
                used to calculate covariate distance. Vector of length equal to nb temporal
                points. Defaults to None.
            temporal_kernel_dist (None | torch.Tensor, optional): Temporal kernel covariate
                distance. A two dimensions matrix (nb temporal points x nb temporal points).
                Should be used instead of `temporal_kernel_x` if distance was already
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
            ValueError: If `y` or `omega` first dimension's length are different than
                `spatial_covariate_matrix` first dimension
            ValueError: If `y` or `omega` second dimension's length are different than
                `temporal_covariate_matrix` first dimension
        """

        # Tensor assignation
        self.y = TSR.tensor(y)
        self.omega = TSR.tensor(omega)
        spatial_covariates = TSR.tensor(spatial_covariate_matrix)
        temporal_covariates = TSR.tensor(temporal_covariate_matrix)
        self.tau = 1 / TSR.tensor(sigma_r)
        temporal_kernel_x_tsr = TSR.get_tensor_or_none(temporal_kernel_x)
        temporal_kernel_dist_tsr = TSR.get_tensor_or_none(temporal_kernel_dist)
        spatial_kernel_x_tsr = TSR.get_tensor_or_none(spatial_kernel_x)
        spatial_kernel_dist_tsr = TSR.get_tensor_or_none(spatial_kernel_dist)

        # Verify input dimensions
        self._verify_input_dimensions(
            y,
            omega,
            spatial_covariates,
            temporal_covariates,
            spatial_kernel_x_tsr,
            spatial_kernel_dist_tsr,
            temporal_kernel_x_tsr,
            temporal_kernel_dist_tsr,
        )

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

        # Reshape covariates
        self._reshape_covariates(spatial_covariates, temporal_covariates)

        # Kernel assignation
        self.spatial_kernel = spatial_kernel
        self.temporal_kernel = temporal_kernel
        self.spatial_kernel.set_distance_matrix(spatial_kernel_x, spatial_kernel_dist)
        self.temporal_kernel.set_distance_matrix(temporal_kernel_x, temporal_kernel_dist)
        # Create first kernels
        self.spatial_kernel.kernel_gen()
        self.temporal_kernel.kernel_gen()

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

    @property
    def beta_estimates(self):
        if self.result_logger is None:
            raise RuntimeError('Beta estimates can only be accessed after MCMC sampling.')
        return self.result_logger.beta_estimates

    @property
    def beta_stdev(self):
        if self.result_logger is None:
            raise RuntimeError('Beta standard dev can only be accessed after MCMC sampling.')
        return self.result_logger.beta_stdev

    @property
    def y_estimates(self):
        if self.result_logger is None:
            raise RuntimeError('Y estimates can only be accessed after MCMC sampling.')
        return self.result_logger.y_estimates

    @staticmethod
    def _verify_kernel_inputs(
        kernel_x: torch.Tensor | None,
        kernel_dist: torch.Tensor | None,
        nb_input_points: int,
        kernel_type: Literal['spatial', 'temporal'],
    ):
        """Verify if kernel inputs are valid and align with covariates dimension.

        Args:
            kernel_x (torch.Tensor | None): Kernel x to be provided to a kernel.
            kernel_dist (torch.Tensor | None): Kernel distance to be provided to kernel.
            nb_input_points (int): Number of spatial/temporal points in covariates.
            kernel_type (Literal[&#39;spatial&#39;, &#39;temporal&#39;]): Type of kernel.

        Raises:
            ValueError: If kernel both or none of kernel_x and kernel_dist are provided.
            ValueError: If kernel_x is provided and its size is not appropriate.
            ValueError: If kernel_dist is provided and its size is not appropriate.
        """
        if (kernel_x is None) == (kernel_dist is None):
            raise ValueError(
                'Either `{kernel_type}_kernel_x` or `{kernel_type}_kernel_dist` must be provided'
            )
        if kernel_x is not None and nb_input_points != kernel_x.shape[0]:
            raise ValueError(
                f'`{kernel_type}_kernel_x` first input dimension must have the same'
                f' length as the number of {kernel_type} points.'
            )
        if kernel_dist is not None and not (
            nb_input_points == kernel_dist.shape[0] == kernel_dist.shape[1]
        ):
            raise ValueError(
                f'`{kernel_type}_kernel_dist` first and second input dimensions must have the same'
                f' length as the number of {kernel_type} points.'
            )

    @classmethod
    def _verify_input_dimensions(
        cls,
        y: torch.Tensor,
        omega: torch.Tensor,
        spatial_covariates: torch.Tensor,
        temporal_covariates: torch.Tensor,
        spatial_kernel_x: torch.Tensor | None,
        spatial_kernel_dist: torch.Tensor | None,
        temporal_kernel_x: torch.Tensor | None,
        temporal_kernel_dist: torch.Tensor | None,
    ):
        """Verify the validity of BKTR tensor inputs' dimensions

        Args:
            y (torch.Tensor): y in __init__
            omega (torch.Tensor):  omega in __init__
            spatial_covariates (torch.Tensor): spatial_covariates in __init__
            temporal_covariates (torch.Tensor): temporal_covariates in __init__
            spatial_kernel_x (torch.Tensor | None): spatial_kernel_x in __init__
            spatial_kernel_dist (torch.Tensor | None): spatial_kernel_dist in __init__
            temporal_kernel_x (torch.Tensor | None): temporal_kernel_x in __init__
            temporal_kernel_dist (torch.Tensor | None): temporal_kernel_dist in __init__

        Raises:
            ValueError: If omega and y do not respect the dimension of spatial covariates
            ValueError: If omega and y do not respect the dimension of temporal covariates
        """
        nb_spatial_points = spatial_covariates.shape[0]
        nb_temporal_points = temporal_covariates.shape[0]
        if not (nb_spatial_points == omega.shape[0] == y.shape[0]):
            raise ValueError(
                'Y and omega matrices should have first dimensions of same length'
                ' as the first dimension of the spatial covariate matrix'
            )
        if not (nb_temporal_points == omega.shape[1] == y.shape[1]):
            raise ValueError(
                'Y and omega matrices should have second dimensions of same length'
                ' as the first dimension of the temporal covariate matrix'
            )
        cls._verify_kernel_inputs(
            spatial_kernel_x, spatial_kernel_dist, nb_spatial_points, 'spatial'
        )
        cls._verify_kernel_inputs(
            temporal_kernel_x, temporal_kernel_dist, nb_temporal_points, 'temporal'
        )

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
        # Calcultate first likelihoods
        self._calc_spatial_marginal_ll()
        self._calc_temporal_marginal_ll()
