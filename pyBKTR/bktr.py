from typing import Literal

import pandas as pd
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
        'results_export_suffix',
        'spatial_labels',
        'temporal_labels',
        'feature_labels',
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
    # Labels
    spatial_labels: list
    temporal_labels: list
    feature_labels: list

    # Constant string needed for dataframe index names
    spatial_index_name = 'location'
    temporal_index_name = 'time'

    def __init__(
        self,
        covariates_df: pd.DataFrame,
        y_df: pd.DataFrame,
        rank_decomp: int,
        burn_in_iter: int,
        sampling_iter: int,
        spatial_kernel: Kernel = KernelMatern(smoothness_factor=3),
        spatial_x_df: None | pd.DataFrame = None,
        spatial_dist_df: None | pd.DataFrame = None,
        temporal_kernel: Kernel = KernelSE(),
        temporal_x_df: None | pd.DataFrame = None,
        temporal_dist_df: None | pd.DataFrame = None,
        sigma_r: float = 1e-2,
        a_0: float = 1e-6,
        b_0: float = 1e-6,
        sampled_beta_indexes: list[int] = [],
        sampled_y_indexes: list[int] = [],
        results_export_dir: str | None = None,
        results_export_suffix: str | None = None,
    ):
        """Create a new *BKTRRegressor* object.

        Args:
            covariates_df (pd.DataFrame):  A dataframe containing all the covariates
                through time and space. It is important that the dataframe has a two
                indexes named `location` and `time` respectively. The dataframe should
                also contain every possible combinations of `location` and `time`
                (i.e. even missing rows should be filled present but filled with NaN).
                So if the dataframe has 10 locations and 5 time points, it should have
                50 rows (10 x 5).
            y_df (pd.DataFrame): Response variable (`Y`). Variable that we want to
                predict. A two dimensions dataframe (nb locations x nb time points).
            rank_decomp (int): Rank of the CP decomposition (Paper -- :math:`R`)
            burn_in_iter (int): Number of iteration before sampling (Paper -- :math:`K_1`).
            sampling_iter (int): Number of sampling iterations (Paper -- :math:`K_2`).
            spatial_kernel (Kernel, optional): Spatial kernel Used.
                Defaults to KernelMatern(smoothness_factor=3).
            spatial_x_df (None | pd.DataFrame, optional): Spatial kernel input
                tensor used to calculate covariate distance. Vector of length equal to nb
                location points. Defaults to None.
            spatial_dist_df (None | pd.DataFrame, optional): Spatial kernel
                covariate distance. A two dimensions df (nb location points x nb location
                points).  Should be used instead of `spatial_kernel_x` if distance was already
                calculated. Defaults to None.
            temporal_kernel (Kernel, optional): Temporal kernel used.
                Defaults to KernelSE().
            temporal_x_df (None | pd.DataFrame, optional): Temporal kernel input tensor
                used to calculate covariate distance. Vector of length equal to nb time
                points. Defaults to None.
            temporal_dist_df (None | pd.DataFrame, optional): Temporal kernel covariate
                distance. A two dimensions df (nb time points x nb time points).
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
                will be exported (if None only iteration data is printed). Defaults to None.
            results_export_suffix (str | None, optional): Suffix added at the end of the csv
                file name (if None, no suffix is added). Defaults to None.

        Raises:
            ValueError: If none or both `spatial_x_df` and `spatial_dist_df` are provided
            ValueError: If none or both `temporal_x_df` and `temporal_dist_df` are provided
            ValueError: If `y` index is different than `covariates_df` location index
            ValueError: If `y` columns are different than `covariates_df` time index
        """
        self._verify_input_labels(
            y_df,
            covariates_df,
            spatial_x_df,
            spatial_dist_df,
            temporal_x_df,
            temporal_dist_df,
        )

        # Sort all df indexes
        for df in [
            y_df,
            covariates_df,
            spatial_x_df,
            spatial_dist_df,
            temporal_x_df,
            temporal_dist_df,
        ]:
            if df is not None:
                df.sort_index(inplace=True)
        # Only a subset of dataframes need to have their columns sorted
        for df in [y_df, spatial_dist_df, temporal_dist_df]:
            if df is not None:
                df.sort_index(axis=1, inplace=True)

        # Set labels
        self.spatial_labels = (
            covariates_df.index.get_level_values(self.spatial_index_name).unique().to_list()
        )
        self.temporal_labels = (
            covariates_df.index.get_level_values(self.temporal_index_name).unique().to_list()
        )
        self.feature_labels = covariates_df.columns.to_list()
        # Tensor assignation
        self.omega = TSR.tensor(1 - y_df.isna().to_numpy())
        self.y = TSR.tensor(y_df.to_numpy(na_value=0))
        covariates = TSR.tensor(covariates_df.to_numpy())
        self.tau = 1 / TSR.tensor(sigma_r)
        temporal_x_tsr = TSR.get_df_tensor_or_none(temporal_x_df)
        temporal_dist_tsr = TSR.get_df_tensor_or_none(temporal_dist_df)
        spatial_x_tsr = TSR.get_df_tensor_or_none(spatial_x_df)
        spatial_dist_tsr = TSR.get_df_tensor_or_none(spatial_dist_df)

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
        self.results_export_suffix = results_export_suffix

        # Reshape covariates
        self._reshape_covariates(covariates, len(self.spatial_labels), len(self.temporal_labels))

        # Kernel assignation
        self.spatial_kernel = spatial_kernel
        self.temporal_kernel = temporal_kernel
        self.spatial_kernel.set_distance_matrix(spatial_x_tsr, spatial_dist_tsr)
        self.temporal_kernel.set_distance_matrix(temporal_x_tsr, temporal_dist_tsr)
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
    def _verify_kernel_labels(
        kernel_x: pd.DataFrame | None,
        kernel_dist: pd.DataFrame | None,
        expected_labels: set,
        kernel_type: Literal['spatial', 'temporal'],
    ):
        """Verify if kernel inputs are valid and align with covariates labels.

        Args:
            kernel_x (pd.DataFrame | None): Kernel x to be provided to a kernel.
            kernel_dist (pd.DataFrame | None): Kernel distance to be provided to kernel.
            expected_labels (set): List of spatial/temporal labels used in covariates.
            kernel_type (Literal[&#39;spatial&#39;, &#39;temporal&#39;]): Type of kernel.

        Raises:
            ValueError: If both or none of kernel_x and kernel_dist are provided.
            ValueError: If kernel_x is provided and its size is not appropriate.
            ValueError: If kernel_dist is provided and its size is not appropriate.
        """
        cov_related_indx_name = 'location' if kernel_type == 'spatial' else 'time'
        if (kernel_x is None) == (kernel_dist is None):
            raise ValueError(
                f'Either `{kernel_type}_kernel_x` or `{kernel_type}_kernel_dist` must be provided'
            )
        if kernel_x is not None and expected_labels != set(kernel_x.index):
            raise ValueError(
                f'`{kernel_type}_x` must have the same index as the covariates\''
                f' {cov_related_indx_name} index.'
            )
        if kernel_dist is not None and not (
            expected_labels == set(kernel_dist.index) == set(kernel_dist.columns)
        ):
            raise ValueError(
                f'`{kernel_type}_dist` index and columns must have the same values as the'
                f' covariates\' {cov_related_indx_name} index.'
            )

    @classmethod
    def _verify_input_labels(
        cls,
        y: pd.DataFrame,
        covariates_df: pd.DataFrame,
        spatial_kernel_x: pd.DataFrame | None,
        spatial_kernel_dist: pd.DataFrame | None,
        temporal_kernel_x: pd.DataFrame | None,
        temporal_kernel_dist: pd.DataFrame | None,
    ):
        """Verify the validity of BKTR dataframe inputs' labels

        Args:
            y (pd.DataFrame): y in __init__
            covariates_df (pd.DataFrame): covariates_df in __init__
            spatial_kernel_x (pd.DataFrame | None): spatial_kernel_x in __init__
            spatial_kernel_dist (pd.DataFrame | None): spatial_kernel_dist in __init__
            temporal_kernel_x (pd.DataFrame | None): temporal_kernel_x in __init__
            temporal_kernel_dist (pd.DataFrame | None): temporal_kernel_dist in __init__

        Raises:
            ValueError: If y index do not correspond with spatial covariates index
            ValueError: If y columns do not correspond with temporal covariates index
        """
        if covariates_df.index.names != ['location', 'time']:
            raise ValueError(
                'The covariates dataframe must have a [`location`, `time`] multiindex.'
            )
        loc_set = set(covariates_df.index.get_level_values('location'))
        time_set = set(covariates_df.index.get_level_values('time'))

        if len(covariates_df) != len(loc_set) * len(time_set):
            raise ValueError(
                'The covariates dataframe must have a row for every possible'
                ' combination of location and time. Even if values are missing (NaN).'
            )
        if loc_set != set(y.index):
            raise ValueError('The covariates location and the y dataframe index must be the same.')
        if time_set != set(y.columns):
            raise ValueError(
                'The covariates time index should hold the same values as the'
                'y dataframe column names.'
            )
        cls._verify_kernel_labels(spatial_kernel_x, spatial_kernel_dist, loc_set, 'spatial')
        cls._verify_kernel_labels(temporal_kernel_x, temporal_kernel_dist, time_set, 'temporal')

    def _reshape_covariates(
        self, covariate_tensor: torch.Tensor, nb_locations: int, nb_times: int
    ):
        """Reshape the covariate tensors into one single tensor and set it as a property

        Args:
            covariate_tensor (torch.Tensor): Tensor of covariates in a (M*N) x P shape
            nb_locations (int): The number of different locations (M)
            nb_times (int): The number of different times (N)
        """
        nb_covariates = covariate_tensor.shape[1]
        self.covariates_dim = {
            'nb_spaces': nb_locations,  # S
            'nb_times': nb_times,  # T
            'nb_covariates': 1 + nb_covariates,  # P
        }

        covs = covariate_tensor.reshape([nb_locations, nb_times, nb_covariates])
        intersect_covs = TSR.ones([nb_locations, nb_times, 1])
        self.covariates = torch.dstack([intersect_covs, covs])

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
            results_export_suffix=self.results_export_suffix,
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
