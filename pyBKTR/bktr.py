import itertools
from copy import deepcopy
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from formulaic import Formula, model_matrix
from formulaic.errors import FormulaicError

from pyBKTR._likelihood_evaluator import MarginalLikelihoodEvaluator
from pyBKTR._result_logger import ResultLogger
from pyBKTR._samplers import (
    KernelParamSampler,
    PrecisionMatrixSampler,
    TauSampler,
    get_cov_decomp_chol,
    sample_norm_multivariate,
)
from pyBKTR.kernels import Kernel, KernelMatern, KernelSE
from pyBKTR.tensor_ops import TSR


class BKTRRegressor:
    """Class encapsulating the BKTR regression elements

    A BKTRRegressor holds all the key elements to accomplish the MCMC sampling
    algorithm (**Algorithm 1** of the paper).
    """

    __slots__ = [
        'data_df',
        'y',
        'omega',
        'covariates',
        'covariates_dim',
        'tau',
        'spatial_decomp',
        'temporal_decomp',
        'covs_decomp',
        'result_logger',
        'has_completed_sampling',
        'spatial_kernel',
        'temporal_kernel',
        'spatial_positions_df',
        'temporal_positions_df',
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
        'formula',
        'spatial_labels',
        'temporal_labels',
        'feature_labels',
    ]
    data_df: pd.DataFrame
    y: torch.Tensor
    omega: torch.Tensor
    covariates: torch.Tensor
    covariates_dim: dict[str, int]
    tau: float
    # Covariate decompositions (change during iter)
    spatial_decomp: torch.Tensor  # U
    temporal_decomp: torch.Tensor  # V
    covs_decomp: torch.Tensor  # C or W
    # Kernels
    spatial_kernel: Kernel
    temporal_kernel: Kernel
    spatial_positions_df: pd.DataFrame
    temporal_positions_df: pd.DataFrame
    # Result Logger
    result_logger: ResultLogger
    has_completed_sampling: bool
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
    # Labels
    spatial_labels: list
    temporal_labels: list
    feature_labels: list

    # Constant string needed for dataframe index names
    spatial_index_name = 'location'
    temporal_index_name = 'time'

    def __init__(
        self,
        data_df: pd.DataFrame,
        spatial_positions_df: pd.DataFrame,
        temporal_positions_df: pd.DataFrame,
        rank_decomp: int,
        burn_in_iter: int,
        sampling_iter: int,
        formula: None | str = None,
        spatial_kernel: Kernel = KernelMatern(smoothness_factor=3),
        temporal_kernel: Kernel = KernelSE(),
        sigma_r: float = 1e-2,
        a_0: float = 1e-6,
        b_0: float = 1e-6,
    ):
        """Create a new *BKTRRegressor* object.

        Args:
            data_df (pd.DataFrame):  A dataframe containing all the covariates
                through time and space. It is important that the dataframe has a two
                indexes named `location` and `time` respectively. The dataframe should
                also contain every possible combinations of `location` and `time`
                (i.e. even missing rows should be filled present but filled with NaN).
                So if the dataframe has 10 locations and 5 time points, it should have
                50 rows (10 x 5). If formula is None, the dataframe should contain
                the response variable `Y` as the first column. Note that the covariate
                columns cannot contain NaN values, but the response variable can.
            formula (str | None, optional): A Wilkinson formula string to specify the relation
                between the response variable `Y` and the covariates (compatible with the
                Formulaic package).  If None, the first column of the data frame will be
                used as the response variable and all the other columns will be used as the
                covariates.  Defaults to None.
            rank_decomp (int): Rank of the CP decomposition (Paper -- :math:`R`)
            burn_in_iter (int): Number of iteration before sampling (Paper -- :math:`K_1`).
            sampling_iter (int): Number of sampling iterations (Paper -- :math:`K_2`).
            spatial_positions_df (pd.DataFrame): Spatial kernel input
                tensor used to calculate covariates' distance. Vector of length equal to
                the number of location points.
            temporal_positions_df (pd.DataFrame): Temporal kernel input tensor
                used to calculate covariate distance. Vector of length equal to
                the number of time points.
            spatial_kernel (Kernel, optional): Spatial kernel Used.
                Defaults to KernelMatern(smoothness_factor=3).
            temporal_kernel (Kernel, optional): Temporal kernel used.
                Defaults to KernelSE().
            sigma_r (float, optional): Variance of the white noise process TODO
                (Paper -- :math:`\\tau^{-1}`). Defaults to 1e-2.
            a_0 (float, optional): Initial value for the shape (:math:`\\alpha`) in the gamma
                function generating tau. Defaults to 1e-6.
            b_0 (float, optional): Initial value for the rate (:math:`\\beta`) in the gamma
                function generating tau. Defaults to 1e-6.
        """
        self.has_completed_sampling = False
        self._verify_input_labels(data_df, spatial_positions_df, temporal_positions_df)

        # Sort all df indexes
        for df in [data_df, spatial_positions_df, temporal_positions_df]:
            df.sort_index(inplace=True)
        self.data_df = data_df
        self.spatial_positions_df = spatial_positions_df
        self.temporal_positions_df = temporal_positions_df

        # Set formula and get model's matrix
        y_df, x_df = self._get_x_and_y_dfs_from_formula(data_df, formula)

        # Set labels
        self.spatial_labels = (
            y_df.index.get_level_values(self.spatial_index_name).unique().to_list()
        )
        self.temporal_labels = (
            y_df.index.get_level_values(self.temporal_index_name).unique().to_list()
        )
        self.feature_labels = x_df.columns.to_list()
        # Tensor assignation
        y_arr = y_df.to_numpy().reshape(len(self.spatial_labels), len(self.temporal_labels))
        self.omega = TSR.tensor(1 - np.isnan(y_arr))
        self.y = TSR.tensor(np.where(np.isnan(y_arr), 0, y_arr))
        covariates = TSR.tensor(x_df.to_numpy())
        self.tau = 1 / TSR.tensor(sigma_r)

        # Param assignation
        self.rank_decomp = rank_decomp
        self.burn_in_iter = burn_in_iter
        self.sampling_iter = sampling_iter
        self.max_iter = self.burn_in_iter + self.sampling_iter
        self.a_0 = a_0
        self.b_0 = b_0

        # Reshape covariates
        self._reshape_covariates(covariates, len(self.spatial_labels), len(self.temporal_labels))

        # Kernel assignation
        self.spatial_kernel = spatial_kernel
        self.temporal_kernel = temporal_kernel
        self.spatial_kernel.set_positions(spatial_positions_df)
        self.temporal_kernel.set_positions(temporal_positions_df)
        # Create first kernels
        self.spatial_kernel.kernel_gen()
        self.temporal_kernel.kernel_gen()

    def mcmc_sampling(self) -> dict[str, float | torch.Tensor]:
        """Launch the MCMC sampling process for a predefined number of iterations

        1. Sample spatial kernel hyperparameters
        2. Sample temporal kernel hyperparameters
        3. Sample the precision matrix from a wishart distribution
        4. Sample a new spatial covariate decomposition
        5. Sample a new feature covariate decomposition
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
            self._set_errors_and_sample_precision_tau(i)
            self._collect_iter_values(i)
        self._log_final_iter_results()

    def predict(
        self,
        new_data_df: pd.DataFrame,
        new_spatial_positions_df: pd.DataFrame | None = None,
        new_temporal_positions_df: pd.DataFrame | None = None,
        jitter=None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predict the beta coefficients and response values for new data.

        Args:
            new_data_df (pd.DataFrame): New covariates. Must have the same columns as
                the covariates used to fit the model. The index should contain the combination
                of all old spatial coordinates with all new temporal coordinates, the combination
                of all new spatial coordinates with all old temporal coordinates, and the
                combination of all new spatial coordinates with all new temporal coordinates.
            new_spatial_positions_df (pd.DataFrame | None, optional): New spatial coordinates.
                If None, there should be no new spatial covariates. Defaults to None.
            new_temporal_positions_df (pd.DataFrame | None, optional): New temporal coordinates.
                If None, there should be no new temporal covariates. Defaults to None.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple of two dataframes. The first
            represents the beta forecasted for all new spatial locations or temporal points.
            The second represents the forecasted response for all new spatial locations or
            temporal points.
        """
        self._pred_valid_and_sort_data(
            new_data_df, new_spatial_positions_df, new_temporal_positions_df
        )

        spatial_positions_df = (
            pd.concat([self.spatial_positions_df, new_spatial_positions_df], axis=0)
            if new_spatial_positions_df is not None
            else self.spatial_positions_df
        )
        temporal_positions_df = (
            pd.concat([self.temporal_positions_df, new_temporal_positions_df], axis=0)
            if new_temporal_positions_df is not None
            else self.temporal_positions_df
        )
        data_df = pd.concat([self.data_df, new_data_df], axis=0)
        data_df_index = list(
            itertools.product(
                spatial_positions_df.index.to_list(), temporal_positions_df.index.to_list()
            )
        )
        data_df = data_df.loc[data_df_index, :]
        self._verify_input_labels(
            data_df,
            spatial_positions_df=spatial_positions_df,
            temporal_positions_df=temporal_positions_df,
        )
        all_betas = TSR.zeros(
            [
                len(spatial_positions_df),
                len(temporal_positions_df),
                len(self.formula.rhs),
                self.sampling_iter,
            ]
        )
        for i in range(self.sampling_iter):
            new_spa_decomp = self._pred_simu_new_decomp(
                'spatial', i, spatial_positions_df, new_spatial_positions_df, jitter
            )
            new_temp_decomp = self._pred_simu_new_decomp(
                'temporal', i, temporal_positions_df, new_temporal_positions_df, jitter
            )
            covs_decomp = self.result_logger.covs_decomp_per_iter[:, :, i]
            all_betas[:, :, :, i] = torch.einsum(
                'il,jl,kl->ijk', [new_spa_decomp, new_temp_decomp, covs_decomp]
            )

        new_betas = all_betas.mean(dim=-1)
        _, x_df = self._get_x_and_y_dfs_from_formula(data_df, self.formula)
        covariates = TSR.tensor(x_df.to_numpy()).reshape(
            [len(spatial_positions_df), len(temporal_positions_df), -1]
        )
        new_y_est = torch.einsum('ijk,ijk->ij', [new_betas, covariates])
        new_beta_df = pd.DataFrame(
            new_betas.flatten(0, 1),
            index=data_df_index,
            columns=x_df.columns,
        )
        new_y_df = pd.DataFrame(
            new_y_est.flatten(),
            index=data_df_index,
            columns=['y'],
        )
        return new_y_df, new_beta_df

    @property
    def summary(self) -> str:
        """Returns a summary of the MCMC regressor results

        Raises:
            RuntimeError: If the MCMC sampling has not been run yet

        Returns:
            str: A summary of the MCMC regressor results containing information about the
                MCMC sampling process and the estimated model's parameters.
        """
        if not self.has_completed_sampling:
            raise RuntimeError('Summary can only be accessed after MCMC sampling.')
        return self.result_logger.summary()

    @property
    def beta_covariates_summary_df(self) -> pd.DataFrame:
        if not self.has_completed_sampling:
            raise RuntimeError(
                'Covariate summary dataframe can only be accessed after MCMC sampling.'
            )
        return self.result_logger.beta_covariates_summary_df

    @property
    def y_estimates(self) -> pd.DataFrame:
        if not self.has_completed_sampling:
            raise RuntimeError('Y estimates can only be accessed after MCMC sampling.')
        y_est = self.result_logger.y_estimates_df.copy()
        y_est.iloc[:, 0].mask(self.omega.flatten().cpu() == 0, inplace=True)
        return y_est

    @property
    def imputed_y_estimates(self) -> pd.DataFrame:
        if not self.has_completed_sampling:
            raise RuntimeError('Imputed Y estimates can only be accessed after MCMC sampling.')
        return self.result_logger.y_estimates_df

    @property
    def beta_estimates(self) -> pd.DataFrame:
        if not self.has_completed_sampling:
            raise RuntimeError('Beta estimates can only be accessed after MCMC sampling.')
        return self.result_logger.beta_estimates_df

    def get_iterations_betas(
        self, spatial_label: Any, temporal_label: Any, feature_label: Any
    ) -> list[float]:
        """Return all sampled betas through sampling iterations for a given set of spatial,
            temporal and feature labels. Useful for plotting the distribution
            of sampled beta values.

        Args:
            spatial_label (Any): The spatial label for which we want to get the betas
            temporal_label (Any): The temporal label for which we want to get the betas
            feature_label (Any): The feature label for which we want to get the betas

        Returns:
            list[float]: The sampled betas through iteration for the given labels
        """
        if not self.has_completed_sampling:
            raise RuntimeError('Beta values can only be accessed after MCMC sampling.')
        beta_per_iter_tensor = self.result_logger.get_iteration_betas_tensor(
            [spatial_label], [temporal_label], [feature_label]
        )[0]
        return list(beta_per_iter_tensor.numpy())

    def get_beta_summary_df(
        self,
        spatial_labels: list[Any] = None,
        temporal_labels: list[Any] = None,
        feature_labels: list[Any] = None,
    ) -> pd.DataFrame:
        """Get a summary of estimated beta values. If no labels are given, then
        the summary is for all the betas. If labels are given, then the summary
        is for the given labels.

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
        if not self.has_completed_sampling:
            raise RuntimeError('Beta values can only be accessed after MCMC sampling.')
        return self.result_logger.get_beta_summary_df(
            spatial_labels, temporal_labels, feature_labels
        )

    @property
    def hyperparameters_per_iter_df(self):
        if not self.has_completed_sampling:
            raise RuntimeError('Hyperparameters trace can only be accessed after MCMC sampling.')
        return self.result_logger.hyperparameters_per_iter_df

    @staticmethod
    def _verify_kernel_labels(
        kernel_positions: pd.DataFrame,
        expected_labels: set,
        kernel_type: Literal['spatial', 'temporal'],
    ):
        """Verify if kernel inputs are valid and align with covariates labels.

        Args:
            kernel_positions (pd.DataFrame): Kernel position vector to be provided (math:`x`).
            expected_labels (set): List of spatial/temporal labels used in covariates.
            kernel_type (Literal[&#39;spatial&#39;, &#39;temporal&#39;]): Type of kernel.

        Raises:
            ValueError: If kernel_positions size is not appropriate.
        """
        cov_related_indx_name = 'location' if kernel_type == 'spatial' else 'time'
        if kernel_positions.index.name != cov_related_indx_name:
            raise ValueError(
                f'`{kernel_type}_positions_df` must have a `{cov_related_indx_name}` index.'
            )
        if expected_labels != set(kernel_positions.index):
            raise ValueError(
                f'`{kernel_type}_positions_df` must contain in its {cov_related_indx_name}',
                f'index the unique values located in `data_df` {cov_related_indx_name} index.',
            )

    @classmethod
    def _verify_input_labels(
        cls,
        data_df: pd.DataFrame,
        spatial_positions_df: pd.DataFrame,
        temporal_positions_df: pd.DataFrame,
    ):
        """Verify the validity of BKTR dataframe inputs' labels

        Args:
            data_df (pd.DataFrame): The data_df from BKTRRegressor __init__
            spatial_positions_df (pd.DataFrame | None): spatial_positions_df from __init__
            temporal_positions_df (pd.DataFrame | None): temporal_positions_df from __init__

        Raises:
            ValueError: If data_df does not contain a multiindex with ['location', 'time']
            ValueError: If data_df location index do not correspond with spatial_positions_df
            ValueError: If data_df time index do not correspond with temporal_positions_df
        """
        if data_df.index.names != ['location', 'time']:
            raise ValueError('The data_df dataframe must have a [`location`, `time`] multiindex.')
        loc_set = set(data_df.index.get_level_values('location'))
        time_set = set(data_df.index.get_level_values('time'))
        product_set = set(itertools.product(loc_set, time_set))
        data_df_index_set = set(data_df.index)

        if data_df_index_set != product_set:
            raise ValueError(
                'The data_df dataframe must have a row for every possible'
                ' combination of location and time. Even if response values are missing (NaN).'
            )
        cls._verify_kernel_labels(spatial_positions_df, loc_set, 'spatial')
        cls._verify_kernel_labels(temporal_positions_df, time_set, 'temporal')

    def _get_x_and_y_dfs_from_formula(
        self, data_df: pd.DataFrame, formula: str | None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Use formula to get x and y dataframes.

        Args:
            data_df (pd.DataFrame): The initial dataframe used to obtain the x and y dataframes.
            formula (str | None): Formula to give the y and X dataframes matrix. If formula is
                None, use the first column as y and all other columns as covariates.

        Raises:
            ValueError: The formula provided is not valid according to the formulaic package.
            ValueError: The formula provided does not contain 1 response variable.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The y and x dataframes respectively.
        """
        try:
            if formula is None:
                covariate_cols = [f'`{x}`' for x in data_df.columns[1:].to_list()]
                covariate_cols_str = ' + '.join(covariate_cols)
                self.formula = Formula(f'{data_df.columns[0]} ~ {covariate_cols_str}')
            else:
                self.formula = Formula(formula)

            y, x = model_matrix(self.formula, data_df, na_action='ignore')
        except FormulaicError as e:
            raise ValueError(
                'An error linked to the formula parameter occured.'
                ' Please check the formula syntax to comply with the formulaic package.'
                '\nFor more info see: https://matthewwardrop.github.io/formulaic/guides/'
                f'\nThe Formulaic error was:\n\t{e}'
            )
        y, x = pd.DataFrame(y), pd.DataFrame(x)
        if len(y.columns) != 1:
            raise ValueError(
                'The formula provided to the regressor is not valid.'
                ' It must contain one and only one response variable.'
            )
        return y, x

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
            'nb_covariates': nb_covariates,  # P
        }
        self.covariates = covariate_tensor.reshape([nb_locations, nb_times, nb_covariates])

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
            nb_burn_in_iter=self.burn_in_iter,
            nb_sampling_iter=self.sampling_iter,
            rank_decomp=self.rank_decomp,
            formula=self.formula,
            spatial_labels=self.spatial_labels,
            temporal_labels=self.temporal_labels,
            feature_labels=self.feature_labels,
            spatial_kernel=self.spatial_kernel,
            temporal_kernel=self.temporal_kernel,
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
            self.spatial_kernel.covariance_matrix, self.temporal_decomp, self.covs_decomp, self.tau
        )

    def _calc_temporal_marginal_ll(self):
        """Calculate the temporal marginal likelihood"""
        return self.temporal_ll_evaluator.calc_likelihood(
            self.temporal_kernel.covariance_matrix, self.spatial_decomp, self.covs_decomp, self.tau
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

    def _set_errors_and_sample_precision_tau(self, iter: int):
        """Sample a new tau and set errors"""
        self.result_logger.set_y_and_beta_estimates(self._decomposition_tensors, iter)
        self.result_logger.set_error_metrics()
        self.tau = self.tau_sampler.sample(self.result_logger.total_sq_error)

    @property
    def _decomposition_tensors(self):
        """Get a dict of current iteration decomposition needed to calculate estimated betas"""
        # TODO this could be arguments in the result logger
        return {
            'spatial_decomp': self.spatial_decomp,
            'temporal_decomp': self.temporal_decomp,
            'covs_decomp': self.covs_decomp,
        }

    def _collect_iter_values(self, iter: int):
        """Collect current iteration values inside the historical data tensor list"""
        self.result_logger.collect_iter_samples(iter, float(self.tau))

    def _log_final_iter_results(self):
        self.result_logger.log_final_iter_results()
        self.has_completed_sampling = True

    def _initialize_params(self):
        """Initialize all parameters that are needed before we start the MCMC sampling"""
        self._init_covariate_decomp()
        self._create_result_logger()
        self._create_likelihood_evaluators()
        self._create_hparam_samplers()
        # Calcultate first likelihoods
        self._calc_spatial_marginal_ll()
        self._calc_temporal_marginal_ll()

    def _pred_simu_new_decomp(
        self,
        pred_type: Literal['spatial', 'temporal'],
        iter_no: int,
        position_df: pd.DataFrame,
        new_position_df: pd.DataFrame | None,
        jitter: float | None,
    ) -> torch.Tensor:
        """Predict new decomposition values for a given iteration using the interpolation
            algorithm described in the paper.

        Args:
            pred_type (Literal[&#39;spatial&#39;, &#39;temporal&#39;]): Type of prediction.
            iter_no (int): Iteration number for which we want to predict new decomposition values.
            position_df (pd.DataFrame): The position dataframe containing the positions (locations
                or time points depending on the prediction type) used during the MCMC sampling.
            new_position_df (pd.DataFrame | None): The position dataframe containing the positions
            (locations or time points) for which we want to predict new decomposition values.
            jitter (float | None): The jitter value to use for the interpolation algorithm.

        Returns:
            torch.Tensor: A tensor containing the predicted decomposition values for the new
                positions.
        """
        old_decomp = (
            self.result_logger.spatial_decomp_per_iter[:, :, iter_no]
            if pred_type == 'spatial'
            else self.result_logger.temporal_decomp_per_iter[:, :, iter_no]
        )
        if new_position_df is None:
            return old_decomp
        nb_new_pos = len(new_position_df)
        old_kernel = self.spatial_kernel if pred_type == 'spatial' else self.temporal_kernel
        new_kernel = deepcopy(old_kernel)
        for param in new_kernel.parameters:
            if not param.is_fixed:
                param_full_repr = f'{pred_type.capitalize()} - {param.full_name}'
                param.value = self.result_logger.hyperparameters_per_iter_df.loc[
                    iter_no + 1, param_full_repr
                ]
        new_kernel.set_positions(position_df)
        cov_mat = new_kernel.kernel_gen()
        old_cov = cov_mat[:-nb_new_pos, :-nb_new_pos]
        new_old_cov = cov_mat[-nb_new_pos:, :-nb_new_pos]
        old_new_cov = cov_mat[:-nb_new_pos, -nb_new_pos:]
        new_cov = cov_mat[-nb_new_pos:, -nb_new_pos:]
        new_decomp_mus = new_old_cov @ old_cov.inverse() @ old_decomp
        new_decomp_cov = new_cov - (new_old_cov @ old_cov.inverse() @ old_new_cov)
        new_decomp_cov = (new_decomp_cov + new_decomp_cov.T) / 2
        if jitter is not None:
            new_decomp_cov += jitter * torch.eye(new_decomp_cov.shape[0])
        new_decomp = (
            torch.distributions.MultivariateNormal(new_decomp_mus.T, new_decomp_cov).sample().T
        )
        return torch.concat([old_decomp, new_decomp], dim=0)

    @staticmethod
    def _check_pred_dfs_integrity(
        old_df_name: str, old_df: pd.DataFrame, new_df: pd.DataFrame
    ) -> None:
        """Check that the new dataframe has the same columns and index as the old one,
        that the new df index labels are unique and that the new df index labels are
        not a subset of the old df index labels.

        Args:
            old_df_name (str): Previous name for the dataframe use in regression.
            old_df (pd.DataFrame): Previous dataframe use in regression.
            new_df (pd.DataFrame): New dataframe use in for prediction.
        """
        if old_df.columns.to_list() != new_df.columns.to_list():
            raise ValueError(
                f'The `new_{old_df_name}` columns should correspond with `{old_df_name}` columns.'
            )
        new_pos_labels_set = set(new_df.index.unique().to_list())
        old_pos_labels_set = set(old_df.index.unique().to_list())
        if len(old_pos_labels_set) != len(old_df):
            raise ValueError(f'All index labels in new_{old_df_name} should be unique.')
        if new_pos_labels_set & old_pos_labels_set:
            raise ValueError(
                f'The index labels in new_{old_df_name} should not exists in {old_df_name}.'
            )

    def _pred_valid_and_sort_data(
        self,
        new_data_df: pd.DataFrame,
        new_spatial_positions_df: pd.DataFrame | None,
        new_temporal_positions_df: pd.DataFrame | None,
    ) -> None:
        """Check that the new data and covariates are valid and sort them by index.

        Args:
            new_data_df (pd.DataFrame): A dataframe with the new data to predict.
            new_spatial_positions_df (pd.DataFrame | None): A dataframe with new spatial locations.
            new_temporal_positions_df (pd.DataFrame | None): A dataframe with new time positions.
        """
        if new_spatial_positions_df is None and new_temporal_positions_df is None:
            raise ValueError(
                'At least one of new_positions_spatial_df and'
                ' new_positions_temporal_df must be provided.'
            )
        for df in [
            new_data_df,
            new_spatial_positions_df,
            new_temporal_positions_df,
        ]:
            if df is not None:
                df.sort_index(inplace=True)

        self._check_pred_dfs_integrity('data_df', self.data_df, new_data_df)
        data_df_labs_set = set(new_data_df.index.to_list())

        # Check spatial locations dataframes integrity
        x_spa_loc_labels = None
        new_spa_needed_indx = set()
        if new_spatial_positions_df is not None:
            self._check_pred_dfs_integrity(
                'spatial_positions_df', self.spatial_positions_df, new_spatial_positions_df
            )
            x_spa_loc_labels = new_spatial_positions_df.index.unique().to_list()
            old_time_labels = self.temporal_positions_df.index.to_list()
            new_spa_needed_indx = set(itertools.product(x_spa_loc_labels, old_time_labels))
            if not new_spa_needed_indx.issubset(data_df_labs_set):
                raise ValueError(
                    'The index of `new_data_df` should include the combination of all previous'
                    ' time points and all new locations provided in `new_spatial_positions_df`.'
                )

        # Check temporal points dataframes integrity
        x_temp_loc_labels = None
        new_temp_needed_indx = set()
        if new_temporal_positions_df is not None:
            self._check_pred_dfs_integrity(
                'temporal_positions_df', self.temporal_positions_df, new_temporal_positions_df
            )
            x_temp_loc_labels = new_temporal_positions_df.index.unique().to_list()
            old_spa_labels = self.spatial_positions_df.index.to_list()
            new_temp_needed_indx = set(itertools.product(old_spa_labels, x_temp_loc_labels))
            if not new_temp_needed_indx.issubset(data_df_labs_set):
                raise ValueError(
                    'The index of `new_data_df` should include the combination of all previous'
                    ' locations and all new time points provided in `new_temporal_positions_df`.'
                )

        # Check combination of new spatial and temporal dataframes integrity
        combi_needed_indx = set()
        if x_spa_loc_labels and x_temp_loc_labels:
            combi_needed_indx = set(itertools.product(x_spa_loc_labels, x_temp_loc_labels))
            if not combi_needed_indx.issubset(data_df_labs_set):
                raise ValueError(
                    'The index of`new_data_df` should include the combination of all new time'
                    'points in `new_temporal_positions_df` and all new locations'
                    ' in `new_spatial_positions_df`.'
                )

        if data_df_labs_set != new_spa_needed_indx | new_temp_needed_indx | combi_needed_indx:
            raise ValueError(
                'The index of `data_df` should only contain:'
                '\n\t1. The combination of all new locations and old time points;'
                '\n\t2. The combination of all new time points and old locations;'
                '\n\t3. The combination of all new time poins and new locations;'
            )
