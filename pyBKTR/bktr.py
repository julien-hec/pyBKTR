from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from formulaic import Formula, model_matrix
from formulaic.errors import FormulaicError

from pyBKTR.kernels import Kernel, KernelMatern, KernelSE
from pyBKTR.likelihood_evaluator import MarginalLikelihoodEvaluator
from pyBKTR.plots import BKTRBetaPlotMaker
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
        'results_export_dir',
        'results_export_suffix',
        'formula',
        'spatial_labels',
        'temporal_labels',
        'feature_labels',
        'spatial_coord',
        'plot_maker',
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
    results_export_dir: str | None
    # Labels
    spatial_labels: list
    temporal_labels: list
    feature_labels: list
    spatial_coord: pd.DataFrame
    # Plot Maker
    plot_maker: BKTRBetaPlotMaker | None

    # Constant string needed for dataframe index names
    spatial_index_name = 'location'
    temporal_index_name = 'time'

    def __init__(
        self,
        rank_decomp: int,
        burn_in_iter: int,
        sampling_iter: int,
        data_df: pd.DataFrame,
        formula: None | str = None,
        spatial_kernel: Kernel = KernelMatern(smoothness_factor=3),
        spatial_x_df: None | pd.DataFrame = None,
        spatial_dist_df: None | pd.DataFrame = None,
        temporal_kernel: Kernel = KernelSE(),
        temporal_x_df: None | pd.DataFrame = None,
        temporal_dist_df: None | pd.DataFrame = None,
        sigma_r: float = 1e-2,
        a_0: float = 1e-6,
        b_0: float = 1e-6,
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
                50 rows (10 x 5). If formula is None, the dataframe should contain
                the response variable `Y` as the first column. Note that the covariate
                columns cannot contain NaN values, but the response variable can.
            formula (str | None, optional): A Wilkinson formula string to specify the
                response variate `Y` and the covariates to use (compatible with the Formulaic
                package).  If None, the first column of the data frame will be used as the
                response variable and all the other columns will be used as the covariates.
                Defaults to None.
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
            data_df,
            spatial_x_df,
            spatial_dist_df,
            temporal_x_df,
            temporal_dist_df,
        )

        # Sort all df indexes
        for df in [
            data_df,
            spatial_x_df,
            spatial_dist_df,
            temporal_x_df,
            temporal_dist_df,
        ]:
            if df is not None:
                df.sort_index(inplace=True)
        # Only a subset of dataframes need to have their columns sorted
        for df in [spatial_dist_df, temporal_dist_df]:
            if df is not None:
                df.sort_index(axis=1, inplace=True)

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
        if spatial_x_df is not None:
            self.spatial_coord = spatial_x_df.copy()
        # Tensor assignation
        y_arr = y_df.to_numpy().reshape(len(self.spatial_labels), len(self.temporal_labels))
        self.omega = TSR.tensor(1 - np.isnan(y_arr))
        self.y = TSR.tensor(np.where(np.isnan(y_arr), 0, y_arr))
        covariates = TSR.tensor(x_df.to_numpy())
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
            self._set_errors_and_sample_precision_tau(i)
            self._collect_iter_values(i)
        self._log_final_iter_results()

    @property
    def summary(self) -> str:
        """Returns a summary of the MCMC regressor results

        Raises:
            RuntimeError: If the MCMC sampling has not been run yet

        Returns:
            str: A summary of the MCMC regressor results containing information about the
                MCMC sampling process and the estimated model's parameters.
        """
        if self.result_logger is None:
            raise RuntimeError('Summary can only be accessed after MCMC sampling.')
        return self.result_logger.summary()

    @property
    def beta_covariates_summary_df(self) -> pd.DataFrame:
        if self.result_logger is None:
            raise RuntimeError(
                'Covariate summary dataframe can only be accessed after MCMC sampling.'
            )
        return self.result_logger.beta_covariates_summary_df

    @property
    def y_estimates(self) -> pd.DataFrame:
        if self.result_logger is None:
            raise RuntimeError('Y estimates can only be accessed after MCMC sampling.')
        y_est = self.result_logger.y_estimates_df.copy()
        y_est.iloc[:, 0].mask(self.omega.flatten().cpu() == 0, inplace=True)
        return y_est

    @property
    def imputed_y_estimates(self) -> pd.DataFrame:
        if self.result_logger is None:
            raise RuntimeError('Imputed Y estimates can only be accessed after MCMC sampling.')
        return self.result_logger.y_estimates_df

    @property
    def beta_estimates(self) -> pd.DataFrame:
        if self.result_logger is None:
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
        if self.result_logger is None:
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
        if self.result_logger is None:
            raise RuntimeError('Beta values can only be accessed after MCMC sampling.')
        return self.result_logger.get_beta_summary_df(
            spatial_labels, temporal_labels, feature_labels
        )

    @property
    def hyperparameters_per_iter_df(self):
        if self.result_logger is None:
            raise RuntimeError('Hyperparameters can only be accessed after MCMC sampling.')
        return self.result_logger.hyperparameters_per_iter_df

    def plot_temporal_betas(
        self,
        plot_feature_labels: list[str],
        spatial_point_label: str,
        show_figure: bool = True,
        fig_width: int = 850,
        fig_height: int = 550,
    ):
        """Create a plot of the beta values through time for a given spatial point and a set of
            feature labels.

        Args:
            plot_feature_labels (list[str]): List of feature labels to plot.
            spatial_point_label (str): Spatial point label to plot.
            show_figure (bool, optional): Whether to show the figure. Defaults to True.
            fig_width (int, optional): Figure width. Defaults to 850.
            fig_height (int, optional): Figure height. Defaults to 550.
        """
        if self.plot_maker is None:
            raise RuntimeError('Plots can only be accessed after MCMC sampling.')
        self.plot_maker.plot_temporal_betas(
            plot_feature_labels,
            spatial_point_label,
            show_figure,
            fig_width,
            fig_height,
        )

    def plot_spatial_betas(
        self,
        plot_feature_labels: list[str],
        temporal_point_label: str,
        geo_coordinates: pd.DataFrame | None = None,
        nb_cols: int = 1,
        mapbox_zoom: int = 9,
        use_dark_mode: bool = True,
        show_figure: bool = True,
        fig_width: int = 850,
        fig_height: int = 550,
    ):
        """Create a plot of beta values through space for a given temporal point and a set of
            feature labels.

        Args:
            plot_feature_labels (list[str]): List of feature labels to plot.
            temporal_point_label (str): Temporal point label to plot.
            geo_coordinates (pd.DataFrame): Geo coordinates dataframe. If None, the coordinates
                are deemed to be passed through the regressor `x_spatial_df`. Defaults to None.
            nb_cols (int, optional): Number of columns in the plot. Defaults to 1.
            mapbox_zoom (int, optional): Mapbox zoom. Defaults to 9.
            use_dark_mode (bool, optional): Whether to use dark mode. Defaults to True.
            show_figure (bool, optional): Whether to show the figure. Defaults to True.
            fig_width (int, optional): Figure width. Defaults to 850.
            fig_height (int, optional): Figure height. Defaults to 550.
        """
        if self.plot_maker is None:
            raise RuntimeError('Plots can only be accessed after MCMC sampling.')
        if geo_coordinates is None and self.spatial_coord is None:
            raise RuntimeError(
                'If `x_spatial_df` is not provided at the bktr regressor creation, then the'
                ' geo coordinates must be passed to the `create_spatial_beta_plot` method.'
            )
        geo_coord = geo_coordinates if geo_coordinates is not None else self.spatial_coord
        self.plot_maker.plot_spatial_betas(
            plot_feature_labels,
            temporal_point_label,
            geo_coord,
            nb_cols,
            mapbox_zoom,
            use_dark_mode,
            show_figure,
            fig_width,
            fig_height,
        )

    def plot_beta_dists(
        self,
        labels_list: list[tuple[Any, Any, Any]],
        show_figure: bool = True,
        fig_width: int = 900,
        fig_height: int = 600,
    ):
        """Plot the distribution of beta values for a given list of labels.

        Args:
            labels_list (list[tuple[Any, Any, Any]]): List of labels (spatial, temporal, feature)
                for which to plot the beta distribution through iterations.
            show_figure (bool, optional): Whether to show the figure. Defaults to True.
            fig_width (int, optional): Figure width. Defaults to 900.
            fig_height (int, optional): Figure height. Defaults to 600.
        """
        if self.plot_maker is None:
            raise RuntimeError('Plots can only be accessed after MCMC sampling.')
        betas_list = [self.get_iterations_betas(*lab) for lab in labels_list]
        return self.plot_maker.plot_beta_dists(
            labels_list, betas_list, show_figure, fig_width, fig_height
        )

    def plot_covariates_beta_dists(
        self,
        feature_labels: list[Any] | None = None,
        show_figure: bool = True,
        fig_width: int = 900,
        fig_height: int = 600,
    ):
        """Plot the distribution of beta estimates regrouped by covariates.

        Args:
            feature_labels (list[Any] | None, optional): List of feature labels labels for
                which to plot the beta estimates distribution. If None plot for all features.
            show_figure (bool, optional): Whether to show the figure. Defaults to True.
            fig_width (int, optional): Figure width. Defaults to 900.
            fig_height (int, optional): Figure height. Defaults to 600.
        """
        if self.plot_maker is None:
            raise RuntimeError('Plots can only be accessed after MCMC sampling.')
        return self.plot_maker.plot_covariates_beta_dists(
            feature_labels, show_figure, fig_width, fig_height
        )

    def plot_hyperparameters_distributions(
        self,
        hyperparameters: list[str] | None = None,
        show_figure: bool = True,
        fig_width: int = 900,
        fig_height: int = 600,
    ):
        """Plot the distribution of hyperparameters through iterations.

        Args:
            hyperparameters (list[str] | None, optional): List of hyperparameters to plot.
                If None, plot all hyperparameters. Defaults to None.
            show_figure (bool, optional): Whether to show the figure. Defaults to True.
            fig_width (int, optional): Figure width. Defaults to 900.
            fig_height (int, optional): Figure height. Defaults to 600.
        """
        if self.plot_maker is None:
            raise RuntimeError('Plots can only be accessed after MCMC sampling.')
        return self.plot_maker.plot_hyperparameters_distributions(
            hyperparameters, show_figure, fig_width, fig_height
        )

    def plot_hyperparameters_per_iter(
        self,
        hyperparameters: list[str] | None = None,
        show_figure: bool = True,
        fig_width: int = 800,
        fig_height: int = 550,
    ):
        """Plot the distribution of hyperparameters through iterations.

        Args:
            hyperparameters (list[str] | None, optional): List of hyperparameters to plot.
                If None, plot all hyperparameters. Defaults to None.
            show_figure (bool, optional): Whether to show the figure. Defaults to True.
            fig_width (int, optional): Figure width. Defaults to 1000.
            fig_height (int, optional): Figure height. Defaults to 600.
        """
        if self.plot_maker is None:
            raise RuntimeError('Plots can only be accessed after MCMC sampling.')
        return self.plot_maker.plot_hyperparameters_per_iter(
            hyperparameters, show_figure, fig_width, fig_height
        )

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
        data_df: pd.DataFrame,
        spatial_kernel_x: pd.DataFrame | None,
        spatial_kernel_dist: pd.DataFrame | None,
        temporal_kernel_x: pd.DataFrame | None,
        temporal_kernel_dist: pd.DataFrame | None,
    ):
        """Verify the validity of BKTR dataframe inputs' labels

        Args:
            data_df (pd.DataFrame): data_df in __init__
            spatial_kernel_x (pd.DataFrame | None): spatial_kernel_x in __init__
            spatial_kernel_dist (pd.DataFrame | None): spatial_kernel_dist in __init__
            temporal_kernel_x (pd.DataFrame | None): temporal_kernel_x in __init__
            temporal_kernel_dist (pd.DataFrame | None): temporal_kernel_dist in __init__

        Raises:
            ValueError: If y index do not correspond with spatial covariates index
            ValueError: If y columns do not correspond with temporal covariates index
        """
        if data_df.index.names != ['location', 'time']:
            raise ValueError('The data_df dataframe must have a [`location`, `time`] multiindex.')
        loc_set = set(data_df.index.get_level_values('location'))
        time_set = set(data_df.index.get_level_values('time'))

        if len(data_df) != len(loc_set) * len(time_set):
            raise ValueError(
                'The data_df dataframe must have a row for every possible'
                ' combination of location and time. Even if response values are missing (NaN).'
            )
        cls._verify_kernel_labels(spatial_kernel_x, spatial_kernel_dist, loc_set, 'spatial')
        cls._verify_kernel_labels(temporal_kernel_x, temporal_kernel_dist, time_set, 'temporal')

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

    def _set_errors_and_sample_precision_tau(self, iter: int):
        """Sample a new tau and set errors"""
        self.result_logger.set_y_and_beta_estimates(self._decomposition_tensors, iter)
        self.result_logger._set_error_metrics()
        self.tau = self.tau_sampler.sample(self.result_logger.total_sq_error)

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
        self.result_logger.collect_iter_samples(iter, float(self.tau))

    def _log_final_iter_results(self):
        self.result_logger.log_final_iter_results()
        self.plot_maker = BKTRBetaPlotMaker(
            self.result_logger.get_beta_summary_df,
            self.beta_estimates,
            self.hyperparameters_per_iter_df,
            self.spatial_labels,
            self.temporal_labels,
            self.feature_labels,
        )

    def _initialize_params(self):
        """Initialize all parameters that are needed before we start the MCMC sampling"""
        self._init_covariate_decomp()
        self._create_result_logger()
        self._create_likelihood_evaluators()
        self._create_hparam_samplers()
        # Calcultate first likelihoods
        self._calc_spatial_marginal_ll()
        self._calc_temporal_marginal_ll()
