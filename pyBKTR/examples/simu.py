import torch
from pkg_resources import resource_stream

from pyBKTR.bktr import BKTRRegressor
from pyBKTR.bktr_config import BKTRConfig
from pyBKTR.distances import DIST_TYPE
from pyBKTR.kernel_generators import KernelMatern, KernelSE
from pyBKTR.tensor_ops import TSR
from pyBKTR.utils import load_numpy_array_from_csv


class SimuBKTRRegressor(BKTRRegressor):
    def __init__(self, *args, **kwargs):
        unrelated_covariates = kwargs.pop('unrelated_covariates')
        super().__init__(*args, **kwargs)
        unrelated_covariates = TSR.tensor(unrelated_covariates)
        self.covariates = torch.dstack([self.covariates, unrelated_covariates])
        # Add a layer of randomly generated and uncorelated covariates
        self.covariates_dim['nb_covariates'] = self.covariates_dim['nb_covariates'] + 1


def run_simu_bktr(
    results_export_dir: str,
    nb_runs: int = 1,
    burn_in_iter: int = 1000,
    max_iter: int = 1500,
    torch_seed: int | None = None,
    torch_device: str = 'cpu',
    torch_dtype: torch.dtype = torch.float64,
) -> None:

    bktr_config = BKTRConfig(
        rank_decomp=10,
        max_iter=max_iter,
        burn_in_iter=burn_in_iter,
        sampled_beta_indexes=[860, 1949, 5519, 6338, 9207, 10435, 15034, 15868, 16059, 16944],
        sampled_y_indexes=[52, 4046, 12979, 13129, 15673, 17807],
        results_export_dir=results_export_dir,
    )

    def get_source_file_name(csv_name: str) -> str:
        return resource_stream(__name__, f'../data/{csv_name}.csv')

    y_full_matrix = load_numpy_array_from_csv(
        get_source_file_name('simu_y_full'), has_header=False
    )
    omega = y_full_matrix != 0
    temporal_covariates = load_numpy_array_from_csv(
        get_source_file_name('simu_temporal_covariates'), has_header=False
    )
    spatial_covariates = load_numpy_array_from_csv(
        get_source_file_name('simu_spatial_covariates'), has_header=False
    )
    distance_matrix = load_numpy_array_from_csv(
        get_source_file_name('simu_distance_matrix'), has_header=False
    )
    unrelated_covariates = load_numpy_array_from_csv(
        get_source_file_name('simu_simulated_covariates'), has_header=False
    )

    time_resolution = 1 / 10
    temporal_kernel_x = TSR.arange(0, temporal_covariates.shape[0]) * time_resolution
    temporal_kernel = KernelSE(temporal_kernel_x)
    spatial_kernel = KernelMatern(
        TSR.tensor([0, 1]), smoothness_factor=3, distance_type=DIST_TYPE.LINEAR
    )  # TODO get x vector
    spatial_kernel.distance_matrix = TSR.tensor(distance_matrix)

    for i in range(1, nb_runs + 1):
        TSR.set_params(torch_dtype, torch_device, torch_seed * i)
        bktr_regressor = SimuBKTRRegressor(
            bktr_config,
            temporal_covariate_matrix=temporal_covariates,
            spatial_covariate_matrix=spatial_covariates,
            y=y_full_matrix,
            omega=omega,
            spatial_kernel=spatial_kernel,
            temporal_kernel=temporal_kernel,
            unrelated_covariates=unrelated_covariates,
        )
        bktr_regressor.mcmc_sampling()
