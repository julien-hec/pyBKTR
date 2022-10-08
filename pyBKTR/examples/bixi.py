import numpy as np
import torch
from pkg_resources import resource_stream

from pyBKTR.bktr import BKTRRegressor
from pyBKTR.bktr_config import BKTRConfig
from pyBKTR.distances import DIST_TYPE
from pyBKTR.kernel_generators import KernelMatern, KernelParameter, KernelPeriodic, KernelSE
from pyBKTR.tensor_ops import TSR
from pyBKTR.utils import load_numpy_array_from_csv


def run_bixi_bktr(
    results_export_dir: str,
    run_id_from: int = 1,
    run_id_to: int = 1,
    burn_in_iter: int = 10,
    max_iter: int = 20,
    torch_seed: int | None = None,
    torch_device: str = 'cpu',
    torch_dtype: torch.dtype = torch.float32,
) -> None:

    # Set tensor backend according to config
    TSR.set_params(torch_dtype, torch_device, torch_seed)

    def get_source_file_name(csv_name: str) -> str:
        return resource_stream(__name__, f'../data/{csv_name}.csv')

    date_columns = ['day', 'month', 'weekday']
    departure_matrix = load_numpy_array_from_csv(
        get_source_file_name('bike_station_departures'),
        columns_to_drop=date_columns,
        rows_to_keep=list(range(1, 197)),
        fill_na_with_zeros=True,
        transpose_matrix=True,
    )

    # TODO Check those additionnal data manipulations
    departure_matrix[departure_matrix > np.quantile(departure_matrix, 0.9)] = 0
    departure_matrix[:, [41, 132, 165]] = 0

    # Create response variable and omega from departure data
    bixi_y = departure_matrix / np.max(departure_matrix)
    bixi_omega = departure_matrix > 0

    bixi_weather_matrix = load_numpy_array_from_csv(
        get_source_file_name('montreal_weather_data'),
        columns_to_drop=date_columns,
        rows_to_keep=list(range(1, 197)),
        fill_na_with_zeros=True,
        min_max_normalization=True,
    )

    bixi_station_matrix = load_numpy_array_from_csv(
        get_source_file_name('bike_station_features'),
        columns_to_keep=list(range(8, 21)),
        fill_na_with_zeros=True,
        transpose_matrix=False,
        min_max_normalization=True,
    )

    bktr_config = BKTRConfig(
        rank_decomp=10,
        max_iter=max_iter,
        burn_in_iter=burn_in_iter,
        sampled_beta_indexes=[230, 450],
        sampled_y_indexes=[100, 325],
        results_export_dir=results_export_dir,
    )

    spatial_kernel_x = TSR.tensor(
        load_numpy_array_from_csv(
            get_source_file_name('bike_station_features'), columns_to_keep=[2, 3]
        )
    )
    temporal_kernel_x = TSR.arange(0, bixi_weather_matrix.shape[0])

    temporal_kernel = (
        KernelPeriodic(period_length=KernelParameter(7, 'period length', is_constant=True))
        * KernelSE()
    )
    spatial_kernel = KernelMatern(smoothness_factor=3, distance_type=DIST_TYPE.HAVERSINE)

    for _ in range(run_id_from, run_id_to + 1):
        bktr_regressor = BKTRRegressor(
            bktr_config,
            temporal_covariate_matrix=bixi_weather_matrix,
            spatial_covariate_matrix=bixi_station_matrix,
            y=bixi_y,
            omega=bixi_omega,
            spatial_kernel=spatial_kernel,
            spatial_kernel_x=spatial_kernel_x,
            temporal_kernel=temporal_kernel,
            temporal_kernel_x=temporal_kernel_x,
        )

        bktr_regressor.mcmc_sampling()
