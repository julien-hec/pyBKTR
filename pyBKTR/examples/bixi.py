import numpy as np
from pkg_resources import resource_stream

from pyBKTR.bktr import BKTRRegressor
from pyBKTR.bktr_config import BKTRConfig
from pyBKTR.utils import load_numpy_array_from_csv, log


def run_bixi_bktr(
    results_export_dir: str,
    run_id_from: int = 1,
    run_id_to: int = 1,
    burn_in_iter: int = 100,
    max_iter: int = 200,
) -> None:

    for r_id in range(run_id_from, run_id_to + 1):
        bktr_config = BKTRConfig(
            rank_decomp=10,
            max_iter=max_iter,
            burn_in_iter=burn_in_iter,
            decay_max_hparam_val=log(2),
            period_max_hparam_val=log(2),
            sampled_beta_indexes=[230, 450],
            sampled_y_indexes=[100, 325],
            results_export_dir=results_export_dir,
        )

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

        bixi_distance_matrix = load_numpy_array_from_csv(
            get_source_file_name('bike_station_distances'), has_header=False
        )

        bktr_regressor = BKTRRegressor(
            bktr_config,
            temporal_covariate_matrix=bixi_weather_matrix,
            spatial_covariate_matrix=bixi_station_matrix,
            spatial_distance_matrix=bixi_distance_matrix,
            y=bixi_y,
            omega=bixi_omega,
        )

        # import cProfile
        # cProfile.run('bktr_regressor.mcmc_sampling()')
        # bktr_result = bktr_regressor.mcmc_sampling()
        bktr_regressor.mcmc_sampling()
