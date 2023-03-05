import pandas as pd
import torch
from pkg_resources import resource_stream

from pyBKTR.bktr import BKTRRegressor
from pyBKTR.distances import DIST_TYPE
from pyBKTR.kernels import KernelMatern, KernelParameter, KernelPeriodic, KernelSE
from pyBKTR.tensor_ops import TSR


def get_source_df(csv_name: str) -> pd.DataFrame:
    file_name = resource_stream(__name__, f'../data/cleaned/{csv_name}.csv')
    return pd.read_csv(file_name, index_col=0)


def run_bixi_bktr(
    results_export_dir: str,
    run_id_from: int = 1,
    run_id_to: int = 1,
    burn_in_iter: int = 10,
    sampling_iter: int = 10,
    torch_seed: int | None = None,
    torch_device: str = 'cpu',
    torch_dtype: torch.dtype = torch.float32,
) -> None:

    # Set tensor backend according to config
    TSR.set_params(torch_dtype, torch_device, torch_seed)

    departure_df = get_source_df('bike_station_departures')
    weather_df = get_source_df('montreal_weather_data')
    station_df = get_source_df('bike_station_features')
    spatial_x = get_source_df('spatial_locations')
    temporal_x = get_source_df('temporal_locations')

    temporal_kernel = (
        KernelPeriodic(period_length=KernelParameter(7, 'period length', is_constant=True))
        * KernelSE()
    )
    spatial_kernel = KernelMatern(smoothness_factor=3, distance_type=DIST_TYPE.HAVERSINE)

    for _ in range(run_id_from, run_id_to + 1):
        bktr_regressor = BKTRRegressor(
            temporal_covariate=weather_df,
            spatial_covariate=station_df,
            y=departure_df,
            rank_decomp=10,
            burn_in_iter=burn_in_iter,
            sampling_iter=sampling_iter,
            spatial_kernel=spatial_kernel,
            spatial_x=spatial_x,
            temporal_kernel=temporal_kernel,
            temporal_x=temporal_x,
            results_export_dir=results_export_dir,
            sampled_beta_indexes=[230, 450],
            sampled_y_indexes=[100, 325],
        )

        bktr_regressor.mcmc_sampling()
