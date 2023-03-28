import pandas as pd
import torch
from pkg_resources import resource_stream

from pyBKTR.bktr import BKTRRegressor
from pyBKTR.distances import DIST_TYPE
from pyBKTR.kernels import KernelMatern, KernelParameter, KernelPeriodic, KernelSE
from pyBKTR.tensor_ops import TSR


class BixiData:
    departure_df: pd.DataFrame
    weather_df: pd.DataFrame
    station_df: pd.DataFrame
    spatial_x_df: pd.DataFrame
    temporal_x_df: pd.DataFrame
    covariates_df: pd.DataFrame

    def __init__(self):
        self.departure_df = self.get_source_df('bixi_station_departures')
        self.weather_df = self.get_source_df('bixi_montreal_weather')
        self.station_df = self.get_source_df('bixi_station_features')
        self.spatial_x_df = self.get_source_df('bixi_spatial_locations')
        self.temporal_x_df = self.get_source_df('bixi_temporal_locations')

        # Merge covariates df TODO move to utils function and const for index names
        spa_index_name = 'location'
        temp_index_name = 'time'
        station_df = self.station_df.copy()
        weather_df = self.weather_df.copy()
        station_df.index.name = spa_index_name
        station_df.reset_index(inplace=True)
        weather_df.index.name = temp_index_name
        weather_df.reset_index(inplace=True)
        self.covariates_df = pd.merge(station_df, weather_df, how='cross')
        self.covariates_df.set_index([spa_index_name, temp_index_name], inplace=True)
        self.covariates_df.sort_index(inplace=True, ascending=False)

    @staticmethod
    def get_source_df(csv_name: str) -> pd.DataFrame:
        file_name = resource_stream('pyBKTR', f'data/{csv_name}.csv')
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

    bixi_data = BixiData()

    temporal_kernel = (
        KernelPeriodic(period_length=KernelParameter(7, 'period length', is_constant=True))
        * KernelSE()
    )
    spatial_kernel = KernelMatern(smoothness_factor=3, distance_type=DIST_TYPE.HAVERSINE)

    for _ in range(run_id_from, run_id_to + 1):
        bktr_regressor = BKTRRegressor(
            temporal_covariates_df=bixi_data.weather_df,
            spatial_covariates_df=bixi_data.station_df,
            y_df=bixi_data.departure_df,
            rank_decomp=10,
            burn_in_iter=burn_in_iter,
            sampling_iter=sampling_iter,
            spatial_kernel=spatial_kernel,
            spatial_x_df=bixi_data.spatial_x_df,
            temporal_kernel=temporal_kernel,
            temporal_x_df=bixi_data.temporal_x_df,
            results_export_dir=results_export_dir,
            sampled_beta_indexes=[230, 450],
            sampled_y_indexes=[100, 325],
        )

        bktr_regressor.mcmc_sampling()
