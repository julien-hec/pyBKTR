import numpy as np
import pandas as pd
from pkg_resources import resource_stream

from pyBKTR.utils import reshape_covariate_dfs


class BixiData:
    departure_df: pd.DataFrame
    weather_df: pd.DataFrame
    station_df: pd.DataFrame
    spatial_positions_df: pd.DataFrame
    temporal_positions_df: pd.DataFrame
    data_df: pd.DataFrame

    def __init__(self):
        self.departure_df = self.get_source_df('bixi_station_departures')
        # Normalize departure counts
        max_val = np.max(self.departure_df.to_numpy(na_value=0))
        self.departure_df = self.departure_df / max_val

        w_df = self.get_source_df('bixi_temporal_features')
        # Normalize each column of the weather data
        self.weather_df = (w_df - w_df.min()) / (w_df.max() - w_df.min())

        s_df = self.get_source_df('bixi_spatial_features')
        # Normalize each column of the station data
        self.station_df = (s_df - s_df.min()) / (s_df.max() - s_df.min())

        self.spatial_positions_df = self.get_source_df('bixi_spatial_locations')
        self.temporal_positions_df = self.get_source_df('bixi_temporal_locations')
        self.data_df = reshape_covariate_dfs(
            spatial_df=self.station_df,
            temporal_df=self.weather_df,
            y_df=self.departure_df,
            y_column_name='nb_departure',
        )

    @staticmethod
    def get_source_df(csv_name: str) -> pd.DataFrame:
        file_name = resource_stream('pyBKTR', f'data/{csv_name}.csv')
        return pd.read_csv(file_name, index_col=0)
