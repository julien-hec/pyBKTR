from datetime import datetime

import numpy as np
import pandas as pd
from pkg_resources import resource_stream


def get_source_file_name(csv_name: str) -> str:
    return resource_stream('pyBKTR', f'data/raw_data/{csv_name}.csv')


EXPORT_PATH = '/Users/juju/Projects/pyBKTR/pyBKTR/data'
date_columns = ['day', 'month', 'weekday']

# Temporal features
df = pd.read_csv(get_source_file_name('montreal_weather_data'))
df.index = (
    df['day']
    .apply(lambda x: datetime.strptime(f'{2019}-{x}', '%Y-%j').strftime('%Y-%m-%d'))
    .to_list()
)
df = df.drop(columns=date_columns)
df.index.name = 'time'
df = df[1:197]
df = df.rename(
    columns={
        'Mean Temp (°C)': 'mean_temp_c',
        'Total Precip (mm)': 'total_precip_mm',
        'max_temp': 'max_temp_f',
    }
)
df.to_csv(f'{EXPORT_PATH}/bixi_temporal_features.csv')

# Temporal Locations
df['time_index'] = range(0, len(df))
df = df[['time_index']]
df.to_csv(f'{EXPORT_PATH}/bixi_temporal_locations.csv')


# Spatial features
df = pd.read_csv(get_source_file_name('bike_station_features'))
df.index = df['Code'].astype('str') + ' - ' + df['name']
df.index.name = 'location'
s_df = df
df = df[df.columns[8:21]]
df.to_csv(f'{EXPORT_PATH}/bixi_spatial_features.csv')

# Spatial Locations
df = s_df[['latitude', 'longitude']]
df.to_csv(f'{EXPORT_PATH}/bixi_spatial_locations.csv')

# Used to replace station names in bike_station_departures.csv
station_names = df.index.to_list()


# Departures (Response Variable)
df = pd.read_csv(get_source_file_name('bike_station_departures'))
df.index = (
    df['day']
    .apply(lambda x: datetime.strptime(f'{2019}-{x}', '%Y-%j').strftime('%Y-%m-%d'))
    .to_list()
)
df = df.drop(columns=date_columns)
df = df[1:197]
df = df.transpose()
q_90 = np.quantile(df.to_numpy(na_value=0), 0.9)
df = df.applymap(lambda x: np.NaN if x > q_90 else x)
df[['2019-05-26', '2019-08-25', '2019-09-27']] = np.NaN
df.index = station_names
df.index.name = 'location'
df.to_csv(f'{EXPORT_PATH}/bixi_station_departures.csv')

# # Min Max scaling (0 - 1)
# df = (df - df.min()) / (df.max() - df.min())

# max_val = np.max(df.to_numpy(na_value=0))
# df = df / max_val
