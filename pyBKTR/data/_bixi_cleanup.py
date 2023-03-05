from datetime import datetime

import numpy as np
import pandas as pd
from pkg_resources import resource_stream


def get_source_file_name(csv_name: str) -> str:
    return resource_stream('pyBKTR', f'data/{csv_name}.csv')


EXPORT_PATH = '/Users/juju/Projects/pyBKTR/pyBKTR/data/cleaned'
date_columns = ['day', 'month', 'weekday']

# Temporal features
df = pd.read_csv(get_source_file_name('montreal_weather_data'))
df.index = (
    df['day']
    .apply(lambda x: datetime.strptime(f'{2019}-{x}', '%Y-%j').strftime('%Y-%m-%d'))
    .to_list()
)
df = df.drop(columns=date_columns)
df.index.name = 'date'
df = df[1:197]
# Min Max scaling (0 - 1)
df = (df - df.min()) / (df.max() - df.min())
df.to_csv(f'{EXPORT_PATH}/montreal_weather_data.csv')

# Temporal Locations
df['time_index'] = range(0, len(df))
df = df[['time_index']]
df.to_csv(f'{EXPORT_PATH}/temporal_locations.csv')


# Spatial features
df = pd.read_csv(get_source_file_name('bike_station_features'))
df.index = df['Code'].astype('str') + ' - ' + df['name']
df.index.name = 'station'
s_df = df
df = df[df.columns[8:21]]
# Min Max scaling (0 - 1)
df = (df - df.min()) / (df.max() - df.min())
df.to_csv(f'{EXPORT_PATH}/bike_station_features.csv')

# Spatial Locations
df = s_df[['latitude', 'longitude']]
df.to_csv(f'{EXPORT_PATH}/spatial_locations.csv')

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
max_val = np.max(df.to_numpy(na_value=0))
df = df / max_val
df.index = station_names
df.index.name = 'station'
df.to_csv(f'{EXPORT_PATH}/bike_station_departures.csv')
