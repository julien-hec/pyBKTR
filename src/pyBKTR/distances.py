import torch

from pyBKTR.tensor_ops import TSR


def check_dist_tensor_dimensions(
    x1: torch.Tensor,
    x2: torch.Tensor,
    expected_nb_dim: int = 2,
    expected_last_dim_shape: int | None = None,
):
    """Check that two tensors have valid dimensions for distance computation."""
    if not (TSR.is_tensor(x1) and TSR.is_tensor(x2)):
        raise ValueError('Distance params must be tensors')
    if not (x1.ndim == x2.ndim and x2.ndim == expected_nb_dim):
        raise ValueError(f'Distance params should have {expected_nb_dim} dimension(s)')
    if expected_last_dim_shape is not None and not (
        expected_last_dim_shape == x1.shape[-1] == x2.shape[-1]
    ):
        raise ValueError(
            f'Distance params last dimension should contain {expected_last_dim_shape} elements'
        )


def get_euclidean_dist_tsr(x: torch.Tensor) -> torch.Tensor:
    """Function to compute the euclidean distance between a tensor and its transpose."""
    check_dist_tensor_dimensions(x, x)
    xu1, xu2 = x.unsqueeze(1), x.unsqueeze(1).transpose(0, 1)
    return (xu1 - xu2).pow(2).sum(2).sqrt()


class GeoMercatorProjector:
    """Class to project coordinates with mercator projection on a 2D plane
    Project coordinates with mercator projection on a 2D plane for
    a given scale. Keep track of the scale and the center of the projection to
    be able to project new coordinates which is useful during interpolation.
    """

    ini_df = None
    x_mid_point = None
    y_mid_point = None
    coords_scale = None
    scale = None
    scaled_ini_df = None
    EARTH_RADIUM_KM = 6371

    def __init__(self, df, scale=10.0):
        self.ini_df = df
        km_df = self._km_from_coords_df(df)
        lon_x = km_df['lon_x']
        lat_y = km_df['lat_y']
        x_min, x_max = min(lon_x), max(lon_x)
        y_max, y_min = max(lat_y), min(lat_y)
        self.x_mid_point = (x_min + x_max) / 2
        self.y_mid_point = (y_min + y_max) / 2
        self.coords_scale = max(x_max - x_min, y_max - y_min)
        self.scale = scale
        self.scaled_ini_df = self._scale_and_center_df(km_df)

    def project_new_coords(self, df):
        km_df = self._km_from_coords_df(df)
        return self._scale_and_center_df(km_df)

    def _scale_and_center_df(self, df):
        new_df = df.copy()
        scaling_factor = self.scale / self.coords_scale
        new_df['lon_x'] = (new_df['lon_x'] - self.x_mid_point) * scaling_factor
        new_df['lat_y'] = (new_df['lat_y'] - self.y_mid_point) * scaling_factor
        return new_df

    def _km_from_coords_df(self, df):
        if not ('latitude' in df.columns and 'longitude' in df.columns):
            raise ValueError('Dataframe must have columns "latitude" and "longitude"')
        new_df = df.copy()
        lons = TSR.tensor(df['longitude'].values)
        lats = TSR.tensor(df['latitude'].values)
        x = (self.EARTH_RADIUM_KM / (2 * torch.pi)) * torch.deg2rad(lons)
        merc_n_y = torch.log(torch.tan(torch.pi / 4 + torch.deg2rad(lats) / 2))
        y = (self.EARTH_RADIUM_KM / (2 * torch.pi)) * merc_n_y
        new_df['lon_x'] = x.cpu().numpy()
        new_df['lat_y'] = y.cpu().numpy()
        new_df = new_df.drop(columns=['latitude', 'longitude'])
        return new_df
