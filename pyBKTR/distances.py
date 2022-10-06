from enum import Enum

import torch


class DIST_TYPE(Enum):
    LINEAR = 'linear'
    EUCLIDEAN = 'euclidean'
    HAVERSINE = 'haversine'


class DistanceCalculator:
    @classmethod
    def get_matrix(cls, x: torch.Tensor, distance_type: DIST_TYPE, earth_radius=6371.2):
        x_u = x.unsqueeze(1)
        match distance_type:
            case DIST_TYPE.LINEAR:
                return x_u - x_u.t()

            case DIST_TYPE.EUCLIDEAN:
                return (x_u - x_u.transpose(0, 1)).pow(2).sum(2).sqrt()

            case DIST_TYPE.HAVERSINE:
                x_u = torch.deg2rad(x_u)
                x_ut = x_u.transpose(0, 1)
                dist = (x_u - x_ut).abs()
                a = (dist[:, :, 0] / 2).sin() ** 2 + x_u[:, :, 0].cos() * x_ut[:, :, 0].cos() * (
                    dist[:, :, 1] / 2
                ).sin() ** 2
                return earth_radius * 2 * torch.atan2(a.sqrt(), (1 - a).sqrt())
