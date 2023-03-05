from enum import Enum

import torch

EARTH_RADIUS_KM = 6371.2


class DIST_TYPE(Enum):
    LINEAR = 'linear'
    EUCLIDEAN = 'euclidean'
    HAVERSINE = 'haversine'
    DOT_PRODUCT = 'dot product'


class DistanceCalculator:
    @classmethod
    def get_matrix(cls, x: torch.Tensor, distance_type: DIST_TYPE, earth_radius=EARTH_RADIUS_KM):
        match distance_type:
            case DIST_TYPE.LINEAR:
                return cls.calc_linear_dist(x, x)
            case DIST_TYPE.EUCLIDEAN:
                return cls.calc_euclidean_dist(x, x)
            case DIST_TYPE.HAVERSINE:
                return cls.calc_haversine_dist(x, x, earth_radius)
            case DIST_TYPE.DOT_PRODUCT:
                return cls.calc_dotproduct_dist(x, x)

    @staticmethod
    def check_tensor_dimensions(
        x1: torch.Tensor,
        x2: torch.Tensor,
        expected_nb_dim: int,
        expected_last_dim_shape: None | int = None,
    ):
        if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
            raise ValueError('Distance params must be tensors')
        if not (x1.ndim == x2.ndim == expected_nb_dim):
            raise RuntimeError(f'Distance params should have {expected_nb_dim} dimension(s)')
        if expected_last_dim_shape is not None and not (
            expected_last_dim_shape == x1.shape[-1] == x2.shape[-1]
        ):
            raise RuntimeError(
                f'Distance params last dimension should contain {expected_last_dim_shape} elements'
            )

    @classmethod
    def calc_linear_dist(cls, x1: torch.Tensor, x2: torch.Tensor):
        cls.check_tensor_dimensions(x1, x2, expected_nb_dim=2, expected_last_dim_shape=1)
        return (x1 - x2.t()).abs()

    @classmethod
    def calc_euclidean_dist(cls, x1: torch.Tensor, x2: torch.Tensor):
        cls.check_tensor_dimensions(x1, x2, expected_nb_dim=2)
        xu1, xu2 = x1.unsqueeze(1), x2.unsqueeze(1)
        return (xu1 - xu2.transpose(0, 1)).pow(2).sum(2).sqrt()

    @classmethod
    def calc_haversine_dist(
        cls, x1: torch.Tensor, x2: torch.Tensor, earth_radius: float = EARTH_RADIUS_KM
    ):
        cls.check_tensor_dimensions(x1, x2, expected_nb_dim=2, expected_last_dim_shape=2)
        xu1, xu2 = x1.unsqueeze(1), x2.unsqueeze(1)
        xu1, xu2 = torch.deg2rad(xu1), torch.deg2rad(xu2)
        xu2 = xu2.transpose(0, 1)
        dist = (xu1 - xu2).abs()
        a = (dist[:, :, 0] / 2).sin() ** 2 + xu1[:, :, 0].cos() * xu2[:, :, 0].cos() * (
            dist[:, :, 1] / 2
        ).sin() ** 2
        return earth_radius * 2 * torch.atan2(a.sqrt(), (1 - a).sqrt())

    @classmethod
    def calc_dotproduct_dist(cls, x1: torch.Tensor, x2: torch.Tensor):
        if x1.shape != x2.shape:
            raise RuntimeError('Distance params should have same dimension')
        xu1, xu2 = x1.unsqueeze(0), x2.unsqueeze(1)
        dist = xu1 * xu2
        if x1.ndim > 1:
            dist = dist.sum(-1)
        return dist
