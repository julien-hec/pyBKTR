from typing import Any

import torch


class TSR:
    """
    Class containing all tensor operations used in BKTR. It serves also as a factory
    for all new tensors (giving the right type and device to each of them).
    """

    dtype: torch.dtype = torch.float64
    device: str = 'cpu'

    @classmethod
    @property
    def default_jitter(cls) -> float:
        match cls.dtype:
            case torch.float64:
                return 1e-8
            case torch.float32:
                return 1e-4
            case _:
                raise ValueError('The dtype used by TSR has no default mapped jitter value')

    @classmethod
    def set_params(
        cls,
        dtype: torch.TensorType | None = None,
        device: str | None = None,
        seed: int | None = None,
    ):
        if dtype is not None:
            cls.dtype = dtype
        if device is not None:
            cls.device = device
        if seed is not None:
            torch.manual_seed(seed)

    @classmethod
    def tensor(cls, tensor_data: Any) -> torch.Tensor:
        return torch.tensor(tensor_data, dtype=cls.dtype, device=cls.device)

    @classmethod
    def eye(cls, n: int):
        return torch.eye(n, dtype=cls.dtype, device=cls.device)

    @classmethod
    def ones(cls, tsr_dim: int | tuple[int]):
        return torch.ones(tsr_dim, dtype=cls.dtype, device=cls.device)

    @classmethod
    def zeros(cls, tsr_dim: int | tuple[int]):
        return torch.zeros(tsr_dim, dtype=cls.dtype, device=cls.device)

    @classmethod
    def rand(cls, tsr_dim: int | tuple[int]):
        return torch.rand(tsr_dim, dtype=cls.dtype, device=cls.device)

    @classmethod
    def randn(cls, tsr_dim: tuple[int]) -> torch.Tensor:
        return torch.randn(tsr_dim, dtype=cls.dtype, device=cls.device)

    @classmethod
    def randn_like(cls, input_tensor: torch.Tensor):
        return torch.randn_like(input_tensor, dtype=cls.dtype, device=cls.device)

    @classmethod
    def arange(cls, start: int, end: int, step: int = 1):
        return torch.arange(start, end, step, dtype=cls.dtype, device=cls.device)

    @staticmethod
    def kronecker_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.kron(a, b)

    @staticmethod
    def khatri_rao_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_cols, b_cols = a.shape[1], b.shape[1]
        if a_cols != b_cols:
            raise ValueError(
                'Matrices must have the same number of columns to perform'
                f'khatri rao product, got {a_cols} and {b_cols}'
            )
        return torch.reshape(torch.einsum('ac,bc->abc', [a, b]), [-1, a.shape[1]])
