from typing import Any

import pandas as pd
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
                return 1e-5
            case _:
                raise ValueError('The dtype used by TSR has no default mapped jitter value')

    @classmethod
    def set_params(cls, dtype: torch.TensorType, device: str, seed: int | None = None):
        cls.dtype = dtype
        cls.device = device
        if seed is not None:
            torch.manual_seed(seed)

    @staticmethod
    def kronecker_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        kron_prod = torch.einsum('ab,cd->acbd', [a, b])
        a_rows, a_cols = a.shape
        b_rows, b_cols = b.shape
        kron_shape = [a_rows * b_rows, a_cols * b_cols]
        return torch.reshape(kron_prod, kron_shape)

    @staticmethod
    def khatri_rao_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_cols, b_cols = a.shape[1], b.shape[1]
        if a_cols != b_cols:
            raise ValueError(
                'Matrices must have the same number of columns to perform'
                f'khatri rao product, got {a_cols} and {b_cols}'
            )
        return torch.reshape(torch.einsum('ac,bc->abc', [a, b]), [-1, a.shape[1]])

    @classmethod
    def tensor(cls, tensor_data: Any) -> torch.Tensor:
        return torch.tensor(tensor_data, dtype=cls.dtype, device=cls.device)

    @classmethod
    def eye(cls, n: int):
        return torch.eye(n, dtype=cls.dtype, device=cls.device)

    @classmethod
    def ones(cls, n: int | tuple[int]):
        return torch.ones(n, dtype=cls.dtype, device=cls.device)

    @classmethod
    def zeros(cls, n: int | tuple[int]):
        return torch.zeros(n, dtype=cls.dtype, device=cls.device)

    @classmethod
    def rand(cls, size: int | tuple[int]):
        return torch.rand(size, dtype=cls.dtype, device=cls.device)

    @classmethod
    def randn(cls, size: tuple[int]) -> torch.Tensor:
        return torch.randn(size, dtype=cls.dtype, device=cls.device)

    @classmethod
    def randn_like(cls, input_tensor: torch.Tensor):
        return torch.randn_like(input_tensor, dtype=cls.dtype, device=cls.device)

    @classmethod
    def arange(cls, start: int, end: int, step: int = 1):
        return torch.arange(start, end, step, dtype=cls.dtype, device=cls.device)

    @classmethod
    def get_df_tensor_or_none(cls, input_df: pd.DataFrame | None) -> torch.Tensor | None:
        """Util function to get none if a value is none or a tensor instance of the df.

        Args:
            input_df (pd.DataFrame | None): Input dataframe
        """
        if input_df is None:
            return None
        return cls.tensor(input_df.to_numpy())
