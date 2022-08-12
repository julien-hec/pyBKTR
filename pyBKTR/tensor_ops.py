from typing import Any

import torch


class TSR:
    """
    Class containing all tensor operations used in BKTR
    """

    def __init__(self, dtype: torch.TensorType, device: str, seed: int):
        self.dtype = dtype
        self.device = device
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

    def tensor(self, tensor_data: Any) -> torch.Tensor:
        return torch.tensor(tensor_data, dtype=self.dtype, device=self.device)

    def eye(self, n: int):
        return torch.eye(n, dtype=self.dtype, device=self.device)

    def ones(self, n: int | tuple[int]):
        return torch.ones(n, dtype=self.dtype, device=self.device)

    def rand(self, size: int | tuple[int]):
        return torch.rand(size, dtype=self.dtype, device=self.device)

    def randn(self, size: tuple[int]) -> torch.Tensor:
        return torch.randn(size, dtype=self.dtype, device=self.device)

    def randn_like(self, input_tensor: torch.Tensor):
        return torch.randn_like(input_tensor, dtype=self.dtype, device=self.device)

    def arange(self, start: int, end: int, step: int = 1):
        return torch.arange(start, end, step, dtype=self.dtype, device=self.device)
