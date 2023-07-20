from typing import Any, Literal

import torch


class TSR:
    """
    Class containing all tensor operations used in BKTR. It serves also as a factory
    for all new tensors (giving the right type and device to each of them).
    """

    fp_type: Literal['float64', 'float32'] = 'float64'
    _dtype: torch.TensorType = torch.float64
    fp_device: str = 'cpu'
    _device: str = 'cpu'

    @classmethod
    @property
    def default_jitter(cls) -> float:
        match cls._dtype:
            case torch.float64:
                return 1e-8
            case torch.float32:
                return 1e-4
            case _:
                raise ValueError('The dtype used by TSR has no default mapped jitter value')

    @classmethod
    def set_params(
        cls,
        fp_type: Literal['float32', 'float64'] | None = None,
        fp_device: str | None = None,
        seed: int | None = None,
    ):
        if fp_type is not None:
            match fp_type:
                case 'float32':
                    cls._dtype = torch.float32
                case 'float64':
                    cls._dtype = torch.float64
                case _:
                    raise ValueError('fp_type must be either `float32` or `float64`')
        if fp_device is not None:
            cls._device = fp_device
            cls.fp_device = fp_device
        if seed is not None:
            torch.manual_seed(seed)

    @classmethod
    def tensor(cls, tensor_data: Any) -> torch.Tensor:
        return torch.tensor(tensor_data, dtype=cls._dtype, device=cls._device)

    @classmethod
    def is_tensor(cls, tensor_data: Any) -> bool:
        return torch.is_tensor(tensor_data)

    @classmethod
    def eye(cls, n: int):
        return torch.eye(n, dtype=cls._dtype, device=cls._device)

    @classmethod
    def ones(cls, tsr_dim: int | tuple[int]):
        return torch.ones(tsr_dim, dtype=cls._dtype, device=cls._device)

    @classmethod
    def zeros(cls, tsr_dim: int | tuple[int]):
        return torch.zeros(tsr_dim, dtype=cls._dtype, device=cls._device)

    @classmethod
    def rand(cls, tsr_dim: int | tuple[int]):
        return torch.rand(tsr_dim, dtype=cls._dtype, device=cls._device)

    @classmethod
    def randn(cls, tsr_dim: tuple[int]) -> torch.Tensor:
        return torch.randn(tsr_dim, dtype=cls._dtype, device=cls._device)

    @classmethod
    def randn_like(cls, input_tensor: torch.Tensor):
        return torch.randn_like(input_tensor, dtype=cls._dtype, device=cls._device)

    @classmethod
    def arange(cls, start: int, end: int, step: int = 1):
        return torch.arange(start, end, step, dtype=cls._dtype, device=cls._device)

    @classmethod
    def rand_choice(cls, choices_tsr: torch.Tensor, nb_sample: int, use_replace: bool = False):
        choices_indx = torch.multinomial(choices_tsr, nb_sample, replacement=use_replace)
        return choices_tsr[choices_indx]

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
