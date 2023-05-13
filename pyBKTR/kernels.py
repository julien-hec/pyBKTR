from __future__ import annotations

import abc
import math
from enum import Enum
from functools import cached_property
from typing import Callable, Literal

import torch
from plotly import express as px

from pyBKTR.distances import DIST_TYPE, DistanceCalculator
from pyBKTR.tensor_ops import TSR
from pyBKTR.utils import log

DEFAULT_LBOUND = 1e-3
DEFAULT_UBOUND = 1e3


class KernelParameter:

    value: float
    """The hyperparameter mean's prior value (Paper -- :math:`\\phi`) or its fixed value"""
    is_fixed: bool = False
    """Says if the kernel parameter is fixed or not (if it is fixed, there is no sampling)"""
    lower_bound: float = DEFAULT_LBOUND
    """The hyperparameter's minimal admissible value in sampling (Paper -- :math:`\\phi_{min}`)"""
    upper_bound: float = DEFAULT_UBOUND
    """The hyperparameter's maximal admissible value in sampling (Paper -- :math:`\\phi_{max}`)"""
    slice_sampling_scale: float
    """The sampling range's amplitude (Paper -- :math:`\\rho`)"""
    hparam_precision: float
    """WHAT IS THAT? TODO"""
    kernel: Kernel | None
    """The kernel associated with the parameter (it is set at kernel instanciation)"""

    def __init__(
        self,
        value: float,
        is_fixed: bool = False,
        lower_bound: float = DEFAULT_LBOUND,
        upper_bound: float = DEFAULT_UBOUND,
        slice_sampling_scale: float = log(10),
        hparam_precision: float = 1.0,
    ):
        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_fixed = is_fixed
        self.slice_sampling_scale = slice_sampling_scale
        self.hparam_precision = hparam_precision

    def set_kernel(self, kernel: Kernel, param_name: str):
        self.kernel = kernel
        self.name = param_name
        self.kernel.parameters.append(self)

    @property
    def full_name(self) -> str:
        if self.kernel is None:
            return self.name
        return f'{self.kernel._name} - {self.name}'

    def __repr__(self):
        rep_attrs = [f'val={self.value}']
        if self.is_fixed:
            rep_attrs.append('is_const=True')
        if self.lower_bound != DEFAULT_LBOUND:
            rep_attrs.append(f'lbound={self.lower_bound}')
        if self.upper_bound != DEFAULT_UBOUND:
            rep_attrs.append(f'ubound={self.upper_bound}')
        return f'KernelParam({", ".join(rep_attrs)})'


class Kernel(abc.ABC):
    """
    Abstract Class Template for kernels
    """

    kernel_variance: float
    jitter_value: float | None
    distance_matrix: torch.Tensor | None = None
    parameters: list[KernelParameter] = []
    kernel: torch.Tensor
    distance_type: type[DIST_TYPE]

    @property
    @abc.abstractmethod
    def _name(self) -> str:
        pass

    @abc.abstractmethod
    def _core_kernel_fn(self) -> torch.Tensor:
        pass

    def __init__(
        self, kernel_variance: float, distance_type: type[DIST_TYPE], jitter_value: float | None
    ) -> None:
        self.parameters = []
        self.kernel_variance = kernel_variance
        self.distance_type = distance_type
        self.jitter_value = jitter_value

    def add_jitter_to_kernel(self):
        if self.jitter_value == 0:
            return
        jitter_val = TSR.default_jitter if self.jitter_value is None else self.jitter_value
        self.kernel += jitter_val * TSR.eye(self.kernel.shape[0])

    def kernel_gen(self) -> torch.Tensor:
        if self.distance_matrix is None:
            raise RuntimeError(
                'Set kernel distance via `set_distance_matrix` before kernel evaluation.'
            )
        self.kernel = self.kernel_variance * self._core_kernel_fn()
        self.add_jitter_to_kernel()
        return self.kernel

    def set_distance_matrix(
        self, x: None | torch.Tensor = None, distance_matrix: None | torch.Tensor = None
    ):
        if (x is None) == (distance_matrix is None):
            raise ValueError('Either `x` or `distance_matrix` must be provided')
        elif x is not None:
            self.distance_matrix = DistanceCalculator.get_matrix(x, self.distance_type)
        else:
            self.distance_matrix = distance_matrix

    def plot(self, show_figure: bool = True):
        fig = px.imshow(
            self.kernel_gen(),
            title=f'{self._name} Covariance Matrix',
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        fig.update_layout(
            xaxis_title='<b>x</b>',
            yaxis_title="<b>x</b>'",
            coloraxis_colorbar={'title': 'Covariance Value'},
            height=700,
            width=700,
        )
        if show_figure:
            fig.show()
            return
        return fig

    def __mul__(self, other) -> KernelComposed:
        return KernelComposed(self, other, f'({self._name} * {other._name})', CompositionOps.MUL)

    def __add__(self, other) -> KernelComposed:
        return KernelComposed(self, other, f'({self._name} + {other._name})', CompositionOps.ADD)


class KernelWhiteNoise(Kernel):
    """White Noise Kernel"""

    variance: KernelParameter
    distance_matrix: torch.Tensor
    _name: str = 'White Noise Kernel'

    def __init__(
        self,
        variance: KernelParameter = KernelParameter(1, is_fixed=True),
        kernel_variance: float = 1,
        jitter_value: float | None = None,
    ) -> None:
        super().__init__(kernel_variance, DIST_TYPE.NONE, jitter_value)
        self.variance = variance
        self.variance.set_kernel(self, 'variance')

    def _core_kernel_fn(self) -> torch.Tensor:
        return self.distance_matrix * self.variance.value


class KernelLinear(Kernel):
    """Linear Kernel"""

    # TODO Why is the variance a kernel param here?????
    variance: KernelParameter
    distance_matrix: torch.Tensor
    _name: str = 'Linear Kernel'

    def __init__(
        self,
        variance: KernelParameter = KernelParameter(1, lower_bound=0),
        kernel_variance: float = 1,
        jitter_value: float | None = None,
    ) -> None:
        super().__init__(kernel_variance, DIST_TYPE.DOT_PRODUCT, jitter_value)
        self.variance = variance
        self.variance.set_kernel(self, 'variance')

    def _core_kernel_fn(self) -> torch.Tensor:
        return self.distance_matrix + self.variance.value


class KernelSE(Kernel):
    """Squared Exponential Kernel"""

    lengthscale: KernelParameter
    distance_matrix: torch.Tensor
    _name: str = 'SE Kernel'

    def __init__(
        self,
        lengthscale: KernelParameter = KernelParameter(log(2)),
        kernel_variance: float = 1,
        distance_type: type[DIST_TYPE] = DIST_TYPE.LINEAR,
        jitter_value: float | None = None,
    ) -> None:
        super().__init__(kernel_variance, distance_type, jitter_value)
        self.lengthscale = lengthscale
        self.lengthscale.set_kernel(self, 'lengthscale')

    def _core_kernel_fn(self) -> torch.Tensor:
        return torch.exp(-self.distance_matrix**2 / (2 * self.lengthscale.value**2))


class KernelRQ(Kernel):
    """Rational Quadratic Kernel"""

    lengthscale: KernelParameter
    alpha: KernelParameter
    distance_matrix: torch.Tensor
    _name: str = 'RQ Kernel'

    def __init__(
        self,
        lengthscale: KernelParameter = KernelParameter(log(2)),
        alpha: KernelParameter = KernelParameter(log(2)),
        kernel_variance: float = 1,
        distance_type: type[DIST_TYPE] = DIST_TYPE.LINEAR,
        jitter_value: float | None = None,
    ) -> None:
        super().__init__(kernel_variance, distance_type, jitter_value)
        self.lengthscale = lengthscale
        self.lengthscale.set_kernel(self, 'lengthscale')
        self.alpha = alpha
        self.alpha.set_kernel(self, 'alpha')

    def _core_kernel_fn(self) -> torch.Tensor:
        return (
            1 + self.distance_matrix**2 / (2 * self.alpha.value * self.lengthscale.value**2)
        ) ** -self.alpha.value


class KernelPeriodic(Kernel):
    """Periodic Kernel"""

    lengthscale: KernelParameter
    period_length: KernelParameter
    distance_matrix: torch.Tensor
    _name: str = 'Periodic Kernel'

    def __init__(
        self,
        lengthscale: KernelParameter = KernelParameter(log(2)),
        period_length: KernelParameter = KernelParameter(log(2)),
        kernel_variance: float = 1,
        distance_type: type[DIST_TYPE] = DIST_TYPE.LINEAR,
        jitter_value: float | None = None,
    ) -> None:
        super().__init__(kernel_variance, distance_type, jitter_value)
        self.lengthscale = lengthscale
        self.lengthscale.set_kernel(self, 'lengthscale')
        self.period_length = period_length
        self.period_length.set_kernel(self, 'period length')

    def _core_kernel_fn(self) -> torch.Tensor:
        return torch.exp(
            -2
            * torch.sin(torch.pi * self.distance_matrix / self.period_length.value) ** 2
            / self.lengthscale.value**2
        )


class KernelMatern(Kernel):
    lengthscale: KernelParameter
    smoothness_factor: Literal[1, 3, 5]

    def __init__(
        self,
        smoothness_factor: Literal[1, 3, 5],
        lengthscale: KernelParameter = KernelParameter(log(2)),
        kernel_variance: float = 1,
        distance_type: type[DIST_TYPE] = DIST_TYPE.HAVERSINE,
        jitter_value: float | None = None,
    ) -> None:
        if smoothness_factor not in {1, 3, 5}:
            raise ValueError('smoothness factor should be one of the following values 1, 3 or 5')
        super().__init__(kernel_variance, distance_type, jitter_value)
        self.smoothness_factor = smoothness_factor
        self.lengthscale = lengthscale
        self.lengthscale.set_kernel(self, 'lengthscale')

    @property
    def _name(self) -> str:
        return f'Matern {self.smoothness_factor}/2 Kernel'

    @cached_property
    def smoothness_kernel_fn(self) -> Callable:
        """
        Get the right kernel function associated with the right smoothness factor
        """
        match self.smoothness_factor:
            case 1:
                return lambda t: 1
            case 3:
                return lambda t: 1 + t
            case 5:
                return lambda t: 1 + t * (1 + t / 3)
            case _:
                raise ValueError('Kernel function for this smoothness factor is not implemented')

    def _core_kernel_fn(self) -> torch.Tensor:
        temp_kernel = (
            self.distance_matrix * math.sqrt(self.smoothness_factor) / self.lengthscale.value
        )
        return self.smoothness_kernel_fn(temp_kernel) * (-temp_kernel).exp()


class CompositionOps(Enum):
    MUL = 'mul'
    ADD = 'add'


class KernelComposed(Kernel):
    _name: str = ''
    parameters: list = []
    left_kernel = Kernel
    right_kernel = Kernel

    def __init__(
        self,
        left_kernel: Kernel,
        right_kernel: Kernel,
        new_name: str,
        composition_operation: CompositionOps,
    ) -> None:
        composed_variance = 1
        if left_kernel.distance_type != right_kernel.distance_type:
            raise RuntimeError('Composed kernel must have the same distance type')
        new_jitter_val = max(
            left_kernel.jitter_value or TSR.default_jitter,
            right_kernel.jitter_value or TSR.default_jitter,
        )
        super().__init__(
            composed_variance,
            left_kernel.distance_type,
            new_jitter_val,
        )
        self.left_kernel = left_kernel
        self.right_kernel = right_kernel
        self._name = new_name
        self.parameters = self.left_kernel.parameters + self.right_kernel.parameters
        self.composition_operation = composition_operation

    def _core_kernel_fn(self) -> torch.Tensor:
        match self.composition_operation:
            case CompositionOps.ADD:
                return self.left_kernel._core_kernel_fn() + self.right_kernel._core_kernel_fn()
            case CompositionOps.MUL:
                return self.left_kernel._core_kernel_fn() * self.right_kernel._core_kernel_fn()
        raise RuntimeError('Composition operation not implemented')

    def set_distance_matrix(
        self, x: None | torch.Tensor = None, distance_matrix: None | torch.Tensor = None
    ):
        super().set_distance_matrix(x, distance_matrix)
        self.left_kernel.set_distance_matrix(x, distance_matrix)
        self.right_kernel.set_distance_matrix(x, distance_matrix)
