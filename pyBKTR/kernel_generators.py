import abc
import math
from typing import Callable

import torch

from pyBKTR.tensor_ops import TSR


class KernelGenerator(abc.ABC):
    """
    Abstract Class Template for kernel generators
    """

    @abc.abstractmethod
    def kernel_gen(self) -> torch.Tensor:
        pass


class TemporalKernelGenerator:
    """Class to generate temporal kernels

    TemporalKernelGenerator can create temporal kernels according to provided parameters.
    A temporal kernel can use a periodic kernel function, a squared exponential (SE)
    kernel function or a mixture of both.
    """

    __slots__ = (
        'time_distances',
        'period_length',
        'kernel',
        'kernel_variance',
        'periodic_length_scale',
        'decay_time_scale',
        '_core_kernel_fn',
        'tsr',
        'has_stabilizing_diag',
        'time_segment_duration',
    )

    STAB_DIAG_MULTIPLIER = 1e-12

    time_distances: torch.Tensor
    """A matrix expressing the distance between two time segments"""
    period_length: int
    kernel: torch.Tensor
    """The kernel values generated by the latest :func:`kernel_gen` call"""
    kernel_variance: float
    periodic_length_scale: float
    decay_time_scale: float
    _core_kernel_fn: Callable
    tsr: TSR
    has_stabilizing_diag: bool
    time_segment_duration: float

    def __init__(
        self,
        kernel_fn_name: str,
        nb_time_segments: int,
        period_length: int,
        kernel_variance: float,
        tsr_instance: TSR,
        periodic_length_scale: float = 0,
        decay_time_scale: float = 0,
        has_stabilizing_diag: bool = True,
        time_segment_duration: float = 0.1,
    ) -> None:
        """Initializing a TemporalKernelGenerator instance

        Args:
            kernel_fn_name (str): Name of the kernel function
                (choices are ``periodic``, ``se`` or ``periodic_se``)
            nb_time_segments (int): The number of time segments used in the kernel,
                this will define the dimension of the kernel matrix (:math:`T \\times T`)
            period_length (int): The period length used in the kernel for periodicity
                (i.e. to show weekly periodicity we could use a value of 7)
            kernel_variance (float): The variance of the kernel
            periodic_length_scale (float, optional): Periodic length scale value
                (Paper -- :math:`\\gamma_1`). Defaults to 0.
            decay_time_scale (float, optional): Decay time scale value
                (Paper -- :math:`\\gamma_2`). Defaults to 0.
            has_stabilizing_diag (bool, optional): Indicate if we are adding a stabilizing diagonal
                to help cholesky decomposition. Defaults to False.
            time_segment_duration (float, optional): Duration between all segments in the kernel,
                TODO check. Defaults to 1.
        """
        self.tsr = tsr_instance
        self._core_kernel_fn = self._get_kernel_fn(kernel_fn_name)
        self._set_time_distance_matrix(nb_time_segments, time_segment_duration)
        self.period_length = period_length
        self.kernel_variance = kernel_variance
        self.periodic_length_scale = periodic_length_scale
        self.decay_time_scale = decay_time_scale
        self.has_stabilizing_diag = has_stabilizing_diag

    def _get_kernel_fn(self, kernel_fn_name: str) -> Callable:
        """
        Get the right kernel generator function associated with the function name input
        """
        match kernel_fn_name:
            case 'periodic':
                return self._periodic_kernel_gen
            case 'se':
                return self._se_kernel_gen
            case 'periodic_se':
                return self._periodic_se_kernel_gen
            case _:
                raise ValueError('Please choose a valid kernel function name')

    def _set_time_distance_matrix(
        self, nb_time_segments: int, time_segment_duration: float = 1
    ) -> None:
        """
        Create and set a time distance matrix according to the number of time segments
        """
        time_segments = self.tsr.arange(0, nb_time_segments).unsqueeze(1) * time_segment_duration
        self.time_distances = time_segments - time_segments.t()

    def _periodic_kernel_gen(self) -> Callable:
        """
        Core kernel function for a periodic kernel
        """
        return torch.exp(
            -2
            * torch.sin(torch.pi * self.time_distances / self.period_length) ** 2
            / self.periodic_length_scale**2
        )

    def _se_kernel_gen(self) -> Callable:
        """
        Core kernel function for a Squared Exponential (SE) kernel
        """
        return torch.exp(-self.time_distances**2 / (2 * self.decay_time_scale**2))

    def _periodic_se_kernel_gen(self) -> Callable:
        """
        Core kernel function for a Periodic and Squared Exponential (SE) kernel.
        """
        return self._periodic_kernel_gen() * self._se_kernel_gen()

    def kernel_gen(self) -> torch.Tensor:
        """
        Method that generates, sets and return a kernel for a given
        temporal kernel configuration

        Returns:
            torch.tensor: Kernel values for the current instance configuration
        """
        self.kernel = self.kernel_variance * self._core_kernel_fn()
        if self.has_stabilizing_diag:
            self.kernel = self.kernel + self.STAB_DIAG_MULTIPLIER * self.tsr.eye(
                self.kernel.shape[0]
            )
        return self.kernel


class SpatialKernelGenerator(KernelGenerator):
    """Class to generate spatial kernels

    A SpatialKernelGenerator can create spatial kernels according
    to provided parameters.
    """

    __slots__ = (
        'smoothness_factor',
        'kernel_variance',
        'kernel',
        '_core_kernel_fn',
        'spatial_length_scale',
    )

    distance_matrix: torch.Tensor
    smoothness_factor: int
    kernel_variance: float
    spatial_length_scale: float
    kernel: torch.Tensor
    _core_kernel_fn: Callable

    def __init__(
        self,
        distance_matrix: torch.Tensor,
        smoothness_factor: int,
        kernel_variance: float,
        spatial_length_scale: float = 0,
    ):
        """Initializing a SpatialKernelGenerator instance

        Args:
            distance_matrix (torch.Tensor): Matrix containing the distance between each evaluated
                spatial coordinates
            smoothness_factor (int): Smoothness factor for the Matern kernel. Can be 1, 3 or 5.
                For a Matern 3/2 we need to use 3.
            kernel_variance (float): The variance of the kernel
            spatial_length_scale (float, optional): Spatial length scale (Paper -- :math:`\\phi`).
                Defaults to 0.
        """
        self.distance_matrix = distance_matrix
        self.smoothness_factor = smoothness_factor
        self.kernel_variance = kernel_variance
        self._core_kernel_fn = self._get_core_kernel_smoothness_fn(smoothness_factor)
        self.spatial_length_scale = spatial_length_scale

    @staticmethod
    def _get_core_kernel_smoothness_fn(smoothness_factor):
        """
        Get the right kernel generator function associated with the right smoothness factor
        """
        match smoothness_factor:
            case 1:
                return lambda t: 1
            case 3:
                return lambda t: 1 + t
            case 5:
                return lambda t: 1 + t * (1 + t / 3)
            case _:
                raise ValueError('Kernel function for this smoothness factor is not implemented')

    def kernel_gen(self) -> torch.Tensor:
        """
        Method that generates, sets and return a kernel for a given spatial kernel configuration

        Returns:
            torch.tensor: Kernel values for the current instance configuration
        """
        kernel_results = (
            self.distance_matrix * math.sqrt(self.smoothness_factor) / self.spatial_length_scale
        )
        self.kernel = (
            self.kernel_variance * self._core_kernel_fn(kernel_results) * (-kernel_results).exp()
        )
        return self.kernel
