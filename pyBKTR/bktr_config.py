from dataclasses import dataclass, field

import torch

from pyBKTR.sampler_config import KernelSamplerConfig
from pyBKTR.utils import log


@dataclass
class BKTRConfig:
    """Class that contains the configuration for a BKTR process

    A BKTRConfig object contains various information that are central to the
    BKTR sampling process. It is notably used to configure the tensor backend, initialize values
    for the algorithm (e,g. hyperparameters) or to define key algorithm parameters (e,g. number
    of iterations or decomposition rank).
    """

    rank_decomp: int
    """Rank of the CP decomposition (Paper -- :math:`R`)"""
    max_iter: int
    """Maximum number of iteration used by MCMC sampling (Paper -- :math:`K_1 + K_2`)"""
    burn_in_iter: int
    """Number of iteration before the sampling starts (Paper -- :math:`K_1`)"""
    temporal_period_length: int = 7
    """Period used in the periodic kernel (Paper -- :math:`T`)"""
    spatial_smoothness_factor: int = 3
    """Smoothness factor used in Matern kernel (Choice in 1, 3 or 5 -- 3 for Matern 3/2)"""
    kernel_variance: float = 1
    """Variance used for all kernels (Paper -- :math:`\\sigma^2_s, \\sigma^2_t`)"""
    sigma_r: float = 1e-2
    """Variance of the white noise process TODO (Paper -- :math:`\\tau^{-1}`)"""
    a_0: float = 1e-6
    """Initial value for the shape (:math:`\\alpha`) in the gamma function generating tau"""
    b_0: float = 1e-6
    """Initial value for the rate (:math:`\\beta`) in the gamma function generating tau"""

    period_slice_sampling_scale: float = log(10)
    """Slice sampling scale of the periodic length scale"""
    period_min_hparam_val: float = log(1e-3)
    """Minimum value for periodic length scale hyperparameter"""
    period_max_hparam_val: float = log(1e3)
    """Maximum value for periodic length scale hyperparameter"""
    period_hparam_mu_prior: float = 0
    """Initial value for periodic length scale mean"""
    period_hparam_precision_prior: float = 1
    """Initial value for periodic length scale precision"""

    decay_slice_sampling_scale: float = log(10)
    """Slice sampling scale of the decay time scale"""
    decay_min_hparam_val: float = log(1e-3)
    """Minimum value for decay time scale hyperparameter"""
    decay_max_hparam_val: float = log(1e3)
    """Maximum value for decay time scale hyperparameter"""
    decay_hparam_mu_prior: float = 0
    """Initial value for decay time scale mean"""
    decay_hparam_precision_prior: float = 1
    """Initial value for decay time scale precision"""

    spatial_slice_sampling_scale: float = log(10)
    """Slice sampling scale of the spatial length scale"""
    spatial_min_hparam_val: float = log(1e-3)
    """Minimum value for spatial length scale hyperparameter"""
    spatial_max_hparam_val: float = log(1e3)
    """Maximum value for spatial length scale hyperparameter"""
    spatial_hparam_mu_prior: float = 0
    """Initial value for spatial length scale mean"""
    spatial_hparam_precision_prior: float = 1
    """Initial value for spatial length scale precision"""

    # Torch Params
    torch_dtype: torch.dtype = torch.float64
    """Type used for floating points in the tensor backend"""
    torch_device: torch.device = 'cpu'
    """Device used by the tensor backend for calculation 'cuda' or 'cpu'"""
    torch_seed: int | None = None
    """Seed used by the torch backend (If None, no seed is used)"""

    config_periodic_scale: KernelSamplerConfig = field(init=False)
    """Periodic length scale's config (Paper -- :math:`\\gamma_1`) created via init inputs"""
    config_decay_scale: KernelSamplerConfig = field(init=False)
    """Decay time scale's config (Paper -- :math:`\\gamma_2`) created via init inputs"""
    config_spatial_length: KernelSamplerConfig = field(init=False)
    """Spatial length-scale's config (Paper -- :math:`\\phi`) created via init inputs"""

    # Output Params
    sampled_beta_indexes: list[int] = field(default_factory=list)
    """Indexes of beta estimates that need to be sampled through iterations"""
    sampled_y_indexes: list[int] = field(default_factory=list)
    """Indexes of y estimates that need to be sampled through iterations"""
    results_export_dir: str | None = None
    """Path of the folder where the csv file will be exported (if None it is printed)"""

    def __post_init__(self):
        self.periodic_scale_config = KernelSamplerConfig(
            self.period_slice_sampling_scale,
            self.period_min_hparam_val,
            self.period_max_hparam_val,
            self.period_hparam_mu_prior,
            self.period_hparam_precision_prior,
        )

        self.decay_scale_config = KernelSamplerConfig(
            self.decay_slice_sampling_scale,
            self.decay_min_hparam_val,
            self.decay_max_hparam_val,
            self.decay_hparam_mu_prior,
            self.decay_hparam_precision_prior,
        )

        self.spatial_length_config = KernelSamplerConfig(
            self.spatial_slice_sampling_scale,
            self.spatial_min_hparam_val,
            self.spatial_max_hparam_val,
            self.spatial_hparam_mu_prior,
            self.spatial_hparam_precision_prior,
        )
