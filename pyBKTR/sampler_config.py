import math
from dataclasses import dataclass


@dataclass
class KernelSamplerConfig:
    """Class representing a hyperparameter sampling's configuration

    A KernelSamplerConfig object contains all the information necessary to the process
    of sampling kernel hyperparameters. (Used in **Algorithm 2** of the paper)
    """

    slice_sampling_scale: float = math.log(10)
    """The sampling range's amplitude (Paper -- :math:`\\rho`)"""
    min_hyper_value: float = math.log(1e-3)
    """The hyperparameter's minimal admissible value (Paper -- :math:`\\phi_{min}`)"""
    max_hyper_value: float = math.log(1e3)
    """The hyperparameter's maximal admissible value (Paper -- :math:`\\phi_{max}`)"""
    hyper_mu_prior: float = 0
    """The hyperparameter mean's prior value (Paper -- :math:`\\phi`)"""
    hyper_precision_prior: float = 1
    """Hyperparameter precision's prior value"""
