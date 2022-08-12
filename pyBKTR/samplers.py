from typing import Callable

import numpy as np
import torch

from pyBKTR.kernel_generators import KernelGenerator
from pyBKTR.sampler_config import KernelSamplerConfig
from pyBKTR.tensor_ops import TSR


class KernelParamSampler:
    """Class for kernel's hyperparameter sampling

    The KernelParamSampler encapsulate all the behavior related to kernel
    hyperparameters' sampling

    """

    __slots__ = (
        'config',
        'kernel_generator',
        'marginal_ll_eval_fn',
        'kernel_hparam_name',
        'theta_value',
        '_theta_min',
        '_theta_max',
    )

    config: KernelSamplerConfig
    """Kernel sampler configuration"""
    kernel_generator: KernelGenerator
    """Kernel Generator used for the hyperparameter sampling process"""
    marginal_ll_eval_fn: Callable
    """Marginal Likelihood Evaluator used in the sampling process"""
    kernel_hparam_name: str
    """The name used in the hyperparameter sampling process"""
    theta_value: float
    """Current value of Theta held by the sampler"""
    _theta_min: float
    _theta_max: float

    def __init__(
        self,
        config: KernelSamplerConfig,
        kernel_generator: KernelGenerator,
        marginal_ll_eval_fn: Callable,
        kernel_hparam_name: str,
    ):
        self.config = config
        self.kernel_generator = kernel_generator
        self.marginal_ll_eval_fn = marginal_ll_eval_fn
        self.kernel_hparam_name = kernel_hparam_name
        self._set_theta_value(config.hyper_mu_prior)

    def _set_theta_value(self, theta: float):
        """Set the theta value for the sampler and for its respective kernel generator

        Args:
            theta (float): _description_
        """
        self.theta_value = theta
        setattr(self.kernel_generator, self.kernel_hparam_name, np.exp(theta))

    def initialize_theta_bounds(self):
        """Initialize sampling bounds according to current theta value and sampling scale"""
        theta_range = self.config.slice_sampling_scale * float(torch.rand(1))
        self._theta_min = max(self.theta_value - theta_range, self.config.min_hyper_value)
        self._theta_max = min(
            self._theta_min + self.config.slice_sampling_scale, self.config.max_hyper_value
        )

    def _prior_fn(self, theta: float) -> float:
        """Prior likelihood function for a given hyperparameter value"""
        return -0.5 * self.config.hyper_precision_prior * (theta - self.config.hyper_mu_prior) ** 2

    def sample_rand_theta_value(self):
        """Sample a random theta value within the sampling bounds"""
        return self._theta_min + (self._theta_max - self._theta_min) * float(torch.rand(1))

    def sample(self):
        """The complete kernel hyperparameter sampling process"""
        initial_theta = self.theta_value
        self.initialize_theta_bounds()
        self.kernel_generator.kernel_gen()
        initial_marginal_likelihood = self.marginal_ll_eval_fn() + self._prior_fn(self.theta_value)

        density_threshold = float(torch.rand(1))

        while True:
            new_theta = self.sample_rand_theta_value()
            self._set_theta_value(new_theta)
            self.kernel_generator.kernel_gen()
            new_marginal_likelihood = self.marginal_ll_eval_fn() + self._prior_fn(new_theta)

            marg_ll_diff = new_marginal_likelihood - initial_marginal_likelihood
            if np.exp(np.clip(marg_ll_diff, -709.78, 709.78)) > density_threshold:
                return np.exp(new_theta)
            if new_theta < initial_theta:
                self._theta_min = new_theta
            else:
                self._theta_max = new_theta


# TODO See if we directly use the norm multivariate from torch
def sample_norm_multivariate(
    mean_vec: torch.Tensor, precision_upper_tri: torch.Tensor
) -> torch.Tensor:
    """_summary_

    Args:
        mean_vec (torch.Tensor): A vector of normal distribution means
        precision_upper_tri (torch.Tensor): An upper triangular matrix of
            precision between the distributions

    Returns:
        torch.Tensor: A sampled vector from the given normal multivariate information
    """
    return (
        torch.linalg.solve_triangular(
            precision_upper_tri, torch.randn_like(mean_vec).unsqueeze(1), upper=True
        ).squeeze()
        + mean_vec
    )


def get_cov_decomp_chol(
    spatial_decomp: torch.Tensor,
    time_decomp: torch.Tensor,
    covs: torch.Tensor,
    rank_cp: int,
    omega: torch.Tensor,
    tau: float,
    y: torch.Tensor,
    wish_precision_tensor: torch.Tensor,
):
    y_masked = omega * y
    # TODO Merge some parts with marginal ll of spatial and temporal
    # get corresponding norm multivariate mean
    b = TSR.khatri_rao_prod(spatial_decomp, time_decomp).reshape(
        [spatial_decomp.shape[0], time_decomp.shape[0], rank_cp]
    )
    psi_c = torch.einsum('ijk,ijl->ijlk', (covs, b))
    psi_c_mask = psi_c * omega.unsqueeze(2).unsqueeze(3).expand_as(psi_c)
    psi_c_mask = psi_c_mask.permute([1, 0, 2, 3]).reshape(
        [psi_c.shape[0] * psi_c.shape[1], psi_c.shape[2] * psi_c.shape[3]]
    )
    inv_s = TSR.kronecker_prod(torch.eye(rank_cp), wish_precision_tensor)
    lambda_c = tau * psi_c_mask.t().matmul(psi_c_mask) + inv_s
    chol_lc = torch.linalg.cholesky(lambda_c)
    cc = torch.linalg.solve(chol_lc, psi_c_mask.t().matmul(y_masked.t().flatten()))
    return {'chol_lc': chol_lc, 'cc': cc}


class TauSampler:
    """Sampler class for the Tau precision parameter of the BKTR Algorithm"""

    b_0: float
    a_tau: float

    def __init__(self, a_0: float, b_0: float, nb_observations: int):
        self.b_0 = b_0
        self.a_tau = a_0 + 0.5 * nb_observations

    def sample(self, total_sq_error):
        b_tau = self.b_0 + 0.5 * total_sq_error
        return torch.distributions.Gamma(self.a_tau.cpu(), b_tau).sample()


class PrecisionMatrixSampler:
    """Sample class to get new precision matrices from Wishart distributions"""

    nb_covariates: int
    wish_df: int
    wish_precision_tensor: torch.Tensor

    def __init__(self, nb_covariates: int, rank_cp: int):
        self.nb_covariates = nb_covariates
        self.wish_df = nb_covariates + rank_cp

    def sample(self, covs_decomp):
        w = covs_decomp.matmul(covs_decomp.t()) + torch.eye(self.nb_covariates)
        wish_sigma = ((w + w.t()) * 0.5).inverse().cpu()
        wish_precision_matrix = torch.distributions.Wishart(
            df=self.wish_df, covariance_matrix=wish_sigma
        ).sample()
        self.wish_precision_tensor = wish_precision_matrix
        return self.wish_precision_tensor
