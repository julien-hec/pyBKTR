import math
from typing import Callable

import numpy as np
import torch

from pyBKTR.kernels import Kernel, KernelParameter
from pyBKTR.tensor_ops import TSR


class KernelParamSampler:
    """Class for kernel's hyperparameter sampling

    The KernelParamSampler encapsulate all the behavior related to kernel
    hyperparameters' sampling

    """

    __slots__ = ('kernel', 'marginal_ll_eval_fn')

    kernel: Kernel
    """Kernel used for the hyperparameter sampling process"""
    marginal_ll_eval_fn: Callable

    def __init__(
        self,
        kernel: Kernel,
        marginal_ll_eval_fn: Callable,
    ):
        self.kernel = kernel
        self.marginal_ll_eval_fn = marginal_ll_eval_fn

    @staticmethod
    def initialize_theta_bounds(param: KernelParameter) -> tuple[float, float]:
        """Initialize sampling bounds according to current theta value and sampling scale"""
        theta_range = param.slice_sampling_scale * float(TSR.rand(1))
        theta_min = max(math.log(param.value) - theta_range, param.lower_bound)
        theta_max = min(theta_min + param.slice_sampling_scale, param.uppder_bound)
        return theta_min, theta_max

    def _prior_fn(self, param: KernelParameter) -> float:
        """
        Prior likelihood function for a given hyperparameter value
        TODO Check if we should always use the same hyper mu prior from the config?
        """
        # return -0.5 * self.config.hypr_precision_prior * (theta - self.config.hypr_mu_prior) ** 2
        return -0.5 * param.hparam_precision * (math.log(param.value)) ** 2

    @staticmethod
    def sample_rand_theta_value(theta_min, theta_max):
        """Sample a random theta value within the sampling bounds"""
        return theta_min + (theta_max - theta_min) * float(TSR.rand(1))

    def sample_param(self, param: KernelParameter):
        """The complete kernel hyperparameter sampling process"""
        theta_min, theta_max = self.initialize_theta_bounds(param)
        initial_theta = math.log(param.value)
        self.kernel.kernel_gen()
        initial_marginal_likelihood = self.marginal_ll_eval_fn() + self._prior_fn(param)
        density_threshold = float(TSR.rand(1))

        while True:
            new_theta = self.sample_rand_theta_value(theta_min, theta_max)
            param.value = math.exp(new_theta)
            self.kernel.kernel_gen()
            new_marginal_likelihood = self.marginal_ll_eval_fn() + self._prior_fn(param)

            marg_ll_diff = new_marginal_likelihood - initial_marginal_likelihood
            if np.exp(np.clip(marg_ll_diff, -709.78, 709.78)) > density_threshold:
                return param.value
            if new_theta < initial_theta:
                theta_min = new_theta
            else:
                theta_max = new_theta

    def sample(self):
        for param in self.kernel.parameters:
            self.sample_param(param)


def sample_norm_multivariate(
    mean_vec: torch.Tensor, precision_upper_tri: torch.Tensor
) -> torch.Tensor:
    """Sample a vector from a normal multivariate distribution

    Args:
        mean_vec (torch.Tensor): A vector of normal distribution means
        precision_upper_tri (torch.Tensor): An upper triangular matrix of
            precision between the distributions

    Returns:
        torch.Tensor: A sampled vector from the given normal multivariate information
    """
    return (
        torch.linalg.solve_triangular(
            precision_upper_tri, TSR.randn_like(mean_vec).unsqueeze(1), upper=True
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
    inv_s = TSR.kronecker_prod(TSR.eye(rank_cp), wish_precision_tensor)
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

    def sample(self, covs_decomp: torch.Tensor):
        w = covs_decomp.matmul(covs_decomp.t()) + TSR.eye(self.nb_covariates)
        # TODO check if we can use cov instead of precision #14
        w_inv = w.inverse()
        wish_sigma = (w_inv + w_inv.t()) * 0.5
        wish_precision_matrix = torch.distributions.Wishart(
            df=self.wish_df, covariance_matrix=wish_sigma
        ).sample()
        self.wish_precision_tensor = wish_precision_matrix
        return self.wish_precision_tensor
