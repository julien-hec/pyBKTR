import torch
from pyBKTR.tensor_ops import TSR


class MarginalLikelihoodEvaluator:
    """Class that evaluate the marginal likelihood of the kernel hyperparameter

    MarginalLikelihoodEvaluator enable the calculation of the marginal
    likelihood of the kernel hyperparameters. This likelihood is used during the sampling
    process.

    """
    axis_permutation: list[int]
    rank_decomp: int
    nb_covariates: int
    covariates: torch.Tensor
    omega: torch.Tensor
    y_masked: torch.Tensor

    inv_k: torch.Tensor
    chol_k: torch.Tensor
    chol_lu: torch.Tensor
    # TODO What is uu
    uu: torch.Tensor
    likelihood: float

    def __init__(
        self,
        rank_decomp: int,
        nb_covariates: int,
        covariates: torch.Tensor,
        omega: torch.Tensor,
        y: torch.Tensor,
        is_transposed: bool
    ):
        self.rank_decomp = rank_decomp
        self.nb_covariates = nb_covariates
        self.covariates = covariates
        self.omega = omega
        self.axis_permutation = [1, 0] if is_transposed else [0, 1]
        self.y_masked = y * omega

    def calc_likelihood(
        self,
        kernel_values: torch.Tensor,
        decomp_values: torch.Tensor,
        covs_decomp: torch.Tensor,
        tau: float
    ) -> float:
        """Calculate and set the marginal likelihood according to current parameter values

        Args:
            kernel_values (torch.Tensor): Current values of the related kernel for the
            decomp_values (torch.Tensor): Values of the related tensor decomposition
            covs_decomp (torch.Tensor): Complete tensor of the covariates
            tau (float): current value of the white noise precision

        Returns:
            float: The calculated marginal likelihood
        """
        rank_decomp = self.rank_decomp
        kernel_size = kernel_values.shape[0]
        lambda_size = kernel_size * self.rank_decomp

        psi_u = torch.einsum("ijk,jkl->ilj", [
            self.covariates.permute([*self.axis_permutation, 2]),
            TSR.khatri_rao_prod(decomp_values, covs_decomp).reshape(
                [-1, self.nb_covariates, rank_decomp]
            )
        ])
        psi_u_mask = psi_u * (
            self.omega
                .permute(self.axis_permutation)
                .unsqueeze(1)
                .expand_as(psi_u)
        )

        self.chol_k = torch.linalg.cholesky(kernel_values)
        self.inv_k = TSR.kronecker_prod(
            torch.eye(rank_decomp),
            (kernel_values).inverse()
        )  # I_R Kron inv(Ks)

        lambda_u = tau * torch.einsum('ijk,ilk->ijl', [psi_u_mask, psi_u_mask])  # tau * H_T * H_T'
        # TODO Check broadcasting here, we can simplify it (We just want to diagonalize lambda_u)
        lambda_u = (
            lambda_u.permute([1, 0, 2])
            .flatten(start_dim=0, end_dim=1)
            .unsqueeze(1)
            .expand([-1, kernel_size, -1])
            .permute([0, 2, 1])
            .reshape([lambda_size, lambda_size])
        )
        lambda_u = TSR.kronecker_prod(
            torch.ones([rank_decomp, rank_decomp]),
            torch.eye(kernel_size)
        ) * lambda_u
        lambda_u = lambda_u + self.inv_k

        self.chol_lu = torch.linalg.cholesky(lambda_u)
        self.uu = torch.linalg.solve(
            self.chol_lu,
            torch.einsum(
                'ijk,ik->ji',
                [psi_u_mask, self.y_masked.permute(self.axis_permutation)]
            ).flatten()
        )
        self.likelihood = float((
            0.5 * tau ** 2 * self.uu.t().matmul(self.uu)
            - self.chol_lu.diag().log().sum()
            - torch.tensor(rank_decomp) * self.chol_k.diag().log().sum()
        ).cpu())

        return self.likelihood
