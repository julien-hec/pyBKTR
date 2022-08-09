import torch


class TSR:
    """
    Class containing all tensor operations used in BKTR
    """

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
