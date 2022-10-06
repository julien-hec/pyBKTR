from dataclasses import dataclass, field


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
    sigma_r: float = 1e-2
    """Variance of the white noise process TODO (Paper -- :math:`\\tau^{-1}`)"""
    a_0: float = 1e-6
    """Initial value for the shape (:math:`\\alpha`) in the gamma function generating tau"""
    b_0: float = 1e-6
    """Initial value for the rate (:math:`\\beta`) in the gamma function generating tau"""

    # Output Params
    sampled_beta_indexes: list[int] = field(default_factory=list)
    """Indexes of beta estimates that need to be sampled through iterations"""
    sampled_y_indexes: list[int] = field(default_factory=list)
    """Indexes of y estimates that need to be sampled through iterations"""
    results_export_dir: str | None = None
    """Path of the folder where the csv file will be exported (if None it is printed)"""
