import math

import numpy as np
import pandas as pd


class log(float):
    """Log wrapper for math.log to get better documentation"""

    def __new__(cls, log_val, *args, **kwargs):
        return super(log, cls).__new__(cls, math.log(log_val))

    def __repr__(cls):
        return f'Log({round(math.exp(cls), 10)})'


def min_max_normalize(arr: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    """Function that min-max normalize a dataframe

    Args:
        arr (np.ndarray): The dataframe of interest
        new_min (float): The new min wanted for each column
        new_max (float): The new max wanted for each column

    Returns:
        np.ndarray: The normalized dataframe
    """
    arr_min, arr_max = arr.min(0), arr.max(0)
    curr_spread = arr_max - arr_min
    new_spread = new_max - new_min
    return (arr - arr_min) * new_spread / curr_spread + new_min


def load_numpy_array_from_csv(
    file_location: str,
    has_header: bool = True,
    columns_to_drop: list[str] = [],
    columns_to_keep: list[int] = None,
    rows_to_keep: list[int] = None,
    fill_na_with_zeros: bool = False,
    min_max_normalization: bool = False,
    transpose_matrix: bool = False,
) -> np.ndarray:
    """Utility function to load a csv as a numpy array with common transformations

    Args:
        file_location (str): The file path of the source CSV
        has_header (bool, optional): Indicator about the presence of headers in the CSV.
            Defaults to True.
        columns_to_drop (list[str], optional): A list of column name that can be removed.
            Defaults to [].
        columns_to_keep (list[int], optional): A list of column name that can remain in
            the matrix. Defaults to None.
        rows_to_keep (list[int], optional): A list of row index that can remain in the
            matrix. Defaults to None.
        fill_na_with_zeros (bool, optional): Replace missing values with zeros. Defaults to False.
        min_max_normalization (bool, optional): Apply a (0-1) min-max normalization
            on the matrix. Defaults to False.
        transpose_matrix (bool, optional): Transpose the matrix after all its transformations.
            Defaults to False.

    Returns:
        np.ndarray: A new matrix on which transformations have been applied
    """
    df = pd.read_csv(file_location, header=(0 if has_header else None))
    df.drop(columns_to_drop, axis=1, inplace=True)
    if columns_to_keep is not None:
        df = df.iloc[:, columns_to_keep]
    if rows_to_keep is not None:
        df = df.iloc[rows_to_keep, :]
    if fill_na_with_zeros:
        df.fillna(0, inplace=True)
    mat = df.to_numpy()
    if transpose_matrix:
        mat = mat.transpose()
    if min_max_normalization:
        mat = min_max_normalize(mat, 0, 1)
    return mat
