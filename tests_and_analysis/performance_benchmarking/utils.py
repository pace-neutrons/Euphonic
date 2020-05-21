import numpy as np
import os
from typing import List


def get_data_path() -> str:
    """
    Returns
    -------
    str
        The path to the data files for use in performance benchmarking
    """
    return os.path.join(os.path.dirname(__file__), "data")


def get_seednames() -> List[str]:
    """
    Returns
    -------
    List[str]
        A list of the seednames to test with
    """
    return ["Nb-242424-s0.25", "quartz", "La2Zr2O7"]


def get_threads() -> List[int]:
    """
    Returns
    -------
    List[int]
        A list of the number of threads to test with
    """
    return [1, 2, 4, 8, 12, 16, 24]


def get_qpts() -> np.ndarray:
    """
    Returns
    -------
    np.ndarray
        A numpy array of 10,000 q-points
    """
    qpts_npy_file = os.path.join(get_data_path(), "qpts_10000.npy")
    return np.load(qpts_npy_file)


def get_san_storage() -> str:
    """
    Returns
    -------
    str
        The server location of the SAN storage.
    """
    return (
        r'\\isis.cclrc.ac.uk\Shares\PACE_Project_Tool_Source'
        r'\euphonic_performance_benchmarking'
    )
