import os
from typing import List, Tuple
import numpy as np


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


def get_qpts() -> np.ndarray:
    """
    Returns
    -------
    np.ndarray
        A numpy array of 100 q-points
    """
    qpts_npy_file = os.path.join(get_data_path(), "qpts_10000.npy")
    return np.load(qpts_npy_file)[:10]
