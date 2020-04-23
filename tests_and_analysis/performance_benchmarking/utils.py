import os
from typing import List, Tuple
import numpy as np


def get_data_path() -> str:
    """
    Returns
    -------
    str: The path to the data files for use in performance benchmarking
    """
    return os.path.join(os.path.dirname(__file__), "data")


def get_structure_factor_data_file() -> str:
    """
    Returns
    -------
    str: The path to the data files containing benchmark bounding data for the
     InterpolationData.calculate_structure_factor method
    """
    return os.path.join(get_data_path(), "structure_factor_benchmark_data.json")


def get_fine_phonon_data_file() -> str:
    """
    Returns
    -------
    str: The path to the data files containing benchmark bounding data for the
     InterpolationData.calculate_fine_phonons method
    """
    return os.path.join(get_data_path(), "fine_phonons_benchmark_data.json")


def get_seednames() -> List[str]:
    """
    Returns
    -------
    List[str]: A list of the seednames to test with
    """
    return ["Nb-242424-s0.25", "quartz", "La2Zr2O7"]


def get_use_c_and_n_threads() -> List[Tuple[bool, List[int]]]:
    """
    Returns
    -------
    List[Tuple[bool, List[int]]]: A list of tuples which contain a first boolean element describing whetheror not to
     use c in the calcuations and a second element that is a list of the number of threads that should be tried
    """
    return [(True, [1, 2, 4, 8, 12, 16, 24]), (False, [1])]


def get_qpts() -> np.ndarray:
    """
    Returns
    -------
    np.ndarray: A numpy array of 100 q-points
    """
    qpts_npy_file = os.path.join(get_data_path(), "qpts_10000.npy")
    return np.load(qpts_npy_file)[:10]


def get_structure_factor_num_of_repeats() -> int:
    """
    Returns
    -------
    int: The amount of times to repeat InterpolationData.calculate_structure_factor runs.
    """
    return 10


def get_fine_phonon_num_of_repeats() -> int:
    """
    Returns
    -------
    int: The amount of times to repeat InterpolationData.calculate_fine_phonons runs.
    """
    return 5
