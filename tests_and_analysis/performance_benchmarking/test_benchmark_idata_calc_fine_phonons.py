import pytest
import os
from timeit import timeit
from .utils import get_data_path


def build_setup_calc_fine_phonons(data_path: str, seedname: str, qpts_npy_file: str, num_of_qpts: int) -> str:
    """
    Build the setup code for timing the InterpolationData.calculate_fine_phonons method.

    Parameters
    ----------
    data_path : str
        The path to the data (castep_bin file)
    seedname : str
        The name of the castep_bin file
    qpts_npy_file : str
        The string path to an npy file containing q-points
    num_of_qpts : int
        The number of q-points to run InterpolationData.calculate_fine_phonons against

    Returns
    -------
    A string containing code for timeit to run as setup for the timed function with an exec(...) call.
    """
    return "from euphonic.data.interpolation import InterpolationData; " \
           "import numpy as np; " \
           "idata = InterpolationData.from_castep('{}', path='{}'); " \
           "qpts = np.load('{}')[:{}]".format(seedname, data_path, qpts_npy_file, num_of_qpts)


def build_calc_fine_phonons(use_c: bool, n_threads: int = 1) -> str:
    """
    Build the code for timing the InterpolationData.calculate_fine_phonons method.

    Parameters
    ----------
    use_c : bool
        True if we are timing the InterpolationData.calculate_fine_phonons with the c extension,
         False if we wish to use Python.
    n_threads : int, optional, default 1
        The number of threads to use when looping over q-points in C. Only applicable if use_c=True

    Returns
    -------
    str : A string containing code for timeit to run and time with an exec(...) call.
    """
    if use_c:
        return "idata.calculate_fine_phonons(qpts, use_c=True, fall_back_on_python=False, n_threads={n_threads})".\
            format(n_threads=str(n_threads))
    else:
        return "idata.calculate_fine_phonons(qpts, use_c=False)"


def run_calc_fine_phonons_benchmark(use_c: bool, data_path: str, seedname: str, qpts_npy_file: str,
                                    num_of_qpts: int, n_threads: int = 1, num_of_repeats: int = 5) -> float:
    """
    Run a benchmark for the InterpolationData.calculate_fine_phonons method.

    Parameters
    ----------
    use_c : True if we are timing the InterpolationData.calculate_fine_phonons with the c extension,
         False if we wish to use Python.
    data_path : str
        The path to the data (castep_bin file)
    seedname : str
        The name of the castep_bin file
    qpts_npy_file : str
        The string path to an npy file containing q-points
    num_of_qpts : int
        The number of q-points to run InterpolationData.calculate_fine_phonons against
    n_threads : int, optional, default 1
        The number of threads to use when looping over q-points in C. Only applicable if use_c=True
    num_of_repeats : int, optional, default 5
        The number of times to repeat the benchmark to get an average

    Returns
    -------
    float : The average time taken to run the InterpolationData.calculate_fine_phonons method.
    """
    return timeit(
        stmt=build_calc_fine_phonons(use_c, n_threads=n_threads),
        setup=build_setup_calc_fine_phonons(data_path, seedname, qpts_npy_file, num_of_qpts),
        number=num_of_repeats
    )


def get_upper_bound_for_benchmark() -> float:
    """
    Get the upper bound for benchmarking the InterpolationData.calculate_fine_phonons method.

    Returns
    -------
    float : Fail the test if the average time for running the InterpolationData.calculate_fine_phonons
     method is longer than this value.
    """
    return 1.0


@pytest.mark.parametrize(("use_c", "n_threads_list"), [(True, [1, 2, 4, 12, 24]), (False, [1])])
def test_calc_fine_phonons(use_c, n_threads_list):
    qpts_npy_file = os.path.join(get_data_path(), "qpts_10000.npy").replace("\\", "\\\\")
    for n_threads in n_threads_list:
        time = run_calc_fine_phonons_benchmark(use_c=use_c, data_path=get_data_path(), seedname="Nb-242424-s0.25",
                                               qpts_npy_file=qpts_npy_file, num_of_qpts=100, n_threads=n_threads,
                                               num_of_repeats=5)
        assert time < get_upper_bound_for_benchmark()
