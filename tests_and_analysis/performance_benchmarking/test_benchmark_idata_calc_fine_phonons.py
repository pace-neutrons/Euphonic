import pytest
import os
import numpy as np
from timeit import default_timer as timeittimer
from .utils import get_data_path
from euphonic.data.interpolation import InterpolationData


def run_calc_fine_phonons_benchmark(use_c: bool, data_path: str, seedname: str, qpts_npy_file: str,
                                    num_of_qpts: int, n_threads: int = 1, num_of_repeats: int = 5) -> float:
    """
    Get the average time for running the InterpolationData.calculate_fine_phonons with the given parameters.

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
    # Setup
    idata = InterpolationData.from_castep(seedname, path=data_path)
    qpts = np.load(qpts_npy_file)[:num_of_qpts]
    if use_c:
        def calc_fine_phonons():
            idata.calculate_fine_phonons(qpts, use_c=True, fall_back_on_python=False, n_threads=n_threads)
    else:
        def calc_fine_phonons():
            idata.calculate_fine_phonons(qpts, use_c=False)
    # Time and run
    times = np.zeros(num_of_repeats)
    for repeat in range(num_of_repeats):
        start = timeittimer()
        calc_fine_phonons()
        end = timeittimer()
        times[repeat] = end - start
    # Return average
    return np.mean(times)


def get_upper_bound_for_benchmark() -> float:
    """
    Get the upper bound for benchmarking the InterpolationData.calculate_fine_phonons method.

    Returns
    -------
    float : Fail the test if the average time for running the InterpolationData.calculate_fine_phonons
     method is longer than this value.
    """
    return 1.0


@pytest.mark.parametrize("seedname", ["Nb-242424-s0.25", "quartz", "La2Zr2O7"])
@pytest.mark.parametrize(("use_c", "n_threads_list"), [(True, [1, 2, 4, 12, 24]), (False, [1])])
def test_calc_fine_phonons(seedname, use_c, n_threads_list):
    qpts_npy_file = os.path.join(get_data_path(), "qpts_10000.npy")
    for n_threads in n_threads_list:
        time = run_calc_fine_phonons_benchmark(use_c=use_c, data_path=get_data_path(), seedname=seedname,
                                               qpts_npy_file=qpts_npy_file, num_of_qpts=100, n_threads=n_threads,
                                               num_of_repeats=1)
        assert time < get_upper_bound_for_benchmark()
