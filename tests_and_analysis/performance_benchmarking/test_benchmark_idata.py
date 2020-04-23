import pytest
import numpy as np
from timeit import default_timer as timeittimer
from utils import get_data_path, get_seednames, get_use_c_and_n_threads, get_qpts
from euphonic.data.interpolation import InterpolationData


def get_calc_fine_phonons_mean_runtime(use_c: bool, data_path: str, seedname: str,
                                       n_threads: int = 1, num_of_repeats: int = 5) -> float:
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
    qpts = get_qpts()
    if use_c:
        def calc_fine_phonons():
            idata.calculate_fine_phonons(qpts, use_c=True, fall_back_on_python=False, n_threads=n_threads,
                                         asr='reciprocal', eta_scale=0.75)
    else:
        def calc_fine_phonons():
            idata.calculate_fine_phonons(qpts, use_c=False, asr='reciprocal', eta_scale=0.75)
    # Time and run
    times = np.zeros(num_of_repeats)
    for repeat in range(num_of_repeats):
        start = timeittimer()
        calc_fine_phonons()
        end = timeittimer()
        times[repeat] = end - start
    # Return average
    return np.mean(times)


def get_upper_bound_for_calc_fine_phonons(seedname: str, use_c: bool) -> float:
    """
    Get the upper bound for benchmarking the InterpolationData.calculate_fine_phonons method.

    Returns
    -------
    float : Fail the test if the average time for running the InterpolationData.calculate_fine_phonons
     method is longer than this value.
    """
    return 1.0


@pytest.mark.parametrize("seedname", get_seednames())
@pytest.mark.parametrize(("use_c", "n_threads_list"), get_use_c_and_n_threads())
def test_calc_fine_phonons(seedname, use_c, n_threads_list):
    for n_threads in n_threads_list:
        time = get_calc_fine_phonons_mean_runtime(use_c=use_c, data_path=get_data_path(), seedname=seedname,
                                                  n_threads=n_threads, num_of_repeats=1)
        assert time <= get_upper_bound_for_calc_fine_phonons()


def get_calc_structure_factor_mean_runtime(idata: InterpolationData, num_of_repeats: int = 5) -> float:
    """
    Get the average time for running the InterpolationData.calculate_structure_factor with the given parameters.

    Parameters
    ----------
    idata : InterpolationData
        The idata with the qpts already calculated to run the structure factor calculations on.
    num_of_repeats : int, optional, default 5
        The number of times to repeat the benchmark to get an average

    Returns
    -------
    float : The average time taken to run the InterpolationData.calculate_structure_factor method.
    """
    # Setup
    sl = {
        'La': 8.24, 'Zr': 7.16, 'O': 5.803, 'C': 6.646, 'Si': 4.1491,
        'H': -3.7390, 'N': 9.36, 'S': 2.847, 'Nb': 7.054
    }
    # Time and run
    times = np.zeros(num_of_repeats)
    for repeat in range(num_of_repeats):
        start = timeittimer()
        idata.calculate_structure_factor(scattering_lengths=sl, T=100)
        end = timeittimer()
        times[repeat] = end - start
    # Return average
    return np.mean(times)


def get_upper_bound_for_calc_structure_factor() -> float:
    """
    Get the upper bound for benchmarking the InterpolationData.calculate_structure_factor method.

    Returns
    -------
    float : Fail the test if the average time for running the InterpolationData.calculate_structure_factor
     method is longer than this value.
    """
    return 1.0


@pytest.mark.parametrize("seedname", ["Nb-242424-s0.25", "quartz", "La2Zr2O7"])
def test_benchmark_calc_structure_factor(seedname):
    qpts = get_qpts()
    idata = InterpolationData.from_castep(seedname, path=get_data_path())
    idata.calculate_fine_phonons(qpts, use_c=True, fall_back_on_python=False, n_threads=5)
    time = get_calc_structure_factor_mean_runtime(idata, num_of_repeats=3)
    assert time <= get_upper_bound_for_calc_structure_factor()
