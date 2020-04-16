import os
import numpy as np

from timeit import timeit

from euphonic.data.interpolation import InterpolationData


def get_data_path() -> str:
    """
    Returns
    -------
    str: The path to the data files for use in performance benchmarking
    """
    return os.path.join(os.path.dirname(__file__), "data")


def build_setup_calc_fine_phonons(data_path: str, seedname: str, qpts_npy_file: str, num_of_qpts: int):
    return "from euphonic.data.interpolation import InterpolationData; " \
           "import numpy as np; " \
           "idata = InterpolationData.from_castep('{}', path='{}'); " \
           "qpts = np.load('{}')[:{}]".format(seedname, data_path, qpts_npy_file, num_of_qpts)


def build_calc_fine_phonons(use_c: bool):
    if use_c:
        return "idata.calculate_fine_phonons(qpts, use_c=True, fall_back_on_python=False)"
    else:
        return "idata.calculate_fine_phonons(qpts, use_c=False)"


if __name__ == "__main__":

    qpts_npy_file = os.path.join(get_data_path(), "qpts_10000.npy")

    result = timeit(stmt=build_calc_fine_phonons(use_c=True),
                    setup=build_setup_calc_fine_phonons(get_data_path(), "Nb-242424-s0.25", qpts_npy_file, 10),
                    number=1)

    print(result)
