import json
from itertools import product
from utils import get_qpts, get_seednames, get_data_path, get_structure_factor_data_file, get_fine_phonon_data_file,\
    get_use_c_and_n_threads, get_structure_factor_num_of_repeats, get_fine_phonon_num_of_repeats
from euphonic.data.interpolation import InterpolationData
from test_benchmark_idata import get_calc_structure_factor_mean_runtime, get_calc_fine_phonons_mean_runtime


def generate_structure_factor_data():
    print("### GENERATING STRUCTURE FACTOR DATA ###")
    qpts = get_qpts()
    data = {}
    for seedname in get_seednames():
        idata = InterpolationData.from_castep(seedname, path=get_data_path())
        idata.calculate_fine_phonons(qpts, use_c=True, fall_back_on_python=False, n_threads=5)
        time = get_calc_structure_factor_mean_runtime(idata=idata, num_of_repeats=get_structure_factor_num_of_repeats())
        data[seedname] = time
        print("seedname={}, time={}".format(seedname, time))
    with open(get_structure_factor_data_file(), "w+") as data_file:
        json.dump(data, data_file)


def generate_fine_phonons_data():
    print("### GENERATING FINE PHONON DATA ###")
    data = {}
    for seedname, (use_c, n_threads_list) in product(get_seednames(), get_use_c_and_n_threads()):
        if seedname not in data:
            data[seedname] = {}
        for n_threads in n_threads_list:
            if use_c not in data[seedname]:
                data[seedname][use_c] = {}
            time = get_calc_fine_phonons_mean_runtime(
                use_c=use_c, data_path=get_data_path(), seedname=seedname,
                num_of_repeats=get_fine_phonon_num_of_repeats(), n_threads=n_threads
            )
            data[seedname][use_c][n_threads] = time
            print("seedname={}, use_c={}, n_threads={}, time={}".format(
                seedname, use_c, n_threads, time))
    with open(get_fine_phonon_data_file(), "w+") as data_file:
        json.dump(data, data_file)


if __name__ == "__main__":
    generate_structure_factor_data()
    generate_fine_phonons_data()

