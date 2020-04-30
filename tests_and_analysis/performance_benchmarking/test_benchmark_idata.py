import pytest
from utils import get_data_path, get_seednames,\
    get_qpts
from euphonic.data.interpolation import InterpolationData


@pytest.mark.parametrize("seedname", get_seednames())
@pytest.mark.parametrize("use_c", [True, False])
@pytest.mark.parametrize("n_threads", [1, 2, 4, 8, 12, 16, 24])
def test_calc_fine_phonons(seedname, use_c, n_threads, benchmark):
    # Set up
    idata = InterpolationData.from_castep(seedname, path=get_data_path())
    qpts = get_qpts()
    # Benchmark
    if use_c:
        benchmark(
            idata.calculate_fine_phonons,
            qpts, use_c=True,
            fall_back_on_python=False,
            n_threads=n_threads,
            asr='reciprocal', eta_scale=0.75
        )
    elif n_threads == 1:
        benchmark(
            idata.calculate_fine_phonons,
            qpts, use_c=False,
            asr='reciprocal', eta_scale=0.75
        )


@pytest.mark.parametrize("seedname", get_seednames())
def test_benchmark_calc_structure_factor(seedname, benchmark):
    # Set up
    qpts = get_qpts()
    idata = InterpolationData.from_castep(seedname, path=get_data_path())
    idata.calculate_fine_phonons(
        qpts, use_c=True, fall_back_on_python=False, n_threads=5
    )
    scattering_lengths = {
        'La': 8.24, 'Zr': 7.16, 'O': 5.803, 'C': 6.646, 'Si': 4.1491,
        'H': -3.7390, 'N': 9.36, 'S': 2.847, 'Nb': 7.054
    }
    # Benchmark
    benchmark(
        idata.calculate_structure_factor,
        scattering_lengths=scattering_lengths, T=100
    )
