import os
import pytest
from utils import get_data_path, get_seednames,\
    get_qpts

from euphonic import ureg, ForceConstants


@pytest.mark.parametrize("seedname", get_seednames())
@pytest.mark.parametrize("use_c", [True, False])
@pytest.mark.parametrize("n_threads", [1, 2, 4, 8, 12, 16, 24])
def test_calculate_qpoint_phonon_modes(seedname, use_c, n_threads, benchmark):
    # Set up
    fc = ForceConstants.from_castep(
            os.path.join(get_data_path(), f'{seedname}.castep_bin'))
    qpts = get_qpts()
    # Benchmark
    if use_c:
        benchmark(
            fc.calculate_qpoint_phonon_modes,
            qpts, use_c=True,
            fall_back_on_python=False,
            n_threads=n_threads,
            asr='reciprocal', eta_scale=0.75
        )
    elif n_threads == 1:
        benchmark(
            fc.calculate_qpoint_phonon_modes,
            qpts, use_c=False,
            asr='reciprocal', eta_scale=0.75
        )


@pytest.mark.parametrize("seedname", get_seednames())
def test_calculate_structure_factor(seedname, benchmark):
    # Set up
    qpts = get_qpts()
    fc = ForceConstants.from_castep(
            os.path.join(get_data_path(), f'{seedname}.castep_bin'))
    phonons = fc.calculate_qpoint_phonon_modes(
        qpts, use_c=True, fall_back_on_python=False, n_threads=5
    )
    fm = ureg('fm')
    scattering_lengths = {
        'La': 8.24*fm, 'Zr': 7.16*fm, 'O': 5.803*fm, 'C': 6.646*fm,
        'Si': 4.1491*fm, 'H': -3.7390*fm, 'N': 9.36*fm, 'S': 2.847*fm,
        'Nb': 7.054*fm
    }
    # Benchmark
    benchmark(
        phonons.calculate_structure_factor,
        scattering_lengths=scattering_lengths
    )
