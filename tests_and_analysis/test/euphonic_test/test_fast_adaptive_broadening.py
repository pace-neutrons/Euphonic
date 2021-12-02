import os
import json
import pytest
import numpy as np
from scipy.integrate import simps

from euphonic.fast_adaptive_broadening import find_coeffs
from euphonic import ureg, QpointFrequencies, Spectrum1D
from tests_and_analysis.test.euphonic_test.test_force_constants\
    import get_fc_dir
from tests_and_analysis.test.utils import get_data_path


def get_qpt_freqs_dir(material):
    return os.path.join(get_data_path(), 'qpoint_frequencies', material)

def get_qpt_freqs(material, file):
    return QpointFrequencies.from_json_file(
        os.path.join(get_qpt_freqs_dir(material), file))

@pytest.mark.parametrize(
        ('material, qpt_freqs_json, mode_widths_json, ebins'), [
            ('quartz', 'quartz_554_full_qpoint_frequencies.json',
             'quartz_554_full_mode_widths.json',
             np.arange(0, 155, 0.1)*ureg('meV')),
            ('LZO', 'lzo_222_full_qpoint_frequencies.json',
             'lzo_222_full_mode_widths.json',
             np.arange(0, 100, 0.1)*ureg('meV'))])
def test_area_unchanged_for_broadened_dos(material, qpt_freqs_json,
                                          mode_widths_json, ebins):
    """
    Test that the area is approximately equal for unbroadened
    and broadened dos
    """
    qpt_freqs = get_qpt_freqs(material, qpt_freqs_json)
    with open(os.path.join(get_fc_dir(), mode_widths_json), 'r') as fp:
        modw_dict = json.load(fp)
    mode_widths = modw_dict['mode_widths']*ureg(
        modw_dict['mode_widths_unit'])
    dos = qpt_freqs._calculate_dos(ebins)
    adaptively_broadened_dos = qpt_freqs._calculate_dos(
        ebins, mode_widths=mode_widths, adaptive_method='fast')
    ebins_centres = \
        Spectrum1D._bin_edges_to_centres(ebins.to('hartree').magnitude)
    dos_area = simps(dos, ebins_centres)
    adaptively_broadened_dos_area = simps(adaptively_broadened_dos,
                                          ebins_centres)
    assert adaptively_broadened_dos_area == pytest.approx(dos_area, abs=1e-4)

@pytest.mark.parametrize(('spacing','expected_coeffs'),
    [(2, [-0.14384022, 1.17850278, -3.52853758, 3.49403704]),
    (np.sqrt(2), [-0.35630207, 2.81589259, -7.6385478, 6.17879599])])
def test_find_coeffs(spacing, expected_coeffs):
    """Test find_coeffs against expected coefficients"""
    coeffs = find_coeffs(spacing)
    assert coeffs == pytest.approx(expected_coeffs)
