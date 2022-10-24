import pytest
import numpy as np
import numpy.testing as npt
from scipy.integrate import simps

from euphonic.broadening import (find_coeffs,
                                 width_interpolated_broadening,
                                 polynomial_broadening)
from euphonic import ureg
from tests_and_analysis.test.euphonic_test.test_force_constants\
    import get_fc_path
from tests_and_analysis.test.euphonic_test.test_qpoint_frequencies\
    import get_qpt_freqs
from tests_and_analysis.test.utils import get_mode_widths
from tests_and_analysis.test.euphonic_test.test_spectrum1d\
    import get_expected_spectrum1d


def test_polynomial_close_to_exact():
    """Check polynomial broadening agrees with exact for trivial case"""
    from numpy.random import RandomState
    from scipy.ndimage import gaussian_filter
    from numpy.polynomial import Polynomial

    rng = RandomState(123)

    bins = np.linspace(0, 100, 200)
    bin_width = bins[1] - bins[0]
    x = (bins[1:] + bins[:-1]) / 2
    y = np.zeros_like(x)
    y[rng.randint(0, len(x), 20)] = rng.rand(20)

    sigma = 2.
    exact = gaussian_filter(y, (sigma / bin_width), mode='constant')

    poly_broadened = polynomial_broadening(
        bins=(bins * ureg('meV')),
        x=(x * ureg('meV')),
        width_polynomial=(Polynomial([sigma, 0., 0.]), ureg('meV')),
        weights=(y * bin_width),  # Convert from spectrum heights to counts
        adaptive_error=1e-5)

    npt.assert_allclose(exact, poly_broadened.to('1/meV').magnitude,
                        atol=1e-4)


@pytest.mark.parametrize(
        ('material, qpt_freqs_json, mode_widths_json, ebins'), [
            ('quartz', 'quartz_554_full_qpoint_frequencies.json',
             'quartz_554_full_mode_widths.json',
             np.arange(0, 155, 0.1)*ureg('meV'))])
def test_area_unchanged_for_broadened_dos(material, qpt_freqs_json,
                                          mode_widths_json, ebins):
    """
    Test that the area is approximately equal for unbroadened and
    broadened dos
    """
    qpt_freqs = get_qpt_freqs(material, qpt_freqs_json)
    mode_widths = get_mode_widths(get_fc_path(mode_widths_json))
    dos = qpt_freqs.calculate_dos(ebins)
    weights = np.full(qpt_freqs.frequencies.shape,
                      1/(qpt_freqs.n_qpts*qpt_freqs.crystal.n_atoms))
    variable_width_broaden = width_interpolated_broadening(
                                ebins,
                                qpt_freqs.frequencies,
                                mode_widths, weights,
                                0.01)
    ebins_centres = ebins.magnitude[:-1] + 0.5*np.diff(ebins.magnitude)
    assert dos.y_data.units == 1/ebins.units
    dos_area = simps(dos.y_data.magnitude, ebins_centres)
    assert variable_width_broaden.units == 1/ebins.units
    adaptively_broadened_dos_area = simps(variable_width_broaden.magnitude,
                                          ebins_centres)
    assert adaptively_broadened_dos_area == pytest.approx(dos_area, rel=0.01)


@pytest.mark.parametrize(
        ('material, qpt_freqs_json, mode_widths_json,'
         'expected_dos_json, ebins'), [
            ('quartz', 'toy_quartz_qpoint_frequencies.json',
             'toy_quartz_mode_widths.json',
             'toy_quartz_adaptive_dos.json',
             np.arange(0, 40, 0.1)*ureg('meV'))])
def test_lower_bound_widths_broadened(material, qpt_freqs_json,
                                      mode_widths_json,
                                      expected_dos_json, ebins):
    """
    Test to ensure that points with mode width equal to
    min(width_samples) are broadened
    """
    qpt_freqs = get_qpt_freqs(material, qpt_freqs_json)
    mode_widths = get_mode_widths(get_fc_path(mode_widths_json))
    weights = np.ones(qpt_freqs.frequencies.shape) * \
        np.full(qpt_freqs.n_qpts, 1/qpt_freqs.n_qpts)[:, np.newaxis]
    dos = width_interpolated_broadening(ebins, qpt_freqs.frequencies,
                                        mode_widths, weights, 0.01)
    expected_dos = get_expected_spectrum1d(expected_dos_json)
    npt.assert_allclose(expected_dos.y_data.magnitude, dos.magnitude)


def test_uneven_bin_width_raises_warning():
    qpt_freqs = get_qpt_freqs('quartz',
                              'quartz_554_full_qpoint_frequencies.json')
    mode_widths = get_mode_widths(get_fc_path(
        'quartz_554_full_mode_widths.json'))
    weights = np.ones(qpt_freqs.frequencies.shape) * \
        np.full(qpt_freqs.n_qpts, 1/qpt_freqs.n_qpts)[:, np.newaxis]
    ebins = np.concatenate((np.arange(0, 100, 0.1),
                             np.arange(100, 155, 0.2)))*ureg('meV')
    with pytest.warns(UserWarning):
        width_interpolated_broadening(ebins, qpt_freqs.frequencies,
                                      mode_widths, weights, 0.01)

@pytest.mark.parametrize(('spacing','expected_coeffs'),
    [(2, [-0.1883858, 1.46930932, -4.0893793, 3.80872458]),
    (np.sqrt(2), [-0.60874207, 4.10599029, -9.63986481, 7.14267836])])
def test_find_coeffs(spacing, expected_coeffs):
    """Test find_coeffs against expected coefficients"""
    coeffs = find_coeffs(spacing)
    assert coeffs == pytest.approx(expected_coeffs)
