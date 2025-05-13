from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial
from numpy.random import RandomState
import numpy.testing as npt
import pytest
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter

from euphonic import Quantity, ureg
from euphonic.broadening import (
    find_coeffs,
    variable_width_broadening,
    width_interpolated_broadening,
)
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc_path,
)
from tests_and_analysis.test.euphonic_test.test_qpoint_frequencies import (
    get_qpt_freqs,
)
from tests_and_analysis.test.euphonic_test.test_spectrum1d import (
    get_expected_spectrum1d,
)
from tests_and_analysis.test.utils import get_mode_widths


@dataclass
class ExampleData:
    bins: Quantity
    x: Quantity
    y: Quantity

    def __post_init__(self) -> None:
        self.bin_width = self.bins[1] - self.bins[0]

@pytest.fixture
def example() -> ExampleData:
    rng = RandomState(123)

    bins = np.linspace(0, 100, 200) * ureg('meV')
    x = (bins[1:] + bins[:-1]) / 2

    y = np.zeros_like(x) * ureg('1/meV')
    y[rng.randint(0, len(x), 20)] = rng.rand(20) * ureg('1/meV')

    return ExampleData(bins, x, y)


def test_variable_close_to_exact(example):
    """Check variable-width broadening agrees with exact for trivial case"""

    sigma = 2. * ureg('meV')
    exact = gaussian_filter(example.y.magnitude,
                            (sigma / example.bin_width).to('dimensionless').magnitude,
                            mode='constant')

    def width_function(x):
        poly = Polynomial([sigma.to('meV').magnitude, 0., 0.])
        return poly(x.to('meV').magnitude) * ureg('meV')

    poly_broadened = variable_width_broadening(
        bins=example.bins,
        x=example.x,
        width_function=width_function,
        width_convention='STD',
        weights=(example.y * example.bin_width).to('dimensionless').magnitude,  # Convert from spectrum heights to counts
        adaptive_error=2e-4,
        fit='cheby-log')

    npt.assert_allclose(exact, poly_broadened.to('1/meV').magnitude,
                        atol=1e-4)


@pytest.mark.parametrize(
    'width_convention, fit, adaptive_error, message',
    [('STD', 'cheby-log', 0.001, 'Standard deviation unavailable'),
     ('UNIMPLEMENTED', 'cheby-log', 0.001, 'must be "std" or "fwhm"'),
     ('FWHM', 'UNIMPLEMENTED', 0.001, 'Fit "UNIMPLEMENTED" is not available'),
     ('FWHM', 'cheby-log', 0.00001, 'Target error is out of fit range'),
     ])
def test_variable_width_broadening_errors(example, width_convention, fit, adaptive_error, message):

    with pytest.raises(ValueError, match=message):
        variable_width_broadening(
            bins=example.bins,
            x=example.x,
            width_function=(lambda x: x),
            width_convention=width_convention,
            weights=example.y.magnitude,
            shape='lorentz',
            fit=fit,
            adaptive_error=adaptive_error)


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
                                0.01,
                                fit='cheby-log')
    ebins_centres = ebins.magnitude[:-1] + 0.5*np.diff(ebins.magnitude)
    assert dos.y_data.units == 1/ebins.units
    dos_area = simpson(dos.y_data.magnitude, x=ebins_centres)
    assert variable_width_broaden.units == 1/ebins.units
    adaptively_broadened_dos_area = simpson(
        variable_width_broaden.magnitude, x=ebins_centres)
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
                                        mode_widths, weights, 0.01,
                                        fit='cubic')
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
                                      mode_widths, weights, 0.01,
                                      fit='cubic')


@pytest.mark.parametrize(
    ('spacing', 'expected_coeffs'),
    [(2, [-0.1883858, 1.46930932, -4.0893793, 3.80872458]),
     (np.sqrt(2), [-0.60874207, 4.10599029, -9.63986481, 7.14267836])])
def test_find_coeffs(spacing, expected_coeffs):
    """Test find_coeffs against expected coefficients"""
    coeffs = find_coeffs(spacing)
    assert coeffs == pytest.approx(expected_coeffs)

@pytest.mark.parametrize(
        'adaptive_error', [(1e-6,), (0.2,)])
def test_bad_range_raises_error(adaptive_error):
    qpt_freqs = get_qpt_freqs('quartz',
                              'toy_quartz_qpoint_frequencies.json')
    mode_widths = np.ones(qpt_freqs.frequencies.shape) * ureg('meV')
    weights = np.ones(qpt_freqs.frequencies.shape)
    ebins = np.arange(0, 150, 0.1) * ureg('meV')
    with pytest.raises(ValueError):
        width_interpolated_broadening(ebins, qpt_freqs.frequencies,
                                      mode_widths, weights,
                                      adaptive_error=adaptive_error)
