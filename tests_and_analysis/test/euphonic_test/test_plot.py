import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

from euphonic import ureg
from euphonic.plot import _plot_1d_core
from euphonic.spectra import Spectrum1D, Spectrum1DCollection


@pytest.fixture
def figure():
    fig = plt.figure()
    yield fig
    plt.close(fig)

@pytest.fixture
def axes(figure):
    ax = figure.add_subplot()
    return ax

class TestPlot1DCore:
    @pytest.mark.parametrize('spectra, expected_error',
                             [('wrong_type', TypeError),])
    def test_1d_core_errors(self, spectra, expected_error, axes):
        with pytest.raises(expected_error):
            _plot_1d_core(spectra, axes)

    @pytest.mark.parametrize('spectrum_params, expected_data',
                        [# Case 1: Trivial
                         ((np.array([0., 1., 2.]) * ureg('meV'),
                           np.array([2., 3., 2.]) * ureg('angstrom^-2')),
                          ([[0., 1., 2.], [2., 3., 2]],)),
                         # Case 2: Split points create new line
                         ((np.array([0., 1., 1., 2.]) * ureg('meV'),
                           np.array([2., 3., 2., 4.]) * ureg('angstrom^-2')),
                          ([[0., 1.], [2., 3.]], [[1., 2.], [2., 4.]]))
                          ])
    def test_plot_single_spectrum(self, spectrum_params, expected_data, axes):
        _plot_1d_core(Spectrum1D(*spectrum_params), axes)

        assert len(expected_data) == len(axes.lines)
        for line, expected in zip(axes.lines, expected_data):

            npt.assert_allclose(line.get_xdata(), expected[0])
            npt.assert_allclose(line.get_ydata(), expected[1])

    # def test_plot_collection(self, simple_spectrum1d, axes):
    #     _plot_1d_core(simple_spectrum1d, axes)

    #     assert len(axes.lines) == 1
    #     npt.assert_allclose(axes.lines[0].get_xdata(),
    #                         simple_spectrum1d.x_data.magnitude)
    #     npt.assert_allclose(axes.lines[0].get_ydata(),
    #                         simple_spectrum1d.y_data.magnitude)
