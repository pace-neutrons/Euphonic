import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

from euphonic import ureg
from euphonic.plot import _plot_1d_core, plot_1d
from euphonic.spectra import Spectrum1D, Spectrum1DCollection


@pytest.fixture
def figure():
    fig = plt.figure()
    yield fig
    plt.close(fig)


@pytest.fixture
def axes(figure):
    ax = figure.add_subplot(1, 1, 1)
    return ax


@pytest.mark.unit
class TestPlot1DCore:
    @pytest.mark.parametrize('spectra, expected_error',
                             [('wrong_type', TypeError), ])
    def test_1d_core_errors(self, spectra, expected_error, axes):
        with pytest.raises(expected_error):
            _plot_1d_core(spectra, axes)

    @pytest.mark.parametrize(
        'spectrum_params, spectrum_kwargs, expected_data, expected_ticks',
        [  # Case 1: Trivial
         ((np.array([0., 1., 2.]) * ureg('meV'),
           np.array([2., 3., 2.]) * ureg('angstrom^-2')),
          {'x_tick_labels': [(0, 'A'), (2, 'B')]},
          ([[0., 1., 2.], [2., 3., 2]],),
          [(0., 'A'), (2., 'B')]),
         # Case 2: Split points create new line
         ((np.array([0., 1., 1., 2.]) * ureg('meV'),
           np.array([2., 3., 2., 4.]) * ureg('angstrom^-2')),
          {'x_tick_labels': [(1, 'B'), (2, 'B'), (3, 'C')]},
          ([[0., 1.], [2., 3.]], [[1., 2.], [2., 4.]]),
          #  Note that duplicated points get plotted twice. Weird but harmless.
          [(1., 'B'), (1., 'B'), (2., 'C')]
          )])
    def test_plot_single_spectrum(self, spectrum_params, spectrum_kwargs,
                                  expected_data, expected_ticks, axes):
        _plot_1d_core(Spectrum1D(*spectrum_params, **spectrum_kwargs), axes)

        assert len(expected_data) == len(axes.lines)
        for line, expected in zip(axes.lines, expected_data):

            npt.assert_allclose(line.get_xdata(), expected[0])
            npt.assert_allclose(line.get_ydata(), expected[1])

        tick_locs, tick_labels = zip(*expected_ticks)
        npt.assert_allclose(axes.get_xticks(), tick_locs)
        assert [label.get_text() for label in axes.get_xticklabels()
                ] == list(tick_labels)

    @pytest.mark.parametrize(
        'spectrum_params, expected_data',
        [  # Case 1: Two lines
         ((np.array([0., 1., 2.]) * ureg('meV'),
           np.array([[2., 3., 2.],
                     [3., 4., 3.]]) * ureg('angstrom^-2')),
          ([[0., 1., 2.], [2., 3., 2.]],
           [[0., 1., 2.], [3., 4., 3.]])),
         # Case 2: Two lines with split points
         ((np.array([0., 1., 1., 2.]) * ureg('meV'),
           np.array([[2., 3., 2., 4.],
                     [5., 4., 3., 2.]]) * ureg('angstrom^-2')),
          ([[0., 1.], [2., 3.]], [[1., 2.], [2., 4.]],
           [[0., 1.], [5., 4.]], [[1., 2.], [3., 2.]]))
           ])
    def test_plot_collection(self, spectrum_params, expected_data, axes):
        _plot_1d_core(Spectrum1DCollection(*spectrum_params), axes)
        assert len(expected_data) == len(axes.lines)
        for line, expected in zip(axes.lines, expected_data):

            npt.assert_allclose(line.get_xdata(), expected[0])
            npt.assert_allclose(line.get_ydata(), expected[1])

@pytest.mark.unit
class TestPlot1D:
    @staticmethod
    def mock_core(mocker):
        return mocker.patch('euphonic.plot._plot_1d_core',
                            return_value=None)

    @pytest.mark.parametrize(
        'spectrum',
        [Spectrum1D([0., 1., 2.] * ureg('meV'),
                    [1., 2., 1.] * ureg('angstrom^-2')),
         Spectrum1DCollection([0., 1., 2.] * ureg('meV'),
                              [[1., 2., 1.],
                               [2., 3., 2.]] * ureg('angstrom^-2'),
                               x_tick_labels=[(1, 'A')]),
         ])
    def test_plot_single(self, mocker, spectrum):
        core = self.mock_core(mocker)

        fig = plot_1d(spectrum)
        # Check args were as expected
        assert core.call_args[0][0] == spectrum
        assert core.call_args[0][1] in fig.axes
        
        plt.close(fig)
