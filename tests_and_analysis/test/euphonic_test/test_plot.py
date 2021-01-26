import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

from euphonic import ureg
import euphonic.plot
from euphonic.plot import plot_1d, _plot_1d_core
from euphonic.spectra import Spectrum1D, Spectrum1DCollection

from ..script_tests.utils import get_ax_image_data


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
def test_missing_matplotlib(mocker):
    from builtins import __import__ as builtins_import
    from importlib import reload
    import euphonic.plot

    def mocked_import(name, *args, **kwargs):
        if name == 'matplotlib':
            raise ModuleNotFoundError
        return builtins_import(name, *args, **kwargs)

    mocker.patch('builtins.__import__', side_effect=mocked_import)
    with pytest.raises(ModuleNotFoundError) as mnf_error:
        reload(euphonic.plot)

    assert ("Cannot import Matplotlib for plotting"
            in mnf_error.value.args[0])


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

    @pytest.fixture
    def band_segments(self):
        spec1 = Spectrum1D([0., 1., 2.] * ureg('angstrom^-1'),
                           [1., 2., 1.] * ureg('meV'))
        spec2 = Spectrum1D([4., 6., 7.] * ureg('angstrom^-1'),
                           [1., 2., 1.] * ureg('meV'))
        return [spec1, spec2]

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

    @pytest.mark.parametrize('spec1_units', [('angstrom^-1', 'hartree'),
                                             ('bohr^-1', 'meV')])
    def test_plot_badunits(self, spec1_units, band_segments):
        spec1, spec2 = band_segments
        spec1.x_data_unit, spec1.y_data_unit = spec1_units
        spec2.x_data_unit, spec2.y_data_unit = ('angstrom^-1', 'meV')
        with pytest.raises(ValueError):
            plot_1d([spec1, spec2])

    @pytest.mark.parametrize('kwargs', [{'labels': ['band A', 'band B']},
                                        {'x_label': 'X_LABEL',
                                         'y_label': 'Y_LABEL',
                                         'title': 'TITLE'}])
    def test_plot_multi(self, mocker, band_segments, kwargs):

        suptitle = mocker.patch.object(matplotlib.figure.Figure, 'suptitle')

        fig = plot_1d(band_segments, **kwargs)

        if 'labels' in kwargs:
            legend = fig.axes[0].get_legend()
            assert legend is not None

            for text, label in zip(legend.get_texts(), kwargs['labels']):
                assert text.get_text() == label

        if 'x_label' in kwargs:
            assert fig.axes[-1].get_xlabel() == kwargs['x_label']
        if 'y_label' in kwargs:
            assert fig.axes[-1].get_ylabel() == kwargs['y_label']
        if 'title' in kwargs:
            suptitle.assert_called_once_with(kwargs['title'])

        for ax in fig.axes[:-1]:
            if 'y_min' in kwargs:
                assert ax.get_ylim()[0] == pytest.approx(kwargs.get('y_min'))
            if 'y_max' in kwargs:
                assert ax.get_ylim()[0] == pytest.approx(kwargs.get('y_max'))


@pytest.mark.unit
class TestPlot2D:
    @staticmethod
    def mock_core(mocker):
        return mocker.patch('euphonic.plot._plot_2d_core',
                            return_value=None)

    @pytest.fixture
    def spectrum(self):
        from .test_spectrum2d import get_spectrum2d as get_ref_spectrum2d
        return get_ref_spectrum2d('quartz_bandstructure_sqw.json')

    @pytest.mark.parametrize('kwargs', [{'cmap': 'magma',
                                         'interpolation': 'nearest',
                                         'norm': 'some-norm'},
                                        {'cmap': 'magma',
                                         'interpolation': 'bilinear',
                                         'norm': None}])
    def test_plot(self, axes, spectrum, kwargs, mocker):
        mock_set_norm = mocker.patch.object(matplotlib.image.NonUniformImage,
                                            'set_norm')

        euphonic.plot._plot_2d_core(spectrum, axes, **kwargs)

        if kwargs['norm']:
            mock_set_norm.assert_called_with(kwargs['norm'])

        image_data = get_ax_image_data(axes)

        if 'cmap' in kwargs:
            assert image_data['cmap'] == kwargs['cmap']

        expected_extent = [0., 5.280359268188477, 0.5, 153.5]
        npt.assert_allclose(image_data['extent'], expected_extent)

        npt.assert_allclose(
            image_data['data_1'],
            spectrum.z_data.magnitude[image_data['size'][1] // 2, :])

    @pytest.mark.parametrize('kwargs',
                             [{'title': 'TITLE',
                               'x_label': 'X_LABEL',
                               'y_label': 'Y_LABEL'},
                              {'vmin': 1.,
                               'vmax': 2.,
                               'cmap': 'magma'}])
    def test_plot_single(self, mocker, spectrum, kwargs):
        core = self.mock_core(mocker)
        suptitle = mocker.patch.object(matplotlib.figure.Figure, 'suptitle')

        fig = euphonic.plot.plot_2d(spectrum, **kwargs)
        # Check args were as expected
        assert core.call_args[0][0] == spectrum
        assert core.call_args[0][1] in fig.axes

        if 'cmap' in kwargs:
            assert core.call_args[1]['cmap'] == kwargs['cmap']

        norm = core.call_args[1]['norm']
        for attr in ('vmin', 'vmax'):
            if attr in kwargs:
                assert getattr(norm, attr) == pytest.approx(kwargs[attr])

        if 'x_label' in kwargs:
            assert fig.axes[-1].get_xlabel() == kwargs['x_label']
        if 'y_label' in kwargs:
            assert fig.axes[-1].get_ylabel() == kwargs['y_label']
        if 'title' in kwargs:
            suptitle.assert_called_once_with(kwargs['title'])

        plt.close(fig)

    def test_plot_multi(self, mocker, spectrum):
        core = self.mock_core(mocker)

        spectra = spectrum.split(indices=[10])
        euphonic.plot.plot_2d(spectra)

        call_1, call_2 = core.call_args_list

        assert call_1[0][0] == spectra[0]
        assert call_2[0][0] == spectra[1]


@pytest.mark.unit
@pytest.mark.parametrize('labels, rotate',
                         [([(1, 'A'), (3, 'B'), (4, 'CDEF')], False),
                          ([(0, 'A'), (3, 'THISISALONGLABEL')], True)])
def test_set_x_tick_labels(axes, labels, rotate):
    from euphonic.plot import _set_x_tick_labels

    x_data = np.array([0., 1., 2., 3., 4.]) * ureg('angstrom^-1')

    _set_x_tick_labels(axes, labels, x_data)

    if rotate:
        angle = 90.
    else:
        angle = 0.

    label_x_indices, label_values = zip(*labels)
    plotted_labels = axes.get_xticklabels()
    plotted_positions = axes.get_xticks()

    for x_index, x, text, plotted_label in zip(
            label_x_indices, plotted_positions, label_values, plotted_labels):

        assert plotted_label.get_text() == text
        assert x == pytest.approx(x_data.magnitude[x_index])
        assert plotted_label.get_rotation() == pytest.approx(angle)
