import numpy as np
import numpy.testing as npt
import pytest

from euphonic import ureg
from euphonic.spectra import Spectrum1D, Spectrum1DCollection

from ..script_tests.utils import get_ax_image_data

# Allow tests with matplotlib marker to be collected and
# deselected if Matplotlib is not installed
pytestmark = pytest.mark.matplotlib
try:
    import matplotlib  # noqa: ICN001
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import euphonic.plot
    from euphonic.plot import plot_1d, plot_1d_to_axis
except ModuleNotFoundError:
    pass

@pytest.fixture
def figure():
    fig = plt.figure()
    yield fig
    plt.close(fig)


@pytest.fixture
def axes(figure):
    return figure.add_subplot(1, 1, 1)

@pytest.fixture
def axes_with_line_and_legend(axes):
    axes.plot(np.array([0., 1., 2.]), np.array([3., 4., 5.]), label='Line A')
    axes.legend()
    return axes

def test_missing_matplotlib(mocker):
    from builtins import __import__ as builtins_import
    from importlib import reload

    import euphonic.plot

    def mocked_import(name, *args, **kwargs):
        if name.split('.')[0] == 'matplotlib':
            raise ModuleNotFoundError
        return builtins_import(name, *args, **kwargs)

    mocker.patch('builtins.__import__', side_effect=mocked_import)
    with pytest.raises(ModuleNotFoundError) as mnf_error:
        reload(euphonic.plot)

    assert ("Cannot import Matplotlib for plotting"
            in mnf_error.value.args[0])


class TestPlot1DCore:

    def teardown_method(self):
        # Ensure figures are closed
        matplotlib.pyplot.close('all')

    @pytest.mark.parametrize('spectra, expected_error',
                             [('wrong_type', TypeError), ])
    def test_1d_core_errors(self, spectra, expected_error, axes):
        with pytest.raises(expected_error):
            plot_1d_to_axis(spectra, axes)

    spec1d_args = (np.array([0., 1., 2.])*ureg('meV'),
                   np.array([2., 3., 2.])*ureg('angstrom^-2'))

    spec1d_split_args = (np.array([0., 1., 1., 2.]) * ureg('meV'),
                         np.array([2., 3., 2., 4.]) * ureg('angstrom^-2'))

    @pytest.mark.parametrize(
        'spectrum_args, spectrum_kwargs, expected_data, expected_ticks',
        [  # Case 1: Trivial
         (spec1d_args,
          {'x_tick_labels': [(0, 'A'), (2, 'B')]},
          ([[0., 1., 2.], [2., 3., 2]],),
          [(0., 'A'), (2., 'B')]),
         # Case 2: Split points create new line
         (spec1d_split_args,
          {'x_tick_labels': [(1, 'B'), (2, 'B'), (3, 'C')]},
          ([[0., 1.], [2., 3.]], [[1., 2.], [2., 4.]]),
          #  Note that duplicated points get plotted twice. Weird but harmless.
          [(1., 'B'), (1., 'B'), (2., 'C')]
          )])
    def test_plot_single_spectrum(self, spectrum_args, spectrum_kwargs,
                                  expected_data, expected_ticks, axes):
        plot_1d_to_axis(Spectrum1D(*spectrum_args, **spectrum_kwargs), axes)

        assert len(expected_data) == len(axes.lines)
        for line, expected in zip(
                axes.lines, expected_data, strict=True):

            npt.assert_allclose(line.get_xdata(), expected[0])
            npt.assert_allclose(line.get_ydata(), expected[1])

        tick_locs, tick_labels = zip(*expected_ticks, strict=True)
        npt.assert_allclose(axes.get_xticks(), tick_locs)
        assert [label.get_text() for label in axes.get_xticklabels()
                ] == list(tick_labels)


    spec1dcol_args = (np.array([0., 1., 2.])*ureg('meV'),
                      np.array([[2., 3., 2.],
                                [3., 4., 3.]])*ureg('angstrom^-2'))

    spec1dcol_split_args = (np.array([0., 1., 1., 2.])*ureg('meV'),
                            np.array([[2., 3., 2., 4.],
                                      [5., 4., 3., 2.]])*ureg('angstrom^-2'))
    @pytest.mark.parametrize(
        'spectrum_args, expected_data',
        [  # Case 1: Two lines
         (spec1dcol_args,
          ([[0., 1., 2.], [2., 3., 2.]],
           [[0., 1., 2.], [3., 4., 3.]])),
         # Case 2: Two lines with split points
         (spec1dcol_split_args,
          ([[0., 1.], [2., 3.]], [[1., 2.], [2., 4.]],
           [[0., 1.], [5., 4.]], [[1., 2.], [3., 2.]]))
           ])
    def test_plot_collection(self, spectrum_args, expected_data, axes):
        plot_1d_to_axis(Spectrum1DCollection(*spectrum_args), axes)
        assert len(expected_data) == len(axes.lines)
        for line, expected in zip(
                axes.lines, expected_data, strict=True):

            npt.assert_allclose(line.get_xdata(), expected[0])
            npt.assert_allclose(line.get_ydata(), expected[1])

    @pytest.mark.parametrize(
        'spec, labels_kwarg, expected_legend_labels',
        [(Spectrum1D(*spec1d_args), None, []),
         (Spectrum1D(*spec1d_args), ['Line B'], ['Line B']),
         (Spectrum1D(*spec1d_args,
                     metadata={'label': 'Line B'}),
          None,
          ['Line B']),
         (Spectrum1D(*spec1d_args,
                     metadata={'label': 'Line B'}),
          ['Line C'],
          ['Line C']),
         (Spectrum1D(*spec1d_split_args,
                     metadata={'label': 'Line B'}),
          'Line C',
          ['Line C']),
         (Spectrum1DCollection(*spec1dcol_args,
                               metadata={'line_data': [
                                            {'label': 'Line B'},
                                            {'label': 'Line C'}]}),
          None,
          ['Line B', 'Line C']),
         (Spectrum1DCollection(*spec1dcol_args,
                               metadata={'line_data': [
                                            {'label': 'Line B'},
                                            {'label': 'Line C'}]}),
          ['Line D', 'Line E'],
          ['Line D', 'Line E']),
         (Spectrum1DCollection(*spec1dcol_args,
                               metadata={'line_data': [
                                            {'label': 'Line B'},
                                            {'label': 'Line C'}]}),
          ['', ''],
          []),
         (Spectrum1DCollection(*spec1dcol_split_args,
                               metadata={'line_data': [
                                            {'label': 'Line B'},
                                            {'label': 'Line C'}]}),
          None,
          ['Line B', 'Line C'])])
    def test_plot_labels(self, spec, labels_kwarg, expected_legend_labels,
                         axes_with_line_and_legend):
        existing_line_label = ['Line A']
        plot_1d_to_axis(spec, axes_with_line_and_legend, labels=labels_kwarg)
        legend = axes_with_line_and_legend.get_legend()
        assert ([x.get_text() for x in legend.get_texts()]
                == existing_line_label + expected_legend_labels)

    @pytest.mark.parametrize('spec, labels', [
        (Spectrum1D(*spec1d_args), ['Line B', 'Line C']),
        (Spectrum1DCollection(*spec1dcol_args), ['Line B'])])
    def test_incorrect_length_labels_raises_value_error(
            self, spec, labels, axes):
        with pytest.raises(ValueError):
            plot_1d_to_axis(spec, axes, labels=labels)

    @pytest.mark.parametrize('kwargs', [
        ({'ls': '--'}),
        ({'color': 'r', 'ms': '+'})
    ])
    def test_extra_plot_kwargs(self, mocker, axes, kwargs):
        mock = mocker.patch('matplotlib.axes.Axes.plot',
                            return_value=None)
        spec = Spectrum1D(np.array([0., 1., 2.])*ureg('meV'),
                          np.array([2., 3., 2.])*ureg('angstrom^-2'))
        plot_1d_to_axis(spec, axes, **kwargs)

        expected_kwargs = {**{'color': None, 'label': None}, **kwargs}
        assert mock.call_args[1] == expected_kwargs

class TestPlot1D:

    def teardown_method(self):
        # Ensure figures are closed
        matplotlib.pyplot.close('all')

    @staticmethod
    def mock_core(mocker):
        return mocker.patch('euphonic.plot.plot_1d_to_axis',
                            return_value=None)

    @pytest.fixture
    def band_segments(self):
        spec1 = Spectrum1D([0., 1., 2.] * ureg('angstrom^-1'),
                           [1., 2., 1.] * ureg('meV'))
        spec2 = Spectrum1D([4., 6., 7.] * ureg('angstrom^-1'),
                           [1., 2., 1.] * ureg('meV'))
        return [spec1, spec2]

    spec1d_args = ([0., 1., 2.] * ureg('meV'),
                   [1., 2., 1.] * ureg('angstrom^-2'))
    spec1dcol_args = ([0., 1., 2.] * ureg('meV'),
                      [[1., 2., 1.], [2., 3., 2.]] * ureg('angstrom^-2'))

    @pytest.mark.parametrize(
        'spectrum, labels, kwargs',
        [(Spectrum1D(*spec1d_args),
          ['Line A'],
          {'lw': 2}),
         (Spectrum1D(*spec1d_args),
          'Line A',
          {'lw': 2}),
          (Spectrum1D(*spec1d_args, metadata={'label': ['Line B']}),
          None,
          {'ls': '--'}),
         (Spectrum1DCollection(*spec1dcol_args, x_tick_labels=[(1, 'A')]),
         ['Line D', 'Line E'],
         {'ls': '.-'}),
         (Spectrum1DCollection(*spec1dcol_args,
                               metadata={'line_data': [
                                            {'label': 'Line B'},
                                            {'label': 'Line C'}]}),
         ['Line D', 'Line E'],
         {})
         ])
    def test_plot_single(self, mocker, spectrum, labels, kwargs):
        core = self.mock_core(mocker)

        fig = plot_1d(spectrum, labels=labels, **kwargs)
        # Check args were as expected
        assert core.call_args[0][0] == spectrum
        assert core.call_args[0][1] in fig.axes
        assert core.call_args[0][2] == labels
        assert core.call_args[1] == kwargs

        plt.close(fig)

    @pytest.mark.parametrize('spec1_units', [('angstrom^-1', 'hartree'),
                                             ('bohr^-1', 'meV')])
    def test_plot_badunits(self, spec1_units, band_segments):
        spec1, spec2 = band_segments
        spec1.x_data_unit, spec1.y_data_unit = spec1_units
        spec2.x_data_unit, spec2.y_data_unit = ('angstrom^-1', 'meV')
        with pytest.raises(ValueError):
            plot_1d([spec1, spec2])

    @pytest.mark.parametrize('kwargs, expected_labels',
        [({'labels': ['band A'], 'xlabel': 'X_LABEL', 'ylabel': 'Y_LABEL',
           'title': 'TITLE'},
         ['band A']),
         ({'ymin': 0, 'ymax': 1.5}, None),
         ({},
          None)])
    def test_plot_multi_segments(
            self, mocker, band_segments, kwargs, expected_labels):

        suptitle = mocker.patch.object(matplotlib.figure.Figure, 'suptitle')
        fig = plot_1d(band_segments, **kwargs)

        legend = fig.axes[0].get_legend()
        if expected_labels is None:
            assert legend == None
        else:
            for text, label in zip(
                    legend.get_texts(), expected_labels, strict=True):
                assert text.get_text() == label
        # Ensure legend only on first subplot
        for ax in fig.axes[1:]:
            assert ax.get_legend() == None

        if 'xlabel' in kwargs:
            assert fig.axes[-1].get_xlabel() == kwargs['xlabel']
        if 'ylabel' in kwargs:
            assert fig.axes[-1].get_ylabel() == kwargs['ylabel']
        if 'title' in kwargs:
            suptitle.assert_called_once_with(kwargs['title'])

        for ax in fig.axes[:-1]:
            if 'ymin' in kwargs:
                assert ax.get_ylim()[0] == pytest.approx(kwargs.get('ymin'))
            if 'ymax' in kwargs:
                assert ax.get_ylim()[1] == pytest.approx(kwargs.get('ymax'))

    def test_plot_with_incorrect_labels_raises_valueerror(self, band_segments):
        with pytest.raises(ValueError):
            fig = plot_1d(band_segments, labels=['Band A', 'Band B'])

    @pytest.mark.parametrize('spec, kwargs', [
        (Spectrum1D(*spec1d_args), {'ls': '.-'}),
        (Spectrum1D(*spec1d_args), {'ms': '*', 'color': 'g'}),
        (Spectrum1DCollection(*spec1dcol_args), {'ms': '*', 'color': 'g'}),
        (Spectrum1DCollection(*spec1dcol_args),
         {'label': 'Line A', 'color': 'k'})
    ])
    def test_plot_kwargs(self, mocker, spec, kwargs):
        mock = mocker.patch('matplotlib.axes.Axes.plot',
                            return_value=None)
        plot_1d(spec, **kwargs)

        expected_kwargs = {**{'color': None, 'label': None}, **kwargs}
        for mock_call_args in mock.call_args_list:
            assert mock_call_args[1] == expected_kwargs

class TestPlot2D:

    def teardown_method(self):
        # Ensure figures are closed
        matplotlib.pyplot.close('all')

    @staticmethod
    def mock_core(mocker):
        return mocker.patch('euphonic.plot.plot_2d_to_axis',
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

        euphonic.plot.plot_2d_to_axis(spectrum, axes, **kwargs)

        if kwargs['norm']:
            mock_set_norm.assert_called_with(kwargs['norm'])

        image_data = get_ax_image_data(axes)

        if 'cmap' in kwargs:
            assert image_data['cmap'] == kwargs['cmap']

        expected_extent = [0., 5.280359268188477, 4., 1244.]
        npt.assert_allclose(image_data['extent'], expected_extent)

        npt.assert_allclose(
            image_data['data_1'],
            spectrum.z_data.magnitude[image_data['size'][1] // 2, :])

    @pytest.mark.parametrize('kwargs',
                             [{'title': 'TITLE',
                               'xlabel': 'X_LABEL',
                               'ylabel': 'Y_LABEL'},
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

        if 'xlabel' in kwargs:
            assert fig.axes[-1].get_xlabel() == kwargs['xlabel']
        if 'ylabel' in kwargs:
            assert fig.axes[-1].get_ylabel() == kwargs['ylabel']
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

    label_x_indices, label_values = zip(*labels, strict=True)
    plotted_labels = axes.get_xticklabels()
    plotted_positions = axes.get_xticks()

    for x_index, x, text, plotted_label in zip(
            label_x_indices, plotted_positions, label_values, plotted_labels,
            strict=True):

        assert plotted_label.get_text() == text
        assert x == pytest.approx(x_data.magnitude[x_index])
        assert plotted_label.get_rotation() == pytest.approx(angle)
