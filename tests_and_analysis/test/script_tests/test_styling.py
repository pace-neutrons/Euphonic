from argparse import Namespace

import pytest
import matplotlib.pyplot
from numpy.testing import assert_allclose

import euphonic.cli.dos
from euphonic.cli.utils import _compose_style

from tests_and_analysis.test.utils import get_castep_path

compose_style_cases = [
    ({'user_args': Namespace(unused=1, no_base_style=False, style=None),
      'base': None}, [{}]),
    ({'user_args': Namespace(unused=1, no_base_style=False, style=None),
      'base': ['dark_background']}, ['dark_background', {}]),
    ({'user_args': Namespace(unused=1, no_base_style=True, style=None),
      'base': ['dark_background']}, [{}]),
    ({'user_args': Namespace(no_base_style=False, style=None),
      'base': ['my/imaginary/file', 'dark_background']},
     ['my/imaginary/file', 'dark_background', {}]),
    ({'user_args': Namespace(no_base_style=False, style=['ggplot', 'seaborn']),
      'base': ['my/imaginary/file', 'dark_background']},
     ['my/imaginary/file', 'dark_background', 'ggplot', 'seaborn', {}]),
    ({'user_args': Namespace(unused=1, no_base_style=False, style=['ggplot'],
                             cmap='bone', fontsize=12, font='Comic Sans',
                             linewidth=4, figsize=[1, 2], figsize_unit='inch'),
      'base': None}, ['ggplot', {'image.cmap': 'bone',
                                 'font.size': 12,
                                 'font.family': 'sans-serif',
                                 'font.sans-serif': 'Comic Sans',
                                 'lines.linewidth': 4,
                                 'figure.figsize': [1, 2]}]),
    ({'user_args': Namespace(unused=1, no_base_style=False, style=['ggplot'],
                             figsize=[2.54, 2.54], figsize_unit='cm'),
      'base': None}, ['ggplot', {'figure.figsize': [1., 1.]}]),
    ]


@pytest.mark.unit
@pytest.mark.parametrize('kwargs,expected_style', compose_style_cases)
def test_compose_style(kwargs, expected_style):
    """Internal function which interprets matplotlib style options"""
    assert _compose_style(**kwargs) == expected_style


@pytest.mark.integration
class TestDOSStyling:

    @pytest.fixture
    def inject_mocks(self, mocker):
        # Prevent calls to show so we can get the current figure using
        # gcf()
        mocker.patch('matplotlib.pyplot.show')
        mocker.resetall()

    def teardown_method(self):
        # Ensure figures are closed
        matplotlib.pyplot.close('all')

    @pytest.mark.parametrize('dos_args',
        [[get_castep_path('NaH', 'NaH.phonon'),
          '--style=dark_background',
          '--linewidth=5.',
          '--fontsize=11.',
          '--figsize', '4', '4',
          '--figsize-unit', 'inch'],
         [get_castep_path('NaH', 'NaH.phonon'),
          '--style=dark_background',
          '--line-width=5.',
          '--font-size=11.',
          '--fig-size', '4', '4',
          '--fig-size-unit', 'inch']])
    def test_dos_styling(self, inject_mocks, dos_args):
        euphonic.cli.dos.main(params=dos_args)

        fig = matplotlib.pyplot.gcf()

        assert_allclose([0., 0., 0., 1.], fig.get_facecolor())
        assert fig.axes[0].lines[0].get_linewidth() == pytest.approx(5.)
        assert_allclose([4., 4.], fig.get_size_inches())
        # Font size: base size 11 * "small" factor 0.833
        assert (fig.axes[0].get_xticklabels()[0].get_fontsize()
                == pytest.approx(0.833 * 11))
