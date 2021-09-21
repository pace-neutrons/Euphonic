from argparse import Namespace
import pytest

from euphonic.cli.utils import _compose_style

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
