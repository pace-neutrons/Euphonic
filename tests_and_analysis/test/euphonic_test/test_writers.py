import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.testing import assert_allclose
import pytest

from euphonic.qpoint_phonon_modes import QpointPhononModes
from euphonic.spectra import XTickLabels
from euphonic.writers.phonon_website import (
    PhononWebsiteData, write_phonon_website_json)
from tests_and_analysis.test.utils import get_data_path


class WritePhononWebsiteKwargs(TypedDict):
    output_file: str | Path
    name: str
    x_tick_labels: XTickLabels | None


@pytest.fixture
def modes_data(
) -> tuple[QpointPhononModes, WritePhononWebsiteKwargs, PhononWebsiteData]:
    """Get modes and reference website data for writer test"""

    input_filename = get_data_path('writers', 'NaCl_minimal_modes.json')
    modes = QpointPhononModes.from_json_file(input_filename)

    ref_filename = get_data_path(
        'writers', 'NaCl_minimal_modes_web_defaults.json')
    with open(ref_filename, 'r', encoding='utf8') as fd:
        ref_data = json.load(fd)

    kwargs = {'output_file': 'phonons.json',
              'name': 'Euphonic export',
              'x_tick_labels': None}

    return (modes, kwargs, ref_data)


def test_phonon_website_writer(
        modes_data: tuple[
            QpointPhononModes, WritePhononWebsiteKwargs, PhononWebsiteData]
) -> None:
    modes, kwargs, ref_data = modes_data

    write_phonon_website_json(modes, **kwargs)

    with open(kwargs['output_file'], 'r', encoding='utf8') as fd:
        output = json.load(fd)

    assert output == pytest.approx(ref_data)


class TestPhononWebsiteWriterInternals:
    def test_remove_breaks(self) -> None:
        """Test internal _remove_breaks method"""
        from euphonic.writers.phonon_website import _remove_breaks

        distances = np.array([0.1, 0.2, 0.3, 4.3, 4.4, 8.4, 8.5])

        breakpoints = _remove_breaks(distances)

        assert breakpoints == [3, 5]
        assert_allclose(distances, [0.1, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5])

    def test_find_duplicates(self) -> None:
        """Test internal _find_duplicates method

        The handling of 3+ duplicates is not useful, but also doesn't come
        up much in practice. We test it here so that "surprises" are visible;
        if a refactor changes it to another harmless behaviour, it's fine to
        update this test.

        """

        from euphonic.writers.phonon_website import _find_duplicates

        distances = np.array([0.0, 0.0, 0.1, 0.4, 0.4, 0.5, 0.6, 0.6, 0.6])
        duplicates = _find_duplicates(distances)

        assert duplicates == [1, 4, 7, 8]

    @pytest.mark.parametrize('x_tick_labels,expected', [
        # Normal case:
        ([(1, "X"), (2, "X"), (4, "A"), (7, "Y"), (8, "Z")],
         [(2, "X"), (4, "A"), (8, "Y|Z")]),
        # Do-nothing case:
        ([(1, r"$\Gamma$"), (5, r"X")],
         [(1, r"$\Gamma$"), (5, r"X")]),
        # 3-way merge:
        ([(1, "X"), (2, "Y"), (3, "Z"), (5, "OK")],
         [(3, "X|Y|Z"), (5, "OK")])
        ])
    def test_combine_neighbouring_labels(
            self, x_tick_labels: XTickLabels, expected: XTickLabels) -> None:
        """Test internal _combine_neighbouring_labels method"""
        from euphonic.writers.phonon_website import (
            _combine_neighbouring_labels)

        assert _combine_neighbouring_labels(x_tick_labels) == expected
