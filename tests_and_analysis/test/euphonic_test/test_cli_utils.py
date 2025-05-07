"""Additional tests of euphonic.cli.utils internals"""


import pytest

from euphonic.cli.utils import (
    _get_energy_bins,
    _get_q_distance,
    _load_phonopy_file,
)
from euphonic.ureg import Quantity
from tests_and_analysis.test.utils import get_data_path


def test_get_q_distance():
    """Test private function _get_q_distance"""

    assert _get_q_distance('mm', 1) == Quantity(1, '1 / mm')

    with pytest.raises(ValueError, match='Length unit not known'):
        _get_q_distance('elephant', 4)


@pytest.mark.phonopy_reader
def test_load_phonopy_errors():
    fc_file = get_data_path(
        'phonopy_files', 'CaHgO2', 'full_force_constants.hdf5')

    with pytest.raises(
            ValueError, match='must be accompanied by phonopy.yaml'):
        _load_phonopy_file(fc_file)


def test_get_energy_bins_error():
    with pytest.raises(
            ValueError,
            match='Maximum energy should be greater than minimum.'):
        _get_energy_bins(modes=None, n_ebins=0, emin=1, emax=0)
