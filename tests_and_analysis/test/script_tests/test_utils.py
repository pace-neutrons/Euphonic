import pytest

from euphonic import ForceConstants, QpointPhononModes
from euphonic.cli.utils import force_constants_from_file, modes_from_file

from tests_and_analysis.test.utils import get_castep_path
from ..euphonic_test.test_force_constants import check_force_constants
from ..euphonic_test.test_qpoint_phonon_modes import check_qpt_ph_modes

# This file can be deleted once the deprecated
# force_constants_from_file and modes_from_file have been removed
quartz_fc_file = get_castep_path('quartz', 'quartz.castep_bin')
lzo_modes_file = get_castep_path('LZO', 'La2Zr2O7.phonon')


def test_force_constants_from_file():
    fc = force_constants_from_file(quartz_fc_file)
    expected_fc = ForceConstants.from_castep(quartz_fc_file)
    check_force_constants(fc, expected_fc)


def test_force_constants_from_file_warns():
    with pytest.warns(DeprecationWarning):
        force_constants_from_file(quartz_fc_file)


def test_fc_from_file_with_qpt_ph_modes_file_raises_type_error():
    with pytest.raises(TypeError):
        force_constants_from_file(lzo_modes_file)


def test_modes_from_file():
    qpt_ph_modes = modes_from_file(lzo_modes_file)
    expected_qpt_ph_modes = QpointPhononModes.from_castep(lzo_modes_file)
    check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes)


def test_modes_from_file_warns():
    with pytest.warns(DeprecationWarning):
        modes_from_file(lzo_modes_file)


def test_modes_from_file_with_fc_file_raises_type_error():
    with pytest.raises(TypeError):
        modes_from_file(quartz_fc_file)
