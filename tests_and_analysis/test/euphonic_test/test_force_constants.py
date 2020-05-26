from euphonic import ForceConstants
import pytest
import os
from ..utils import get_data_path


@pytest.mark.unit
class TestObjectCreation:

    @pytest.mark.parametrize("castep_bin_file", [
        os.path.join('LZO', 'La2Zr2O7.castep_bin'),
        os.path.join('graphite', 'graphite.castep_bin'),
        os.path.join('quartz', 'quartz.castep_bin')
    ])
    def test_creation_from_castep(self, castep_bin_file):
        filepath = os.path.join(
            get_data_path(), 'interpolation', castep_bin_file
        )
        ForceConstants.from_castep(filepath)

    @pytest.mark.parametrize("phonopy_args", [
        {"fc_name": "FORCE_CONSTANTS"},
        {"fc_name": "FULL_FORCE_CONSTANTS"},
        {"fc_name": "full_force_constants.hdf5"},
        {"summary_name": "phonopy_nofc_noborn.yaml"}
    ])
    def test_creation_from_phonopy(self, phonopy_args):
        phonopy_args["path"] = os.path.join(
            get_data_path(), 'phonopy_data', 'NaCl', 'interpolation'
        )
        ForceConstants.from_phonopy(**phonopy_args)


