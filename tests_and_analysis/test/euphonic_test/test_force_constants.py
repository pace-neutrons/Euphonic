from euphonic import ForceConstants
import pytest
import os
from ..utils import get_data_path


@pytest.mark.unit
class TestObjectCreation:

    @pytest.mark.parametrize("castep_bin_file", [
        os.path.join('interpolation', 'LZO', 'La2Zr2O7.castep_bin'),
        os.path.join('interpolation', 'graphite', 'graphite.castep_bin'),
        os.path.join('interpolation', 'quartz', 'quartz.castep_bin')
    ])
    def test_creation_from_castep(self, castep_bin_file):
        filepath = os.path.join(get_data_path(), castep_bin_file)
        ForceConstants.from_castep(filepath)

    @pytest.mark.parametrize("phonopy_args", [
        {
            "path": os.path.join('phonopy_data', 'NaCl', 'interpolation'),
            "fc_name": "FORCE_CONSTANTS"
        },
        {
            "path": os.path.join('phonopy_data', 'NaCl', 'interpolation'),
            "fc_name": "FULL_FORCE_CONSTANTS"
        },
        {
            "path": os.path.join('phonopy_data', 'NaCl', 'interpolation'),
            "fc_name": "full_force_constants.hdf5"
        },
        {
            "path": os.path.join('phonopy_data', 'NaCl', 'interpolation'),
            "summary_name": 'phonopy_nofc_noborn.yaml'
        }
    ])
    def test_creation_from_phonopy(self, phonopy_args):
        phonopy_args["path"] = os.path.join(
            get_data_path(), phonopy_args["path"]
        )
        ForceConstants.from_phonopy(**phonopy_args)
