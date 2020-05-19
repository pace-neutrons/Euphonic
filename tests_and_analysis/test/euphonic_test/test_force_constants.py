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
