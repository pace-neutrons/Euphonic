import os
import json

import pytest
import numpy as np
import numpy.testing as npt

from euphonic.util import direction_changed, mp_grid, get_qpoint_labels
from euphonic.readers.castep import read_phonon_dos_data
from tests_and_analysis.test.utils import get_data_path, get_castep_path
from .test_crystal import ExpectedCrystal, check_crystal
from euphonic import ureg


def get_filepath(filename):
    return os.path.join(get_data_path(), 'readers', filename)

class TestReadPhononDosData:

    @pytest.mark.parametrize(
        'material, phonon_dos_file, expected_data_file', [
        ('LZO', 'La2Zr2O7-222-full.phonon_dos',
         'lzo_222_full_phonon_dos.json'),
        ('quartz', 'quartz-554-full.phonon_dos',
         'quartz_554_full_phonon_dos.json')])
    def test_read_data(self, material, phonon_dos_file, expected_data_file):
        dos_data = read_phonon_dos_data(
            get_castep_path(material, phonon_dos_file))
        with open(get_filepath(expected_data_file), 'r') as fp:
            expected_dos_data = json.loads(fp.read())

        dos_crystal = ExpectedCrystal(dos_data.pop('crystal'))
        expected_dos_crystal = ExpectedCrystal(
                expected_dos_data.pop('crystal'))
        check_crystal(dos_crystal, expected_dos_crystal)

        def check_dict(dct, expected_dct):
            for exp_key, exp_val in expected_dct.items():
                if isinstance(exp_val, list):
                    npt.assert_allclose(dct[exp_key], np.array(exp_val))
                elif isinstance(exp_val, dict):
                    check_dict(dct[exp_key], exp_val)
                else:
                    assert dct[exp_key] == exp_val
        # Temporary solution until test data has been regenerated
        # units have changed
        expected_dos_data['dos_unit'] = 'meV'
        dos_conv = (1*ureg('1/cm').to('meV')).magnitude
        for key, value in expected_dos_data['dos'].items():
            expected_dos_data['dos'][key] = [x/dos_conv for x in value]
        check_dict(dos_data, expected_dos_data)
