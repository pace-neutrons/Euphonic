import os
import json

import pytest
import numpy as np
import numpy.testing as npt
from toolz.dicttoolz import valmap

from euphonic.io import _from_json_dict
from euphonic.util import direction_changed, mp_grid, get_qpoint_labels
from euphonic.readers.castep import read_phonon_data, read_phonon_dos_data
from tests_and_analysis.test.utils import get_data_path, get_castep_path
from .test_crystal import ExpectedCrystal, check_crystal
from euphonic import ureg


class TestReadPhononDosData:

    @pytest.mark.parametrize(
        'material, phonon_dos_file, expected_data_file', [
        ('LZO', 'La2Zr2O7-222-full.phonon_dos',
         'lzo_222_full_phonon_dos.json'),
        ('quartz', 'quartz-554-full.phonon_dos',
         'quartz_554_full_phonon_dos.json'),
                             ])
    def test_read_data(self, material, phonon_dos_file, expected_data_file):
        dos_data = read_phonon_dos_data(
            get_castep_path(material, phonon_dos_file))
        with open(get_data_path('readers', expected_data_file), 'r') as fp:
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

        # Convert ref data from cm to 1/meV to match current behaviour
        expected_dos_data['dos_unit'] = '1/meV'

        def transform(dos: list[float]) -> list[float]:
            return (ureg.Quantity(dos, "cm")
                    .to("1 / meV", "reciprocal_spectroscopy")
                    .magnitude
                    .tolist())

        expected_dos_data["dos"] = valmap(transform, expected_dos_data["dos"])

        check_dict(dos_data, expected_dos_data)


class TestReadPhononData:
    @pytest.mark.parametrize(
        'material, phonon_file, expected_data_file, kwargs', [
            ('ZnS', 'loto_directions.phonon', 'loto_directions_average.json',
             {'average_repeat_points': True, 'prefer_non_loto': False}),
            ('ZnS', 'loto_directions.phonon', 'loto_directions_no_loto.json',
             {'average_repeat_points': True, 'prefer_non_loto': True}),
            ('ZnS', 'loto_directions.phonon', 'loto_directions_no_loto.json',
             {'average_repeat_points': False, 'prefer_non_loto': True}),
                              ])
    def test_read_data(
        self, material, phonon_file, expected_data_file, kwargs):
        data = read_phonon_data(
            get_castep_path(material, phonon_file),
            **kwargs)

        with open(get_data_path('readers', expected_data_file), 'r') as fp:
            expected_data = _from_json_dict(
                json.loads(fp.read()),
                type_dict={'eigenvectors': np.complex128})

        crystal = ExpectedCrystal(data.pop('crystal'))
        expected_crystal = ExpectedCrystal(
                expected_data.pop('crystal'))
        check_crystal(crystal, expected_crystal)

        def check_dict(dct, expected_dct):
            for exp_key, exp_val in expected_dct.items():
                if isinstance(exp_val, (list, np.ndarray)):
                    npt.assert_allclose(dct[exp_key], np.array(exp_val))
                elif isinstance(exp_val, dict):
                    check_dict(dct[exp_key], exp_val)
                else:
                    assert dct[exp_key] == exp_val

        check_dict(data, expected_data)
