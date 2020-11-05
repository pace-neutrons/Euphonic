import json
import os

import pytest
import numpy as np
import numpy.testing as npt

from euphonic import ForceConstants, Crystal, ureg
from euphonic.readers.phonopy import ImportPhonopyReaderError
from tests_and_analysis.test.euphonic_test.test_crystal import (
    get_crystal, ExpectedCrystal, check_crystal)
from tests_and_analysis.test.utils import (
    get_data_path, check_unit_conversion, check_json_metadata)


class ExpectedForceConstants:

    def __init__(self, force_constants_json_file: str):
        with open(force_constants_json_file) as fd:
            self.data = json.load(fd)

    @property
    def crystal(self):
        return ExpectedCrystal(self.data['crystal'])

    @property
    def sc_matrix(self):
        return np.array(self.data['sc_matrix'])

    @property
    def cell_origins(self):
        return np.array(self.data['cell_origins'])

    @property
    def born(self):
        if 'born' in self.data.keys():
            return np.array(self.data['born'])*ureg(
                self.data['born_unit'])
        else:
            return None

    @property
    def force_constants(self):
        return np.array(self.data['force_constants'])*ureg(
            self.data['force_constants_unit'])

    @property
    def dielectric(self):
        if 'dielectric' in self.data.keys():
            return np.array(self.data['dielectric'])*ureg(
                self.data['dielectric_unit'])
        else:
            return None

    @property
    def n_cells_in_sc(self):
        return self.data['n_cells_in_sc']

    def to_dict(self):
        d = {
            'crystal': self.crystal.to_dict(),
            'force_constants': self.force_constants.magnitude,
            'force_constants_unit': str(self.force_constants.units),
            'n_cells_in_sc': self.n_cells_in_sc,
            'sc_matrix': self.sc_matrix,
            'cell_origins': self.cell_origins}
        if self.born is not None:
            d['born'] = self.born.magnitude
            d['born_unit'] = str(self.born.units)
        if self.dielectric is not None:
            d['dielectric'] = self.dielectric.magnitude
            d['dielectric_unit'] = str(self.dielectric.units)
        return d

    def to_constructor_args(self, crystal=None, force_constants=None,
                            sc_matrix=None, cell_origins=None,
                            born=None, dielectric=None):
        if crystal is None:
            crystal = Crystal(*self.crystal.to_constructor_args())
        if force_constants is None:
            force_constants = self.force_constants
        if sc_matrix is None:
            sc_matrix = self.sc_matrix
        if cell_origins is None:
            cell_origins = self.cell_origins
        if born is None:
            born = self.born
        if dielectric is None:
            dielectric = self.dielectric

        kwargs = {}
        if born is not None:
            kwargs['born'] = born
        if dielectric is not None:
            kwargs['dielectric'] = dielectric

        return (crystal, force_constants, sc_matrix, cell_origins), kwargs


def get_fc_dir(material):
    return os.path.join(get_data_path(), 'force_constants', material)


def get_json_file(material):
    return os.path.join(get_fc_dir(material),
                        f'{material}_force_constants.json')


def get_expected_fc(material):
    return ExpectedForceConstants(get_json_file(material))


def get_fc_from_json_file(filepath):
    return ForceConstants.from_json_file(filepath)


def get_fc(material):
    return get_fc_from_json_file(get_json_file(material))


def check_force_constants(
        actual_force_constants, expected_force_constants):
    check_crystal(actual_force_constants.crystal,
                  expected_force_constants.crystal)

    assert (actual_force_constants.n_cells_in_sc
            == expected_force_constants.n_cells_in_sc)

    npt.assert_allclose(
        actual_force_constants.force_constants.magnitude,
        expected_force_constants.force_constants.magnitude,
        atol=np.finfo(np.float64).eps)
    assert (actual_force_constants.force_constants.units
            == expected_force_constants.force_constants.units)

    npt.assert_array_equal(
        actual_force_constants.sc_matrix,
        expected_force_constants.sc_matrix)

    npt.assert_array_equal(
        actual_force_constants.cell_origins,
        expected_force_constants.cell_origins)

    if expected_force_constants.born is None:
        assert actual_force_constants.born is None
    else:
        npt.assert_allclose(
            actual_force_constants.born.magnitude,
            expected_force_constants.born.magnitude,
            atol=np.finfo(np.float64).eps)
        assert (actual_force_constants.born.units
                == expected_force_constants.born.units)

    if expected_force_constants.dielectric is None:
        assert actual_force_constants.dielectric is None
    else:
        npt.assert_allclose(
            actual_force_constants.dielectric.magnitude,
            expected_force_constants.dielectric.magnitude)
        assert (actual_force_constants.dielectric.units
                == expected_force_constants.dielectric.units)


@pytest.mark.unit
class TestForceConstantsCreation:

    @pytest.fixture(params=[get_expected_fc('quartz'),
                            get_expected_fc('LZO'),
                            get_expected_fc('NaCl')])
    def create_from_constructor(self, request):
        expected_fc = request.param
        args, kwargs = expected_fc.to_constructor_args()
        fc = ForceConstants(*args, **kwargs)
        return fc, expected_fc

    @pytest.fixture(params=[
        ('LZO', 'La2Zr2O7.castep_bin'),
        ('graphite', 'graphite.castep_bin'),
        ('Si2-sc-skew', 'Si2-sc-skew.castep_bin'),
        ('quartz', 'quartz.castep_bin')])
    def create_from_castep(self, request):
        material, castep_bin_file = request.param
        expected_fc = get_expected_fc(material)
        castep_filepath = os.path.join(get_fc_dir(material),  castep_bin_file)
        fc = ForceConstants.from_castep(castep_filepath)
        return fc, expected_fc

    @pytest.fixture(params=['LZO', 'graphite', 'Si2-sc-skew', 'quartz',
                            'CaHgO2', 'NaCl'])
    def create_from_json(self, request):
        material = request.param
        expected_fc = get_expected_fc(material)
        fc = ForceConstants.from_json_file(get_json_file(material))
        return fc, expected_fc

    @pytest.fixture(params=['LZO', 'graphite', 'quartz'])
    def create_from_dict(self, request):
        material = request.param
        expected_fc = get_expected_fc(material)
        fc = ForceConstants.from_dict(expected_fc.to_dict())
        return fc, expected_fc

    @pytest.fixture(params=[
        # Test all combinations of reading from .yaml with/without force
        # constants/born and different file formats. Extra files (e.g.
        # FORCE_CONSTANTS) have been renamed from their defaults to
        # avoid a false positive
        ('NaCl', {'summary_name': 'phonopy_nacl.yaml'}),
        ('NaCl', {'summary_name': 'phonopy_full_fc.yaml'}),
        ('NaCl', {'summary_name': 'phonopy_nofc.yaml',
                  'fc_name': 'FORCE_CONSTANTS_nacl'}),
        ('NaCl', {'summary_name': 'phonopy_nofc.yaml',
                  'fc_name': 'FULL_FORCE_CONSTANTS'}),
        ('NaCl', {'summary_name': 'phonopy_nofc.yaml',
                  'fc_name': 'FULL_FORCE_CONSTANTS_single_number'}),
        ('NaCl', {'summary_name': 'phonopy_nofc.yaml',
                  'fc_name': 'full_force_constants.hdf5'}),
        ('NaCl', {'summary_name': 'phonopy_nofc.yaml',
                  'fc_name': 'force_constants.hdf5'}),
        ('NaCl', {'summary_name': 'phonopy_nofc.yaml',
                  'fc_name': 'force_constants_hdf5.test',
                  'fc_format': 'hdf5'}),
        ('NaCl', {'summary_name': 'phonopy_nofc_noborn.yaml',
                  'fc_name': 'FORCE_CONSTANTS_nacl',
                  'born_name': 'BORN_nacl'}),
        # Explicitly test the default behaviour (if fc/born aren't found
        # in phonopy.yaml they should be read from BORN, FORCE_CONSTANTS).
        # This must be done in a separate directory to the above tests,
        # again to avoid false positives
        ('NaCl_default', {}),
        ('NaCl_prim', {'summary_name': 'phonopy_nacl.yaml'}),
        ('CaHgO2', {'summary_name': 'mp-7041-20180417.yaml'}),
        ('CaHgO2', {'summary_name': 'phonopy_fullfc.yaml'}),
        ('CaHgO2', {'summary_name': 'phonopy_nofc.yaml',
                    'fc_name': 'FULL_FORCE_CONSTANTS'}),
        ('CaHgO2', {'summary_name': 'phonopy_nofc.yaml',
                    'fc_name': 'full_force_constants.hdf5'})])
    def create_from_phonopy(self, request):
        material, phonopy_args = request.param
        phonopy_args['path'] = get_fc_dir(material)
        fc = ForceConstants.from_phonopy(**phonopy_args)
        expected_fc = get_expected_fc(material)
        return fc, expected_fc

    @pytest.mark.parametrize(('force_constants_creator'), [
        pytest.lazy_fixture('create_from_constructor'),
        pytest.lazy_fixture('create_from_dict'),
        pytest.lazy_fixture('create_from_json'),
        pytest.lazy_fixture('create_from_phonopy'),
        pytest.lazy_fixture('create_from_castep')])
    def test_correct_object_creation(self, force_constants_creator):
        force_constants, expected_force_constants = force_constants_creator
        check_force_constants(force_constants, expected_force_constants)

    @pytest.fixture(params=[
        ('sc_matrix',
         get_expected_fc('quartz').sc_matrix[:2],
         ValueError),
        ('cell_origins',
         get_expected_fc('quartz').sc_matrix,
         ValueError),
        ('force_constants',
         get_expected_fc('quartz').force_constants[:2],
         ValueError),
        ('born',
         get_expected_fc('quartz').born[:2],
         ValueError),
        ('dielectric',
         get_expected_fc('quartz').dielectric[:2],
         ValueError),
        ('sc_matrix',
         list(get_expected_fc('quartz').sc_matrix),
         TypeError),
        ('cell_origins',
         get_expected_fc('quartz').sc_matrix*ureg.meter,
         TypeError),
        ('crystal',
         get_crystal('LZO'),
         ValueError),
        ('force_constants',
         get_expected_fc('quartz').force_constants.magnitude,
         TypeError),
        ('born',
         list(get_expected_fc('quartz').born.magnitude),
         TypeError),
        ('dielectric',
         get_expected_fc('quartz').dielectric.shape,
         TypeError)])
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        expected_fc = get_expected_fc('quartz')
        # Inject the faulty value and get a tuple of constructor arguments
        args, kwargs = expected_fc.to_constructor_args(**{faulty_arg: faulty_value})
        return args, kwargs, expected_exception

    def test_faulty_object_creation(self, inject_faulty_elements):
        faulty_args, faulty_kwargs, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            ForceConstants(*faulty_args, **faulty_kwargs)

    @pytest.mark.parametrize('phonopy_args', [
        ({'summary_name': 'phonopy_nofc.yaml',
          'fc_name': 'force_constants.hdf5',
          'path': get_fc_dir('NaCl')})])
    def test_create_from_phonopy_without_installed_modules_raises_err(
            self, phonopy_args, mocker):
        # Mock import of yaml, h5py to raise ModuleNotFoundError
        import builtins
        real_import = builtins.__import__
        def mocked_import(name, *args, **kwargs):
            if name == 'h5py' or name == 'yaml':
                raise ModuleNotFoundError
            return real_import(name, *args, **kwargs)
        mocker.patch('builtins.__import__', side_effect=mocked_import)
        with pytest.raises(ImportPhonopyReaderError):
            ForceConstants.from_phonopy(**phonopy_args)

    @pytest.mark.parametrize('phonopy_args, err', [
        ({'summary_name': 'phonopy_nofc.yaml',
          'fc_name': 'force_constants.hdf5',
          'path': get_fc_dir('NaCl'),
          'fc_format': 'nonsense'},
         ValueError),
        ({'summary_name': '../CaHgO2/phonopy_nofc.yaml',
          'fc_name': 'force_constants.hdf5',
          'path': get_fc_dir('NaCl')},
         ValueError)])
    def test_create_from_phonopy_with_bad_inputs_raises_err(
            self, phonopy_args, err):
        with pytest.raises(err):
            ForceConstants.from_phonopy(**phonopy_args)

    def test_create_from_castep_with_no_fc_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            ForceConstants.from_castep(
                os.path.join(get_fc_dir('h-BN'),
                             'h-BN_no_force_constants.castep_bin'))

    @pytest.mark.parametrize('material, phonopy_args', [
        ('CaHgO2', {'summary_name': 'mp-7041-20180417.yaml'})])
    def test_create_from_phonopy_without_cloader_is_ok(
            self, material, phonopy_args, mocker):
        # Mock 'from yaml import CLoader as Loader' to raise ImportError
        import builtins
        real_import = builtins.__import__
        def mocked_import(name, globals, locals, fromlist, level):
            if name == 'yaml':
                if fromlist is not None and fromlist[0] == 'CSafeLoader':
                    raise ImportError
            return real_import(name, globals, locals, fromlist, level)
        mocker.patch('builtins.__import__', side_effect=mocked_import)

        phonopy_args['path'] = get_fc_dir(material)
        fc = ForceConstants.from_phonopy(**phonopy_args)
        expected_fc = get_expected_fc(material)
        check_force_constants(fc, expected_fc)


@pytest.mark.unit
class TestForceConstantsSerialisation:

    @pytest.mark.parametrize('fc', [
        get_fc('quartz'),
        get_fc('LZO'),
        get_fc('NaCl')])
    def test_serialise_to_json_file(self, fc, tmpdir):
        output_file = str(tmpdir.join('tmp.test'))
        fc.to_json_file(output_file)
        check_json_metadata(output_file, 'ForceConstants')
        deserialised_fc = ForceConstants.from_json_file(output_file)
        check_force_constants(fc, deserialised_fc)

    @pytest.fixture(params=[
        (get_fc('quartz'), get_expected_fc('quartz')),
        (get_fc('LZO'), get_expected_fc('LZO')),
        (get_fc('NaCl'), get_expected_fc('NaCl'))])
    def serialise_to_dict(self, request):
        fc, expected_fc = request.param
        # Convert to dict, then back to object to test
        fc_dict = fc.to_dict()
        fc_from_dict = ForceConstants.from_dict(fc_dict)
        return fc_from_dict, expected_fc

    def test_serialise_to_dict(self, serialise_to_dict):
        fc, expected_fc = serialise_to_dict
        check_force_constants(fc, expected_fc)


@pytest.mark.unit
class TestForceConstantsUnitConversion:

    @pytest.mark.parametrize('material, attr, unit_val', [
        ('quartz', 'force_constants', 'hartree/bohr**2'),
        ('quartz', 'dielectric', 'e**2/(angstrom*eV)'),
        ('quartz', 'born', 'C')])
    def test_correct_unit_conversion(self, material, attr, unit_val):
        fc = get_fc(material)
        check_unit_conversion(fc, attr, unit_val)

    @pytest.mark.parametrize('material, unit_attr, unit_val, err', [
        ('quartz', 'force_constants_unit', 'hartree', ValueError),
        ('quartz', 'dielectric_unit', 'angstrom', ValueError),
        ('quartz', 'born_unit', '1/cm', ValueError)])
    def test_incorrect_unit_conversion(self, material, unit_attr,
                                       unit_val, err):
        fc = get_fc(material)
        with pytest.raises(err):
            setattr(fc, unit_attr, unit_val)
