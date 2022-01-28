import os
import json

import pytest
import numpy as np
import numpy.testing as npt
from pint import DimensionalityError

from euphonic import ureg, Crystal, QpointPhononModes, Spectrum1DCollection
from euphonic.readers.phonopy import ImportPhonopyReaderError
from tests_and_analysis.test.euphonic_test.test_crystal import (
    ExpectedCrystal, get_crystal, check_crystal)
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc_path)
from tests_and_analysis.test.euphonic_test.test_debye_waller import (
    get_expected_dw, check_debye_waller)
from tests_and_analysis.test.euphonic_test.test_qpoint_frequencies import (
    get_expected_qpt_freqs, check_qpt_freqs)
from tests_and_analysis.test.euphonic_test.test_spectrum1d import (
    get_expected_spectrum1d, check_spectrum1d)
from tests_and_analysis.test.euphonic_test.test_spectrum1dcollection import (
    get_expected_spectrum1dcollection, check_spectrum1dcollection)
from tests_and_analysis.test.utils import (
    get_data_path, get_castep_path, get_phonopy_path,
    check_frequencies_at_qpts, check_unit_conversion,
    check_json_metadata, check_property_setters, get_mode_widths)


class ExpectedQpointPhononModes:

    def __init__(self, qpoint_phonon_modes_json_file: str):
        with open(qpoint_phonon_modes_json_file) as fd:
            self.data = json.load(fd)

    @property
    def crystal(self):
        return ExpectedCrystal(self.data['crystal'])

    @property
    def qpts(self):
        return np.array(self.data['qpts'])

    @property
    def frequencies(self):
        return np.array(self.data['frequencies'])*ureg(
            self.data['frequencies_unit'])

    @property
    def eigenvectors(self):
        n_atoms = self.crystal.n_atoms
        evecs = np.array(self.data['eigenvectors'], dtype=np.float64)
        return evecs.view(np.complex128).squeeze().reshape(
            (-1, 3*n_atoms, n_atoms, 3))

    @property
    def weights(self):
        # Weights are optional, so if they are not found in .json
        # file, assign equal weights, simulating expected behaviour
        if 'weights' in self.data:
            return np.array(self.data['weights'])
        else:
            return np.full(len(self.qpts), 1/len(self.qpts))

    def to_dict(self):
        d = {
            'crystal': self.crystal.to_dict(),
            'qpts': self.qpts,
            'frequencies': self.frequencies.magnitude,
            'frequencies_unit': str(self.frequencies.units),
            'eigenvectors': self.eigenvectors,
            'weights': self.weights}
        return d

    def to_constructor_args(self, crystal=None, qpts=None, frequencies=None,
                            eigenvectors=None, weights=None):
        if crystal is None:
            crystal = Crystal(*self.crystal.to_constructor_args())
        if qpts is None:
            qpts = self.qpts
        if frequencies is None:
            frequencies = self.frequencies
        if eigenvectors is None:
            eigenvectors = self.eigenvectors
        if weights is None:
            weights = self.weights

        kwargs = {}
        # Allow setting weights=False to not include weights in kwargs, to test
        # object creation when weights is not supplied
        if weights is not False:
            kwargs['weights'] = weights

        return (crystal, qpts, frequencies, eigenvectors), kwargs


def get_qpt_ph_modes_path(*subpaths):
    return get_data_path('qpoint_phonon_modes', *subpaths)

def get_json_file(material):
    return f'{material}_reciprocal_qpoint_phonon_modes.json'


def get_expected_qpt_ph_modes(material):
    return ExpectedQpointPhononModes(
        get_qpt_ph_modes_path(material, get_json_file(material)))


def get_qpt_ph_modes_from_json(material, json_file):
    return QpointPhononModes.from_json_file(
        get_qpt_ph_modes_path(material, json_file))


def get_qpt_ph_modes(material):
    return get_qpt_ph_modes_from_json(material, get_json_file(material))


def check_qpt_ph_modes(
        qpoint_phonon_modes, expected_qpoint_phonon_modes,
        frequencies_atol=np.finfo(np.float64).eps,
        frequencies_rtol=1e-7,
        acoustic_gamma_atol=None,
        check_evecs=False):
    check_crystal(qpoint_phonon_modes.crystal,
                  expected_qpoint_phonon_modes.crystal)

    npt.assert_allclose(
        qpoint_phonon_modes.qpts,
        expected_qpoint_phonon_modes.qpts,
        atol=np.finfo(np.float64).eps)

    assert (qpoint_phonon_modes.frequencies.units
            == expected_qpoint_phonon_modes.frequencies.units)
    # Check frequencies
    check_frequencies_at_qpts(
        qpoint_phonon_modes.qpts,
        qpoint_phonon_modes.frequencies.magnitude,
        expected_qpoint_phonon_modes.frequencies.magnitude,
        atol=frequencies_atol,
        rtol=frequencies_rtol,
        gamma_atol=acoustic_gamma_atol)

    # Note by default the eigenvectors aren't checked, as these cannot
    # be directly compared from different force constants calculations
    # in the case of degenerate frequencies, and are better compared by
    # checking a resultant observable such as the structure factor.
    # However, it can be useful to test if eigenvectors read from a
    # .phonon and .json are the same, for example.
    if check_evecs:
        npt.assert_allclose(qpoint_phonon_modes.eigenvectors,
                            expected_qpoint_phonon_modes.eigenvectors,
                            atol=np.finfo(np.complex128).eps)

    npt.assert_allclose(
        qpoint_phonon_modes.weights,
        expected_qpoint_phonon_modes.weights,
        atol=np.finfo(np.float64).eps)


class TestQpointPhononModesCreation:

    @pytest.mark.parametrize('expected_qpt_ph_modes', [
        get_expected_qpt_ph_modes('quartz'),
        get_expected_qpt_ph_modes('LZO'),
        get_expected_qpt_ph_modes('NaCl')])
    def test_create_from_constructor(self, expected_qpt_ph_modes):
        args, kwargs = expected_qpt_ph_modes.to_constructor_args()
        qpt_ph_modes = QpointPhononModes(*args, **kwargs)
        check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes,
                           check_evecs=True)

    @pytest.mark.parametrize('expected_qpt_ph_modes', [
        get_expected_qpt_ph_modes('quartz'),
        get_expected_qpt_ph_modes('NaCl')])
    def test_create_from_constructor_without_weights(
            self, expected_qpt_ph_modes):
        args, kwargs = expected_qpt_ph_modes.to_constructor_args(weights=False)
        qpt_ph_modes = QpointPhononModes(*args, **kwargs)
        check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes,
                           check_evecs=True)

    @pytest.mark.parametrize('material, phonon_file, json_file', [
        ('LZO', 'La2Zr2O7.phonon',
         'LZO_from_castep_qpoint_phonon_modes.json'),
        ('Si2-sc-skew', 'Si2-sc-skew.phonon',
         'Si2-sc-skew_from_castep_qpoint_phonon_modes.json'),
        ('quartz', 'quartz_nosplit.phonon',
         'quartz_from_castep_qpoint_phonon_modes.json'),
        ('quartz', 'quartz_split_qpts.phonon',
         'quartz_split_from_castep_qpoint_phonon_modes.json')])
    def test_create_from_castep(self, material, phonon_file, json_file):
        qpt_ph_modes = QpointPhononModes.from_castep(
            get_castep_path(material, phonon_file))
        expected_qpt_ph_modes = ExpectedQpointPhononModes(
            get_qpt_ph_modes_path(material, json_file))
        check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes,
                           check_evecs=True)

    @pytest.mark.phonopy_reader
    @pytest.mark.parametrize('material, subdir, phonopy_args, json_file', [
        ('NaCl', 'band', {'summary_name': 'should_not_be_read',
                          'phonon_name': 'band.yaml'},
         'NaCl_band_yaml_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'band', {'summary_name': 'phonopy.yaml',
                          'phonon_name': 'band.hdf5'},
         'NaCl_band_hdf5_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'mesh', {'summary_name': 'should_not_be_read',
                          'phonon_name': 'mesh.yaml'},
         'NaCl_mesh_yaml_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'mesh', {'summary_name': 'phonopy.yaml',
                          'phonon_name': 'mesh.hdf5'},
         'NaCl_mesh_hdf5_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.yaml'},
         'NaCl_qpoints_yaml_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints_yaml.test',
                             'phonon_format': 'yaml'},
         'NaCl_qpoints_yaml_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints_hdf5.test',
                             'phonon_format': 'hdf5'},
         'NaCl_qpoints_hdf5_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.hdf5'},
         'NaCl_qpoints_hdf5_from_phonopy_qpoint_phonon_modes.json'),
        ('CaHgO2', '', {'summary_name': 'mp-7041-20180417.yaml',
                        'phonon_name': 'qpoints.yaml'},
         'CaHgO2_from_phonopy_qpoint_phonon_modes.json')])
    def test_create_from_phonopy(self, material, subdir, phonopy_args, json_file):
        phonopy_args['path'] = get_phonopy_path(material, subdir)
        qpt_ph_modes = QpointPhononModes.from_phonopy(**phonopy_args)
        json_path = get_qpt_ph_modes_path(material, json_file)
        expected_qpt_ph_modes = ExpectedQpointPhononModes(json_path)
        check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes,
                           check_evecs=True)

    @pytest.mark.parametrize('material', ['LZO', 'quartz', 'NaCl'])
    def test_create_from_json(self, material):
        expected_qpt_ph_modes = get_expected_qpt_ph_modes(material)
        qpt_ph_modes = get_qpt_ph_modes(material)
        check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes,
                           check_evecs=True)

    @pytest.mark.parametrize('material', ['LZO', 'quartz', 'NaCl'])
    def test_create_from_dict(self, material):
        expected_qpt_ph_modes = get_expected_qpt_ph_modes(material)
        qpt_ph_modes = QpointPhononModes.from_dict(
            expected_qpt_ph_modes.to_dict())
        check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes,
                           check_evecs=True)

    @pytest.mark.parametrize('faulty_arg, faulty_value, expected_exception', [
        ('qpts',
         get_expected_qpt_ph_modes('quartz').qpts[:3],
         ValueError),
        ('frequencies',
         get_expected_qpt_ph_modes('quartz').frequencies[:2],
         ValueError),
        ('frequencies',
         get_expected_qpt_ph_modes('quartz').frequencies[:, :10],
         ValueError),
        ('frequencies',
         get_expected_qpt_ph_modes('quartz').frequencies.magnitude,
         TypeError),
        ('frequencies',
         get_expected_qpt_ph_modes('quartz').frequencies.magnitude*ureg('kg'),
         DimensionalityError),
        ('eigenvectors',
         get_expected_qpt_ph_modes('quartz').eigenvectors[:4],
         ValueError),
        ('weights',
         get_expected_qpt_ph_modes('quartz').weights[:5],
         ValueError),
        ('crystal',
         get_crystal('LZO'),
         ValueError)])
    def test_faulty_object_creation(
            self, faulty_arg, faulty_value, expected_exception):
        expected_qpt_ph_modes = get_expected_qpt_ph_modes('quartz')
        # Inject the faulty value and get a tuple of constructor arguments
        args, kwargs = expected_qpt_ph_modes.to_constructor_args(
            **{faulty_arg: faulty_value})
        with pytest.raises(expected_exception):
            QpointPhononModes(*args, **kwargs)

    @pytest.mark.parametrize('material, subdir, phonopy_args', [
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.yaml'}),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.hdf5'})])
    def test_create_from_phonopy_without_installed_modules_raises_err(
            self, material, subdir, phonopy_args, mocker):
        phonopy_args['path'] = get_phonopy_path(material, subdir)
        # Mock import of yaml, h5py to raise ModuleNotFoundError
        import builtins
        real_import = builtins.__import__
        def mocked_import(name, *args, **kwargs):
            if name == 'h5py' or name == 'yaml':
                raise ModuleNotFoundError
            return real_import(name, *args, **kwargs)
        mocker.patch('builtins.__import__', side_effect=mocked_import)
        with pytest.raises(ImportPhonopyReaderError):
            QpointPhononModes.from_phonopy(**phonopy_args)

    @pytest.mark.phonopy_reader
    @pytest.mark.parametrize('material, subdir, phonopy_args, err', [
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints_hdf5.test'},
         ValueError),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.hdf5',
                             'phonon_format': 'nonsense'},
         ValueError),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints_no_evec.yaml'},
         RuntimeError),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints_no_evec.hdf5'},
         RuntimeError),
        ('NaCl', 'band', {'summary_name': 'phonopy.yaml',
                          'phonon_name': 'band_no_evec.hdf5'},
         RuntimeError),
        ('NaCl', 'qpoints', {
            'summary_name': '../../CaHgO2/mp-7041-20180417.yaml',
            'phonon_name': 'qpoints.hdf5'},
         ValueError)])
    def test_create_from_phonopy_with_bad_inputs_raises_err(
            self, material, subdir, phonopy_args, err):
        phonopy_args['path'] = get_phonopy_path(material, subdir)
        with pytest.raises(err):
            QpointPhononModes.from_phonopy(**phonopy_args)

    @pytest.mark.phonopy_reader
    @pytest.mark.parametrize('material, subdir, phonopy_args, json_file', [
        ('CaHgO2', '', {'summary_name': 'mp-7041-20180417.yaml',
                        'phonon_name': 'qpoints.yaml'},
         'CaHgO2_from_phonopy_qpoint_phonon_modes.json')])
    def test_create_from_phonopy_without_cloader_is_ok(
            self, material, subdir, phonopy_args, json_file, mocker):
        # Mock 'from yaml import CLoader as Loader' to raise ImportError
        import builtins
        real_import = builtins.__import__
        def mocked_import(name, globals, locals, fromlist, level):
            if name == 'yaml':
                if fromlist is not None and fromlist[0] == 'CSafeLoader':
                    raise ImportError
            return real_import(name, globals, locals, fromlist, level)
        mocker.patch('builtins.__import__', side_effect=mocked_import)

        phonopy_args['path'] = get_phonopy_path(material, subdir)
        qpt_ph_modes = QpointPhononModes.from_phonopy(**phonopy_args)
        json_path = get_qpt_ph_modes_path(material, json_file)
        expected_qpt_ph_modes = ExpectedQpointPhononModes(json_path)
        check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes,
                           check_evecs=True)

class TestQpointPhononModesSerialisation:

    @pytest.mark.parametrize('qpt_ph_modes', [
        get_qpt_ph_modes('quartz'),
        get_qpt_ph_modes('Si2-sc-skew'),
        get_qpt_ph_modes('NaCl')])
    def test_serialise_to_json_file(self, qpt_ph_modes, tmpdir):
        output_file = str(tmpdir.join('tmp.test'))
        qpt_ph_modes.to_json_file(output_file)
        check_json_metadata(output_file, 'QpointPhononModes')
        deserialised_qpt_ph_modes = QpointPhononModes.from_json_file(
            output_file)
        check_qpt_ph_modes(qpt_ph_modes, deserialised_qpt_ph_modes,
                           check_evecs=True)

    @pytest.fixture(params=['quartz', 'Si2-sc-skew', 'NaCl'])
    def serialise_to_dict(self, request):
        qpt_ph_modes = get_qpt_ph_modes(request.param)
        # Convert to dict, then back to object to test
        qpt_ph_modes_dict = qpt_ph_modes.to_dict()
        qpt_ph_modes_from_dict = QpointPhononModes.from_dict(qpt_ph_modes_dict)
        return qpt_ph_modes, qpt_ph_modes_from_dict

    def test_serialise_to_dict(self, serialise_to_dict):
        qpt_ph_modes, qpt_ph_modes_from_dict = serialise_to_dict
        check_qpt_ph_modes(qpt_ph_modes, qpt_ph_modes_from_dict,
                           check_evecs=True)

    @pytest.mark.parametrize('material, qpt_ph_modes_json, qpt_freqs_json', [
        ('quartz',
         'quartz_bandstructure_qpoint_phonon_modes.json',
         'quartz_bandstructure_qpoint_frequencies.json'),
        ('NaCl',
         'NaCl_mesh_yaml_from_phonopy_qpoint_phonon_modes.json',
         'NaCl_mesh_yaml_from_phonopy_qpoint_frequencies.json')])
    def test_to_qpoint_frequencies(
            self, material, qpt_ph_modes_json, qpt_freqs_json):
        qpt_ph_modes = get_qpt_ph_modes_from_json(material, qpt_ph_modes_json)
        qpt_freqs = qpt_ph_modes.to_qpoint_frequencies()
        expected_qpt_freqs = get_expected_qpt_freqs(material, qpt_freqs_json)
        check_qpt_freqs(qpt_freqs, expected_qpt_freqs)



class TestQpointPhononModesUnitConversion:

    @pytest.mark.parametrize('material, attr, unit_val', [
        ('quartz', 'frequencies', '1/cm')])
    def test_correct_unit_conversion(self, material, attr, unit_val):
        qpt_ph_modes = get_qpt_ph_modes(material)
        check_unit_conversion(qpt_ph_modes, attr, unit_val)

    @pytest.mark.parametrize('material, unit_attr, unit_val, err', [
        ('quartz', 'frequencies_unit', 'kg', ValueError)])
    def test_incorrect_unit_conversion(self, material, unit_attr,
                                       unit_val, err):
        qpt_ph_modes = get_qpt_ph_modes(material)
        with pytest.raises(err):
            setattr(qpt_ph_modes, unit_attr, unit_val)


class TestQpointPhononModesSetters:

    @pytest.mark.parametrize('material, attr, unit, scale', [
        ('quartz', 'frequencies', '1/cm', 2.),
        ('quartz', 'frequencies', 'meV', 3.)
        ])
    def test_setter_correct_units(self, material, attr, unit, scale):
        qpt_ph_modes = get_qpt_ph_modes(material)
        check_property_setters(qpt_ph_modes, attr, unit, scale)

    @pytest.mark.parametrize('material, attr, unit, err', [
        ('quartz', 'frequencies', '1/cm**2', ValueError)])
    def test_incorrect_unit_conversion(self, material, attr, unit, err):
        qpt_ph_modes = get_qpt_ph_modes(material)
        new_attr = getattr(qpt_ph_modes, attr).magnitude*ureg(unit)
        with pytest.raises(err):
            setattr(qpt_ph_modes, attr, new_attr)


class TestQpointPhononModesReorderFrequencies:

    @pytest.mark.parametrize(
        'material, qpt_ph_modes_file, reordered_qpt_ph_modes_file, kwargs', [
            ('LZO', 'La2Zr2O7_before_reorder_qpoint_phonon_modes.json',
             'La2Zr2O7_after_reorder_qpoint_phonon_modes.json', {}),
            ('quartz', 'quartz_before_reorder_qpoint_phonon_modes.json',
             'quartz_after_reorder_nogamma_qpoint_phonon_modes.json',
             {'reorder_gamma': False})])
    def test_reorder_frequencies(self, material, qpt_ph_modes_file,
                                 reordered_qpt_ph_modes_file, kwargs):
        qpt_ph_modes = get_qpt_ph_modes_from_json(material, qpt_ph_modes_file)
        qpt_ph_modes.reorder_frequencies(**kwargs)
        expected_qpt_ph_modes = get_qpt_ph_modes_from_json(
            material, reordered_qpt_ph_modes_file)
        check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes)


class TestQpointPhononModesCalculateDebyeWaller:

    @pytest.mark.parametrize(
        'material, qpt_ph_modes_file, expected_dw_json, temperature, kwargs', [
            ('quartz', 'quartz-666-grid.phonon',
                'quartz_666_0K_debye_waller.json', 0, {'symmetrise': False}),
            ('quartz', 'quartz-666-grid.phonon',
             'quartz_666_0K_debye_waller_10mev_lim.json', 0,
             {'frequency_min': 10*ureg('meV'), 'symmetrise': False}),
            ('quartz', 'quartz-666-grid.phonon',
                'quartz_666_300K_debye_waller.json', 300,
                {'symmetrise': False}),
            ('quartz', 'quartz-666-grid.phonon',
                'quartz_666_300K_symm_debye_waller.json', 300,
                {'symmetrise': True}),
            ('quartz', 'quartz-777-grid.phonon',
                'quartz_777_300K_debye_waller.json', 300,
                {'symmetrise': False}),
            ('Si2-sc-skew', 'Si2-sc-skew-666-grid.phonon',
                'Si2-sc-skew_666_300K_debye_waller.json', 300,
                {'symmetrise': False}),
            ('Si2-sc-skew', 'Si2-sc-skew-666-grid.phonon',
                'Si2-sc-skew_666_300K_symm_debye_waller.json', 300,
                {'symmetrise': True}),
        ])
    def test_calculate_debye_waller(self, material, qpt_ph_modes_file,
                                    expected_dw_json, temperature, kwargs):
        qpt_ph_modes = QpointPhononModes.from_castep(
            get_castep_path(material, qpt_ph_modes_file))
        dw = qpt_ph_modes.calculate_debye_waller(
            temperature*ureg('K'), **kwargs)
        expected_dw = get_expected_dw(material, expected_dw_json)
        check_debye_waller(dw, expected_dw, dw_atol=1e-12)

    @pytest.mark.phonopy_reader
    @pytest.mark.parametrize(
        'material, qpt_ph_modes_file, expected_dw_json, temperature, kwargs', [
            ('CaHgO2', 'CaHgO2-666-grid.yaml',
                'CaHgO2_666_300K_debye_waller.json', 300,
                {'symmetrise': False})
        ])
    def test_calculate_debye_waller_from_phonopy(
            self, material, qpt_ph_modes_file,
            expected_dw_json, temperature, kwargs):
        qpt_ph_modes = QpointPhononModes.from_phonopy(
            phonon_name=get_phonopy_path(material, qpt_ph_modes_file))
        dw = qpt_ph_modes.calculate_debye_waller(
            temperature*ureg('K'), **kwargs)
        expected_dw = get_expected_dw(material, expected_dw_json)
        check_debye_waller(dw, expected_dw, dw_atol=1e-12)


class TestQpointPhononModesCalculateDos:

    @pytest.mark.parametrize(
        'material, qpt_ph_modes_file, expected_dos_json, ebins', [
            ('quartz', 'quartz-666-grid.phonon',
             'quartz_666_dos.json', np.arange(0, 155, 0.5)*ureg('meV')),
        ])
    def test_calculate_dos(self, material, qpt_ph_modes_file,
                           expected_dos_json, ebins):
        qpt_ph_modes = QpointPhononModes.from_castep(
            get_castep_path(material, qpt_ph_modes_file))
        dos = qpt_ph_modes.calculate_dos(ebins)
        expected_dos = get_expected_spectrum1d(expected_dos_json)
        check_spectrum1d(dos, expected_dos)

    @pytest.mark.phonopy_reader
    @pytest.mark.parametrize(
        'material, qpt_ph_modes_file, expected_dos_json, ebins', [
            ('CaHgO2', 'CaHgO2-666-grid.yaml',
             'CaHgO2_666_dos.json', np.arange(0, 95, 0.4)*ureg('meV'))
        ])
    def test_calculate_dos_from_phonopy(
            self, material, qpt_ph_modes_file,
            expected_dos_json, ebins):
        qpt_ph_modes = QpointPhononModes.from_phonopy(
            phonon_name=get_phonopy_path(material, qpt_ph_modes_file))
        dos = qpt_ph_modes.calculate_dos(ebins)
        expected_dos = get_expected_spectrum1d(expected_dos_json)
        check_spectrum1d(dos, expected_dos)

    @pytest.mark.parametrize(
        ('material, qpt_ph_modes_json, mode_widths_json, '
         'expected_dos_json, ebins, kwargs'), [
            ('quartz', 'quartz_554_full_qpoint_phonon_modes.json',
             'quartz_554_full_mode_widths.json',
             'quartz_554_full_adaptive_dos.json',
             np.arange(0, 155, 0.1)*ureg('meV'),
             {'adaptive_method':'reference'}),
            ('quartz', 'quartz_554_full_qpoint_phonon_modes.json',
             'quartz_554_full_mode_widths.json',
             'quartz_554_full_adaptive_dos_fast.json',
             np.arange(0, 155, 0.1)*ureg('meV'),
             {'adaptive_method':'fast'})])
    def test_calculate_dos_with_mode_widths(
            self, material, qpt_ph_modes_json, mode_widths_json,
            expected_dos_json, ebins, kwargs):
        qpt_ph_modes = get_qpt_ph_modes_from_json(material, qpt_ph_modes_json)
        mode_widths = get_mode_widths(get_fc_path(mode_widths_json))
        dos = qpt_ph_modes.calculate_dos(
            ebins, mode_widths=mode_widths, **kwargs)
        expected_dos = get_expected_spectrum1d(expected_dos_json)
        check_spectrum1d(dos, expected_dos, tol=1e-13)

    @pytest.mark.parametrize(
        ('material, qpt_ph_modes_json, mode_widths_json, ebins'), [
            ('quartz', 'quartz_554_full_qpoint_phonon_modes.json',
             'quartz_554_full_mode_widths.json',
             np.arange(0, 155, 0.1)*ureg('meV'))])
    def test_calculate_dos_similar_for_ref_and_fast_methods(
            self, material, qpt_ph_modes_json, mode_widths_json, ebins):
        qpt_ph_modes = get_qpt_ph_modes_from_json(material, qpt_ph_modes_json)
        mode_widths = get_mode_widths(get_fc_path(mode_widths_json))
        ref_adaptive_dos = qpt_ph_modes.calculate_dos(
            ebins, mode_widths=mode_widths, adaptive_method='reference')
        fast_adaptive_dos = qpt_ph_modes.calculate_dos(
            ebins, mode_widths=mode_widths, adaptive_method='fast')
        assert fast_adaptive_dos.y_data.magnitude == \
            pytest.approx(ref_adaptive_dos.y_data.magnitude, abs=0.1)

    @pytest.mark.parametrize(
        ('material, qpt_ph_modes_json, mode_widths_json, mode_widths_min, '
         'ebins'), [
            ('LZO', 'lzo_222_full_qpoint_phonon_modes.json',
             'lzo_222_full_mode_widths.json',
             2e-4*ureg('hartree'),
             np.arange(0, 100, 0.1)*ureg('meV'))])
    def test_calculate_dos_with_mode_widths_min(
            self, material, qpt_ph_modes_json, mode_widths_json,
            mode_widths_min, ebins):
        qpt_ph_modes = get_qpt_ph_modes_from_json(material, qpt_ph_modes_json)
        mode_widths = get_mode_widths(get_fc_path(mode_widths_json))
        dos = qpt_ph_modes.calculate_dos(ebins, mode_widths=mode_widths,
                                      mode_widths_min=mode_widths_min)
        mode_widths = np.maximum(
            mode_widths.magnitude,
            mode_widths_min.to(mode_widths.units).magnitude)*mode_widths.units
        expected_dos = qpt_ph_modes.calculate_dos(ebins, mode_widths=mode_widths)
        check_spectrum1d(dos, expected_dos)

    def test_calculate_dos_with_0_inv_cm_bin_doesnt_raise_runtime_warn(self):
        qpt_ph_modes = get_qpt_ph_modes('quartz')
        ebins = np.arange(0, 1300, 4)*ureg('1/cm')
        with pytest.warns(None) as warn_record:
            dos = qpt_ph_modes.calculate_dos(ebins)
        assert len(warn_record) == 0

class TestQpointPhononModesCalculatePdos:

    @pytest.mark.parametrize(
        'material, qpt_ph_modes_file, expected_pdos_json, ebins, kwargs', [
            ('quartz', 'quartz-666-grid.phonon',
             'quartz_666_pdos.json', np.arange(0, 155, 0.5)*ureg('meV'), {}),
            ('quartz', 'quartz-666-grid.phonon',
             'quartz_666_coh_pdos.json', np.arange(0, 155, 0.5)*ureg('meV'),
             {'weighting': 'coherent'}),
            ('quartz', 'quartz-666-grid.phonon',
             'quartz_666_coherent_plus_incoherent_pdos.json',
             np.arange(0, 155, 0.5)*ureg('meV'),
             {'weighting': 'coherent-plus-incoherent'}),
            ('quartz', 'quartz-666-grid.phonon',
             'quartz_666_cs_dict_pdos.json', np.arange(0, 155, 0.5)*ureg('meV'),
             {'cross_sections': {'O': 429*ureg('fm**2'), 'Si': 2.78*ureg('barn')}}),
            ('LZO', 'La2Zr2O7-666-grid.phonon',
             'La2Zr2O7_666_coh_pdos.json', np.arange(0, 100, 0.8)*ureg('meV'),
             {'weighting': 'coherent'}),
            ('LZO', 'La2Zr2O7-666-grid.phonon',
             'La2Zr2O7_666_incoh_pdos.json', np.arange(0, 100, 0.8)*ureg('meV'),
             {'weighting': 'incoherent'})
        ])
    def test_calculate_pdos(
            self, material, qpt_ph_modes_file, expected_pdos_json, ebins,
            kwargs):
        qpt_ph_modes = QpointPhononModes.from_castep(
            get_castep_path(material, qpt_ph_modes_file))
        pdos = qpt_ph_modes.calculate_pdos(ebins, **kwargs)
        expected_pdos = get_expected_spectrum1dcollection(expected_pdos_json)
        check_spectrum1dcollection(pdos, expected_pdos)

    @pytest.mark.parametrize(
        ('material, qpt_ph_modes_json, mode_widths_json, '
         'expected_pdos_json, ebins, kwargs'), [
            ('LZO', 'lzo_222_full_qpoint_phonon_modes.json',
             'lzo_222_full_mode_widths.json',
             'lzo_222_full_adaptive_coh_pdos.json',
             np.arange(0, 100, 0.5)*ureg('meV'),
             {'adaptive_method':'reference'}),
            ('LZO', 'lzo_222_full_qpoint_phonon_modes.json',
             'lzo_222_full_mode_widths.json',
             'lzo_222_full_adaptive_fast_coh_pdos.json',
             np.arange(0, 100, 0.5)*ureg('meV'),
             {'adaptive_method':'fast'})])
    def test_calculate_pdos_with_mode_widths(
            self, material, qpt_ph_modes_json, mode_widths_json,
            expected_pdos_json, ebins, kwargs):
        qpt_ph_modes = get_qpt_ph_modes_from_json(material, qpt_ph_modes_json)
        mode_widths = get_mode_widths(get_fc_path(mode_widths_json))
        pdos = qpt_ph_modes.calculate_pdos(
            ebins, mode_widths=mode_widths,
            weighting='coherent', **kwargs)
        expected_pdos = get_expected_spectrum1dcollection(expected_pdos_json)
        check_spectrum1dcollection(pdos, expected_pdos)

    @pytest.mark.parametrize(
        'material, qpt_ph_modes_file, expected_dos_json, ebins', [
            ('quartz', 'quartz-666-grid.phonon',
             'quartz_666_dos.json', np.arange(0, 155, 0.5)*ureg('meV')),
        ])
    def test_total_dos_from_pdos_same_as_calculate_dos(
            self, material, qpt_ph_modes_file, expected_dos_json, ebins):
        qpt_ph_modes = QpointPhononModes.from_castep(
            get_castep_path(material, qpt_ph_modes_file))
        summed_pdos = qpt_ph_modes.calculate_pdos(ebins).sum()
        expected_total_dos = get_expected_spectrum1d(expected_dos_json)
        check_spectrum1d(summed_pdos, expected_total_dos)

    @pytest.mark.parametrize(
        ('material, qpt_ph_modes_json, mode_widths_json, expected_dos_json, '
         'ebins, kwargs'), [
            ('quartz', 'quartz_554_full_qpoint_phonon_modes.json',
             'quartz_554_full_mode_widths.json',
             'quartz_554_full_adaptive_dos.json',
             np.arange(0, 155, 0.1)*ureg('meV'),
             {'adaptive_method':'reference'}),
            ('quartz', 'quartz_554_full_qpoint_phonon_modes.json',
             'quartz_554_full_mode_widths.json',
             'quartz_554_full_adaptive_dos_fast.json',
             np.arange(0, 155, 0.1)*ureg('meV'),
             {'adaptive_method':'fast'})
        ])
    def test_total_dos_from_pdos_same_as_calculate_dos_with_mode_widths(
            self, material, qpt_ph_modes_json, mode_widths_json,
            expected_dos_json, ebins, kwargs):
        qpt_ph_modes = get_qpt_ph_modes_from_json(material, qpt_ph_modes_json)
        mode_widths = get_mode_widths(get_fc_path(mode_widths_json))
        summed_pdos = qpt_ph_modes.calculate_pdos(
            ebins, mode_widths=mode_widths, **kwargs).sum()
        expected_total_dos = get_expected_spectrum1d(expected_dos_json)
        check_spectrum1d(summed_pdos, expected_total_dos, tol=1e-13)

    def test_invalid_weighting_raises_value_error(self):
        qpt_ph_modes = QpointPhononModes.from_castep(
            get_castep_path('quartz', 'quartz-666-grid.phonon'))
        with pytest.raises(ValueError):
            qpt_ph_modes.calculate_pdos(np.arange(0, 155, 0.5)*ureg('meV'),
                                        weighting='neutron')

    def test_wrong_type_cross_sections_raises_type_error(self):
        qpt_ph_modes = QpointPhononModes.from_castep(
            get_castep_path('quartz', 'quartz-666-grid.phonon'))
        with pytest.raises(TypeError):
            qpt_ph_modes.calculate_pdos(
                np.arange(0, 155, 0.5)*ureg('meV'),
                cross_sections=[4.29, 2.78])

    def test_cross_sections_wrong_units_raises_value_error(self):
        qpt_ph_modes = QpointPhononModes.from_castep(
            get_castep_path('quartz', 'quartz-666-grid.phonon'))
        with pytest.raises(ValueError):
            qpt_ph_modes.calculate_pdos(
                np.arange(0, 155, 0.5)*ureg('meV'),
                cross_sections={'O': 4.29*ureg('fm'), 'Si': 2.78*ureg('fm')})



class TestQpointPhononModesGetDispersion:

    @pytest.mark.parametrize(
        'material, qpt_ph_modes_json, expected_dispersion_json', [
            ('quartz', 'quartz_bandstructure_qpoint_phonon_modes.json',
             'quartz_bandstructure_dispersion.json'),
            ('NaCl', 'NaCl_band_yaml_from_phonopy_qpoint_phonon_modes.json',
             'NaCl_band_yaml_dispersion.json')
        ])
    def test_get_dispersion(
            self, material, qpt_ph_modes_json, expected_dispersion_json):
        qpt_ph_modes = get_qpt_ph_modes_from_json(material, qpt_ph_modes_json)
        disp = qpt_ph_modes.get_dispersion()
        expected_disp = get_expected_spectrum1dcollection(
            expected_dispersion_json)
        check_spectrum1dcollection(disp, expected_disp)
