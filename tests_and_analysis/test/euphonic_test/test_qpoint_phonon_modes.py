import os
import json

import pytest
import numpy as np
import numpy.testing as npt
from pint import DimensionalityError

from euphonic import ureg, Crystal, QpointPhononModes
from euphonic.readers.phonopy import ImportPhonopyReaderError
from tests_and_analysis.test.euphonic_test.test_crystal import (
    ExpectedCrystal, get_crystal, check_crystal)
from tests_and_analysis.test.euphonic_test.test_debye_waller import (
    get_expected_dw, check_debye_waller)
from tests_and_analysis.test.euphonic_test.test_spectrum1d import (
    get_expected_spectrum1d, check_spectrum1d)
from tests_and_analysis.test.utils import (
    get_data_path, check_mode_values_at_qpts, check_unit_conversion)


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
        return np.array(self.data['weights'])

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


def get_qpt_ph_modes_dir(material):
    return os.path.join(get_data_path(), 'qpoint_phonon_modes', material)


def get_json_file(material):
    return f'{material}_reciprocal_qpoint_phonon_modes.json'


def get_expected_qpt_ph_modes(material):
    return ExpectedQpointPhononModes(
        os.path.join(get_qpt_ph_modes_dir(material), get_json_file(material)))


def get_qpt_ph_modes_from_json(material, file):
    return QpointPhononModes.from_json_file(
        os.path.join(get_qpt_ph_modes_dir(material), file))


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
    check_mode_values_at_qpts(
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


@pytest.mark.unit
class TestQpointPhononModesCreation:

    @pytest.fixture(params=[get_expected_qpt_ph_modes('quartz'),
                            get_expected_qpt_ph_modes('LZO'),
                            get_expected_qpt_ph_modes('NaCl')])
    def create_from_constructor(self, request):
        expected_qpt_ph_modes = request.param
        args, kwargs = expected_qpt_ph_modes.to_constructor_args()
        qpt_ph_modes = QpointPhononModes(*args, **kwargs)
        return qpt_ph_modes, expected_qpt_ph_modes

    @pytest.fixture(params=[get_expected_qpt_ph_modes('quartz'),
                            get_expected_qpt_ph_modes('NaCl')])
    def create_from_constructor_without_weights(self, request):
        expected_qpt_ph_modes = request.param
        args, kwargs = expected_qpt_ph_modes.to_constructor_args(weights=False)
        qpt_ph_modes = QpointPhononModes(*args, **kwargs)
        return qpt_ph_modes, expected_qpt_ph_modes

    @pytest.fixture(params=[
        ('LZO', 'La2Zr2O7.phonon',
         'LZO_from_castep_qpoint_phonon_modes.json'),
        ('Si2-sc-skew', 'Si2-sc-skew.phonon',
         'Si2-sc-skew_from_castep_qpoint_phonon_modes.json'),
        ('quartz', 'quartz_nosplit.phonon',
         'quartz_from_castep_qpoint_phonon_modes.json'),
        ('quartz', 'quartz_split_qpts.phonon',
         'quartz_split_from_castep_qpoint_phonon_modes.json')])
    def create_from_castep(self, request):
        material, phonon_file, json_file = request.param
        qpt_ph_modes = QpointPhononModes.from_castep(
            os.path.join(get_qpt_ph_modes_dir(material), phonon_file))
        expected_qpt_ph_modes = ExpectedQpointPhononModes(
            os.path.join(get_qpt_ph_modes_dir(material), json_file))
        return qpt_ph_modes, expected_qpt_ph_modes

    @pytest.fixture(params=[
        ('NaCl', 'band', {'summary_name': 'phonopy.yaml',
                          'phonon_name': 'band.yaml'},
         'NaCl_band_yaml_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'band', {'summary_name': 'phonopy.yaml',
                          'phonon_name': 'band.hdf5'},
         'NaCl_band_hdf5_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'mesh', {'summary_name': 'phonopy.yaml',
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
    def create_from_phonopy(self, request):
        material, subdir, phonopy_args, json_file = request.param
        phonopy_args['path'] = os.path.join(get_qpt_ph_modes_dir(material),
                                            subdir)
        qpt_ph_modes = QpointPhononModes.from_phonopy(**phonopy_args)
        json_path = os.path.join(phonopy_args['path'], json_file)
        expected_qpt_ph_modes = ExpectedQpointPhononModes(json_path)
        return qpt_ph_modes, expected_qpt_ph_modes

    @pytest.fixture(params=['LZO', 'quartz', 'NaCl'])
    def create_from_json(self, request):
        material = request.param
        expected_qpt_ph_modes = get_expected_qpt_ph_modes(material)
        qpt_ph_modes = get_qpt_ph_modes(material)
        return qpt_ph_modes, expected_qpt_ph_modes

    @pytest.fixture(params=['LZO', 'quartz', 'NaCl'])
    def create_from_dict(self, request):
        material = request.param
        expected_qpt_ph_modes = get_expected_qpt_ph_modes(material)
        qpt_ph_modes = QpointPhononModes.from_dict(
            expected_qpt_ph_modes.to_dict())
        return qpt_ph_modes, expected_qpt_ph_modes

    @pytest.mark.parametrize(('qpt_ph_modes_creator'), [
        pytest.lazy_fixture('create_from_constructor'),
        pytest.lazy_fixture('create_from_constructor_without_weights'),
        pytest.lazy_fixture('create_from_dict'),
        pytest.lazy_fixture('create_from_json'),
        pytest.lazy_fixture('create_from_castep'),
        pytest.lazy_fixture('create_from_phonopy')])
    def test_correct_object_creation(self, qpt_ph_modes_creator):
        qpt_ph_modes, expected_qpt_ph_modes = qpt_ph_modes_creator
        check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes,
                           check_evecs=True)

    @pytest.fixture(params=[
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
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        expected_qpt_ph_modes = get_expected_qpt_ph_modes('quartz')
        # Inject the faulty value and get a tuple of constructor arguments
        args, kwargs = expected_qpt_ph_modes.to_constructor_args(
            **{faulty_arg: faulty_value})
        return args, kwargs, expected_exception

    def test_faulty_object_creation(self, inject_faulty_elements):
        faulty_args, faulty_kwargs, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            QpointPhononModes(*faulty_args, **faulty_kwargs)


    @pytest.mark.parametrize('material, subdir, phonopy_args', [
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.yaml'}),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.hdf5'})])
    def test_create_from_phonopy_without_installed_modules_raises_err(
            self, material, subdir, phonopy_args, mocker):
        phonopy_args['path'] = os.path.join(get_qpt_ph_modes_dir(material),
                                            subdir)
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
        phonopy_args['path'] = os.path.join(get_qpt_ph_modes_dir(material),
                                            subdir)
        with pytest.raises(err):
            QpointPhononModes.from_phonopy(**phonopy_args)

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

        phonopy_args['path'] = os.path.join(get_qpt_ph_modes_dir(material),
                                            subdir)
        qpt_ph_modes = QpointPhononModes.from_phonopy(**phonopy_args)
        json_path = os.path.join(phonopy_args['path'], json_file)
        expected_qpt_ph_modes = ExpectedQpointPhononModes(json_path)
        check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes,
                           check_evecs=True)

@pytest.mark.unit
class TestQpointPhononModesSerialisation:

    @pytest.fixture(params=['quartz', 'Si2-sc-skew', 'NaCl'])
    def serialise_to_json_file(self, request, tmpdir):
        qpt_ph_modes = get_qpt_ph_modes(request.param)
        # Write to file then read back to test
        output_file = str(tmpdir.join('tmp.test'))
        qpt_ph_modes.to_json_file(output_file)
        deserialised_qpt_ph_modes = QpointPhononModes.from_json_file(
            output_file)
        return qpt_ph_modes, deserialised_qpt_ph_modes

    def test_serialise_to_file(self, serialise_to_json_file):
        qpt_ph_modes, deserialised_qpt_ph_modes = serialise_to_json_file
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
class TestQpointPhononModesCalculateDebyeWaller:

    @pytest.mark.parametrize(
        'material, qpt_ph_modes_file, expected_dw_json, temperature', [
            ('quartz', 'quartz-666-grid.phonon',
             'quartz_666_0K_debye_waller.json', 0),
            ('quartz', 'quartz-666-grid.phonon',
             'quartz_666_300K_debye_waller.json', 300),
            ('quartz', 'quartz-777-grid.phonon',
             'quartz_777_300K_debye_waller.json', 300),
            ('Si2-sc-skew', 'Si2-sc-skew-666-grid.phonon',
             'Si2-sc-skew_666_300K_debye_waller.json', 300),
            ('CaHgO2', 'CaHgO2-666-grid.yaml',
             'CaHgO2_666_300K_debye_waller.json', 300)
        ])
    def test_calculate_debye_waller(self, material, qpt_ph_modes_file,
                                    expected_dw_json, temperature):
        filepath = os.path.join(get_qpt_ph_modes_dir(material),
                                qpt_ph_modes_file)
        if filepath.endswith('.phonon'):
            qpt_ph_modes = QpointPhononModes.from_castep(filepath)
        else:
            qpt_ph_modes = QpointPhononModes.from_phonopy(phonon_name=filepath)

        dw = qpt_ph_modes.calculate_debye_waller(temperature*ureg('K'))
        expected_dw = get_expected_dw(material, expected_dw_json)
        check_debye_waller(dw, expected_dw)


@pytest.mark.unit
class TestQpointPhononModesCalculateDos:

    @pytest.mark.parametrize(
        'material, qpt_ph_modes_file, expected_dos_json, ebins', [
            ('quartz', 'quartz-666-grid.phonon',
             'quartz_666_dos.json', np.arange(0, 155, 0.5)*ureg('meV')),
            ('CaHgO2', 'CaHgO2-666-grid.yaml',
             'CaHgO2_666_dos.json', np.arange(0, 95, 0.4)*ureg('meV'))
        ])
    def test_calculate_dos(self, material, qpt_ph_modes_file,
                           expected_dos_json, ebins):
        filepath = os.path.join(get_qpt_ph_modes_dir(material),
                                qpt_ph_modes_file)
        if filepath.endswith('.phonon'):
            qpt_ph_modes = QpointPhononModes.from_castep(filepath)
        else:
            qpt_ph_modes = QpointPhononModes.from_phonopy(phonon_name=filepath)
        dos = qpt_ph_modes.calculate_dos(ebins)
        expected_dos = get_expected_spectrum1d(expected_dos_json)
        check_spectrum1d(dos, expected_dos)

    def test_calculate_dos_with_0_inv_cm_bin_doesnt_raise_runtime_warn(self):
        qpt_ph_modes = get_qpt_ph_modes('quartz')
        ebins = np.arange(0, 1300, 4)*ureg('1/cm')
        with pytest.warns(None) as warn_record:
            dos = qpt_ph_modes.calculate_dos(ebins)
        assert len(warn_record) == 0
