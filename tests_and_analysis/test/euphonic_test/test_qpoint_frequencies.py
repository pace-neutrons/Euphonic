import os
import json

import pytest
import numpy as np
import numpy.testing as npt
from pint import DimensionalityError

from euphonic import ureg, Crystal, QpointFrequencies
from euphonic.readers.phonopy import ImportPhonopyReaderError
from tests_and_analysis.test.euphonic_test.test_crystal import (
    ExpectedCrystal, check_crystal)
from tests_and_analysis.test.euphonic_test.test_force_constants import (
    get_fc_dir)
from tests_and_analysis.test.euphonic_test.test_spectrum1d import (
    get_expected_spectrum1d, check_spectrum1d)
from tests_and_analysis.test.euphonic_test.test_spectrum1dcollection import (
    get_expected_spectrum1dcollection, check_spectrum1dcollection)
from tests_and_analysis.test.utils import (
    get_data_path, get_castep_path, get_phonopy_path,
    check_frequencies_at_qpts, check_unit_conversion,
    check_json_metadata)


class ExpectedQpointFrequencies:

    def __init__(self, qpoint_frequencies_json_file: str):
        with open(qpoint_frequencies_json_file) as fd:
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
            'weights': self.weights}
        return d

    def to_constructor_args(self, crystal=None, qpts=None, frequencies=None,
                            weights=None):
        if crystal is None:
            crystal = Crystal(*self.crystal.to_constructor_args())
        if qpts is None:
            qpts = self.qpts
        if frequencies is None:
            frequencies = self.frequencies
        if weights is None:
            weights = self.weights

        kwargs = {}
        # Allow setting weights=False to not include weights in kwargs, to test
        # object creation when weights is not supplied
        if weights is not False:
            kwargs['weights'] = weights

        return (crystal, qpts, frequencies), kwargs

def get_qpt_freqs_dir(material):
    return os.path.join(get_data_path(), 'qpoint_frequencies', material)


def get_qpt_freqs(material, file):
    return QpointFrequencies.from_json_file(
        os.path.join(get_qpt_freqs_dir(material), file))


def get_expected_qpt_freqs(material, file):
    return ExpectedQpointFrequencies(
        os.path.join(get_qpt_freqs_dir(material), file))


def check_qpt_freqs(
        qpoint_frequencies, expected_qpoint_frequencies,
        frequencies_atol=np.finfo(np.float64).eps,
        frequencies_rtol=1e-7,
        acoustic_gamma_atol=None):
    check_crystal(qpoint_frequencies.crystal,
                  expected_qpoint_frequencies.crystal)

    npt.assert_allclose(
        qpoint_frequencies.qpts,
        expected_qpoint_frequencies.qpts,
        atol=np.finfo(np.float64).eps)

    assert (qpoint_frequencies.frequencies.units
            == expected_qpoint_frequencies.frequencies.units)
    # Check frequencies
    check_frequencies_at_qpts(
        qpoint_frequencies.qpts,
        qpoint_frequencies.frequencies.magnitude,
        expected_qpoint_frequencies.frequencies.magnitude,
        atol=frequencies_atol,
        rtol=frequencies_rtol,
        gamma_atol=acoustic_gamma_atol)

    npt.assert_allclose(
        qpoint_frequencies.weights,
        expected_qpoint_frequencies.weights,
        atol=np.finfo(np.float64).eps)


@pytest.mark.unit
class TestQpointFrequenciesCreation:

    @pytest.fixture(params=[
        get_expected_qpt_freqs(
            'quartz', 'quartz_666_qpoint_frequencies.json'),
        get_expected_qpt_freqs(
            'quartz', 'quartz_bandstructure_qpoint_frequencies.json'),
        get_expected_qpt_freqs(
            'quartz', 'quartz_666_cv_only_qpoint_frequencies.json')])
    def create_from_constructor(self, request):
        expected_qpt_freqs = request.param
        args, kwargs = expected_qpt_freqs.to_constructor_args()
        qpt_freqs = QpointFrequencies(*args, **kwargs)
        return qpt_freqs, expected_qpt_freqs

    @pytest.fixture(params=[
        get_expected_qpt_freqs(
            'quartz', 'quartz_bandstructure_qpoint_frequencies.json')])
    def create_from_constructor_without_weights(self, request):
        expected_qpt_freqs = request.param
        args, kwargs = expected_qpt_freqs.to_constructor_args(weights=False)
        qpt_freqs = QpointFrequencies(*args, **kwargs)
        return qpt_freqs, expected_qpt_freqs

    @pytest.fixture(params=[
        ('LZO', 'La2Zr2O7.phonon',
         'LZO_qpoint_frequencies.json'),
        ('quartz', 'quartz-666-grid.phonon',
         'quartz_666_qpoint_frequencies.json'),
        ('quartz', 'quartz_split_qpts.phonon',
         'quartz_split_from_castep_qpoint_frequencies.json')])
    def create_from_castep(self, request):
        material, phonon_file, json_file = request.param
        qpt_freqs = QpointFrequencies.from_castep(
            get_castep_path(material, phonon_file))
        expected_qpt_freqs = ExpectedQpointFrequencies(
            os.path.join(get_qpt_freqs_dir(material), json_file))
        return qpt_freqs, expected_qpt_freqs

    @pytest.fixture(params=[
        ('NaCl', 'band', {'summary_name': 'should_not_be_read',
                          'phonon_name': 'band.yaml'},
         'NaCl_band_yaml_from_phonopy_qpoint_frequencies.json'),
        ('NaCl', 'band', {'summary_name': 'phonopy.yaml',
                          'phonon_name': 'band_no_evec.hdf5'},
         'NaCl_band_no_evec_hdf5_from_phonopy_qpoint_frequencies.json'),
        ('NaCl', 'mesh', {'summary_name': 'should_not_be_read',
                          'phonon_name': 'mesh_no_evec.yaml'},
         'NaCl_mesh_yaml_from_phonopy_qpoint_frequencies.json'),
        ('NaCl', 'mesh', {'summary_name': 'phonopy.yaml',
                          'phonon_name': 'mesh.hdf5'},
         'NaCl_mesh_hdf5_from_phonopy_qpoint_frequencies.json'),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.yaml'},
         'NaCl_qpoints_yaml_from_phonopy_qpoint_frequencies.json'),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints_yaml.test',
                             'phonon_format': 'yaml'},
         'NaCl_qpoints_yaml_from_phonopy_qpoint_frequencies.json'),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints_hdf5.test',
                             'phonon_format': 'hdf5'},
         'NaCl_qpoints_hdf5_from_phonopy_qpoint_frequencies.json'),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.hdf5'},
         'NaCl_qpoints_hdf5_from_phonopy_qpoint_frequencies.json'),
        ('CaHgO2', '', {'summary_name': 'mp-7041-20180417.yaml',
                        'phonon_name': 'qpoints.yaml'},
         'CaHgO2_from_phonopy_qpoint_frequencies.json')])
    def create_from_phonopy(self, request):
        material, subdir, phonopy_args, json_file = request.param
        phonopy_args['path'] = get_phonopy_path(material, subdir)
        qpt_freqs = QpointFrequencies.from_phonopy(**phonopy_args)
        json_path = os.path.join(
            get_qpt_freqs_dir(material), json_file)
        expected_qpt_freqs = ExpectedQpointFrequencies(json_path)
        return qpt_freqs, expected_qpt_freqs

    @pytest.fixture(params=[
        ('quartz', 'quartz_666_qpoint_frequencies.json'),
        ('quartz', 'quartz_bandstructure_qpoint_frequencies.json'),
        ('quartz', 'quartz_666_cv_only_qpoint_frequencies.json')])
    def create_from_json(self, request):
        material, json_file = request.param
        expected_qpt_freqs = get_expected_qpt_freqs(material, json_file)
        qpt_freqs = get_qpt_freqs(material, json_file)
        return qpt_freqs, expected_qpt_freqs

    @pytest.fixture(params=[
        ('quartz', 'quartz_666_qpoint_frequencies.json'),
        ('quartz', 'quartz_bandstructure_qpoint_frequencies.json'),
        ('quartz', 'quartz_666_cv_only_qpoint_frequencies.json')])
    def create_from_dict(self, request):
        material, json_file = request.param
        expected_qpt_freqs = get_expected_qpt_freqs(material, json_file)
        qpt_freqs = QpointFrequencies.from_dict(
            expected_qpt_freqs.to_dict())
        return qpt_freqs, expected_qpt_freqs

    @pytest.mark.parametrize(('qpt_freqs_creator'), [
        pytest.lazy_fixture('create_from_constructor'),
        pytest.lazy_fixture('create_from_constructor_without_weights'),
        pytest.lazy_fixture('create_from_dict'),
        pytest.lazy_fixture('create_from_json'),
        pytest.lazy_fixture('create_from_castep'),
        pytest.lazy_fixture('create_from_phonopy')])
    def test_correct_object_creation(self, qpt_freqs_creator):
        qpt_freqs, expected_qpt_freqs = qpt_freqs_creator
        check_qpt_freqs(qpt_freqs, expected_qpt_freqs)

    @pytest.fixture(params=[
        ('qpts',
         get_expected_qpt_freqs(
             'quartz', 'quartz_666_qpoint_frequencies.json').qpts[:3],
         ValueError),
        ('frequencies',
         get_expected_qpt_freqs(
             'quartz', 'quartz_666_qpoint_frequencies.json').frequencies[:2],
         ValueError),
        ('frequencies',
         get_expected_qpt_freqs(
             'quartz', 'quartz_666_qpoint_frequencies.json'
         ).frequencies.magnitude,
         TypeError),
        ('frequencies',
         get_expected_qpt_freqs(
             'quartz', 'quartz_666_qpoint_frequencies.json'
         ).frequencies.magnitude*ureg('kg'),
         DimensionalityError),
        ('weights',
         get_expected_qpt_freqs(
             'quartz', 'quartz_666_qpoint_frequencies.json').weights[:5],
         ValueError)])
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        expected_qpt_freqs = get_expected_qpt_freqs(
            'quartz', 'quartz_666_qpoint_frequencies.json')
        # Inject the faulty value and get a tuple of constructor arguments
        args, kwargs = expected_qpt_freqs.to_constructor_args(
            **{faulty_arg: faulty_value})
        return args, kwargs, expected_exception

    def test_faulty_object_creation(self, inject_faulty_elements):
        faulty_args, faulty_kwargs, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            QpointFrequencies(*faulty_args, **faulty_kwargs)

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
            QpointFrequencies.from_phonopy(**phonopy_args)

    @pytest.mark.parametrize('material, subdir, phonopy_args, err', [
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints_hdf5.test'},
         ValueError),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.hdf5',
                             'phonon_format': 'nonsense'},
         ValueError),
        ('NaCl', 'qpoints', {
            'summary_name': '../../CaHgO2/mp-7041-20180417.yaml',
            'phonon_name': 'qpoints.hdf5'},
         ValueError)])
    def test_create_from_phonopy_with_bad_inputs_raises_err(
            self, material, subdir, phonopy_args, err):
        phonopy_args['path'] = get_phonopy_path(material, subdir)
        with pytest.raises(err):
            QpointFrequencies.from_phonopy(**phonopy_args)

    @pytest.mark.parametrize('material, subdir, phonopy_args, json_file', [
        ('CaHgO2', '', {'summary_name': 'mp-7041-20180417.yaml',
                        'phonon_name': 'qpoints.yaml'},
         'CaHgO2_from_phonopy_qpoint_frequencies.json')])
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
        qpt_freqs = QpointFrequencies.from_phonopy(**phonopy_args)
        json_path = os.path.join(
            get_qpt_freqs_dir(material), json_file)
        expected_qpt_freqs = ExpectedQpointFrequencies(json_path)
        check_qpt_freqs(qpt_freqs, expected_qpt_freqs)


@pytest.mark.unit
class TestQpointFrequenciesSerialisation:

    @pytest.mark.parametrize('qpt_freqs', [
        get_qpt_freqs('quartz', 'quartz_666_qpoint_frequencies.json'),
        get_qpt_freqs('quartz', 'quartz_666_cv_only_qpoint_frequencies.json')])
    def test_serialise_to_json_file(self, qpt_freqs, tmpdir):
        output_file = str(tmpdir.join('tmp.test'))
        qpt_freqs.to_json_file(output_file)
        check_json_metadata(output_file, 'QpointFrequencies')
        deserialised_qpt_freqs = QpointFrequencies.from_json_file(
            output_file)
        check_qpt_freqs(qpt_freqs, deserialised_qpt_freqs)

    @pytest.fixture(params=[
        ('quartz', 'quartz_666_qpoint_frequencies.json'),
        ('quartz', 'quartz_666_cv_only_qpoint_frequencies.json')])
    def serialise_to_dict(self, request):
        material, json_file = request.param
        qpt_freqs = get_qpt_freqs(material, json_file)
        # Convert to dict, then back to object to test
        qpt_freqs_dict = qpt_freqs.to_dict()
        qpt_freqs_from_dict = QpointFrequencies.from_dict(qpt_freqs_dict)
        return qpt_freqs, qpt_freqs_from_dict

    def test_serialise_to_dict(self, serialise_to_dict):
        qpt_freqs, qpt_freqs_from_dict = serialise_to_dict
        check_qpt_freqs(qpt_freqs, qpt_freqs_from_dict)


@pytest.mark.unit
class TestQpointFrequenciesUnitConversion:

    @pytest.mark.parametrize('material, json_file, attr, unit_val', [
        ('quartz',
         'quartz_666_qpoint_frequencies.json',
         'frequencies',
         '1/cm')])
    def test_correct_unit_conversion(self, material, json_file, attr, unit_val):
        qpt_freqs = get_qpt_freqs(material, json_file)
        check_unit_conversion(qpt_freqs, attr, unit_val)

    @pytest.mark.parametrize('material, json_file, unit_attr, unit_val, err', [
        ('quartz',
         'quartz_666_qpoint_frequencies.json',
         'frequencies_unit',
         'kg', ValueError)])
    def test_incorrect_unit_conversion(self, material, json_file, unit_attr,
                                       unit_val, err):
        qpt_freqs = get_qpt_freqs(material, json_file)
        with pytest.raises(err):
            setattr(qpt_freqs, unit_attr, unit_val)


@pytest.mark.unit
class TestQpointFrequenciesCalculateDos:

    @pytest.mark.parametrize(
        'material, qpt_freqs_json, expected_dos_json, ebins', [
            ('quartz', 'quartz_666_qpoint_frequencies.json',
             'quartz_666_dos.json', np.arange(0, 155, 0.5)*ureg('meV')),
            ('quartz', 'quartz_666_cv_only_qpoint_frequencies.json',
             'quartz_666_dos.json', np.arange(0, 155, 0.5)*ureg('meV')),
            ('CaHgO2', 'CaHgO2_666_qpoint_frequencies.json',
             'CaHgO2_666_dos.json', np.arange(0, 95, 0.4)*ureg('meV')),
        ])
    def test_calculate_dos(
            self, material, qpt_freqs_json, expected_dos_json, ebins):
        qpt_freqs = get_qpt_freqs(material, qpt_freqs_json)
        dos = qpt_freqs.calculate_dos(ebins)
        expected_dos = get_expected_spectrum1d(expected_dos_json)
        # Temporary solution until test data has been regenerated, results
        # have changed by a factor of the number of modes
        dos._y_data = dos._y_data/qpt_freqs.frequencies.shape[1]
        check_spectrum1d(dos, expected_dos)

    @pytest.mark.parametrize(
        ('material, qpt_freqs_json, mode_widths_json, '
         'expected_dos_json, ebins'), [
            ('quartz', 'quartz_554_full_qpoint_frequencies.json',
             'quartz_554_full_mode_widths.json',
             'quartz_554_full_adaptive_dos.json',
             np.arange(0, 155, 0.1)*ureg('meV')),
            ('LZO', 'lzo_222_full_qpoint_frequencies.json',
             'lzo_222_full_mode_widths.json',
             'lzo_222_full_adaptive_dos.json',
             np.arange(0, 100, 0.1)*ureg('meV'))])
    def test_calculate_dos_with_mode_widths(
            self, material, qpt_freqs_json, mode_widths_json,
            expected_dos_json, ebins):
        qpt_freqs = get_qpt_freqs(material, qpt_freqs_json)
        with open(os.path.join(get_fc_dir(), mode_widths_json), 'r') as fp:
            modw_dict = json.load(fp)
        mode_widths = modw_dict['mode_widths']*ureg(
            modw_dict['mode_widths_unit'])
        dos = qpt_freqs.calculate_dos(
            ebins, mode_widths=mode_widths)
        expected_dos = get_expected_spectrum1d(expected_dos_json)
        # Temporary solution until test data has been regenerated, results
        # have changed by a factor of the number of modes
        dos._y_data = dos._y_data/qpt_freqs.frequencies.shape[1]
        check_spectrum1d(dos, expected_dos)

    @pytest.mark.parametrize(
        ('material, qpt_freqs_json, mode_widths_json, mode_widths_min, '
         'ebins'), [
            ('LZO', 'lzo_222_full_qpoint_frequencies.json',
             'lzo_222_full_mode_widths.json',
              5*ureg('meV'),
             np.arange(0, 100, 0.1)*ureg('meV')),
            ('LZO', 'lzo_222_full_qpoint_frequencies.json',
             'lzo_222_full_mode_widths.json',
             2e-4*ureg('hartree'),
             np.arange(0, 100, 0.1)*ureg('meV'))])
    def test_calculate_dos_with_mode_widths_min(
            self, material, qpt_freqs_json, mode_widths_json,
            mode_widths_min, ebins):
        qpt_freqs = get_qpt_freqs(material, qpt_freqs_json)
        with open(os.path.join(get_fc_dir(), mode_widths_json), 'r') as fp:
            modw_dict = json.load(fp)
        mode_widths = modw_dict['mode_widths']*ureg(
            modw_dict['mode_widths_unit'])
        dos = qpt_freqs.calculate_dos(ebins, mode_widths=mode_widths,
                                      mode_widths_min=mode_widths_min)
        mode_widths = np.maximum(
            mode_widths.magnitude,
            mode_widths_min.to(mode_widths.units).magnitude)*mode_widths.units
        expected_dos = qpt_freqs.calculate_dos(ebins, mode_widths=mode_widths)
        check_spectrum1d(dos, expected_dos)

    def test_calculate_dos_with_0_inv_cm_bin_doesnt_raise_runtime_warn(self):
        qpt_freqs = get_qpt_freqs(
            'quartz', 'quartz_666_qpoint_frequencies.json')
        ebins = np.arange(0, 1300, 4)*ureg('1/cm')
        with pytest.warns(None) as warn_record:
            dos = qpt_freqs.calculate_dos(ebins)
        assert len(warn_record) == 0

@pytest.mark.unit
class TestQpointFrequenciesGetDispersion:

    @pytest.mark.parametrize(
        'material, qpt_freqs_json, expected_dispersion_json', [
            ('quartz', 'quartz_bandstructure_qpoint_frequencies.json',
             'quartz_bandstructure_dispersion.json'),
            ('quartz', 'quartz_bandstructure_cv_only_qpoint_frequencies.json',
             'quartz_bandstructure_cv_only_dispersion.json'),
            ('NaCl', 'NaCl_band_yaml_from_phonopy_qpoint_frequencies.json',
             'NaCl_band_yaml_dispersion.json')
        ])
    def test_get_dispersion(
            self, material, qpt_freqs_json, expected_dispersion_json):
        qpt_freqs = get_qpt_freqs(material, qpt_freqs_json)
        disp = qpt_freqs.get_dispersion()
        expected_disp = get_expected_spectrum1dcollection(
            expected_dispersion_json)
        check_spectrum1dcollection(disp, expected_disp)
