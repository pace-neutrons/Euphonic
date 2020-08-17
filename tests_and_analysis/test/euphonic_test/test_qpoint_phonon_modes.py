import os
import json

import pytest
import numpy as np
import numpy.testing as npt
from pint import DimensionalityError

from euphonic import ureg, Crystal, QpointPhononModes
from tests_and_analysis.test.euphonic_test.test_crystal import (
    ExpectedCrystal, get_crystal, check_crystal)
from tests_and_analysis.test.utils import get_data_path


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
    return os.path.join(get_qpt_ph_modes_dir(material),
                        f'{material}_reciprocal_qpoint_phonon_modes.json')


def get_expected_qpt_ph_modes(material):
    return ExpectedQpointPhononModes(get_json_file(material))


def get_qpt_ph_modes_from_json_file(filepath):
    return QpointPhononModes.from_json_file(filepath)


def get_qpt_ph_modes(material):
    return get_qpt_ph_modes_from_json_file(get_json_file(material))


def check_qpt_ph_modes(
        qpoint_phonon_modes, expected_qpoint_phonon_modes,
        frequencies_atol=np.finfo(np.float64).eps,
        frequencies_rtol=1e-7,
        acoustic_gamma_atol=None):
    # Note this doesn't check the eigenvectors, as these cannot be directly
    # compared in the case of degenerate frequencies, and are better compared
    # by checking a resultant observable e.g. the structure factor
    check_crystal(qpoint_phonon_modes.crystal,
                  expected_qpoint_phonon_modes.crystal)

    npt.assert_allclose(
        qpoint_phonon_modes.qpts,
        expected_qpoint_phonon_modes.qpts,
        atol=np.finfo(np.float64).eps)

    assert (qpoint_phonon_modes.frequencies.units
            == expected_qpoint_phonon_modes.frequencies.units)
    # Gamma point acoustic modes are near zero, so can be unstable and get
    # special treatment
    freqs_to_test = np.ones(qpoint_phonon_modes.frequencies.shape, dtype=bool)
    if acoustic_gamma_atol:
        qpts = expected_qpoint_phonon_modes.qpts
        gamma_points = (np.sum(np.absolute(qpts - np.rint(qpts)), axis=-1)
                        < 1e-10)
        freqs_to_test[gamma_points, :3] = False
        assert np.all(np.absolute(
            (qpoint_phonon_modes.frequencies.magnitude[~freqs_to_test]
             < acoustic_gamma_atol)))
    npt.assert_allclose(
        qpoint_phonon_modes.frequencies.magnitude[freqs_to_test],
        expected_qpoint_phonon_modes.frequencies.magnitude[freqs_to_test],
        atol=frequencies_atol, rtol=frequencies_rtol)

    npt.assert_allclose(
        qpoint_phonon_modes.weights,
        expected_qpoint_phonon_modes.weights,
        atol=np.finfo(np.float64).eps)


@pytest.mark.unit
class TestCalculateStructureFactorUsingReferenceData:
    @pytest.fixture
    def quartz_modes(self):
        return get_qpt_ph_modes('quartz')

    @pytest.fixture
    def fake_quartz_data(self):
        return {
            "description": "fake data for testing",
            "physical_property": {"coherent_scattering_length":
                                  {"__units__": "fm",
                                   "Si": {"__complex__": True,
                                          "real": 4.0, "imag": -0.70},
                                   "O": 5.803}}}

    @staticmethod
    def _dump_data(data, tmpdir, filename):
        filename = tmpdir.join(filename)
        with open(filename, 'wt') as fd:
            json.dump(data, fd)
        return str(filename)

    def test_structure_factor_with_named_ref(self, quartz_modes):
        fm = ureg['fm']
        sf_direct = quartz_modes.calculate_structure_factor(
            scattering_lengths={'Si': 4.1491*fm, 'O': 5.803*fm})
        sf_named = quartz_modes.calculate_structure_factor(
            scattering_lengths='Sears1992')

        aa2 = ureg['angstrom']**2
        npt.assert_allclose(sf_direct.structure_factors.to(aa2).magnitude,
                            sf_named.structure_factors.to(aa2).magnitude)

    def test_structure_factor_with_file_ref(self, quartz_modes,
                                            tmpdir, fake_quartz_data):
        fm = ureg['fm']

        filename = self._dump_data(fake_quartz_data, tmpdir, 'fake_data')

        sf_direct = quartz_modes.calculate_structure_factor(
            scattering_lengths={'Si': complex(4., -0.7)*fm, 'O': 5.803*fm})
        sf_from_file = quartz_modes.calculate_structure_factor(
            scattering_lengths=filename)

        aa2 = ureg['angstrom']**2
        npt.assert_allclose(sf_direct.structure_factors.to(aa2).magnitude,
                            sf_from_file.structure_factors.to(aa2).magnitude)


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
         'NaCl_mesh_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'mesh', {'summary_name': 'phonopy.yaml',
                          'phonon_name': 'mesh.hdf5'},
         'NaCl_mesh_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.yaml'},
         'NaCl_qpoints_from_phonopy_qpoint_phonon_modes.json'),
        ('NaCl', 'qpoints', {'summary_name': 'phonopy.yaml',
                             'phonon_name': 'qpoints.hdf5'},
         'NaCl_qpoints_from_phonopy_qpoint_phonon_modes.json')])
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
        qpt_ph_modes = QpointPhononModes.from_json_file(
            get_json_file(material))
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
        pytest.lazy_fixture('create_from_dict'),
        pytest.lazy_fixture('create_from_json'),
        pytest.lazy_fixture('create_from_castep'),
        pytest.lazy_fixture('create_from_phonopy')])
    def test_correct_object_creation(self, qpt_ph_modes_creator):
        qpt_ph_modes, expected_qpt_ph_modes = qpt_ph_modes_creator
        check_qpt_ph_modes(qpt_ph_modes, expected_qpt_ph_modes)

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


@pytest.mark.unit
class TestQpointPhononModesUnitConversion:

    @pytest.mark.parametrize('material, unit_attr, unit_val', [
        ('quartz', 'frequencies_unit', '1/cm')])
    def test_correct_unit_conversion(self, material, unit_attr,
                                     unit_val):
        qpt_ph_modes = get_qpt_ph_modes(material)
        setattr(qpt_ph_modes, unit_attr, unit_val)
        assert getattr(qpt_ph_modes, unit_attr) == unit_val

    @pytest.mark.parametrize('material, unit_attr, unit_val, err', [
        ('quartz', 'frequencies_unit', 'kg', ValueError)])
    def test_incorrect_unit_conversion(self, material, unit_attr,
                                       unit_val, err):
        qpt_ph_modes = get_qpt_ph_modes(material)
        with pytest.raises(err):
            setattr(qpt_ph_modes, unit_attr, unit_val)
