import json
import os

import numpy as np
import numpy.testing as npt
from pint import DimensionalityError
import pytest

from euphonic import ureg, Crystal, StructureFactor
from tests_and_analysis.test.euphonic_test.test_crystal import (
    ExpectedCrystal, get_crystal, check_crystal)
from tests_and_analysis.test.euphonic_test.test_qpoint_frequencies import (
    get_expected_qpt_freqs, check_qpt_freqs)
from tests_and_analysis.test.euphonic_test.test_spectrum2d import (
    get_expected_spectrum2d, check_spectrum2d)
from tests_and_analysis.test.euphonic_test.test_spectrum1d import (
    get_expected_spectrum1d, check_spectrum1d)
from tests_and_analysis.test.euphonic_test.test_spectrum1dcollection import (
    get_expected_spectrum1dcollection, check_spectrum1dcollection)
from tests_and_analysis.test.utils import (
    get_data_path, check_frequencies_at_qpts, check_structure_factors_at_qpts,
    check_unit_conversion, check_json_metadata)


class ExpectedStructureFactor:

    def __init__(self, structure_factor_json_file: str):
        with open(structure_factor_json_file) as fd:
            self.data = json.load(fd)

    @property
    def crystal(self):
        return ExpectedCrystal(self.data['crystal'])

    @property
    def qpts(self):
        return np.array(self.data['qpts'])

    @property
    def structure_factors(self):
        return np.array(self.data['structure_factors'])*ureg(
            self.data['structure_factors_unit'])

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

    @property
    def temperature(self):
        if 'temperature' in self.data.keys():
            return np.array(self.data['temperature'])*ureg(
                self.data['temperature_unit'])
        else:
            return None

    def to_dict(self):
        d = {
            'crystal': self.crystal.to_dict(),
            'qpts': self.qpts,
            'frequencies': self.frequencies.magnitude,
            'frequencies_unit': str(self.frequencies.units),
            'structure_factors': self.structure_factors.magnitude,
            'structure_factors_unit': str(self.structure_factors.units),
            'weights': self.weights}
        if self.temperature is not None:
            d['temperature'] = self.temperature.magnitude
            d['temperature_unit'] = str(self.temperature.units)
        return d

    def to_constructor_args(self, crystal=None, qpts=None, frequencies=None,
                            structure_factors=None, weights=None, temperature=None):
        if crystal is None:
            crystal = Crystal(*self.crystal.to_constructor_args())
        if qpts is None:
            qpts = self.qpts
        if frequencies is None:
            frequencies = self.frequencies
        if structure_factors is None:
            structure_factors = self.structure_factors
        if weights is None:
            weights = self.weights
        if temperature is None:
            temperature = self.temperature

        kwargs = {}
        # Allow setting weights=False to not include weights in kwargs, to test
        # object creation when weights is not supplied
        if weights is not False:
            kwargs['weights'] = weights
        if temperature is not None:
            kwargs['temperature'] = temperature

        return (crystal, qpts, frequencies, structure_factors), kwargs


def get_sf_dir(material):
    return os.path.join(get_data_path(), 'structure_factor', material)


def get_json_file(material, json_file):
    return os.path.join(get_sf_dir(material), json_file)


def get_sf(material, json_file):
    return StructureFactor.from_json_file(get_json_file(material, json_file))


def get_expected_sf(material, json_file):
    return ExpectedStructureFactor(get_json_file(material, json_file))


def get_summed_structure_factors(structure_factors, frequencies, TOL=0.05):
    """
    For degenerate frequency modes, eigenvectors are an arbitrary
    admixture, so the derived structure factors can only be compared
    when summed over degenerate modes. This function performs that
    summation

    Parameters
    ----------
    structure_factors (n_qpts, n_branches) float ndarray
        The plain structure factors magnitudes
    frequencies (n_qpts, n_branches) float ndarray
        The plain frequency magnitudes

    Returns
    -------
    sf_sum (n_qpts, n_branches) float ndarray
        The summed structure factors. As there will be different numbers
        of summed values for each q-point depending on the number of
        degenerate modes, the last few entries for some q-points will be
        zero
    """
    sf_sum = np.zeros(structure_factors.shape)
    for i in range(len(frequencies)):
        diff = np.append(TOL + 1, np.diff(frequencies[i]))
        unique_index = np.where(diff > TOL)[0]
        x = np.zeros(len(frequencies[0]), dtype=np.int32)
        x[unique_index] = 1
        unique_modes = np.cumsum(x) - 1
        sf_sum[i, :len(unique_index)] = np.bincount(unique_modes,
                                                    structure_factors[i])
    return sf_sum


def check_structure_factor(
        sf, expected_sf,
        freq_atol=np.finfo(np.float64).eps,
        freq_rtol=1e-7,
        freq_gamma_atol=None,
        sf_atol=np.finfo(np.float64).eps,
        sf_rtol=1e-7,
        sf_gamma_atol=None,
        sum_sf=True):

    check_crystal(sf.crystal,
                  expected_sf.crystal)

    npt.assert_allclose(
        sf.qpts,
        sf.qpts,
        atol=np.finfo(np.float64).eps)

    npt.assert_allclose(
        sf.weights,
        expected_sf.weights,
        atol=np.finfo(np.float64).eps)

    if expected_sf.temperature is None:
        assert sf.temperature is None
    else:
        assert (str(sf.temperature.units)
                == str(expected_sf.temperature.units))
        npt.assert_almost_equal(
            sf.temperature.magnitude,
            expected_sf.temperature.magnitude)

    assert sf.frequencies.units == expected_sf.frequencies.units
    # Check frequencies
    check_frequencies_at_qpts(
        sf.qpts,
        sf.frequencies.magnitude,
        expected_sf.frequencies.magnitude,
        atol=freq_atol,
        rtol=freq_rtol,
        gamma_atol=freq_gamma_atol)

    assert sf.structure_factors.units == expected_sf.structure_factors.units
    # Sum structure factors over degenerate modes and check
    if sum_sf:
        sf_sum = get_summed_structure_factors(
            sf.structure_factors.magnitude,
            expected_sf.frequencies.magnitude)
        expected_sf_sum = get_summed_structure_factors(
            expected_sf.structure_factors.magnitude,
            expected_sf.frequencies.magnitude)
    else:
        sf_sum = sf.structure_factors.magnitude
        expected_sf_sum = expected_sf.structure_factors.magnitude
    check_structure_factors_at_qpts(
        sf.qpts,
        sf_sum,
        expected_sf_sum,
        atol=sf_atol,
        rtol=sf_rtol,
        gamma_atol=sf_gamma_atol)


@pytest.mark.unit
class TestStructureFactorCreation:

    @pytest.fixture(params=[
        get_expected_sf('quartz', 'quartz_0K_structure_factor.json'),
        get_expected_sf('CaHgO2', 'CaHgO2_300K_structure_factor.json')])
    def create_from_constructor(self, request):
        expected_sf = request.param
        args, kwargs = expected_sf.to_constructor_args()
        sf = StructureFactor(*args, **kwargs)
        return sf, expected_sf

    @pytest.fixture(params=[
        get_expected_sf('quartz', 'quartz_0K_structure_factor.json'),
        get_expected_sf('CaHgO2', 'CaHgO2_300K_structure_factor.json')])
    def create_from_constructor_without_weights(self, request):
        expected_sf = request.param
        args, kwargs = expected_sf.to_constructor_args(weights=False)
        sf = StructureFactor(*args, **kwargs)
        return sf, expected_sf

    @pytest.fixture(params=[
        get_expected_sf('quartz', 'quartz_0K_structure_factor.json'),
        get_expected_sf('CaHgO2', 'CaHgO2_300K_structure_factor.json')])
    def create_from_dict(self, request):
        expected_sf = request.param
        d = expected_sf.to_dict()
        sf = StructureFactor.from_dict(d)
        return sf, expected_sf

    @pytest.fixture(params=[
        ('quartz', 'quartz_0K_structure_factor.json'),
        ('CaHgO2', 'CaHgO2_300K_structure_factor.json')])
    def create_from_json_file(self, request):
        material, json_file = request.param
        sf = get_sf(material, json_file)
        expected_sf = get_expected_sf(material, json_file)
        return sf, expected_sf

    @pytest.mark.parametrize('sf_creator', [
        pytest.lazy_fixture('create_from_constructor'),
        pytest.lazy_fixture('create_from_json_file'),
        pytest.lazy_fixture('create_from_dict')])
    def test_create(self, sf_creator):
        sf, expected_sf = sf_creator
        check_structure_factor(sf, expected_sf, sum_sf=False)

    faulty_elements = [
        ('crystal', get_crystal('LZO'), ValueError),
        ('frequencies',
         get_expected_sf(
             'quartz',
             'quartz_0K_structure_factor.json').frequencies.magnitude,
         TypeError),
        ('frequencies',
         get_expected_sf(
             'quartz',
             'quartz_0K_structure_factor.json').frequencies[:3],
         ValueError),
        ('frequencies', (
         get_expected_sf(
             'quartz',
             'quartz_0K_structure_factor.json').frequencies.magnitude
         *ureg('kg')),
         DimensionalityError),
        ('structure_factors',
         get_expected_sf(
             'quartz',
             'quartz_0K_structure_factor.json').structure_factors.magnitude,
         TypeError),
        ('structure_factors',
         get_expected_sf(
             'quartz',
             'quartz_0K_structure_factor.json').structure_factors[:5],
         ValueError),
        ('structure_factors', (
         get_expected_sf(
             'quartz',
             'quartz_0K_structure_factor.json').structure_factors.magnitude
         *ureg('angstrom')),
         DimensionalityError),
        ('temperature',
         300*ureg('kg'),
         DimensionalityError),
        ('temperature',
         300,
         TypeError)]

    @pytest.fixture(params=faulty_elements)
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        expected_sf = get_expected_sf(
            'quartz', 'quartz_0K_structure_factor.json')
        # Inject the faulty value and get a tuple of constructor arguments
        args, kwargs = expected_sf.to_constructor_args(
            **{faulty_arg: faulty_value})
        return args, kwargs, expected_exception

    def test_faulty_creation(self, inject_faulty_elements):
        faulty_args, faulty_kwargs, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            StructureFactor(*faulty_args, **faulty_kwargs)


@pytest.mark.unit
class TestStructureFactorSerialisation:

    @pytest.mark.parametrize('sf', [
        get_sf('quartz', 'quartz_0K_structure_factor.json'),
        get_sf('CaHgO2', 'CaHgO2_300K_structure_factor.json')])
    def test_serialise_to_json_file(self, sf, tmpdir):
        output_file = str(tmpdir.join('tmp.test'))
        sf.to_json_file(output_file)
        check_json_metadata(output_file, 'StructureFactor')
        deserialised_sf = StructureFactor.from_json_file(output_file)
        check_structure_factor(sf, deserialised_sf)

    @pytest.fixture(params=[
        ('quartz', 'quartz_0K_structure_factor.json'),
        ('CaHgO2', 'CaHgO2_300K_structure_factor.json')])
    def serialise_to_dict(self, request):
        material, json_file = request.param
        sf = get_sf(material, json_file)
        expected_sf = get_expected_sf(material, json_file)
        # Convert to dict, then back to object to test
        sf_dict = sf.to_dict()
        sf_from_dict = StructureFactor.from_dict(sf_dict)
        return sf_from_dict, expected_sf

    def test_serialise_to_dict(self, serialise_to_dict):
        sf, expected_sf = serialise_to_dict
        check_structure_factor(sf, expected_sf, sum_sf=False)

    @pytest.mark.parametrize('material, sf_json, qpt_freqs_json', [
        ('quartz',
         'quartz_666_300K_structure_factor.json',
         'quartz_666_qpoint_frequencies.json'),
        ('CaHgO2',
         'CaHgO2_300K_structure_factor.json',
         'CaHgO2_from_phonopy_qpoint_frequencies.json')])
    def test_to_qpoint_frequencies(
            self, material, sf_json, qpt_freqs_json):
        sf = get_sf(material, sf_json)
        qpt_freqs = sf.to_qpoint_frequencies()
        expected_qpt_freqs = get_expected_qpt_freqs(material, qpt_freqs_json)
        check_qpt_freqs(qpt_freqs, expected_qpt_freqs)


@pytest.mark.unit
class TestStructureFactorUnitConversion:

    @pytest.mark.parametrize('material, json_file, attr, unit_val', [
        ('quartz', 'quartz_0K_structure_factor.json',
         'frequencies', '1/cm'),
        ('quartz', 'quartz_0K_structure_factor.json',
         'structure_factors', 'bohr**2'),
        ('quartz', 'quartz_0K_structure_factor.json',
         'temperature', 'celsius')])
    def test_correct_unit_conversion(self, material, json_file, attr,
                                     unit_val):
        sf = get_sf(material, json_file)
        check_unit_conversion(sf, attr, unit_val)

    @pytest.mark.parametrize('material, json_file, unit_attr, unit_val, err', [
        ('quartz', 'quartz_0K_structure_factor.json',
         'frequencies_unit', '1/cm**2', ValueError),
        ('quartz', 'quartz_0K_structure_factor.json',
         'structure_factors_unit', '1/bohr', ValueError),
        ('quartz', 'quartz_0K_structure_factor.json',
         'temperature_unit', 'kg', ValueError)])
    def test_incorrect_unit_conversion(self, material, json_file, unit_attr,
                                       unit_val, err):
        sf = get_sf(material, json_file)
        with pytest.raises(err):
            setattr(sf, unit_attr, unit_val)


@pytest.mark.unit
class TestStructureFactorCalculateSqwMap:

    @pytest.mark.parametrize(
        'material, sf_json, expected_sqw_json, ebins, kwargs', [
            ('quartz', 'quartz_bandstructure_structure_factor.json',
             'quartz_bandstructure_sqw.json',
             np.arange(0,155)*ureg('meV'), {'calc_bose': False}),
            ('quartz', 'quartz_bandstructure_structure_factor.json',
             'quartz_bandstructure_100K_bose_no_dw_sqw.json',
             np.arange(0,155)*ureg('meV'),
             {'calc_bose': True, 'temperature': 100*ureg('K')}),
            ('CaHgO2', 'CaHgO2_300K_structure_factor.json',
             'CaHgO2_300K_bose_sqw.json',
             np.arange(0,95,0.4)*ureg('meV'), {'calc_bose': True}),
            ('CaHgO2', 'CaHgO2_300K_structure_factor.json',
             'CaHgO2_300K_bose_negative_e_sqw.json',
             np.arange(-95,95,0.4)*ureg('meV'), {'calc_bose': True}),
            ('LZO', 'La2Zr2O7_cut_structure_factor.json',
             'La2Zr2O7_cut_sqw.json',
             np.arange(0,95)*ureg('meV'), {'calc_bose': 'False'})])
    def test_calculate_sqw_map(self, material, sf_json, expected_sqw_json,
                               ebins, kwargs):
        sf = get_sf(material, sf_json)
        sqw = sf.calculate_sqw_map(ebins, **kwargs)
        expected_sqw = get_expected_spectrum2d(expected_sqw_json)
        check_spectrum2d(sqw, expected_sqw)

    @pytest.mark.parametrize(
        'material, sf_json, ebins, kwargs, err', [
            ('CaHgO2', 'CaHgO2_300K_structure_factor.json',
             np.arange(0,95,0.4)*ureg('meV'),
             {'calc_bose': True, 'temperature': 100*ureg('K')}, ValueError)])
    def test_inconsistent_temperatures_raises_error(
            self, material, sf_json, ebins, kwargs, err):
        sf = get_sf(material, sf_json)
        with pytest.raises(err):
            sf.calculate_sqw_map(ebins, **kwargs)

@pytest.mark.unit
class TestStructureFactorCalculate1dAverage:

    @pytest.mark.parametrize(
        'material, sf_json, expected_1d_json, ebins, kwargs', [
            ('quartz', 'quartz_666_300K_structure_factor.json',
             'quartz_666_300K_sf_1d_average.json',
             np.arange(0,156)*ureg('meV'), {}),
            ('quartz', 'quartz_666_300K_structure_factor_noweights.json',
             'quartz_666_300K_sf_1d_average.json',
             np.arange(0,156)*ureg('meV'),
             {'weights': np.load(os.path.join(
                 get_sf_dir('quartz'),
                 'quartz_666_weights.npy'))}),
            ('quartz', 'quartz_666_300K_structure_factor_noweights.json',
             'quartz_666_300K_sf_1d_average_noweights.json',
             np.arange(0,156)*ureg('meV'), {})])
    def test_calculate_1d_average(self, material, sf_json, expected_1d_json,
                                  ebins, kwargs):
        sf = get_sf(material, sf_json)
        sw = sf.calculate_1d_average(ebins, **kwargs)
        expected_sw = get_expected_spectrum1d(expected_1d_json)
        check_spectrum1d(sw, expected_sw)

@pytest.mark.unit
class TestStructureFactorGetDispersion:

    @pytest.mark.parametrize(
        'material, sf_json, expected_dispersion_json', [
            ('quartz', 'quartz_bandstructure_structure_factor.json',
             'quartz_bandstructure_dispersion.json'),
            ('LZO', 'La2Zr2O7_cut_structure_factor.json',
             'LZO_cut_dispersion.json')
        ])
    def test_get_dispersion(
            self, material, sf_json, expected_dispersion_json):
        sf = get_sf(material, sf_json)
        disp = sf.get_dispersion()
        expected_disp = get_expected_spectrum1dcollection(
            expected_dispersion_json)
        check_spectrum1dcollection(disp, expected_disp)

@pytest.mark.unit
class TestStructureFactorCalculateDos:

    @pytest.mark.parametrize(
        'material, sf_json, expected_dos_json, ebins', [
            ('quartz', 'quartz_666_300K_structure_factor.json',
             'quartz_666_dos.json', np.arange(0, 155, 0.5)*ureg('meV'))])
    def test_calculate_dos(
            self, material, sf_json, expected_dos_json, ebins):
        sf = get_sf(material, sf_json)
        dos = sf.calculate_dos(ebins)
        expected_dos = get_expected_spectrum1d(expected_dos_json)
        check_spectrum1d(dos, expected_dos)

    def test_calculate_dos_with_0_inv_cm_bin_doesnt_raise_runtime_warn(self):
        sf = get_sf(
            'quartz', 'quartz_666_300K_structure_factor.json')
        ebins = np.arange(0, 1300, 4)*ureg('1/cm')
        with pytest.warns(None) as warn_record:
            dos = sf.calculate_dos(ebins)
        assert len(warn_record) == 0
