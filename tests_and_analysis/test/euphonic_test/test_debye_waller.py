import json

import numpy as np
import numpy.testing as npt
from pint import DimensionalityError
import pytest
from pytest_lazy_fixtures import lf as lazy_fixture

from euphonic import Crystal, DebyeWaller, ureg
from tests_and_analysis.test.euphonic_test.test_crystal import (
    ExpectedCrystal,
    check_crystal,
    get_crystal,
)
from tests_and_analysis.test.utils import (
    check_json_metadata,
    check_property_setters,
    check_unit_conversion,
    get_data_path,
)

FLOAT64_EPS = np.finfo(np.float64).eps


class ExpectedDebyeWaller:

    def __init__(self, debye_waller_json_file: str):
        with open(debye_waller_json_file) as fd:
            self.data = json.load(fd)

    @property
    def crystal(self):
        return ExpectedCrystal(self.data['crystal'])

    @property
    def debye_waller(self):
        return np.array(self.data['debye_waller'])*ureg(
            self.data['debye_waller_unit'])

    @property
    def temperature(self):
        return np.array(self.data['temperature'])*ureg(
            self.data['temperature_unit'])

    def to_dict(self):
        return {
            'crystal': self.crystal.to_dict(),
            'debye_waller': self.debye_waller.magnitude,
            'debye_waller_unit': str(self.debye_waller.units),
            'temperature': self.temperature.magnitude,
            'temperature_unit': str(self.temperature.units)}

    def to_constructor_args(self, crystal=None, debye_waller=None,
                            temperature=None):
        if crystal is None:
            crystal = Crystal(*self.crystal.to_constructor_args())
        if debye_waller is None:
            debye_waller = self.debye_waller
        if temperature is None:
            temperature = self.temperature

        return crystal, debye_waller, temperature


def get_dw_path(*subpaths):
    return get_data_path('debye_waller', *subpaths)


def get_dw(material, json_file):
    return DebyeWaller.from_json_file(get_dw_path(material, json_file))


def get_expected_dw(material, json_file):
    return ExpectedDebyeWaller(get_dw_path(material, json_file))


def check_debye_waller(
        debye_waller, expected_debye_waller,
        dw_atol=FLOAT64_EPS,
        dw_rtol=1e-7):
    check_crystal(debye_waller.crystal,
                  expected_debye_waller.crystal)

    assert (str(debye_waller.temperature.units)
            == str(expected_debye_waller.temperature.units))
    npt.assert_almost_equal(debye_waller.temperature.magnitude,
                            expected_debye_waller.temperature.magnitude)

    assert (debye_waller.debye_waller.units
            == expected_debye_waller.debye_waller.units)
    npt.assert_allclose(debye_waller.debye_waller.magnitude,
                        expected_debye_waller.debye_waller.magnitude,
                        atol=dw_atol,
                        rtol=dw_rtol)

class TestDebyeWallerCreation:

    @pytest.fixture(params=[
        get_expected_dw('quartz', 'quartz_666_0K_debye_waller.json'),
        get_expected_dw('Si2-sc-skew',
                        'Si2-sc-skew_666_300K_debye_waller.json'),
        get_expected_dw('CaHgO2', 'CaHgO2_666_300K_debye_waller.json')])
    def create_from_constructor(self, request):
        expected_dw = request.param
        dw = DebyeWaller(*expected_dw.to_constructor_args())
        return dw, expected_dw

    @pytest.fixture(params=[
        get_expected_dw('quartz', 'quartz_666_0K_debye_waller.json'),
        get_expected_dw('Si2-sc-skew',
                        'Si2-sc-skew_666_300K_debye_waller.json'),
        get_expected_dw('CaHgO2', 'CaHgO2_666_300K_debye_waller.json')])
    def create_from_dict(self, request):
        expected_dw = request.param
        d = expected_dw.to_dict()
        dw = DebyeWaller.from_dict(d)
        return dw, expected_dw

    @pytest.fixture(params=[
        ('quartz', 'quartz_666_0K_debye_waller.json'),
        ('CaHgO2', 'CaHgO2_666_300K_debye_waller.json')])
    def create_from_json_file(self, request):
        material, json_file = request.param
        dw = get_dw(material, json_file)
        expected_dw = get_expected_dw(material, json_file)
        return dw, expected_dw

    @pytest.mark.parametrize('dw_creator', [
        lazy_fixture('create_from_constructor'),
        lazy_fixture('create_from_json_file'),
        lazy_fixture('create_from_dict'),
    ])
    def test_create(self, dw_creator):
        dw, expected_dw = dw_creator
        check_debye_waller(dw, expected_dw)

    faulty_elements = [
        ('crystal',
         get_crystal('LZO'),
         ValueError),
        ('temperature',
         300*ureg('kg'),
         DimensionalityError),
        ('temperature',
         300,
         TypeError),
        ('debye_waller',
         get_expected_dw(
             'quartz', 'quartz_666_0K_debye_waller.json').debye_waller[:3],
         ValueError),
        ('debye_waller',
         (get_expected_dw(
             'quartz',
             'quartz_666_0K_debye_waller.json').debye_waller.magnitude
          *ureg('K')),
         DimensionalityError),
    ]

    @pytest.fixture(params=faulty_elements)
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        dw = get_expected_dw('quartz', 'quartz_666_0K_debye_waller.json')
        # Inject the faulty value and get a tuple of constructor arguments
        args = dw.to_constructor_args(**{faulty_arg: faulty_value})
        return args, expected_exception

    def test_faulty_creation(self, inject_faulty_elements):
        faulty_args, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            DebyeWaller(*faulty_args)


class TestDebyeWallerSerialisation:

    @pytest.mark.parametrize('dw', [
        get_dw('quartz', 'quartz_666_0K_debye_waller.json'),
        get_dw('Si2-sc-skew', 'Si2-sc-skew_666_300K_debye_waller.json'),
        get_dw('CaHgO2', 'CaHgO2_666_300K_debye_waller.json')])
    def test_serialise_to_json_file(self, dw, tmpdir):
        output_file = str(tmpdir.join('tmp.test'))
        dw.to_json_file(output_file)
        check_json_metadata(output_file, 'DebyeWaller')
        deserialised_dw = DebyeWaller.from_json_file(output_file)
        check_debye_waller(dw, deserialised_dw)

    @pytest.fixture(params=[
        ('quartz', 'quartz_666_0K_debye_waller.json'),
        ('Si2-sc-skew', 'Si2-sc-skew_666_300K_debye_waller.json'),
        ('CaHgO2', 'CaHgO2_666_300K_debye_waller.json'),
    ])
    def serialise_to_dict(self, request):
        material, json_file = request.param
        dw = get_dw(material, json_file)
        expected_dw = get_expected_dw(material, json_file)
        # Convert to dict, then back to object to test
        dw_dict = dw.to_dict()
        dw_from_dict = DebyeWaller.from_dict(dw_dict)
        return dw_from_dict, expected_dw

    def test_serialise_to_dict(self, serialise_to_dict):
        dw, expected_dw = serialise_to_dict
        check_debye_waller(dw, expected_dw)


class TestDebyeWallerUnitConversion:

    @pytest.mark.parametrize('material, json_file, attr, unit_val', [
        ('quartz', 'quartz_666_0K_debye_waller.json',
         'temperature', 'celsius'),
        ('quartz', 'quartz_666_0K_debye_waller.json',
         'debye_waller', 'bohr**2')])
    def test_correct_unit_conversion(self, material, json_file, attr,
                                     unit_val):
        dw = get_dw(material, json_file)
        check_unit_conversion(dw, attr, unit_val)

    @pytest.mark.parametrize('material, json_file, unit_attr, unit_val, err', [
        ('quartz', 'quartz_666_0K_debye_waller.json',
         'temperature_unit', 'angstrom', ValueError),
        ('quartz', 'quartz_666_0K_debye_waller.json',
         'debye_waller_unit', 'kg**2', ValueError)])
    def test_incorrect_unit_conversion(self, material, json_file, unit_attr,
                                       unit_val, err):
        dw = get_dw(material, json_file)
        with pytest.raises(err):
            setattr(dw, unit_attr, unit_val)


class TestDebyeWallerSetters:

    @pytest.mark.parametrize('material, json_file, attr, unit, scale', [
        ('CaHgO2', 'CaHgO2_666_300K_debye_waller.json',
         'debye_waller', 'angstrom**2', 2.),
        ('CaHgO2', 'CaHgO2_666_300K_debye_waller.json',
         'debye_waller', 'mbarn', 3.),
        ('CaHgO2', 'CaHgO2_666_300K_debye_waller.json',
         'temperature', 'K', 2.),
        ])
    def test_setter_correct_units(self, material, json_file, attr,
                                  unit, scale):
        dw = get_dw(material, json_file)
        check_property_setters(dw, attr, unit, scale)

    @pytest.mark.parametrize('material, json_file, attr, unit, err', [
        ('CaHgO2', 'CaHgO2_666_300K_debye_waller.json',
         'debye_waller', '1/cm', ValueError),
        ('CaHgO2', 'CaHgO2_666_300K_debye_waller.json',
         'temperature', 'kg', ValueError)])
    def test_incorrect_unit_conversion(self, material, json_file, attr,
                                       unit, err):
        dw = get_dw(material, json_file)
        new_attr = getattr(dw, attr).magnitude*ureg(unit)
        with pytest.raises(err):
            setattr(dw, attr, new_attr)

