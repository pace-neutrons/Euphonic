import json
import os

import numpy as np
import numpy.testing as npt
import pytest
from pint import DimensionalityError

from euphonic import ureg, Quantity, Crystal
from tests_and_analysis.test.utils import (
    get_data_path, check_unit_conversion, check_json_metadata)


class ExpectedCrystal:

    def __init__(self, crystal_json):
        self.data = crystal_json

    @property
    def cell_vectors(self) -> Quantity:
        return np.array(self.data['cell_vectors'])*ureg(
            self.data['cell_vectors_unit'])

    @property
    def n_atoms(self) -> int:
        return self.data['n_atoms']

    @property
    def atom_r(self) -> np.array:
        if self.n_atoms == 0 and len(self.data['atom_r']) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        else:
            return np.array(self.data['atom_r'])

    @property
    def atom_type(self) -> np.array:
        return np.array(
            self.data['atom_type'],
            dtype='<U2')

    @property
    def atom_mass(self) -> Quantity:
        return np.array(self.data['atom_mass'])*ureg(
            self.data['atom_mass_unit'])

    def to_dict(self):
        return {
            'cell_vectors': self.cell_vectors.magnitude,
            'cell_vectors_unit': str(self.cell_vectors.units),
            'n_atoms': self.n_atoms,
            'atom_r': self.atom_r,
            'atom_type': self.atom_type,
            'atom_mass': self.atom_mass.magnitude,
            'atom_mass_unit': str(self.atom_mass.units)}

    def to_constructor_args(self, cell_vectors=None, atom_r=None,
                            atom_type=None, atom_mass=None):
        if cell_vectors is None:
            cell_vectors = self.cell_vectors
        if atom_r is None:
            atom_r = self.atom_r
        if atom_type is None:
            atom_type = self.atom_type
        if atom_mass is None:
            atom_mass = self.atom_mass
        return (cell_vectors, atom_r, atom_type, atom_mass)


def get_filepath(filename):
    return os.path.join(get_data_path(), 'crystal', filename)


def get_json_file(crystal_name):
    return get_filepath(f'crystal_{crystal_name}.json')


def get_expected_crystal(crystal_name):
    return ExpectedCrystal(json.load(open(get_json_file(crystal_name))))


def get_crystal_from_json_file(filepath):
    return Crystal.from_json_file(filepath)


def get_crystal(crystal):
    return get_crystal_from_json_file(
        get_json_file(crystal))


def check_crystal(crystal, expected_crystal):
    assert crystal.n_atoms == expected_crystal.n_atoms

    assert (crystal.cell_vectors.units
            == expected_crystal.cell_vectors.units)
    npt.assert_allclose(
        crystal.cell_vectors.magnitude,
        expected_crystal.cell_vectors.magnitude,
        atol=np.finfo(np.float64).eps)

    npt.assert_allclose(
        crystal.atom_r,
        expected_crystal.atom_r,
        atol=np.finfo(np.float64).eps)

    npt.assert_equal(
        crystal.atom_type,
        expected_crystal.atom_type)

    assert (crystal.atom_mass.units
            == expected_crystal.atom_mass.units)
    npt.assert_allclose(
        crystal.atom_mass.magnitude,
        expected_crystal.atom_mass.magnitude,
        atol=np.finfo(np.float64).eps)


@pytest.mark.unit
class TestCrystalCreation:

    @pytest.fixture(params=[get_expected_crystal('quartz'),
                            get_expected_crystal('quartz_cv_only'),
                            get_expected_crystal('LZO')])
    def create_from_constructor(self, request):
        expected_crystal = request.param
        crystal = Crystal(*expected_crystal.to_constructor_args())
        return crystal, expected_crystal

    @pytest.fixture(params=[get_expected_crystal('quartz'),
                            get_expected_crystal('quartz_cv_only'),
                            get_expected_crystal('LZO')])
    def create_from_dict(self, request):
        expected_crystal = request.param
        d = expected_crystal.to_dict()
        crystal = Crystal.from_dict(d)
        return crystal, expected_crystal

    @pytest.fixture(params=[
        (get_json_file('quartz'),
         get_expected_crystal('quartz')),
        (get_json_file('quartz_cv_only'),
         get_expected_crystal('quartz_cv_only')),
        (get_json_file('LZO'),
         get_expected_crystal('LZO'))
    ])
    def create_from_json_file(self, request):
        filename, expected_crystal = request.param
        return get_crystal_from_json_file(filename), expected_crystal

    @pytest.fixture(params=[
        (Quantity(np.array([[2.426176, -4.20226, 0.000000],
                            [2.426176,  4.20226, 0.000000],
                            [0.000000,  0.00000, 5.350304]]), 'angstrom'),
         get_expected_crystal('quartz_cv_only'))])
    def create_from_cell_vectors(self, request):
        cell_vectors, expected_crystal = request.param
        return Crystal.from_cell_vectors(cell_vectors), expected_crystal

    @pytest.mark.parametrize('crystal_creator', [
        pytest.lazy_fixture('create_from_constructor'),
        pytest.lazy_fixture('create_from_json_file'),
        pytest.lazy_fixture('create_from_dict'),
        pytest.lazy_fixture('create_from_cell_vectors')
    ])
    def test_create(self, crystal_creator):
        crystal, expected_crystal = crystal_creator
        check_crystal(crystal, expected_crystal)

    faulty_elements = [
        ('cell_vectors',
         np.array([[1.23, 2.45, 0.0],
                   [3.45, 5.66, 7.22]])*ureg('angstrom'),
         ValueError),
        ('cell_vectors',
         get_expected_crystal('quartz').cell_vectors.magnitude*ureg('kg'),
         DimensionalityError),
        ('cell_vectors',
         get_expected_crystal('quartz').cell_vectors.magnitude*ureg(''),
         DimensionalityError),
        ('atom_r',
         np.array([[0.125, 0.125, 0.125],
                   [0.875, 0.875, 0.875]]),
         ValueError),
        ('atom_mass',
         np.array(
             [15.999399987607514, 15.999399987607514, 91.2239999293416])*ureg(
                 'amu'),
         ValueError),
        ('atom_mass',
         get_expected_crystal('quartz').atom_mass.magnitude*ureg('angstrom'),
         DimensionalityError),
        ('atom_mass',
         get_expected_crystal('quartz').atom_mass.magnitude*ureg(''),
         DimensionalityError),
        ('atom_type', np.array(['O', 'Zr', 'La']), ValueError),
    ]

    @pytest.fixture(params=faulty_elements)
    def inject_faulty_elements(self, request):
        faulty_arg, faulty_value, expected_exception = request.param
        crystal = get_expected_crystal('quartz')
        # Inject the faulty value and get a tuple of constructor arguments
        args = crystal.to_constructor_args(**{faulty_arg: faulty_value})
        return args, expected_exception

    def test_faulty_creation(self, inject_faulty_elements):
        faulty_args, expected_exception = inject_faulty_elements
        with pytest.raises(expected_exception):
            Crystal(*faulty_args)


@pytest.mark.unit
class TestCrystalSerialisation:

    @pytest.mark.parametrize('crystal', [
        get_crystal('quartz'),
        get_crystal('quartz_cv_only'),
        get_crystal('LZO')])
    def test_serialise_to_json_file(self, crystal, tmpdir):
        # Serialise
        output_file = str(tmpdir.join('tmp.test'))
        crystal.to_json_file(output_file)
        # Test file metadata
        check_json_metadata(output_file, 'Crystal')
        # Deserialise
        deserialised_crystal = Crystal.from_json_file(output_file)
        return check_crystal(crystal, deserialised_crystal)

    @pytest.fixture(params=[
        (get_crystal('quartz'),
         get_expected_crystal('quartz')),
        (get_crystal('quartz_cv_only'),
         get_expected_crystal('quartz_cv_only')),
        (get_crystal('LZO'),
         get_expected_crystal('LZO'))
    ])
    def serialise_to_dict(self, request):
        crystal, expected_crystal = request.param
        # Convert to dict, then back to object to test
        crystal_dict = crystal.to_dict()
        crystal_from_dict = Crystal.from_dict(crystal_dict)
        return crystal_from_dict, expected_crystal

    def test_serialise_to_dict(self, serialise_to_dict):
        crystal, expected_crystal = serialise_to_dict
        check_crystal(crystal, expected_crystal)


@pytest.mark.unit
class TestCrystalUnitConversion:

    @pytest.mark.parametrize('material, attr, unit_val', [
        ('quartz', 'cell_vectors', 'bohr'),
        ('quartz', 'atom_mass', 'kg')])
    def test_correct_unit_conversion(self, material, attr,
                                     unit_val):
        crystal = get_crystal(material)
        check_unit_conversion(crystal, attr, unit_val)

    @pytest.mark.parametrize('material, unit_attr, unit_val, err', [
        ('quartz', 'cell_vectors_unit', 'kg', ValueError),
        ('quartz', 'atom_mass_unit', 'bohr', ValueError)])
    def test_incorrect_unit_conversion(self, material, unit_attr,
                                       unit_val, err):
        crystal = get_crystal(material)
        with pytest.raises(err):
            setattr(crystal, unit_attr, unit_val)


@pytest.mark.unit
class TestCrystalMethods:

    quartz_reciprocal_cell = np.array([
        [1.29487418, -0.74759597, 0.],
        [1.29487418, 0.74759597, 0.],
        [0., 0., 1.17436043]
    ])*ureg('1/angstrom')

    lzo_reciprocal_cell = np.array([
        [8.28488599e-01, 0.00000000e+00, -5.85829906e-01],
        [-2.01146673e-33, 8.28488599e-01, 5.85829906e-01],
        [2.01146673e-33, -8.28488599e-01, 5.85829906e-01]
    ])*ureg('1/angstrom')

    @pytest.mark.parametrize('crystal,expected_recip', [
        (get_crystal('quartz'), quartz_reciprocal_cell),
        (get_crystal('LZO'), lzo_reciprocal_cell)
    ])
    def test_reciprocal_cell(self, crystal, expected_recip):
        recip = crystal.reciprocal_cell()
        npt.assert_allclose(
            recip.to('1/angstrom').magnitude,
            expected_recip.to('1/angstrom').magnitude
        )

    quartz_cell_volume = 109.09721804482547*ureg('angstrom**3')

    lzo_cell_volume = 308.4359549515967*ureg('angstrom**3')

    @pytest.mark.parametrize('crystal,expected_vol', [
        (get_crystal('quartz'), quartz_cell_volume),
        (get_crystal('LZO'), lzo_cell_volume)
    ])
    def test_cell_volume(self, crystal, expected_vol):
        vol = crystal.cell_volume()
        npt.assert_allclose(
            vol.to('angstrom**3').magnitude,
            expected_vol.to('angstrom**3').magnitude
        )

    quartz_spglib_cell = ([[2.426176, -4.20226, 0.000000],
                           [2.426176,  4.20226, 0.000000],
                           [0.000000,  0.00000, 5.350304]],
                          [[0.411708, 0.275682, 0.279408],
                           [0.724318, 0.136026, 0.946074],
                           [0.863974, 0.588292, 0.612741],
                           [0.136026, 0.724318, 0.053926],
                           [0.588292, 0.863974, 0.387259],
                           [0.275682, 0.411708, 0.720592],
                           [0.464960, 0.000000, 0.166667],
                           [0.000000, 0.464960, 0.833333],
                           [0.535040, 0.535040, 0.500000]],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1])

    quartz_cv_only_spglib_cell = ([[2.426176, -4.20226, 0.000000],
                                   [2.426176,  4.20226, 0.000000],
                                   [0.000000,  0.00000, 5.350304]],
                                   [],
                                   [])

    lzo_spglib_cell = ([[7.58391282,  0.00000000, 0.00000000],
                        [3.79195641,  3.79195641, 5.36263619],
                        [3.79195641, -3.79195641, 5.36263619]],
                       [[0.12500000, 0.12500000, 0.12500000],
                        [0.87500000, 0.87500000, 0.87500000],
                        [0.41861943, 0.41861943, 0.83138057],
                        [0.83138057, 0.83138057, 0.41861943],
                        [0.41861943, 0.83138057, 0.41861943],
                        [0.83138057, 0.41861943, 0.83138057],
                        [0.83138057, 0.41861943, 0.41861943],
                        [0.41861943, 0.83138057, 0.83138057],
                        [0.16861943, 0.58138057, 0.16861943],
                        [0.58138057, 0.16861943, 0.58138057],
                        [0.16861943, 0.16861943, 0.58138057],
                        [0.58138057, 0.58138057, 0.16861943],
                        [0.16861943, 0.58138057, 0.58138057],
                        [0.58138057, 0.16861943, 0.16861943],
                        [0.50000000, 0.50000000, 0.50000000],
                        [0.00000000, 0.50000000, 0.50000000],
                        [0.50000000, 0.00000000, 0.50000000],
                        [0.50000000, 0.50000000, 0.00000000],
                        [0.00000000, 0.00000000, 0.00000000],
                        [0.50000000, 0.00000000, 0.00000000],
                        [0.00000000, 0.50000000, 0.00000000],
                        [0.00000000, 0.00000000, 0.50000000]],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        2, 2, 2, 2,
                        0, 0, 0, 0])

    @pytest.mark.parametrize('crystal,expected_spglib_cell', [
        (get_crystal('quartz'), quartz_spglib_cell),
        (get_crystal('quartz_cv_only'), quartz_cv_only_spglib_cell),
        (get_crystal('LZO'), lzo_spglib_cell)])
    def test_to_spglib_cell(self, crystal, expected_spglib_cell):
        spglib_cell = crystal.to_spglib_cell()
        assert type(spglib_cell) is tuple
        npt.assert_allclose(spglib_cell[0], expected_spglib_cell[0],
                            atol=np.finfo(np.float64).eps)
        npt.assert_allclose(spglib_cell[1], expected_spglib_cell[1],
                            atol=np.finfo(np.float64).eps)
        assert spglib_cell[2] == expected_spglib_cell[2]

    @pytest.mark.parametrize('crystal,kwargs,expected', [
        (get_crystal('quartz'), {}, (15, 15, 12)),
        (get_crystal('quartz'), {'spacing': 0.05 * ureg('1/angstrom')},
         (30, 30, 24)),
        (get_crystal('quartz'), {'spacing': 0.05 * ureg('1/bohr')},
         (16, 16, 13)),
        (get_crystal('LZO'), {'spacing': 0.5 * ureg('1/angstrom')},
         (3, 3, 3))])
    def test_get_mp_grid_spec(self, crystal, kwargs, expected):
        assert crystal.get_mp_grid_spec(**kwargs) == expected
